#!/usr/bin/env python3
"""
Sibyl Smoke Test — Quick GPU probe before committing to the full 48h job.

Tests:
  1. smtTrainer.py can train a model (2 epochs, ~200 benchmarks)
  2. inference.py can produce valid predictions from that model
  3. Predictions are valid config names from the portfolio

Exit code 0 = PASS, 1 = FAIL.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ============================================================================
# Paths (all absolute)
# ============================================================================

PROJECT_DIR = Path("/dcs/23/u5573765/cs351/eval-sibyl")
SIBYL_DIR = Path("/dcs/large/u5573765/artifacts/sibyl")
SIBYL_TRAINER = SIBYL_DIR / "src" / "networks" / "smtTrainer.py"
SIBYL_INFERENCE = SIBYL_DIR / "src" / "networks" / "inference.py"
SIBYL_NETWORKS_DIR = SIBYL_DIR / "src" / "networks"

RESULTS_DIR = PROJECT_DIR / "sibyl_results"
LABELS_PATH = RESULTS_DIR / "labels_QF_LIA.json"  # Use QF_LIA (3000 benchmarks)
DATA_DIR = RESULTS_DIR / "workspace" / "QF_LIA"
PORTFOLIO_PATH = RESULTS_DIR / "portfolio.txt"

CONFIG_ORDER = [
    "z3_baseline_default",
    "z3_baseline_qflia_case_split",
    "z3_baseline_qfbv_sat_euf_sat",
    "cvc5_baseline_default",
    "cvc5_qfnra_05_nl_cov",
    "cvc5_qfnra_08_decision_just",
]

# Smoke test sizing
N_TRAIN = 180  # train samples (needs >50 for upstream round() bug)
N_TEST = 20    # test samples for inference
EPOCHS = 2     # minimal epochs
TIME_STEPS = 2
POOL_TYPE = "attention"

SEED = 12345


def banner(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def fail(msg):
    print(f"\n*** SMOKE TEST FAILED: {msg} ***\n")
    sys.exit(1)


def main():
    banner("Sibyl Smoke Test")
    t0 = time.time()

    # ---- Preflight checks ----
    for p, desc in [
        (SIBYL_TRAINER, "smtTrainer.py"),
        (SIBYL_INFERENCE, "inference.py"),
        (LABELS_PATH, "QF_LIA labels"),
        (DATA_DIR, "QF_LIA workspace"),
        (PORTFOLIO_PATH, "portfolio.txt"),
    ]:
        if not p.exists():
            fail(f"Missing: {p} ({desc})")
    print("  All paths OK")

    # Check GPU
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        if gpu_ok:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: No GPU detected — training will be slow")
    except ImportError:
        fail("PyTorch not importable")

    # ---- Sample benchmarks ----
    banner("Step 1: Sample benchmarks")

    with open(LABELS_PATH) as f:
        all_labels = json.load(f)

    all_keys = list(all_labels["train"].keys())

    # Filter to only benchmarks that have .npz files
    valid_keys = []
    for key in all_keys:
        npz = DATA_DIR / key.replace(".smt2", ".npz")
        if npz.exists():
            valid_keys.append(key)

    print(f"  {len(valid_keys)}/{len(all_keys)} benchmarks have .npz files")

    if len(valid_keys) < N_TRAIN + N_TEST:
        fail(f"Need {N_TRAIN + N_TEST} benchmarks, only {len(valid_keys)} available")

    random.seed(SEED)
    sample = random.sample(valid_keys, N_TRAIN + N_TEST)
    train_keys = sample[:N_TRAIN]
    test_keys = sample[N_TRAIN:]

    print(f"  Sampled: {N_TRAIN} train + {N_TEST} test")

    # ---- Create smoke labels JSON ----
    smoke_dir = PROJECT_DIR / "sibyl_results" / "_smoke_test"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True)

    smoke_labels = {
        "train": {k: all_labels["train"][k] for k in train_keys},
        "test": {k: all_labels["train"][k] for k in test_keys},
    }
    smoke_labels_path = smoke_dir / "smoke_labels.json"
    with open(smoke_labels_path, "w") as f:
        json.dump(smoke_labels, f)

    print(f"  Labels written: {smoke_labels_path}")

    # ---- Step 2: Train with smtTrainer.py ----
    banner("Step 2: Train (smtTrainer.py, 2 epochs)")

    train_cmd = [
        sys.executable, str(SIBYL_TRAINER),
        "--data", str(DATA_DIR.resolve()),
        "--labels", str(smoke_labels_path.resolve()),
        "-t", str(TIME_STEPS),
        "-e", str(EPOCHS),
        "--cross-valid", "0",
        "--pool-type", POOL_TYPE,
        "--data-weight", "best",
    ]
    if not gpu_ok:
        train_cmd.append("--cpu")

    print(f"  CMD: {' '.join(train_cmd[-10:])}")
    train_t0 = time.time()

    # Clean any leftover .pt from previous smoke tests in cwd
    for old_pt in SIBYL_NETWORKS_DIR.glob("*.pt"):
        old_pt.unlink()

    result = subprocess.run(
        train_cmd,
        cwd=str(SIBYL_NETWORKS_DIR),
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min max
    )

    train_elapsed = time.time() - train_t0

    # Show last 15 lines of output
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-15:]:
            print(f"    {line}")

    if result.returncode != 0:
        print("  STDERR:")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                print(f"    {line}")
        fail(f"smtTrainer.py exited with code {result.returncode}")

    print(f"\n  Training completed in {train_elapsed:.0f}s")

    # Find the .pt model
    pt_files = sorted(
        SIBYL_NETWORKS_DIR.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not pt_files:
        fail("No .pt model file produced by smtTrainer.py")

    model_path = pt_files[0]
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Model: {model_path.name} ({model_size_mb:.1f} MB)")

    # Move to smoke dir
    dest_model = smoke_dir / "smoke_model.pt"
    shutil.move(str(model_path), str(dest_model))

    # ---- Step 3: Inference with inference.py ----
    banner("Step 3: Inference (inference.py)")

    valid_predictions = 0
    invalid_predictions = 0
    errors = 0

    for i, key in enumerate(test_keys):
        npz_path = DATA_DIR / key.replace(".smt2", ".npz")
        if not npz_path.exists():
            errors += 1
            continue

        inf_cmd = [
            sys.executable, str(SIBYL_INFERENCE),
            "--model", str(dest_model.resolve()),
            "--query", str(npz_path.resolve()),
            "--portfolio", str(PORTFOLIO_PATH.resolve()),
            "-t", str(TIME_STEPS),
            "--pool-type", POOL_TYPE,
            "--cpu",
        ]

        try:
            inf_result = subprocess.run(
                inf_cmd,
                cwd=str(SIBYL_NETWORKS_DIR),
                capture_output=True,
                text=True,
                timeout=60,
            )

            pred_solver = None
            if inf_result.returncode == 0:
                for line in inf_result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("1:"):
                        pred_solver = line.split(":", 1)[1].strip()
                        break

            if pred_solver and pred_solver in CONFIG_ORDER:
                valid_predictions += 1
                if i < 3:  # Show first 3
                    print(f"    [{i+1}/{N_TEST}] {key[:50]}... -> {pred_solver}")
            else:
                invalid_predictions += 1
                if i < 3:
                    stderr_snip = inf_result.stderr[:100] if inf_result.stderr else ""
                    print(f"    [{i+1}/{N_TEST}] {key[:50]}... -> INVALID "
                          f"(got: {pred_solver!r}, rc={inf_result.returncode}, "
                          f"err={stderr_snip})")

        except subprocess.TimeoutExpired:
            errors += 1
        except Exception as e:
            errors += 1
            print(f"    [{i+1}/{N_TEST}] EXCEPTION: {e}")

    print(f"\n  Results: {valid_predictions} valid, "
          f"{invalid_predictions} invalid, {errors} errors "
          f"(out of {N_TEST})")

    # ---- Verdict ----
    banner("VERDICT")
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f}s")

    # We need at least 75% valid predictions to pass
    min_valid = int(N_TEST * 0.75)
    if valid_predictions >= min_valid:
        print(f"  PASS: {valid_predictions}/{N_TEST} valid predictions "
              f"(threshold: {min_valid})")

        # Check prediction diversity (not all the same solver)
        # Re-run a quick count
        print(f"  Smoke test artifacts saved to: {smoke_dir}")
        print(f"\n  >>> SMOKE TEST PASSED <<<")
        return 0
    else:
        fail(f"Only {valid_predictions}/{N_TEST} valid predictions "
             f"(need >= {min_valid})")
        return 1  # unreachable but explicit


if __name__ == "__main__":
    sys.exit(main())
