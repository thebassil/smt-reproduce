#!/usr/bin/env python3
"""
Sibyl Portfolio Trainer for suite_9k benchmarks (Portfolio 7, 6 configs)

Trains Sibyl GNN models per logic using k-fold cross-validation on the
suite_9k benchmark data.  Each held-out fold gets out-of-fold predictions,
giving honest predictions for all 9,000 instances.

Stages (can be run individually or all at once):
  1. extract  - Pull run data from DB, create labels JSON + portfolio file
  2. graphs   - Build graph representations (.npz) via Sibyl's graph-builder
  3. train    - Train per-logic GNN models (k-fold CV), generate predictions
  4. merge    - Merge per-logic decisions into combined CSV

Usage:
  python3 sibyl_portfolio_trainer.py                          # all stages
  python3 sibyl_portfolio_trainer.py --stage extract
  python3 sibyl_portfolio_trainer.py --stage graphs --cores 8
  python3 sibyl_portfolio_trainer.py --stage train --epochs 20
  python3 sibyl_portfolio_trainer.py --stage merge
"""

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
import time as _time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Portfolio 7: 6 solver-configs run against all 3 logics
PORTFOLIO_ID = 7
SUITE_NAME = "suite_9k"
TIMEOUT_S = 60

# Config order (must be consistent across labels, portfolio file, and predictions)
# These are the config NAMES from the configs table
CONFIG_ORDER = [
    "z3_baseline_default",           # config_id 114
    "z3_baseline_qflia_case_split",  # config_id 116
    "z3_baseline_qfbv_sat_euf_sat",  # config_id 118
    "cvc5_baseline_default",         # config_id 115
    "cvc5_qfnra_05_nl_cov",         # config_id 109
    "cvc5_qfnra_08_decision_just",  # config_id 112
]

LOGICS = ["QF_BV", "QF_LIA", "QF_NRA"]

SIBYL_DIR = Path("/dcs/large/u5573765/artifacts/sibyl")


# ============================================================================
# Database Functions
# ============================================================================

def load_run_data(db_path: str) -> Tuple[dict, int]:
    """
    Load all run data for portfolio 7 / suite_9k.

    Returns:
        data: {file_path: {logic, family, runs: {config_name: (status, runtime_ms)}}}
        timeout_s: int
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    row = cur.execute(
        "SELECT timeout_s FROM portfolios WHERE id = ?", (PORTFOLIO_ID,)
    ).fetchone()
    if not row:
        raise ValueError(f"Portfolio {PORTFOLIO_ID} not found")
    timeout_s = row[0]

    runs = cur.execute("""
        SELECT i.file_path, c.name AS config_name, r.status, r.runtime_ms,
               i.logic, i.family
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c  ON r.config_id = c.id
        WHERE r.portfolio_id = ? AND i.suite_name = ?
    """, (PORTFOLIO_ID, SUITE_NAME)).fetchall()
    conn.close()

    data = {}
    for file_path, config_name, status, runtime_ms, logic, family in runs:
        if file_path not in data:
            data[file_path] = {"logic": logic, "family": family or "unknown", "runs": {}}
        data[file_path]["runs"][config_name] = (status, runtime_ms)

    return data, timeout_s


# ============================================================================
# Stage 1: Extract data, create labels + portfolio
# ============================================================================

def resolve_benchmark_path(file_path: str, base_dir: str) -> str:
    """Resolve a DB file_path to an absolute path."""
    abs_path = os.path.join(base_dir, file_path)
    if os.path.exists(abs_path):
        return os.path.abspath(abs_path)
    # Try without base_dir (already absolute)
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    return None


def create_portfolio_file(output_path: Path):
    """Create portfolio file listing all 6 config names."""
    with open(output_path, "w") as f:
        for cfg in CONFIG_ORDER:
            f.write(f"{cfg}\n")
    print(f"  Created portfolio file: {output_path} ({len(CONFIG_ORDER)} configs)")


def create_labels_and_symlinks(
    data: dict,
    timeout_s: int,
    output_dir: Path,
    base_dir: str,
    logic: str,
) -> Tuple[Path, Path, int]:
    """
    Create labels JSON and symlink directory for a specific logic.

    Labels key format: "family/filename.smt2" (relative to logic dir).
    Symlinks preserve family subdirectory structure.

    Returns: (labels_path, data_dir, num_instances)
    """
    data_dir = output_dir / "workspace" / logic
    data_dir.mkdir(parents=True, exist_ok=True)

    labels = {"train": {}, "test": {}}
    linked = 0
    skipped = 0

    for file_path, bdata in data.items():
        if bdata["logic"] != logic:
            continue

        # Extract relative key: family/filename.smt2
        # file_path is like: data/suite_9k/QF_BV/family/filename.smt2
        parts = file_path.split("/")
        # Find the logic part and take everything after
        try:
            logic_idx = parts.index(logic)
            relative_key = "/".join(parts[logic_idx + 1:])
        except ValueError:
            relative_key = os.path.basename(file_path)

        # Create runtime vector in CONFIG_ORDER
        runtimes = []
        for cfg in CONFIG_ORDER:
            if cfg in bdata["runs"]:
                status, runtime_ms = bdata["runs"][cfg]
                if status in ("sat", "unsat"):
                    runtimes.append(runtime_ms / 1000.0)
                else:
                    runtimes.append(float(timeout_s))
            else:
                runtimes.append(float(timeout_s))

        labels["train"][relative_key] = runtimes

        # Create symlink preserving directory structure
        abs_source = resolve_benchmark_path(file_path, base_dir)
        if abs_source:
            link_path = data_dir / relative_key
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if not link_path.exists():
                try:
                    os.symlink(abs_source, link_path)
                    linked += 1
                except OSError:
                    skipped += 1
        else:
            skipped += 1

    labels_path = output_dir / f"labels_{logic}.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f)

    print(f"  {logic}: {len(labels['train'])} instances, "
          f"{linked} symlinked, {skipped} skipped")
    print(f"    Labels: {labels_path}")
    print(f"    Data dir: {data_dir}")

    return labels_path, data_dir, len(labels["train"])


def stage_extract(db_path: str, output_dir: Path, base_dir: str):
    """Stage 1: Extract data and create all input files."""
    print("\n" + "=" * 70)
    print("STAGE 1: Extract data from database")
    print("=" * 70)

    print(f"\nLoading run data from {db_path}...")
    data, timeout_s = load_run_data(db_path)
    print(f"  Loaded {len(data)} benchmarks, timeout={timeout_s}s")

    # Count per logic
    by_logic = defaultdict(int)
    for bdata in data.values():
        by_logic[bdata["logic"]] += 1
    for logic, count in sorted(by_logic.items()):
        print(f"    {logic}: {count} instances")

    # Create portfolio file
    portfolio_path = output_dir / "portfolio.txt"
    create_portfolio_file(portfolio_path)

    # Create per-logic labels + symlinks
    print("\nCreating labels and symlinks per logic...")
    logic_info = {}
    for logic in LOGICS:
        labels_path, data_dir, n = create_labels_and_symlinks(
            data, timeout_s, output_dir, base_dir, logic
        )
        logic_info[logic] = {
            "labels_path": str(labels_path),
            "data_dir": str(data_dir),
            "num_instances": n,
        }

    # Save logic info for later stages
    info_path = output_dir / "logic_info.json"
    with open(info_path, "w") as f:
        json.dump(logic_info, f, indent=2)
    print(f"\nSaved logic info to {info_path}")


# ============================================================================
# Stage 2: Build graphs
# ============================================================================

def build_graphs_for_logic(data_dir: Path, num_cores: int = 4) -> Tuple[int, int]:
    """
    Build graph representations for all .smt2 files in data_dir.

    Returns: (built, failed) counts
    """
    # Find all .smt2 files that don't have corresponding .npz
    smt2_files = list(data_dir.rglob("*.smt2"))
    need_build = []
    for smt2 in smt2_files:
        npz = smt2.with_suffix(".npz")
        if not npz.exists():
            need_build.append(smt2)

    if not need_build:
        print(f"    All {len(smt2_files)} graphs already built")
        return len(smt2_files), 0

    print(f"    Building {len(need_build)}/{len(smt2_files)} graphs "
          f"({num_cores} cores)...")

    graph_builder = SIBYL_DIR / "src" / "data_handlers" / "graph-builder.py"

    built = 0
    failed = 0

    # Use absolute paths (do NOT resolve symlinks - NPZ must land in workspace)
    abs_need_build = [os.path.abspath(str(smt2)) for smt2 in need_build]
    abs_graph_builder = os.path.abspath(str(graph_builder))

    if num_cores > 1:
        # Use GNU parallel if available
        try:
            filelist = data_dir / "_build_list.txt"
            with open(filelist, "w") as f:
                for abs_path in abs_need_build:
                    f.write(f"{abs_path}\n")

            cmd = (
                f"cat {filelist} | parallel --bar -j {num_cores} "
                f"python3 {abs_graph_builder} {{}} 2>/dev/null"
            )
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=7200  # 2 hour timeout
            )

            # Count results
            for smt2 in need_build:
                npz = smt2.with_suffix(".npz")
                if npz.exists():
                    built += 1
                else:
                    failed += 1

            filelist.unlink(missing_ok=True)

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"    Parallel failed ({e}), falling back to sequential...")
            num_cores = 1

    if num_cores == 1:
        for i, smt2 in enumerate(need_build):
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{len(need_build)} "
                      f"(built={built}, failed={failed})")
            try:
                result = subprocess.run(
                    ["python3", abs_graph_builder, os.path.abspath(str(smt2))],
                    capture_output=True, text=True,
                    timeout=60
                )
                npz = smt2.with_suffix(".npz")
                if npz.exists():
                    built += 1
                else:
                    failed += 1
            except (subprocess.TimeoutExpired, Exception):
                failed += 1

    already = len(smt2_files) - len(need_build)
    print(f"    Done: {already} existing + {built} new = "
          f"{already + built} total, {failed} failed")
    return already + built, failed


def stage_graphs(output_dir: Path, num_cores: int = 4):
    """Stage 2: Build graph representations."""
    print("\n" + "=" * 70)
    print("STAGE 2: Build graph representations")
    print("=" * 70)

    info_path = output_dir / "logic_info.json"
    if not info_path.exists():
        print("ERROR: Run --stage extract first")
        return

    with open(info_path) as f:
        logic_info = json.load(f)

    for logic in LOGICS:
        if logic not in logic_info:
            print(f"  Skipping {logic} (not in logic_info)")
            continue

        data_dir = Path(logic_info[logic]["data_dir"])
        print(f"\n  {logic}: {data_dir}")
        build_graphs_for_logic(data_dir, num_cores)


# ============================================================================
# Stage 3: Train per-logic models (k-fold CV) and predict
# ============================================================================

MAX_GRAPH_NODES = 500_000  # Skip graphs larger than this to avoid GPU OOM

SIBYL_TRAINER = SIBYL_DIR / "src" / "networks" / "smtTrainer.py"
SIBYL_INFERENCE = SIBYL_DIR / "src" / "networks" / "inference.py"
SIBYL_NETWORKS_DIR = SIBYL_DIR / "src" / "networks"


def _run_sibyl_inference(args):
    """Run inference.py on a single benchmark. Used by thread pool."""
    npz_path, model_path, portfolio_path, time_steps, pool_type = args
    cmd = [
        sys.executable, str(SIBYL_INFERENCE),
        "--model", str(Path(model_path).resolve()),
        "--query", str(Path(npz_path).resolve()),
        "--portfolio", str(Path(portfolio_path).resolve()),
        "-t", str(time_steps),
        "--pool-type", pool_type,
        "--cpu",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(SIBYL_NETWORKS_DIR),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("1:"):
                    return line.split(":", 1)[1].strip()
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def train_and_predict_kfold(
    logic: str,
    labels_path: Path,
    data_dir: Path,
    output_dir: Path,
    fold_indices: list = None,
    k_fold: int = 5,
    epochs: int = 15,
    time_steps: int = 2,
    use_gpu: bool = True,
    pool_type: str = "attention",
    seed: int = 42,
) -> Dict[str, str]:
    """
    Train Sibyl GNN with k-fold CV for a single logic.

    Calls the REAL Sibyl CLI as subprocesses:
      - smtTrainer.py  for training  (the actual upstream binary)
      - inference.py   for prediction (the actual upstream binary)

    Our code only handles: fold generation, labels JSON creation, result collection.

    Returns: {relative_key: predicted_config_name} for all instances
    """
    from concurrent.futures import ThreadPoolExecutor

    # Load labels
    with open(labels_path) as f:
        labels = json.load(f)

    all_benchmarks = labels["train"]
    benchmark_names = list(all_benchmarks.keys())
    n_solvers = len(CONFIG_ORDER)

    # ---- Pre-filter: exclude graphs too large for GPU ----
    print(f"    Filtering graphs by size (max {MAX_GRAPH_NODES} nodes)...")
    t0 = _time.time()
    valid_names = []
    skipped_missing = 0
    skipped_large = 0
    for name in benchmark_names:
        npz_path = data_dir / (name[:-5] + ".npz")
        if not npz_path.exists():
            npz_path = data_dir / name.replace(".smt2", ".npz")
        if not npz_path.exists():
            skipped_missing += 1
            continue
        try:
            n_nodes = np.load(str(npz_path))["nodes"].shape[0]
            if n_nodes > MAX_GRAPH_NODES:
                skipped_large += 1
                continue
            valid_names.append(name)
        except Exception:
            skipped_missing += 1

    elapsed = _time.time() - t0
    print(f"    {len(valid_names)} valid benchmarks in {elapsed:.1f}s "
          f"({skipped_missing} missing, {skipped_large} too large)")
    print(f"    {len(valid_names)} benchmarks, {n_solvers} configs, "
          f"{k_fold}-fold CV")

    # ---- K-fold split ----
    valid_set = set(valid_names)
    all_predictions = {}

    if fold_indices is not None:
        folds = []
        for train_all, test_all in fold_indices:
            train_names = [n for n in train_all if n in valid_set]
            test_names = [n for n in test_all if n in valid_set]
            folds.append((train_names, test_names))
        print(f"    Using {len(folds)} pre-computed family-aware folds")
    else:
        from sklearn.model_selection import KFold
        np.random.seed(seed)
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
        np_names = np.array(valid_names)
        folds = []
        for train_idx, test_idx in kf.split(np_names):
            folds.append((list(np_names[train_idx]), list(np_names[test_idx])))
        print(f"    WARNING: Using simple KFold (no family awareness)")

    # Portfolio file for inference.py
    portfolio_path = output_dir / "portfolio.txt"
    if not portfolio_path.exists():
        with open(portfolio_path, "w") as f:
            for cfg in CONFIG_ORDER:
                f.write(f"{cfg}\n")

    for fold_idx, (train_names, test_names) in enumerate(folds):

        print(f"\n    Fold {fold_idx + 1}/{k_fold}: "
              f"train={len(train_names)}, test={len(test_names)}")

        if len(test_names) == 0:
            print(f"    Skipping fold {fold_idx+1} (no test instances)")
            continue

        # ---- Generate fold-specific labels JSON for smtTrainer.py ----
        # smtTrainer.py expects {"train": {...}, "test": {...}}
        # It splits "train" into train/val internally via --cross-valid
        fold_labels = {
            "train": {n: all_benchmarks[n] for n in train_names},
            "test": {n: all_benchmarks[n] for n in test_names},
        }
        fold_labels_path = output_dir / f"_fold_labels_{logic}_{fold_idx}.json"
        with open(fold_labels_path, "w") as f:
            json.dump(fold_labels, f)

        # ---- Call REAL smtTrainer.py ----
        train_cmd = [
            sys.executable, str(SIBYL_TRAINER),
            "--data", str(data_dir.resolve()),
            "--labels", str(fold_labels_path.resolve()),
            "-t", str(time_steps),
            "-e", str(epochs),
            "--cross-valid", "0",
            "--pool-type", pool_type,
            "--data-weight", "best",
        ]
        if not use_gpu:
            train_cmd.append("--cpu")

        print(f"    Running: {' '.join(str(c) for c in train_cmd[-8:])}")

        train_result = subprocess.run(
            train_cmd, cwd=str(SIBYL_NETWORKS_DIR),
            capture_output=True, text=True, timeout=14400,  # 4h per fold
        )

        # Print last portion of training output
        if train_result.stdout:
            lines = train_result.stdout.strip().split("\n")
            for line in lines[-10:]:
                print(f"      {line}")

        if train_result.returncode != 0:
            print(f"    ERROR: smtTrainer.py failed (exit {train_result.returncode})")
            if train_result.stderr:
                for line in train_result.stderr.strip().split("\n")[-5:]:
                    print(f"      {line}")
            continue

        # ---- Find the model .pt file created by smtTrainer.py ----
        # smtTrainer.py saves to cwd (SIBYL_NETWORKS_DIR) with timestamp name
        pt_files = sorted(
            SIBYL_NETWORKS_DIR.glob("*.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not pt_files:
            print(f"    ERROR: No .pt model file found after training")
            continue

        src_model = pt_files[0]  # most recently created
        dest_model = output_dir / "models" / f"sibyl_{logic}_fold{fold_idx}.pt"
        dest_model.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.move(str(src_model), str(dest_model))
        print(f"    Model saved: {dest_model}")

        # Also move the .npz metrics file if present
        src_metrics = src_model.with_suffix(".npz")
        if src_metrics.exists():
            shutil.move(str(src_metrics),
                        str(dest_model.with_suffix(".metrics.npz")))

        # ---- Call REAL inference.py for each held-out test benchmark ----
        print(f"    Running inference on {len(test_names)} test benchmarks...")
        inf_t0 = _time.time()

        # Build inference task list
        inf_args = []
        test_name_order = []
        for test_name in test_names:
            npz_path = data_dir / (test_name[:-5] + ".npz")
            if not npz_path.exists():
                npz_path = data_dir / test_name.replace(".smt2", ".npz")
            if not npz_path.exists():
                continue
            inf_args.append(
                (npz_path, dest_model, portfolio_path, time_steps, pool_type)
            )
            test_name_order.append(test_name)

        # Parallel inference (12 threads — each is a subprocess)
        fold_predictions = 0
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(_run_sibyl_inference, inf_args))

        for test_name, pred_solver in zip(test_name_order, results):
            if pred_solver and pred_solver in CONFIG_ORDER:
                all_predictions[test_name] = pred_solver
                fold_predictions += 1

        inf_elapsed = _time.time() - inf_t0
        print(f"    Fold {fold_idx + 1}: {fold_predictions}/{len(test_names)} "
              f"predictions ({inf_elapsed:.0f}s)")

    return all_predictions


def save_decisions_csv(
    predictions: Dict[str, str],
    data: dict,
    logic: str,
    output_path: Path,
):
    """Save predictions for a logic to CSV."""
    rows = []
    for file_path, bdata in data.items():
        if bdata["logic"] != logic:
            continue

        # Get relative key
        parts = file_path.split("/")
        try:
            logic_idx = parts.index(logic)
            relative_key = "/".join(parts[logic_idx + 1:])
        except ValueError:
            relative_key = os.path.basename(file_path)

        pred_config = predictions.get(relative_key)
        if pred_config:
            rows.append({
                "benchmark": file_path,
                "logic": logic,
                "predicted_config": pred_config,
            })

    fieldnames = ["benchmark", "logic", "predicted_config"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} decisions to {output_path}")
    return rows


def stage_train(
    db_path: str,
    output_dir: Path,
    k_fold: int = 5,
    epochs: int = 15,
    time_steps: int = 2,
    use_gpu: bool = True,
    pool_type: str = "attention",
):
    """Stage 3: Train models and predict using family-aware GroupKFold."""
    from sklearn.model_selection import GroupKFold

    print("\n" + "=" * 70)
    print("STAGE 3: Train Sibyl GNN models (family-aware k-fold CV)")
    print("=" * 70)

    info_path = output_dir / "logic_info.json"
    if not info_path.exists():
        print("ERROR: Run --stage extract first")
        return

    with open(info_path) as f:
        logic_info = json.load(f)

    # Load run data (now includes family)
    data, timeout_s = load_run_data(db_path)

    # ---- Build ONE shared GroupKFold split across all 9k benchmarks ----
    all_paths = sorted(data.keys())
    families = np.array([data[p]["family"] for p in all_paths])
    family_names = sorted(set(families))
    fam_to_idx = {f: i for i, f in enumerate(family_names)}
    groups = np.array([fam_to_idx[f] for f in families])

    print(f"\n  Total benchmarks: {len(all_paths)}")
    print(f"  Families: {len(family_names)}: {family_names}")

    gkf = GroupKFold(n_splits=k_fold)
    np_paths = np.array(all_paths)

    # Pre-compute per-logic fold indices from the global split
    # Key: logic -> list of (train_relative_keys, test_relative_keys)
    logic_fold_indices = {logic: [] for logic in LOGICS}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np_paths, groups=groups)):
        train_paths_set = set(np_paths[train_idx])
        test_paths_set = set(np_paths[test_idx])

        train_fams = sorted(set(families[train_idx]))
        test_fams = sorted(set(families[test_idx]))
        print(f"\n  Fold {fold_idx+1}/{k_fold}: "
              f"train={len(train_idx)} ({len(train_fams)} fam), "
              f"test={len(test_idx)} ({len(test_fams)} fam)")
        print(f"    Test families: {test_fams}")

        for logic in LOGICS:
            # Convert file_paths to relative keys (family/filename.smt2)
            train_keys = []
            test_keys = []
            for file_path, bdata in data.items():
                if bdata["logic"] != logic:
                    continue
                parts = file_path.split("/")
                try:
                    logic_idx = parts.index(logic)
                    relative_key = "/".join(parts[logic_idx + 1:])
                except ValueError:
                    relative_key = os.path.basename(file_path)

                if file_path in train_paths_set:
                    train_keys.append(relative_key)
                elif file_path in test_paths_set:
                    test_keys.append(relative_key)

            logic_fold_indices[logic].append((train_keys, test_keys))

    # ---- Train each logic with the shared splits ----
    for logic in LOGICS:
        if logic not in logic_info:
            continue

        print(f"\n  Training {logic}...")
        labels_path = Path(logic_info[logic]["labels_path"])
        data_dir = Path(logic_info[logic]["data_dir"])

        predictions = train_and_predict_kfold(
            logic=logic,
            labels_path=labels_path,
            data_dir=data_dir,
            output_dir=output_dir,
            fold_indices=logic_fold_indices[logic],
            k_fold=k_fold,
            epochs=epochs,
            time_steps=time_steps,
            use_gpu=use_gpu,
            pool_type=pool_type,
        )

        print(f"\n  {logic}: {len(predictions)} total predictions")

        # Save per-logic decisions CSV
        csv_path = output_dir / f"decisions_{logic}.csv"
        save_decisions_csv(predictions, data, logic, csv_path)


# ============================================================================
# Stage 4: Merge per-logic decisions
# ============================================================================

def stage_merge(output_dir: Path):
    """Stage 4: Merge per-logic decision CSVs."""
    print("\n" + "=" * 70)
    print("STAGE 4: Merge per-logic decisions")
    print("=" * 70)

    all_rows = []
    for logic in LOGICS:
        csv_path = output_dir / f"decisions_{logic}.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                all_rows.extend(rows)
                print(f"  {logic}: {len(rows)} decisions")

    if not all_rows:
        print("  No decisions found!")
        return

    # Write combined CSV
    combined_path = output_dir / "decisions_all.csv"
    fieldnames = ["benchmark", "logic", "predicted_config"]
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Combined: {combined_path} ({len(all_rows)} rows)")

    # Write summary
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for row in all_rows:
        logic = row["logic"]
        pred = row["predicted_config"]
        counts[logic][pred] += 1
        totals[logic] += 1

    summary_path = output_dir / "decisions_summary.csv"
    all_configs = sorted(set(c for lc in counts.values() for c in lc))
    fieldnames = ["logic", "total"] + all_configs
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for logic in sorted(counts.keys()):
            row = {"logic": logic, "total": totals[logic]}
            for cfg in all_configs:
                c = counts[logic].get(cfg, 0)
                pct = c / totals[logic] * 100 if totals[logic] > 0 else 0
                row[cfg] = f"{c} ({pct:.1f}%)" if c > 0 else "-"
            writer.writerow(row)
    print(f"  Summary: {summary_path}")

    # Print summary to console
    print("\n  Selection Distribution:")
    for logic in sorted(counts.keys()):
        print(f"    {logic} ({totals[logic]} instances):")
        for cfg, c in sorted(counts[logic].items(), key=lambda x: -x[1]):
            pct = c / totals[logic] * 100
            print(f"      {cfg}: {c} ({pct:.1f}%)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sibyl GNN Portfolio Trainer for suite_9k"
    )
    parser.add_argument(
        "--db",
        default="benchmark-suite/temp_benchmarks/db/results.sqlite",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir", default="sibyl_results",
        help="Output directory",
    )
    parser.add_argument(
        "--base-dir", default=".",
        help="Base directory for resolving benchmark file paths",
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "graphs", "train", "merge", "all"],
        default="all",
        help="Which stage to run",
    )
    parser.add_argument("--cores", type=int, default=4,
                        help="Parallel cores for graph building")
    parser.add_argument("--k-fold", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs per fold")
    parser.add_argument("--time-steps", type=int, default=2,
                        help="GAT message-passing iterations")
    parser.add_argument("--pool-type", default="attention",
                        choices=["attention", "max", "mean", "add"],
                        help="Graph pooling strategy")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Sibyl GNN Portfolio Trainer")
    print("=" * 70)
    print(f"Database: {args.db}")
    print(f"Output: {output_dir}/")
    print(f"Portfolio: {PORTFOLIO_ID} ({len(CONFIG_ORDER)} configs)")
    print(f"Suite: {SUITE_NAME} ({', '.join(LOGICS)})")
    print(f"Stage: {args.stage}")

    stages = (
        ["extract", "graphs", "train", "merge"]
        if args.stage == "all" else [args.stage]
    )

    for stage in stages:
        if stage == "extract":
            stage_extract(args.db, output_dir, args.base_dir)
        elif stage == "graphs":
            stage_graphs(output_dir, args.cores)
        elif stage == "train":
            stage_train(
                db_path=args.db,
                output_dir=output_dir,
                k_fold=args.k_fold,
                epochs=args.epochs,
                time_steps=args.time_steps,
                use_gpu=not args.cpu,
                pool_type=args.pool_type,
            )
        elif stage == "merge":
            stage_merge(output_dir)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
