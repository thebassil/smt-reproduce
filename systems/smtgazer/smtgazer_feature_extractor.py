#!/usr/bin/env python3
"""
SMTGazer Feature Extraction for suite_9k

Extracts 189 machfea features per benchmark using multiprocessing.
Run on Slurm tiger with ~64 workers (~13 min for 9k benchmarks).

Output: machfea/infer_result/{logic}_train_feature.json
"""

import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

# === Configuration ===
WORKDIR = Path("/dcs/large/u5573765/smtgazer_workdir")
BENCHMARK_BASE = Path("/dcs/large/u5573765/data/benchmarks/suite_9k")
MACHFEA_DIR = Path("/dcs/23/u5573765/cs351/smt-small-sbs-smtgazer/artifacts/smtgazer/SMTGazer/machfea")
LOGICS = ["QF_BV", "QF_LIA", "QF_NRA"]
N_WORKERS = int(os.environ.get("FEATURE_WORKERS", 64))


def init_worker():
    """Initialize machfea in each worker process."""
    sys.path.insert(0, str(MACHFEA_DIR))
    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    import machsmt.benchmark  # noqa: F401 — pre-import
    sys.argv = saved_argv


def extract_single(args):
    """Extract features for a single benchmark. Returns (key, features) or (key, None)."""
    key, full_path = args
    try:
        from machsmt.benchmark import Benchmark as MachBenchmark
        b = MachBenchmark(full_path)
        b.parse()
        features = b.get_features()
        return (key, features)
    except Exception as e:
        return (key, None, str(e))


def extract_logic(logic):
    """Extract features for all benchmarks of one logic."""
    logic_dir = BENCHMARK_BASE / logic
    if not logic_dir.exists():
        print(f"  ERROR: Directory not found: {logic_dir}")
        return {}

    # Build task list: key = family/filename
    tasks = []
    for family_dir in sorted(logic_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        family = family_dir.name
        for smt_file in sorted(family_dir.glob("*.smt2")):
            key = f"{family}/{smt_file.name}"
            tasks.append((key, str(smt_file)))

    print(f"  {logic}: {len(tasks)} benchmarks, {N_WORKERS} workers")
    t0 = time.time()

    features = {}
    failed = []

    with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(extract_single, tasks)):
            if len(result) == 2:
                key, feat = result
                features[key] = feat
            else:
                key, _, err = result
                failed.append((key, err))

            if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"    Progress: {i+1}/{len(tasks)} ({rate:.1f}/s), "
                      f"failed: {len(failed)}, elapsed: {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  {logic}: {len(features)} extracted, {len(failed)} failed in {elapsed:.1f}s")

    if failed:
        print(f"  Failed benchmarks (first 10):")
        for key, err in failed[:10]:
            print(f"    {key}: {err}")

    return features


def main():
    print("=" * 70)
    print("SMTGazer Feature Extraction")
    print("=" * 70)
    print(f"Benchmark base: {BENCHMARK_BASE}")
    print(f"Workers: {N_WORKERS}")
    print(f"Output: {WORKDIR / 'machfea' / 'infer_result'}")
    print()

    output_dir = WORKDIR / "machfea" / "infer_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_features = 0
    for logic in LOGICS:
        print(f"\n--- {logic} ---")
        features = extract_logic(logic)

        if not features:
            print(f"  WARNING: No features extracted for {logic}")
            continue

        # Verify feature dimensions
        dims = set(len(v) for v in features.values())
        print(f"  Feature dimensions: {dims}")

        # Save
        out_path = output_dir / f"{logic}_train_feature.json"
        with open(out_path, 'w') as f:
            json.dump(features, f)
        print(f"  Saved: {out_path.name} ({len(features)} entries)")
        total_features += len(features)

    print("\n" + "=" * 70)
    print(f"FEATURE EXTRACTION COMPLETE: {total_features} total features")
    print("=" * 70)

    # Verify expected counts
    for logic in LOGICS:
        out_path = output_dir / f"{logic}_train_feature.json"
        if out_path.exists():
            with open(out_path) as f:
                data = json.load(f)
            expected = 3000
            actual = len(data)
            status = "OK" if actual >= expected * 0.95 else "LOW"
            print(f"  {logic}: {actual}/{expected} ({status})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
