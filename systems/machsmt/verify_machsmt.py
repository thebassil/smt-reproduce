#!/usr/bin/env python3
"""
Sanity check: build a MachSMT model the same way the CLI bin/machsmt does,
then compare predictions against our saved .pkl model.
"""
import sys
import os
from pathlib import Path

# Add MachSMT to path (same as portfolio trainer)
MACHSMT_PATH = Path(__file__).parent / "artifacts" / "machsmt" / "MachSMT"
sys.path.insert(0, str(MACHSMT_PATH))

# Prevent MachSMT argparse from eating our args
saved_argv = sys.argv
sys.argv = [sys.argv[0]]
from machsmt import MachSMT
from machsmt.config.config import CONFIG_OBJ
sys.argv = saved_argv

CSV_PATH = "machsmt_results/training_QF_BV.csv"
PKL_PATH = "machsmt_results/machsmt_QF_BV.pkl"

# Match the settings our trainer used
CONFIG_OBJ.k = 5
CONFIG_OBJ.cores = min(os.cpu_count() or 1, 8)
CONFIG_OBJ.min_datapoints = 5

print("=" * 60)
print("VERIFICATION: CLI-style build vs saved .pkl model")
print("=" * 60)

# --- Method 1: Build fresh (exactly what CLI bin/machsmt build does) ---
print("\n[1] Building fresh model from CSV (CLI-style)...")
print(f"    MachSMT({CSV_PATH!r})")
fresh = MachSMT(CSV_PATH, train_on_init=True)
fresh_benchmarks = fresh.db.get_benchmarks()
fresh_preds, fresh_confs = fresh.predict(benchmarks=fresh_benchmarks, include_predictions=True)
print(f"    Got {len(fresh_preds)} predictions")

# --- Method 2: Load existing .pkl (what our trainer saved) ---
print(f"\n[2] Loading saved model from {PKL_PATH}...")
saved = MachSMT.load(PKL_PATH)
saved_benchmarks = saved.db.get_benchmarks()
saved_preds, saved_confs = saved.predict(benchmarks=saved_benchmarks, include_predictions=True)
print(f"    Got {len(saved_preds)} predictions")

# --- Compare ---
print(f"\n{'=' * 60}")
print("COMPARISON")
print(f"{'=' * 60}")

# Check benchmark counts match
print(f"\nBenchmark count:  fresh={len(fresh_benchmarks)}  saved={len(saved_benchmarks)}")

# Build path->prediction maps
fresh_map = {b.get_path(): fresh_preds[i].get_name() for i, b in enumerate(fresh_benchmarks)}
saved_map = {b.get_path(): saved_preds[i].get_name() for i, b in enumerate(saved_benchmarks)}

# Find common benchmarks
common = set(fresh_map.keys()) & set(saved_map.keys())
print(f"Common benchmarks: {len(common)}")

matches = 0
mismatches = 0
mismatch_examples = []

for path in sorted(common):
    if fresh_map[path] == saved_map[path]:
        matches += 1
    else:
        mismatches += 1
        if len(mismatch_examples) < 10:
            mismatch_examples.append((path, fresh_map[path], saved_map[path]))

print(f"\nMatching predictions:    {matches}/{len(common)} ({100*matches/len(common):.1f}%)")
print(f"Mismatched predictions:  {mismatches}/{len(common)} ({100*mismatches/len(common):.1f}%)")

if mismatch_examples:
    print(f"\nFirst {len(mismatch_examples)} mismatches:")
    for path, fresh_pred, saved_pred in mismatch_examples:
        short = os.path.basename(path)
        print(f"  {short}: fresh={fresh_pred} vs saved={saved_pred}")

# Also compare confidence distributions
print(f"\n--- Confidence spot-check (first 3 benchmarks) ---")
for i in range(min(3, len(common))):
    path = sorted(common)[i]
    fi = [j for j, b in enumerate(fresh_benchmarks) if b.get_path() == path][0]
    si = [j for j, b in enumerate(saved_benchmarks) if b.get_path() == path][0]

    print(f"\n  {os.path.basename(path)}:")
    print(f"    Fresh:  {fresh_preds[fi].get_name()}")
    for solver, score in sorted(fresh_confs[fi].items(), key=lambda x: -x[1]):
        print(f"      {solver.get_name():<45} {score:.4f}")
    print(f"    Saved:  {saved_preds[si].get_name()}")
    for solver, score in sorted(saved_confs[si].items(), key=lambda x: -x[1]):
        print(f"      {solver.get_name():<45} {score:.4f}")

print(f"\n{'=' * 60}")
if mismatches == 0:
    print("RESULT: PASS — Fresh CLI-style build matches saved .pkl exactly")
else:
    print(f"RESULT: {mismatches} differences found (may be due to AdaBoost")
    print("        stochastic training — check if confidence scores are close)")
print(f"{'=' * 60}")
