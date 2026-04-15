#!/usr/bin/env python3
"""
Ablation Runner for Sibyl (GNN/GAT)

Takes a run-config JSON (reference + one knob override) and:
  1. Creates k-fold splits from labels JSON
  2. For each fold: trains smtTrainer.py, evaluates via inference
  3. Writes per-fold detail CSV + cross-fold summary CSV

Called from sbatch; not invoked directly by users.

Usage:
    python runner.py --config run_config.json --output-dir ablation/results/ABL_EPOCHS/10 --logic QF_BV
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def create_kfold_splits(data: dict, k: int, seed: int = 42) -> list:
    """Create k-fold train/validation splits."""
    random.seed(seed)
    items = list(data.items())
    random.shuffle(items)

    fold_size = len(items) // k
    splits = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(items)
        val_items = items[start:end]
        train_items = items[:start] + items[end:]
        splits.append((dict(train_items), dict(val_items)))
    return splits


def build_trainer_cmd(cfg: dict, labels_path: str, data_dir: str,
                      output_dir: str) -> List[str]:
    """Build the smtTrainer.py command from config."""
    algo = cfg["algorithmic"]

    cmd = [
        "python3", cfg["paths"]["trainer"],
        "-t", str(algo["time_steps"]),
        "-e", str(algo["epochs"]),
        "--data", data_dir,
        "--labels", labels_path,
        "--pool-type", algo["pool_type"],
        "--dropout", str(algo["dropout"]),
        "--data-weight", algo["data_weight"],
        "-m", algo["mode"],
    ]

    # edge_sets
    cmd += ["--edge-sets"] + algo["edge_sets"]

    # no_jump: the CLI flag --no-jump DISABLES jumping.
    # When no_jump=true in our config, jumping IS enabled (don't pass flag).
    # When no_jump=false, jumping IS disabled (pass --no-jump).
    if not algo["no_jump"]:
        cmd.append("--no-jump")

    return cmd


def build_inference_cmd(cfg: dict, model_path: str, graph_path: str,
                        portfolio_path: str) -> List[str]:
    """Build the inference.py command from config."""
    algo = cfg["algorithmic"]

    cmd = [
        "python3", cfg["paths"]["inference"],
        "--model", model_path,
        "--query", graph_path,
        "--portfolio", portfolio_path,
        "-t", str(algo["time_steps"]),
        "--pool-type", algo["pool_type"],
        "--dropout", str(algo["dropout"]),
        "-m", algo["mode"],
    ]

    # edge_sets
    cmd += ["--edge-sets"] + algo["edge_sets"]

    if not algo["no_jump"]:
        cmd.append("--no-jump")

    return cmd


def train_fold(cfg: dict, labels_path: str, data_dir: str,
               work_dir: str) -> Optional[str]:
    """Train a Sibyl model for one fold. Returns model path or None."""
    cmd = build_trainer_cmd(cfg, labels_path, data_dir, work_dir)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=work_dir, timeout=18000,  # 5h per fold (suite_9k needs ~45min/fold default)
        )
        if result.returncode != 0:
            print(f"    Training error: {result.stderr[-500:]}")
            return None
    except subprocess.TimeoutExpired:
        print("    Training timeout (2h)")
        return None

    # Find the most recently created .pt file
    pt_files = list(Path(work_dir).glob("*.pt"))
    if pt_files:
        return str(max(pt_files, key=lambda p: p.stat().st_mtime))
    return None


def run_inference_single(cfg: dict, model_path: str, graph_path: str,
                         portfolio_path: str) -> Optional[str]:
    """Run inference on one graph. Returns predicted config name or None."""
    cmd = build_inference_cmd(cfg, model_path, graph_path, portfolio_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None

        # Parse "Predicted Order:" output
        in_preds = False
        for line in result.stdout.strip().split("\n"):
            if "Predicted Order" in line:
                in_preds = True
                continue
            if in_preds and line.strip():
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None
    except subprocess.TimeoutExpired:
        return None


def evaluate_fold(
    cfg: dict,
    model_path: str,
    val_data: dict,
    configs: List[str],
    benchmark_dir: str,
    portfolio_path: str,
    timeout_s: float,
) -> dict:
    """Evaluate a trained model on a validation fold."""
    metrics = {
        "vbs_par2": 0.0, "vbs_solved": 0,
        "sbs_par2": 0.0, "sbs_solved": 0,
        "sibyl_par2": 0.0, "sibyl_solved": 0,
        "total": 0, "vbs_matches": 0,
    }

    # Find SBS on validation set
    config_totals = defaultdict(float)
    for filename, runtimes in val_data.items():
        for i, runtime in enumerate(runtimes):
            par2 = runtime if runtime < timeout_s else 2 * timeout_s
            config_totals[configs[i]] += par2

    sbs_config = min(config_totals, key=config_totals.get) if config_totals else configs[0]
    sbs_idx = configs.index(sbs_config)

    for filename, runtimes in val_data.items():
        metrics["total"] += 1

        # VBS
        best_runtime = min(runtimes)
        vbs_par2 = best_runtime if best_runtime < timeout_s else 2 * timeout_s
        metrics["vbs_par2"] += vbs_par2
        metrics["vbs_solved"] += 1 if best_runtime < timeout_s else 0

        # SBS
        sbs_runtime = runtimes[sbs_idx]
        sbs_par2 = sbs_runtime if sbs_runtime < timeout_s else 2 * timeout_s
        metrics["sbs_par2"] += sbs_par2
        metrics["sbs_solved"] += 1 if sbs_runtime < timeout_s else 0

        # Sibyl prediction — preserve family subdir if present (e.g. "sage/bench_162.smt2")
        graph_name = str(Path(filename).with_suffix(".npz"))
        graph_path = os.path.join(benchmark_dir, graph_name)

        pred_config = None
        if os.path.exists(graph_path):
            pred_config = run_inference_single(
                cfg, model_path, graph_path, portfolio_path
            )

        if pred_config and pred_config in configs:
            pred_idx = configs.index(pred_config)
            pred_runtime = runtimes[pred_idx]
            sibyl_par2 = pred_runtime if pred_runtime < timeout_s else 2 * timeout_s
            metrics["sibyl_par2"] += sibyl_par2
            metrics["sibyl_solved"] += 1 if pred_runtime < timeout_s else 0

            best_idx = runtimes.index(min(runtimes))
            if pred_idx == best_idx:
                metrics["vbs_matches"] += 1
        else:
            # Fallback to SBS
            metrics["sibyl_par2"] += sbs_par2
            metrics["sibyl_solved"] += 1 if sbs_runtime < timeout_s else 0

    return metrics


def run_kfold(cfg: dict, logic: str, output_dir: Path) -> List[dict]:
    """Run k-fold CV for a given logic and config."""
    data_cfg = cfg["data"]
    timeout_s = cfg["convention"]["timeout_s"]
    k = 5  # standard k-fold
    seed = 42

    # Load labels
    labels_key = f"labels_{logic.lower()}"
    # Map logic to correct labels/portfolio path
    logic_key = logic.lower()
    labels_path = data_cfg[f"labels_{logic_key}"]
    portfolio_path = data_cfg[f"portfolio_{logic_key}"]

    with open(labels_path) as f:
        labels = json.load(f)

    with open(portfolio_path) as f:
        configs = [line.strip() for line in f if line.strip()]

    train_data = labels.get("train", {})

    # Support per-logic benchmark dirs (suite_9k) or shared dir (suite_500)
    logic_bm_key = f"benchmark_dir_{logic.lower()}"
    if logic_bm_key in data_cfg:
        benchmark_dir = data_cfg[logic_bm_key]
    else:
        benchmark_dir = data_cfg["benchmark_dir"]

    # Pre-filter: exclude missing graphs and graphs too large for GPU
    # Matches sibyl_portfolio_trainer.py (max 500k nodes)
    MAX_GRAPH_NODES = 500_000
    filtered_data = {}
    skipped_missing = 0
    skipped_large = 0
    for name, runtimes in train_data.items():
        npz_path = os.path.join(benchmark_dir, name.replace(".smt2", ".npz"))
        if not os.path.exists(npz_path):
            skipped_missing += 1
            continue
        try:
            n_nodes = np.load(npz_path)["nodes"].shape[0]
            if n_nodes > MAX_GRAPH_NODES:
                skipped_large += 1
                continue
        except Exception:
            skipped_missing += 1
            continue
        filtered_data[name] = runtimes
    train_data = filtered_data

    print(f"\n  Logic: {logic}")
    print(f"  Benchmarks: {len(train_data)} valid ({skipped_missing} missing, {skipped_large} too large)")
    print(f"  Configs: {configs}")
    print(f"  K-folds: {k}")

    splits = create_kfold_splits(train_data, k, seed)
    fold_results = []

    for fold_idx, (train_fold, val_fold) in enumerate(splits):
        t0 = time.time()
        print(f"\n  --- Fold {fold_idx + 1}/{k}: train={len(train_fold)}, val={len(val_fold)} ---")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Write fold labels
            fold_labels = {"train": train_fold, "test": val_fold}
            fold_labels_path = os.path.join(temp_dir, "labels.json")
            with open(fold_labels_path, "w") as f:
                json.dump(fold_labels, f)

            # Train
            algo = cfg["algorithmic"]
            print(f"    Training (epochs={algo['epochs']}, t={algo['time_steps']}, "
                  f"dropout={algo['dropout']}, pool={algo['pool_type']}, "
                  f"edges={algo['edge_sets']}, mode={algo['mode']}, "
                  f"jump={algo['no_jump']}, weight={algo['data_weight']})...")

            model_path = train_fold_model(cfg, fold_labels_path, benchmark_dir, temp_dir)

            if model_path is None:
                print("    Training FAILED — skipping fold")
                continue

            # Evaluate
            print(f"    Evaluating on {len(val_fold)} benchmarks...")
            metrics = evaluate_fold(
                cfg, model_path, val_fold, configs,
                benchmark_dir, portfolio_path, timeout_s,
            )

            elapsed = time.time() - t0
            metrics["fold"] = fold_idx + 1
            metrics["wall_seconds"] = round(elapsed, 1)

            n = metrics["total"]
            print(f"    VBS:   {metrics['vbs_solved']}/{n} solved, PAR-2={metrics['vbs_par2']:.1f}")
            print(f"    Sibyl: {metrics['sibyl_solved']}/{n} solved, PAR-2={metrics['sibyl_par2']:.1f}")
            print(f"    SBS:   {metrics['sbs_solved']}/{n} solved, PAR-2={metrics['sbs_par2']:.1f}")
            print(f"    VBS matches: {metrics['vbs_matches']}/{n}")
            print(f"    Wall: {elapsed:.1f}s")

            fold_results.append(metrics)

    return fold_results


def train_fold_model(cfg, labels_path, data_dir, work_dir):
    """Wrapper for train_fold that handles the model path."""
    return train_fold(cfg, labels_path, data_dir, work_dir)


def write_results(fold_results: List[dict], output_dir: Path,
                  experiment_code: str, knob: str, value: str,
                  logic: str):
    """Write per-fold CSV and summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold detail
    perfold_path = output_dir / "perfold.csv"
    fieldnames = [
        "experiment_code", "knob", "value", "logic", "fold",
        "total",
        "vbs_par2", "vbs_solved",
        "sbs_par2", "sbs_solved",
        "sibyl_par2", "sibyl_solved",
        "vbs_matches", "wall_seconds",
    ]
    with open(perfold_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in fold_results:
            row = {
                "experiment_code": experiment_code,
                "knob": knob,
                "value": value,
                "logic": logic,
            }
            row.update(r)
            w.writerow(row)

    # Cross-fold summary
    summary_path = output_dir / "summary.csv"
    n = sum(r["total"] for r in fold_results)

    summary = {
        "experiment_code": experiment_code,
        "knob": knob,
        "value": value,
        "logic": logic,
        "k": len(fold_results),
        "total_instances": n,
        "vbs_par2": sum(r["vbs_par2"] for r in fold_results),
        "vbs_solved": sum(r["vbs_solved"] for r in fold_results),
        "vbs_solved_pct": sum(r["vbs_solved"] for r in fold_results) / n * 100 if n else 0,
        "sbs_par2": sum(r["sbs_par2"] for r in fold_results),
        "sbs_solved": sum(r["sbs_solved"] for r in fold_results),
        "sbs_solved_pct": sum(r["sbs_solved"] for r in fold_results) / n * 100 if n else 0,
        "sibyl_par2": sum(r["sibyl_par2"] for r in fold_results),
        "sibyl_solved": sum(r["sibyl_solved"] for r in fold_results),
        "sibyl_solved_pct": sum(r["sibyl_solved"] for r in fold_results) / n * 100 if n else 0,
        "vbs_matches": sum(r["vbs_matches"] for r in fold_results),
        "total_wall_seconds": sum(r["wall_seconds"] for r in fold_results),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Derived metrics
    if summary["sbs_par2"] > 0:
        summary["sibyl_vs_sbs_pct"] = (
            (summary["sbs_par2"] - summary["sibyl_par2"])
            / summary["sbs_par2"] * 100
        )
    else:
        summary["sibyl_vs_sbs_pct"] = 0.0

    vbs_p = summary["vbs_par2"]
    sbs_p = summary["sbs_par2"]
    sib_p = summary["sibyl_par2"]
    if sbs_p != vbs_p:
        summary["closeness_to_vbs_pct"] = (sbs_p - sib_p) / (sbs_p - vbs_p) * 100
    else:
        summary["closeness_to_vbs_pct"] = 100.0

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(f"\n  Wrote: {perfold_path}")
    print(f"  Wrote: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Sibyl ablation runner")
    parser.add_argument("--config", required=True,
                        help="Path to run-config JSON (reference + override)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for CSV results")
    parser.add_argument("--logic", default="QF_BV",
                        choices=["QF_BV", "QF_LIA", "QF_NRA"],
                        help="Which logic to evaluate")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    experiment_code = cfg.get("experiment_code", "UNKNOWN")
    knob = cfg.get("knob", "unknown")
    value = str(cfg.get("value", "unknown"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Sibyl Ablation Runner")
    print(f"  Experiment: {experiment_code}")
    print(f"  Knob:       {knob} = {value}")
    print(f"  Logic:      {args.logic}")
    print(f"  Output:     {output_dir}")
    print("=" * 70)

    # Run k-fold CV
    fold_results = run_kfold(cfg, args.logic, output_dir)

    if not fold_results:
        print("\nERROR: No folds completed successfully.")
        return 1

    # Write results
    summary = write_results(
        fold_results, output_dir, experiment_code, knob, value, args.logic
    )

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {experiment_code} | {knob}={value} | {args.logic}")
    print(f"  Sibyl: {summary['sibyl_solved']}/{summary['total_instances']} solved "
          f"({summary['sibyl_solved_pct']:.1f}%), PAR-2={summary['sibyl_par2']:.1f}")
    print(f"  vs SBS: {summary['sibyl_vs_sbs_pct']:+.1f}%")
    print(f"  Closeness to VBS: {summary['closeness_to_vbs_pct']:.1f}%")
    print(f"  Wall time: {summary['total_wall_seconds']:.0f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
