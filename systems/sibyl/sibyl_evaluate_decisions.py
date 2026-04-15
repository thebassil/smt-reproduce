#!/usr/bin/env python3
"""
Sibyl Decision Evaluator + DB Writer

Evaluates Sibyl's solver-config selection decisions against baselines:
- VBS (Virtual Best Solver): oracle, always picks the fastest config
- SBS (Single Best Solver): always picks the same config everywhere
- Combined SBS: best config per logic
- Random: random selection (seed=42)

Outputs:
- evaluation_summary.csv     (strategy comparison table)
- full_decision_table.csv    (per-benchmark decisions and PAR-2 scores)
- selection_distribution.csv (how often each config was selected)

Also writes decisions to the live DB (ml_selectors + decisions tables).

Usage:
  python3 sibyl_evaluate_decisions.py
  python3 sibyl_evaluate_decisions.py --decisions-csv sibyl_results/decisions_all.csv
  python3 sibyl_evaluate_decisions.py --skip-db  # CSV only, no DB write
"""

import argparse
import csv
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from ml_db_utils import (
    get_config_map,
    get_instance_map,
    register_ml_selector,
    write_decisions_to_db,
)

# ============================================================================
# Configuration (must match sibyl_portfolio_trainer.py)
# ============================================================================

PORTFOLIO_ID = 7
SUITE_NAME = "suite_9k"

CONFIG_ORDER = [
    "z3_baseline_default",
    "z3_baseline_qflia_case_split",
    "z3_baseline_qfbv_sat_euf_sat",
    "cvc5_baseline_default",
    "cvc5_qfnra_05_nl_cov",
    "cvc5_qfnra_08_decision_just",
]


# ============================================================================
# Data Loading
# ============================================================================

def load_run_data(db_path: str) -> Tuple[dict, int]:
    """Load all run data for portfolio 7 / suite_9k."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    row = cur.execute(
        "SELECT timeout_s FROM portfolios WHERE id = ?", (PORTFOLIO_ID,)
    ).fetchone()
    if not row:
        raise ValueError(f"Portfolio {PORTFOLIO_ID} not found")
    timeout_s = row[0]

    runs = cur.execute("""
        SELECT i.file_path, c.name AS config_name, r.solver, r.status,
               r.runtime_ms, i.logic
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c  ON r.config_id = c.id
        WHERE r.portfolio_id = ? AND i.suite_name = ?
    """, (PORTFOLIO_ID, SUITE_NAME)).fetchall()
    conn.close()

    data = {}
    for file_path, config_name, solver, status, runtime_ms, logic in runs:
        if file_path not in data:
            data[file_path] = {"logic": logic, "runs": {}}
        data[file_path]["runs"][config_name] = (status, runtime_ms)

    return data, timeout_s


def load_sibyl_decisions(decisions_csv: str) -> Dict[str, str]:
    """Load Sibyl decisions from CSV. Returns {file_path: config_name}."""
    decisions = {}
    with open(decisions_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decisions[row["benchmark"]] = row["predicted_config"]
    return decisions


# ============================================================================
# Evaluation Helpers
# ============================================================================

def compute_par2(status: str, runtime_ms: int, timeout_s: int) -> float:
    """Compute PAR-2 score for a single run."""
    if status in ("sat", "unsat"):
        return runtime_ms / 1000.0
    return 2 * timeout_s


def find_vbs(benchmark_data: dict, timeout_s: int) -> Tuple[str, float]:
    """Find the virtual best config for a benchmark."""
    best_cfg = None
    best_score = float("inf")
    for config, (status, runtime_ms) in benchmark_data["runs"].items():
        score = compute_par2(status, runtime_ms, timeout_s)
        if score < best_score:
            best_score = score
            best_cfg = config
    return best_cfg, best_score


def find_sbs(data: dict, timeout_s: int) -> str:
    """Find the single best config across all benchmarks."""
    total_scores = defaultdict(float)
    for bdata in data.values():
        for config, (status, runtime_ms) in bdata["runs"].items():
            score = compute_par2(status, runtime_ms, timeout_s)
            total_scores[config] += score
    return min(total_scores, key=total_scores.get)


def find_combined_sbs(data: dict, timeout_s: int) -> Dict[str, str]:
    """Find the best config per logic."""
    by_logic = defaultdict(lambda: defaultdict(float))
    for bdata in data.values():
        logic = bdata["logic"]
        for config, (status, runtime_ms) in bdata["runs"].items():
            score = compute_par2(status, runtime_ms, timeout_s)
            by_logic[logic][config] += score
    return {
        logic: min(scores, key=scores.get)
        for logic, scores in by_logic.items()
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_decisions(
    data: dict,
    timeout_s: int,
    sibyl_decisions: Dict[str, str],
) -> dict:
    """Evaluate all strategies. Returns comprehensive results dict."""
    random.seed(42)

    sbs_config = find_sbs(data, timeout_s)
    csbs_configs = find_combined_sbs(data, timeout_s)
    all_configs = sorted(set(
        c for bdata in data.values() for c in bdata["runs"]
    ))

    results = {
        "vbs": {"par2": 0, "solved": 0, "decisions": {}},
        "sibyl": {"par2": 0, "solved": 0, "decisions": {}},
        "combined_sbs": {"par2": 0, "solved": 0, "decisions": {}},
        "sbs": {"par2": 0, "solved": 0, "solver_config": sbs_config,
                "decisions": {}},
        "random": {"par2": 0, "solved": 0, "decisions": {}},
    }

    per_benchmark = []

    for file_path, bdata in data.items():
        logic = bdata["logic"]
        runs = bdata["runs"]

        row = {"benchmark": file_path, "logic": logic}

        # VBS
        vbs_cfg, vbs_score = find_vbs(bdata, timeout_s)
        vbs_status = runs[vbs_cfg][0]
        results["vbs"]["par2"] += vbs_score
        results["vbs"]["solved"] += 1 if vbs_status in ("sat", "unsat") else 0
        results["vbs"]["decisions"][file_path] = vbs_cfg
        row["vbs_selection"] = vbs_cfg
        row["vbs_par2"] = f"{vbs_score:.3f}"

        # Combined SBS
        csbs_cfg = csbs_configs.get(logic, sbs_config)
        if csbs_cfg in runs:
            csbs_status, csbs_runtime = runs[csbs_cfg]
            csbs_score = compute_par2(csbs_status, csbs_runtime, timeout_s)
        else:
            csbs_score = 2 * timeout_s
            csbs_status = "unknown"
        results["combined_sbs"]["par2"] += csbs_score
        results["combined_sbs"]["solved"] += (
            1 if csbs_status in ("sat", "unsat") else 0
        )
        results["combined_sbs"]["decisions"][file_path] = csbs_cfg
        row["combined_sbs_selection"] = csbs_cfg
        row["combined_sbs_par2"] = f"{csbs_score:.3f}"

        # SBS
        if sbs_config in runs:
            sbs_status, sbs_runtime = runs[sbs_config]
            sbs_score = compute_par2(sbs_status, sbs_runtime, timeout_s)
        else:
            sbs_score = 2 * timeout_s
            sbs_status = "unknown"
        results["sbs"]["par2"] += sbs_score
        results["sbs"]["solved"] += (
            1 if sbs_status in ("sat", "unsat") else 0
        )
        results["sbs"]["decisions"][file_path] = sbs_config
        row["sbs_selection"] = sbs_config
        row["sbs_par2"] = f"{sbs_score:.3f}"

        # Random
        rand_cfg = random.choice(list(runs.keys()))
        rand_status, rand_runtime = runs[rand_cfg]
        rand_score = compute_par2(rand_status, rand_runtime, timeout_s)
        results["random"]["par2"] += rand_score
        results["random"]["solved"] += (
            1 if rand_status in ("sat", "unsat") else 0
        )
        results["random"]["decisions"][file_path] = rand_cfg
        row["random_selection"] = rand_cfg
        row["random_par2"] = f"{rand_score:.3f}"

        # Sibyl
        sibyl_cfg = sibyl_decisions.get(file_path)
        if sibyl_cfg and sibyl_cfg in runs:
            sib_status, sib_runtime = runs[sibyl_cfg]
            sib_score = compute_par2(sib_status, sib_runtime, timeout_s)
        elif sibyl_cfg:
            sib_score = 2 * timeout_s
            sib_status = "unknown"
        else:
            sib_score = 2 * timeout_s
            sib_status = "unknown"
        results["sibyl"]["par2"] += sib_score
        results["sibyl"]["solved"] += (
            1 if sib_status in ("sat", "unsat") else 0
        )
        results["sibyl"]["decisions"][file_path] = sibyl_cfg or "MISSING"
        row["sibyl_selection"] = sibyl_cfg or "MISSING"
        row["sibyl_par2"] = f"{sib_score:.3f}"
        row["sibyl_matched_vbs"] = "yes" if sibyl_cfg == vbs_cfg else "no"
        row["sibyl_regret"] = f"{sib_score - vbs_score:.3f}"

        per_benchmark.append(row)

    return {
        "summary": results,
        "per_benchmark": per_benchmark,
        "all_configs": all_configs,
        "num_benchmarks": len(data),
        "sbs_config": sbs_config,
        "csbs_configs": csbs_configs,
    }


# ============================================================================
# Output Writers
# ============================================================================

def write_summary(evaluation: dict, output_path: str):
    """Write evaluation_summary.csv and print to console."""
    summary = evaluation["summary"]
    n = evaluation["num_benchmarks"]

    rows = []
    for strategy in ["vbs", "sibyl", "combined_sbs", "sbs", "random"]:
        m = summary[strategy]
        row = {
            "strategy": strategy.upper(),
            "total_par2": f"{m['par2']:.2f}",
            "avg_par2": f"{m['par2'] / n:.3f}" if n > 0 else "N/A",
            "solved": m["solved"],
            "solved_pct": f"{m['solved'] / n * 100:.1f}%" if n > 0 else "N/A",
        }
        if strategy == "sbs":
            row["notes"] = f"Uses: {m.get('solver_config', 'N/A')}"
        elif strategy == "combined_sbs":
            row["notes"] = f"Per-logic: {evaluation['csbs_configs']}"
        rows.append(row)

    fieldnames = ["strategy", "total_par2", "avg_par2", "solved",
                  "solved_pct", "notes"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Print to console
    print(f"\n{'=' * 75}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Strategy':<16} {'Total PAR-2':>14} {'Avg PAR-2':>12} "
          f"{'Solved':>10} {'%':>8}")
    print("-" * 75)
    for row in rows:
        print(f"{row['strategy']:<16} {row['total_par2']:>14} "
              f"{row['avg_par2']:>12} {row['solved']:>10} "
              f"{row['solved_pct']:>8}")
    print("=" * 75)

    # Sibyl vs baselines
    if "sibyl" in summary:
        vbs_p = summary["vbs"]["par2"]
        csbs_p = summary["combined_sbs"]["par2"]
        sbs_p = summary["sbs"]["par2"]
        sib_p = summary["sibyl"]["par2"]

        if csbs_p != vbs_p:
            closeness = (csbs_p - sib_p) / (csbs_p - vbs_p) * 100
        else:
            closeness = 100.0

        print(f"\nSibyl Performance:")
        impr = csbs_p - sib_p
        impr_pct = impr / csbs_p * 100 if csbs_p > 0 else 0
        print(f"  PAR-2 vs Combined SBS: {impr:+.2f} ({impr_pct:+.1f}%)")
        print(f"  Closeness to VBS: {closeness:.1f}%")
        vbs_match = sum(
            1 for b in summary["sibyl"]["decisions"]
            if summary["sibyl"]["decisions"][b] ==
               summary["vbs"]["decisions"].get(b)
        )
        print(f"  VBS matches: {vbs_match} / {n}")
        print(f"  Coverage: {sum(1 for v in summary['sibyl']['decisions'].values() if v != 'MISSING')} / {n}")

    print(f"\nWrote summary to {output_path}")


def write_decision_table(evaluation: dict, output_path: str):
    """Write full_decision_table.csv."""
    rows = evaluation["per_benchmark"]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote decision table to {output_path} ({len(rows)} rows)")


def write_selection_distribution(evaluation: dict, output_path: str):
    """Write selection_distribution.csv."""
    summary = evaluation["summary"]
    all_configs = evaluation["all_configs"]

    counts = {s: defaultdict(int) for s in summary}
    for strategy, m in summary.items():
        for sel in m["decisions"].values():
            if sel and sel != "MISSING":
                counts[strategy][sel] += 1

    n = evaluation["num_benchmarks"]
    fieldnames = ["solver_config"] + list(summary.keys())
    rows = []
    for cfg in all_configs:
        row = {"solver_config": cfg}
        for strategy in summary:
            c = counts[strategy][cfg]
            pct = c / n * 100 if n > 0 else 0
            row[strategy] = f"{c} ({pct:.1f}%)"
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote selection distribution to {output_path}")


# ============================================================================
# DB Write
# ============================================================================

def write_to_database(
    db_path: str,
    sibyl_decisions: Dict[str, str],
    output_dir: Path,
):
    """Register Sibyl ML selector and write all decisions to the DB."""
    print("\n" + "-" * 70)
    print("Writing decisions to database...")

    # Register selector
    training_info = {
        "model_type": "Sibyl_GAT",
        "portfolio_id": PORTFOLIO_ID,
        "suite": SUITE_NAME,
        "num_configs": len(CONFIG_ORDER),
        "config_order": CONFIG_ORDER,
        "num_decisions": len(sibyl_decisions),
        "method": "k-fold_cv_out_of_fold",
    }

    selector_id = register_ml_selector(
        db_path=db_path,
        name=f"sibyl_gat_{SUITE_NAME}",
        model_type="Sibyl_GAT",
        portfolio_id=PORTFOLIO_ID,
        model_path=str(output_dir / "models"),
        training_info=training_info,
    )
    print(f"  Selector ID: {selector_id}")

    # Load maps
    instance_map = get_instance_map(db_path)
    config_map = get_config_map(db_path)

    # Format decisions
    decision_list = [
        {"benchmark": bench, "predicted_config": config}
        for bench, config in sibyl_decisions.items()
        if config and config != "MISSING"
    ]

    written, skipped = write_decisions_to_db(
        db_path, selector_id, decision_list, instance_map, config_map
    )
    print(f"  Written: {written}, Skipped: {skipped}")
    return selector_id


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Sibyl decisions and write to DB"
    )
    parser.add_argument(
        "--db",
        default="benchmark-suite/temp_benchmarks/db/results.sqlite",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--decisions-csv",
        default="sibyl_results/decisions_all.csv",
        help="Path to Sibyl decisions CSV",
    )
    parser.add_argument(
        "--output-dir", default="sibyl_results",
        help="Output directory for evaluation CSVs",
    )
    parser.add_argument(
        "--skip-db", action="store_true",
        help="Skip writing to the live database",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Sibyl Decision Evaluator")
    print("=" * 70)

    # Load run data
    print("\nLoading run data from database...")
    data, timeout_s = load_run_data(args.db)
    print(f"  Loaded {len(data)} benchmarks, timeout={timeout_s}s")

    # Load Sibyl decisions
    if os.path.exists(args.decisions_csv):
        print(f"\nLoading Sibyl decisions from {args.decisions_csv}...")
        sibyl_decisions = load_sibyl_decisions(args.decisions_csv)
        print(f"  Loaded {len(sibyl_decisions)} decisions")
    else:
        print(f"\nERROR: Decisions CSV not found at {args.decisions_csv}")
        print("Run sibyl_portfolio_trainer.py first.")
        return 1

    # Evaluate
    print("\nEvaluating strategies...")
    evaluation = evaluate_decisions(data, timeout_s, sibyl_decisions)

    # Write outputs
    write_summary(evaluation, str(output_dir / "evaluation_summary.csv"))
    write_decision_table(evaluation,
                         str(output_dir / "full_decision_table.csv"))
    write_selection_distribution(
        evaluation, str(output_dir / "selection_distribution.csv")
    )

    # Write to DB
    if not args.skip_db:
        write_to_database(args.db, sibyl_decisions, output_dir)
    else:
        print("\n  Skipping DB write (--skip-db)")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
