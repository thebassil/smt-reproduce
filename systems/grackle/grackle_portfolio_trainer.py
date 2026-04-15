#!/usr/bin/env python3
"""
Grackle Portfolio Trainer for suite_9k / Portfolio 7

Implements Grackle's greedy set cover algorithm for portfolio construction:
- Greedily selects solver-configs that maximize coverage of unsolved problems
- Trains SEPARATE portfolios per logic (QF_BV, QF_LIA, QF_NRA)
- Uses the greedy algorithm from Grackle (ai4reason/grackle)

Portfolio 7 configs (all run cross-logic on 9k instances):
  109 = cvc5_qfnra_05_nl_cov
  112 = cvc5_qfnra_08_decision_just
  114 = z3_baseline_default
  115 = cvc5_baseline_default
  116 = z3_baseline_qflia_case_split
  118 = z3_baseline_qfbv_sat_euf_sat
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_par2(status: str, runtime_ms: int, timeout_s: int) -> float:
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2 * timeout_s


def load_run_data(db_path: str, portfolio_id: int, suite_name: str) -> Tuple[dict, int]:
    """
    Load all run data from the database.

    Returns: (data_dict, timeout_s)
        data_dict: {file_path: {'logic': str, 'family': str,
                     'instance_id': int,
                     'runs': {config_id: (solver_config_str, status, runtime_ms)}}}
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    row = cur.execute(
        "SELECT timeout_s FROM portfolios WHERE id = ?", (portfolio_id,)
    ).fetchone()
    if not row:
        raise ValueError(f"Portfolio {portfolio_id} not found")
    timeout_s = row[0]

    runs = cur.execute("""
        SELECT
            i.file_path,
            c.id   AS config_id,
            c.name AS config_name,
            r.solver,
            r.status,
            r.runtime_ms,
            i.logic,
            i.family,
            i.id   AS instance_id
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c  ON r.config_id   = c.id
        WHERE r.portfolio_id = ?
          AND i.suite_name   = ?
    """, (portfolio_id, suite_name)).fetchall()
    conn.close()

    data = {}
    for file_path, config_id, config_name, solver, status, runtime_ms, logic, family, instance_id in runs:
        solver_config = f"{solver}::{config_name}"
        if file_path not in data:
            data[file_path] = {
                'logic': logic,
                'family': family,
                'instance_id': instance_id,
                'runs': {},
            }
        data[file_path]['runs'][config_id] = (solver_config, status, runtime_ms)

    return data, timeout_s


def build_solved_sets(data: dict) -> Dict[int, Set[str]]:
    """Build solved sets keyed by config_id: {config_id: set(file_paths_solved)}"""
    solved = defaultdict(set)
    for file_path, bdata in data.items():
        for config_id, (sc_str, status, runtime_ms) in bdata['runs'].items():
            if status in ('sat', 'unsat'):
                solved[config_id].add(file_path)
    return dict(solved)


def greedy_cover(solved_sets: Dict[int, Set[str]], max_n: int = None) -> List[Tuple[int, int]]:
    """
    Grackle's greedy set cover algorithm.

    Returns: List of (config_id, num_new_problems_covered)
    """
    results = {cid: set(problems) for cid, problems in solved_sets.items()}
    cover = []

    while results:
        best = max(results, key=lambda s: len(results[s]))
        if len(results[best]) == 0:
            break

        cover.append((best, len(results[best])))
        eaten = frozenset(results[best])

        for s in results:
            results[s].difference_update(eaten)

        results = {s: results[s] for s in results if results[s]}

        if max_n and len(cover) >= max_n:
            break

    return cover


def train_grackle_for_logic(
    data: dict, logic: str, timeout_s: int, config_names: dict, max_configs: int = None
) -> List[int]:
    """Train a Grackle greedy cover for a single logic. Returns ordered list of config_ids."""
    logic_data = {p: d for p, d in data.items() if d['logic'] == logic}
    if not logic_data:
        return []

    solved_sets = build_solved_sets(logic_data)

    print(f"\n  {logic}: {len(logic_data)} benchmarks")
    print(f"    Solver configs: {[(cid, config_names[cid]) for cid in sorted(solved_sets)]}")
    print(f"    Solved counts: {[(config_names[cid], len(s)) for cid, s in sorted(solved_sets.items(), key=lambda x: -len(x[1]))]}")

    print(f"\n    Running greedy set cover...")
    cover = greedy_cover(solved_sets, max_n=max_configs)

    schedule = [cid for cid, count in cover]

    print(f"    Greedy cover ({len(schedule)} configs):")
    for i, (cid, count) in enumerate(cover):
        print(f"      {i+1}. [{cid}] {config_names[cid]}: +{count} problems")

    return schedule


def compute_schedule_predictions(
    data: dict,
    schedule: List[int],
    timeout_s: int,
) -> Dict[str, Tuple[int, float, str]]:
    """
    Apply a schedule to benchmarks.

    Returns: {file_path: (config_id, par2_score, status)}
    """
    predictions = {}

    for file_path, bdata in data.items():
        runs = bdata['runs']

        # Try schedule in order — first solver that solves
        predicted = None
        for config_id in schedule:
            if config_id in runs:
                sc_str, status, runtime_ms = runs[config_id]
                if status in ('sat', 'unsat'):
                    predicted = (config_id, compute_par2(status, runtime_ms, timeout_s), status)
                    break

        # None solved — use first in schedule that was run
        if predicted is None:
            for config_id in schedule:
                if config_id in runs:
                    sc_str, status, runtime_ms = runs[config_id]
                    predicted = (config_id, compute_par2(status, runtime_ms, timeout_s), status)
                    break

        # Fallback
        if predicted is None and schedule:
            predicted = (schedule[0], 2 * timeout_s, 'unknown')

        if predicted:
            predictions[file_path] = predicted

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train per-logic Grackle greedy portfolios for suite_9k / Portfolio 7"
    )
    parser.add_argument("--db", default="/dcs/large/u5573765/db/results.sqlite")
    parser.add_argument("--portfolio-id", type=int, default=7)
    parser.add_argument("--suite", default="suite_9k")
    parser.add_argument("--output-dir", default="grackle_results")
    parser.add_argument("--max-configs", type=int, default=None)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Grackle Portfolio Trainer (Greedy Set Cover)")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Portfolio ID: {args.portfolio_id}")
    print(f"Training Suite: {args.suite}")
    print(f"Output: {output_dir}/")
    print()

    # Load run data
    print("Loading run data from database...")
    data, timeout_s = load_run_data(args.db, args.portfolio_id, args.suite)
    print(f"Loaded {len(data)} benchmarks with timeout={timeout_s}s")

    if not data:
        print("\nERROR: No run data found.")
        return 1

    # Build config_id -> name mapping from data
    config_names = {}
    for bdata in data.values():
        for config_id, (sc_str, status, runtime_ms) in bdata['runs'].items():
            if config_id not in config_names:
                config_names[config_id] = sc_str

    # Group by logic
    by_logic = defaultdict(list)
    for path, bdata in data.items():
        by_logic[bdata['logic']].append(path)

    print(f"Logics found: {dict((l, len(p)) for l, p in by_logic.items())}")

    # Train per-logic greedy covers
    print("\n" + "=" * 60)
    print("Training Greedy Covers per Logic")
    print("=" * 60)

    schedules = {}  # {logic: [config_id, ...]}
    for logic in sorted(by_logic.keys()):
        schedule = train_grackle_for_logic(data, logic, timeout_s, config_names, args.max_configs)
        schedules[logic] = schedule

    # Save schedules as JSON (with both config_id and human-readable names)
    schedule_export = {}
    for logic, sched in schedules.items():
        schedule_export[logic] = [
            {"config_id": cid, "solver_config": config_names[cid]}
            for cid in sched
        ]
    schedule_path = output_dir / "grackle_schedules.json"
    with open(schedule_path, 'w') as f:
        json.dump(schedule_export, f, indent=2)
    print(f"\nSchedules saved to: {schedule_path}")

    # Compute predictions
    print("\n" + "=" * 60)
    print("Computing Predictions")
    print("=" * 60)

    all_predictions = {}
    decision_rows = []

    for logic, sched in schedules.items():
        logic_data = {p: d for p, d in data.items() if d['logic'] == logic}
        predictions = compute_schedule_predictions(logic_data, sched, timeout_s)
        all_predictions.update(predictions)

        for path, (config_id, score, status) in predictions.items():
            decision_rows.append({
                'instance_id': data[path]['instance_id'],
                'logic': logic,
                'selected_config_id': config_id,
                'selected_config': config_names.get(config_id, str(config_id)),
                'runtime_ms': int(score * 1000) if status in ('sat', 'unsat') else int(timeout_s * 1000),
                'status': status,
                'par2': score,
            })

    # Save decisions CSV
    decisions_path = output_dir / "decisions_all.csv"
    with open(decisions_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'instance_id', 'logic', 'selected_config_id', 'selected_config',
            'runtime_ms', 'status', 'par2'
        ])
        writer.writeheader()
        writer.writerows(sorted(decision_rows, key=lambda r: r['instance_id']))
    print(f"Decisions saved to: {decisions_path}")

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("Training Set Performance Summary")
    print("=" * 60)

    total = len(data)

    # VBS
    vbs_par2 = 0
    vbs_solved = 0
    for path, bdata in data.items():
        best_score = float('inf')
        best_status = 'unknown'
        for config_id, (sc_str, status, runtime_ms) in bdata['runs'].items():
            score = compute_par2(status, runtime_ms, timeout_s)
            if score < best_score:
                best_score = score
                best_status = status
        vbs_par2 += best_score
        if best_status in ('sat', 'unsat'):
            vbs_solved += 1

    # SBS (overall best single config)
    config_totals = defaultdict(float)
    config_solved_counts = defaultdict(int)
    for path, bdata in data.items():
        for config_id, (sc_str, status, runtime_ms) in bdata['runs'].items():
            score = compute_par2(status, runtime_ms, timeout_s)
            config_totals[config_id] += score
            if status in ('sat', 'unsat'):
                config_solved_counts[config_id] += 1
    sbs_config = min(config_totals, key=config_totals.get)
    sbs_par2 = config_totals[sbs_config]
    sbs_solved = config_solved_counts[sbs_config]
    print(f"SBS: [{sbs_config}] {config_names[sbs_config]} (PAR-2={sbs_par2:.2f})")

    # CSBS (per-logic best)
    csbs_per_logic = {}
    logic_config_totals = defaultdict(lambda: defaultdict(float))
    for path, bdata in data.items():
        logic = bdata['logic']
        for config_id, (sc_str, status, runtime_ms) in bdata['runs'].items():
            score = compute_par2(status, runtime_ms, timeout_s)
            logic_config_totals[logic][config_id] += score
    for logic, scores in logic_config_totals.items():
        csbs_per_logic[logic] = min(scores, key=scores.get)
    print(f"CSBS: {[(l, config_names[cid]) for l, cid in sorted(csbs_per_logic.items())]}")

    csbs_par2 = 0
    csbs_solved = 0
    for path, bdata in data.items():
        logic = bdata['logic']
        cid = csbs_per_logic[logic]
        if cid in bdata['runs']:
            sc_str, status, runtime_ms = bdata['runs'][cid]
            csbs_par2 += compute_par2(status, runtime_ms, timeout_s)
            if status in ('sat', 'unsat'):
                csbs_solved += 1
        else:
            csbs_par2 += 2 * timeout_s

    # Grackle
    grackle_par2 = 0
    grackle_solved = 0
    for path in data:
        if path in all_predictions:
            config_id, score, status = all_predictions[path]
            grackle_par2 += score
            if status in ('sat', 'unsat'):
                grackle_solved += 1
        else:
            grackle_par2 += 2 * timeout_s

    print(f"\n{'Strategy':<15} {'Total PAR-2':>12} {'Avg PAR-2':>12} {'Solved':>10} {'%':>8}")
    print("-" * 60)
    for name, tp, sv in [
        ('VBS', vbs_par2, vbs_solved),
        ('GRACKLE', grackle_par2, grackle_solved),
        ('CSBS', csbs_par2, csbs_solved),
        ('SBS', sbs_par2, sbs_solved),
    ]:
        print(f"{name:<15} {tp:>12.2f} {tp/total:>12.3f} {sv:>10} {sv/total*100:>7.1f}%")
    print("=" * 60)

    # Save summary
    summary_rows = [
        {'strategy': 'VBS', 'total_par2': vbs_par2, 'avg_par2': vbs_par2/total, 'solved': vbs_solved, 'solved_pct': vbs_solved/total*100},
        {'strategy': 'GRACKLE', 'total_par2': grackle_par2, 'avg_par2': grackle_par2/total, 'solved': grackle_solved, 'solved_pct': grackle_solved/total*100},
        {'strategy': 'CSBS', 'total_par2': csbs_par2, 'avg_par2': csbs_par2/total, 'solved': csbs_solved, 'solved_pct': csbs_solved/total*100},
        {'strategy': 'SBS', 'total_par2': sbs_par2, 'avg_par2': sbs_par2/total, 'solved': sbs_solved, 'solved_pct': sbs_solved/total*100},
    ]

    summary_path = output_dir / "training_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run grackle_kfold_evaluator.py for k-fold CV evaluation")
    print("  2. Run grackle_test_evaluator.py for DB registration")

    return 0


if __name__ == "__main__":
    sys.exit(main())
