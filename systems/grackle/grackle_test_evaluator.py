#!/usr/bin/env python3
"""
Grackle Test Evaluator & DB Registration for suite_9k / Portfolio 7

Full-dataset evaluation of Grackle schedules (no train/test split —
uses schedules trained on full suite_9k data by grackle_portfolio_trainer.py).
Also registers the selector and per-instance decisions in the central DB.

Outputs:
  - grackle_results/decisions_all.csv  (per-instance decisions)
  - grackle_results/grackle_schedules.json  (already created by trainer)
  - DB: ml_selectors entry + decisions table entries
"""

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_par2(status: str, runtime_ms: int, timeout_s: int) -> float:
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2 * timeout_s


def load_run_data(db_path: str, portfolio_id: int, suite_name: str) -> Tuple[dict, int]:
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
            i.file_path, c.id AS config_id, c.name AS config_name,
            r.solver, r.status, r.runtime_ms,
            i.logic, i.family, i.id AS instance_id
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


def apply_schedule(runs: dict, schedule: List[dict], timeout_s: int) -> Tuple[int, float, str]:
    """Apply a schedule (list of {config_id, solver_config}) to one instance."""
    for entry in schedule:
        cid = entry['config_id']
        if cid in runs:
            sc, st, rt = runs[cid]
            if st in ('sat', 'unsat'):
                return (cid, compute_par2(st, rt, timeout_s), st)

    for entry in schedule:
        cid = entry['config_id']
        if cid in runs:
            sc, st, rt = runs[cid]
            return (cid, compute_par2(st, rt, timeout_s), st)

    cid = schedule[0]['config_id'] if schedule else None
    return (cid, 2 * timeout_s, 'unknown')


def register_selector(
    db_path: str,
    name: str,
    portfolio_id: int,
    schedules: dict,
    kfold_metrics: dict,
) -> int:
    """Register (or update) the selector in ml_selectors. Returns selector_id."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    training_info = json.dumps({
        'algorithm': 'greedy_set_cover',
        'suite': 'suite_9k',
        'portfolio_id': portfolio_id,
        'k_folds': 5,
        'seed': 42,
        'family_aware': True,
        'schedules': schedules,
        'metrics': kfold_metrics,
    })

    # Check if selector already exists
    existing = cur.execute(
        "SELECT id FROM ml_selectors WHERE name = ?", (name,)
    ).fetchone()

    ts = get_utc_timestamp()

    if existing:
        selector_id = existing[0]
        cur.execute("""
            UPDATE ml_selectors
            SET training_info = ?, model_path = ?, created_utc = ?
            WHERE id = ?
        """, (training_info, 'grackle_results/grackle_schedules.json', ts, selector_id))
        # Clear old decisions
        cur.execute("DELETE FROM decisions WHERE selector_id = ?", (selector_id,))
        print(f"  Updated existing selector id={selector_id}")
    else:
        cur.execute("""
            INSERT INTO ml_selectors (name, model_type, portfolio_id, model_path, training_info, created_utc)
            VALUES (?, 'Grackle', ?, 'grackle_results/grackle_schedules.json', ?, ?)
        """, (name, portfolio_id, training_info, ts))
        selector_id = cur.lastrowid
        print(f"  Created new selector id={selector_id}")

    conn.commit()
    conn.close()
    return selector_id


def write_decisions(
    db_path: str,
    selector_id: int,
    decisions: List[dict],
):
    """Write per-instance decisions to the decisions table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = get_utc_timestamp()

    rows = []
    for d in decisions:
        rows.append((
            selector_id,
            d['instance_id'],
            d['selected_config_id'],
            d['step_num'],
            d.get('confidence'),
            d.get('confidence_scores'),
            ts,
        ))

    cur.executemany("""
        INSERT OR REPLACE INTO decisions
            (selector_id, instance_id, selected_config_id, step_num, confidence, confidence_scores, ts_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    print(f"  Wrote {len(rows)} decisions to DB")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Grackle on full suite_9k and register in DB"
    )
    parser.add_argument("--db", default="/dcs/large/u5573765/db/results.sqlite")
    parser.add_argument("--portfolio-id", type=int, default=7)
    parser.add_argument("--suite", default="suite_9k")
    parser.add_argument("--model-dir", default="grackle_results")
    parser.add_argument("--output-dir", default="grackle_results")
    parser.add_argument("--selector-name", default="grackle_suite9k_portfolio7")
    parser.add_argument("--skip-db", action="store_true", help="Skip DB registration")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    print("=" * 70)
    print("Grackle Test Evaluator & DB Registration")
    print("=" * 70)
    print(f"Database: {args.db}")
    print(f"Portfolio: {args.portfolio_id}, Suite: {args.suite}")
    print(f"Model directory: {model_dir}")
    print()

    # Load schedules
    schedule_path = model_dir / "grackle_schedules.json"
    if not schedule_path.exists():
        print(f"ERROR: Schedules not found: {schedule_path}")
        print("Run grackle_portfolio_trainer.py first.")
        return 1

    with open(schedule_path) as f:
        schedules = json.load(f)

    print("Loaded schedules:")
    for logic, sched in schedules.items():
        print(f"  {logic}: {len(sched)} configs")
        for i, entry in enumerate(sched):
            print(f"    {i+1}. [{entry['config_id']}] {entry['solver_config']}")

    # Load run data
    print(f"\nLoading run data...")
    data, timeout_s = load_run_data(args.db, args.portfolio_id, args.suite)
    print(f"Loaded {len(data)} benchmarks with timeout={timeout_s}s")

    # Config names
    config_names = {}
    for bdata in data.values():
        for cid, (sc, st, rt) in bdata['runs'].items():
            if cid not in config_names:
                config_names[cid] = sc

    # Apply schedules to all instances
    print("\nApplying schedules...")
    decisions = []
    decision_csv_rows = []

    for path, bdata in sorted(data.items()):
        logic = bdata['logic']
        runs = bdata['runs']
        instance_id = bdata['instance_id']

        schedule = schedules.get(logic, [])
        cid, par2, status = apply_schedule(runs, schedule, timeout_s)

        # For sequential portfolio: record each step
        for step_num, entry in enumerate(schedule, 1):
            ecid = entry['config_id']
            if ecid in runs:
                sc, st, rt = runs[ecid]
                decisions.append({
                    'instance_id': instance_id,
                    'selected_config_id': ecid,
                    'step_num': step_num,
                    'confidence': None,
                    'confidence_scores': None,
                })
                # Only record steps up to the solving one
                if st in ('sat', 'unsat'):
                    break

        # CSV row (best selection only)
        decision_csv_rows.append({
            'instance_id': instance_id,
            'logic': logic,
            'selected_config_id': cid,
            'selected_config': config_names.get(cid, str(cid)),
            'runtime_ms': int(par2 * 1000) if status in ('sat', 'unsat') else int(timeout_s * 1000),
            'status': status,
            'par2': par2,
        })

    # Save decisions CSV
    decisions_csv_path = output_dir / "decisions_all.csv"
    with open(decisions_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'instance_id', 'logic', 'selected_config_id', 'selected_config',
            'runtime_ms', 'status', 'par2'
        ])
        writer.writeheader()
        writer.writerows(sorted(decision_csv_rows, key=lambda r: r['instance_id']))
    print(f"Decisions CSV saved to: {decisions_csv_path}")

    # Summary stats
    total = len(data)
    grackle_par2 = sum(r['par2'] for r in decision_csv_rows)
    grackle_solved = sum(1 for r in decision_csv_rows if r['status'] in ('sat', 'unsat'))

    print(f"\nFull-dataset evaluation:")
    print(f"  Grackle: {grackle_solved}/{total} solved ({grackle_solved/total*100:.1f}%), "
          f"avg PAR-2={grackle_par2/total:.3f}")

    # Load k-fold metrics for DB registration
    kfold_summary_path = output_dir / "kfold_k5_summary.csv"
    kfold_metrics = {}
    if kfold_summary_path.exists():
        with open(kfold_summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['strategy'] == 'GRACKLE':
                    kfold_metrics['avg_par2'] = float(row['avg_par2'])
                    kfold_metrics['solved_pct'] = float(row['solved_pct'])

        # Also read meanstd
        meanstd_path = output_dir / "kfold_k5_meanstd.csv"
        if meanstd_path.exists():
            with open(meanstd_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'ΔPAR-2 vs CSBS' in row['metric']:
                        kfold_metrics['delta_par2_vs_csbs'] = float(row['mean'])
                    elif 'VBS disagreement' in row['metric']:
                        kfold_metrics['vbs_disagree_pct'] = float(row['mean'])

    # Load kfold perlogic for closeness
    kfold_k5_path = output_dir / "kfold_k5_summary.csv"
    if kfold_k5_path.exists():
        with open(kfold_k5_path) as f:
            reader = csv.DictReader(f)
            kfold_data = {row['strategy']: row for row in reader}
        if 'VBS' in kfold_data and 'CSBS' in kfold_data and 'GRACKLE' in kfold_data:
            vbs_p = float(kfold_data['VBS']['total_par2'])
            csbs_p = float(kfold_data['CSBS']['total_par2'])
            grackle_p = float(kfold_data['GRACKLE']['total_par2'])
            if csbs_p != vbs_p:
                kfold_metrics['closeness_to_vbs'] = round((csbs_p - grackle_p) / (csbs_p - vbs_p) * 100, 1)

    print(f"  K-fold metrics for DB: {kfold_metrics}")

    # DB registration
    if not args.skip_db:
        print(f"\nRegistering selector '{args.selector_name}' in DB...")
        selector_id = register_selector(
            args.db, args.selector_name, args.portfolio_id,
            schedules, kfold_metrics,
        )

        print(f"Writing decisions to DB (selector_id={selector_id})...")
        write_decisions(args.db, selector_id, decisions)
    else:
        print("\nSkipping DB registration (--skip-db)")

    print("\n" + "=" * 70)
    print("Evaluation & registration complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
