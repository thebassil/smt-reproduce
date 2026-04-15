#!/usr/bin/env python3
"""
Medley Decision Evaluator

Evaluates Medley's solver-config selection decisions against baselines:
  - VBS  (Virtual Best Solver): oracle best per benchmark
  - SBS  (Single Best Solver): globally best single config
  - Combined SBS: per-logic best config
  - Random: uniform random selection

Reads the k-fold decisions CSV produced by medley_portfolio_trainer.py and
the baseline runs from the database to generate comprehensive metrics.

Outputs:
  - evaluation_summary.csv          – aggregate PAR-2, solved, closeness-to-VBS
  - full_decision_table.csv         – per-benchmark decisions + PAR-2 for every strategy
  - selection_distribution.csv      – config selection frequency per strategy
  - per_logic_summary.csv           – per-logic breakdown of all strategies
"""

import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
import random

# ============================================================================
# Defaults (match trainer)
# ============================================================================
DEFAULT_PORTFOLIO_ID = 7
DEFAULT_TIMEOUT_S = 60


# ============================================================================
# Helpers
# ============================================================================

def par2(status: str, runtime_ms: int, timeout_s: int) -> float:
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2.0 * timeout_s


def load_run_data(db_path: str, portfolio_id: int, suite_name: str):
    """Load all runs for the portfolio+suite.
    Returns: {file_path: {'logic': str, 'runs': {config_name: (status, runtime_ms)}}}
    """
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute("""
        SELECT i.file_path, i.logic, c.name, r.status, r.runtime_ms
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c   ON r.config_id  = c.id
        JOIN portfolio_configs pc ON pc.config_id = c.id AND pc.portfolio_id = ?
        WHERE i.suite_name = ?
          AND r.config_id IN (SELECT config_id FROM portfolio_configs WHERE portfolio_id = ?)
        ORDER BY i.file_path, c.name
    """, (portfolio_id, suite_name, portfolio_id)).fetchall()
    conn.close()

    data = {}
    for fp, logic, cfg, status, rt in rows:
        if fp not in data:
            data[fp] = {'logic': logic, 'runs': {}}
        data[fp]['runs'][cfg] = (status, rt)
    return data


def load_medley_decisions(csv_path: str):
    """Load Medley decisions from decisions_all.csv.
    Returns: {file_path: predicted_config}
    """
    decisions = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            decisions[row['file_path']] = row['predicted_config']
    return decisions


def find_sbs(data: dict, timeout_s: int) -> str:
    totals = defaultdict(float)
    for bd in data.values():
        for cfg, (status, rt) in bd['runs'].items():
            totals[cfg] += par2(status, rt, timeout_s)
    return min(totals, key=totals.get)


def find_combined_sbs(data: dict, timeout_s: int) -> dict:
    logic_totals = defaultdict(lambda: defaultdict(float))
    for bd in data.values():
        logic = bd['logic']
        for cfg, (status, rt) in bd['runs'].items():
            logic_totals[logic][cfg] += par2(status, rt, timeout_s)
    return {logic: min(ct, key=ct.get) for logic, ct in logic_totals.items()}


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(data: dict, timeout_s: int, medley_decisions: dict):
    random.seed(42)

    all_cfgs = set()
    for bd in data.values():
        all_cfgs.update(bd['runs'].keys())
    all_cfgs = sorted(all_cfgs)

    sbs_cfg = find_sbs(data, timeout_s)
    combined_sbs_cfg = find_combined_sbs(data, timeout_s)

    strategies = ['VBS', 'MEDLEY', 'COMBINED_SBS', 'SBS', 'RANDOM']
    totals = {s: {'par2': 0.0, 'solved': 0} for s in strategies}
    logic_totals = {s: defaultdict(lambda: {'par2': 0.0, 'solved': 0, 'n': 0}) for s in strategies}
    selection_counts = {s: defaultdict(int) for s in strategies}

    per_benchmark = []
    benchmarks = sorted(data.keys())

    for fp in benchmarks:
        bd = data[fp]
        logic = bd['logic']
        runs = bd['runs']

        row = {'file_path': fp, 'logic': logic}

        # VBS
        vbs_cfg = min(runs, key=lambda c: par2(runs[c][0], runs[c][1], timeout_s))
        vbs_score = par2(runs[vbs_cfg][0], runs[vbs_cfg][1], timeout_s)
        vbs_solved = runs[vbs_cfg][0] in ('sat', 'unsat')

        # MEDLEY
        med_cfg = medley_decisions.get(fp)
        if med_cfg and med_cfg in runs:
            med_score = par2(runs[med_cfg][0], runs[med_cfg][1], timeout_s)
            med_solved = runs[med_cfg][0] in ('sat', 'unsat')
        else:
            med_cfg = med_cfg or 'N/A'
            med_score = 2.0 * timeout_s
            med_solved = False

        # COMBINED_SBS
        csbs_cfg = combined_sbs_cfg.get(logic, sbs_cfg)
        if csbs_cfg in runs:
            csbs_score = par2(runs[csbs_cfg][0], runs[csbs_cfg][1], timeout_s)
            csbs_solved = runs[csbs_cfg][0] in ('sat', 'unsat')
        else:
            csbs_score = 2.0 * timeout_s
            csbs_solved = False

        # SBS
        if sbs_cfg in runs:
            sbs_score = par2(runs[sbs_cfg][0], runs[sbs_cfg][1], timeout_s)
            sbs_s = runs[sbs_cfg][0] in ('sat', 'unsat')
        else:
            sbs_score = 2.0 * timeout_s
            sbs_s = False

        # RANDOM
        rand_cfg = random.choice(list(runs.keys()))
        rand_score = par2(runs[rand_cfg][0], runs[rand_cfg][1], timeout_s)
        rand_solved = runs[rand_cfg][0] in ('sat', 'unsat')

        # Accumulate
        for strat, cfg, score, solved in [
            ('VBS', vbs_cfg, vbs_score, vbs_solved),
            ('MEDLEY', med_cfg, med_score, med_solved),
            ('COMBINED_SBS', csbs_cfg, csbs_score, csbs_solved),
            ('SBS', sbs_cfg, sbs_score, sbs_s),
            ('RANDOM', rand_cfg, rand_score, rand_solved),
        ]:
            totals[strat]['par2'] += score
            totals[strat]['solved'] += int(solved)
            logic_totals[strat][logic]['par2'] += score
            logic_totals[strat][logic]['solved'] += int(solved)
            logic_totals[strat][logic]['n'] += 1
            selection_counts[strat][cfg] += 1

        row.update({
            'vbs_config': vbs_cfg,       'vbs_par2': f"{vbs_score:.3f}",
            'medley_config': med_cfg,     'medley_par2': f"{med_score:.3f}",
            'csbs_config': csbs_cfg,      'csbs_par2': f"{csbs_score:.3f}",
            'sbs_config': sbs_cfg,        'sbs_par2': f"{sbs_score:.3f}",
            'random_config': rand_cfg,    'random_par2': f"{rand_score:.3f}",
            'matched_vbs': 'yes' if med_cfg == vbs_cfg else 'no',
            'regret': f"{med_score - vbs_score:.3f}",
        })
        per_benchmark.append(row)

    return {
        'totals': totals,
        'logic_totals': logic_totals,
        'selection_counts': selection_counts,
        'per_benchmark': per_benchmark,
        'n': len(benchmarks),
        'all_cfgs': all_cfgs,
        'sbs_cfg': sbs_cfg,
        'combined_sbs_cfg': combined_sbs_cfg,
    }


# ============================================================================
# Writers
# ============================================================================

def write_summary(ev: dict, path: str, timeout_s: int):
    n = ev['n']
    strategies = ['VBS', 'MEDLEY', 'COMBINED_SBS', 'SBS', 'RANDOM']

    rows = []
    for s in strategies:
        t = ev['totals'][s]
        rows.append({
            'strategy': s,
            'total_par2': f"{t['par2']:.2f}",
            'avg_par2': f"{t['par2'] / n:.3f}" if n else '',
            'solved': t['solved'],
            'solved_pct': f"{t['solved'] / n * 100:.1f}%" if n else '',
        })

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct'])
        w.writeheader()
        w.writerows(rows)

    # Console
    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  {'Strategy':<15} {'Total PAR-2':>14} {'Avg PAR-2':>12} {'Solved':>8} {'%':>8}")
    print("  " + "-" * 62)
    for r in rows:
        print(f"  {r['strategy']:<15} {r['total_par2']:>14} {r['avg_par2']:>12} {r['solved']:>8} {r['solved_pct']:>8}")
    print("=" * 72)

    vbs_p = ev['totals']['VBS']['par2']
    sbs_p = ev['totals']['SBS']['par2']
    med_p = ev['totals']['MEDLEY']['par2']
    if sbs_p != vbs_p:
        closeness = (sbs_p - med_p) / (sbs_p - vbs_p) * 100
        print(f"  Medley closeness to VBS: {closeness:.1f}%")
    vbs_matches = sum(1 for r in ev['per_benchmark'] if r['matched_vbs'] == 'yes')
    print(f"  VBS matches: {vbs_matches}/{n} ({vbs_matches/n*100:.1f}%)")
    print(f"  SBS config: {ev['sbs_cfg']}")
    print(f"  Combined SBS: {ev['combined_sbs_cfg']}")


def write_per_logic(ev: dict, path: str):
    strategies = ['VBS', 'MEDLEY', 'COMBINED_SBS', 'SBS', 'RANDOM']
    logics = sorted({r['logic'] for r in ev['per_benchmark']})

    rows = []
    for logic in logics:
        for s in strategies:
            lt = ev['logic_totals'][s][logic]
            n_l = lt['n']
            rows.append({
                'logic': logic,
                'strategy': s,
                'total_par2': f"{lt['par2']:.2f}",
                'avg_par2': f"{lt['par2'] / n_l:.3f}" if n_l else '',
                'solved': lt['solved'],
                'solved_pct': f"{lt['solved'] / n_l * 100:.1f}%" if n_l else '',
                'n': n_l,
            })

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['logic', 'strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct', 'n'])
        w.writeheader()
        w.writerows(rows)

    # Console per-logic
    print("\nPer-logic breakdown:")
    for logic in logics:
        print(f"\n  {logic}:")
        for s in strategies:
            lt = ev['logic_totals'][s][logic]
            n_l = lt['n']
            avg = lt['par2'] / n_l if n_l else 0
            pct = lt['solved'] / n_l * 100 if n_l else 0
            print(f"    {s:<15} PAR-2={lt['par2']:>10.2f}  avg={avg:>8.3f}  solved={lt['solved']:>5}/{n_l} ({pct:.1f}%)")


def write_decision_table(ev: dict, path: str):
    if not ev['per_benchmark']:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ev['per_benchmark'][0].keys())
        w.writeheader()
        w.writerows(ev['per_benchmark'])


def write_selection_distribution(ev: dict, path: str):
    strategies = ['VBS', 'MEDLEY', 'COMBINED_SBS', 'SBS', 'RANDOM']
    n = ev['n']
    rows = []
    for cfg in ev['all_cfgs']:
        row = {'config': cfg}
        for s in strategies:
            c = ev['selection_counts'][s][cfg]
            row[s] = f"{c} ({c/n*100:.1f}%)" if n else '0'
        rows.append(row)

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config'] + strategies)
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Medley decisions against baselines")
    parser.add_argument("--db", default="benchmark-suite/temp_benchmarks/db/results.sqlite")
    parser.add_argument("--portfolio-id", type=int, default=DEFAULT_PORTFOLIO_ID)
    parser.add_argument("--suite", default="suite_9k")
    parser.add_argument("--decisions-csv", default="medley_results/decisions_all.csv")
    parser.add_argument("--output-dir", default="medley_results")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Medley Decision Evaluator")
    print("=" * 72)
    print(f"  Database     : {args.db}")
    print(f"  Portfolio    : {args.portfolio_id}")
    print(f"  Suite        : {args.suite}")
    print(f"  Decisions    : {args.decisions_csv}")
    print(f"  Timeout      : {args.timeout}s")

    # Load
    print("\nLoading run data ...")
    data = load_run_data(args.db, args.portfolio_id, args.suite)
    print(f"  {len(data)} benchmarks loaded")

    if not data:
        print("ERROR: No run data found.")
        return 1

    print("Loading Medley decisions ...")
    if not os.path.exists(args.decisions_csv):
        print(f"ERROR: Decisions file not found: {args.decisions_csv}")
        return 1
    medley_decisions = load_medley_decisions(args.decisions_csv)
    print(f"  {len(medley_decisions)} decisions loaded")

    # Evaluate
    print("\nEvaluating ...")
    ev = evaluate(data, args.timeout, medley_decisions)

    # Write outputs
    write_summary(ev, str(output_dir / "evaluation_summary.csv"), args.timeout)
    write_per_logic(ev, str(output_dir / "per_logic_summary.csv"))
    write_decision_table(ev, str(output_dir / "full_decision_table.csv"))
    write_selection_distribution(ev, str(output_dir / "selection_distribution.csv"))

    print(f"\nOutputs saved to {output_dir}/")
    print("  evaluation_summary.csv")
    print("  per_logic_summary.csv")
    print("  full_decision_table.csv")
    print("  selection_distribution.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
