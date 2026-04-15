#!/usr/bin/env python3
"""
MachSMT Decision Evaluator

Evaluates MachSMT's solver-config selection decisions against:
- Virtual Best Solver (VBS): always picks the fastest solver-config per benchmark
- Single Best Solver (SBS): always picks the same solver-config
- Random: random selection

Outputs comprehensive decision tables and performance metrics.
"""

import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add MachSMT to path
MACHSMT_PATH = Path(__file__).parent / "artifacts" / "machsmt" / "MachSMT"
sys.path.insert(0, str(MACHSMT_PATH))


def load_run_data(db_path: str, portfolio_id: int) -> Tuple[dict, int]:
    """
    Load all run data from the database.
    
    Returns:
        - data: {benchmark_path: {solver_config: (status, runtime_ms)}}
        - timeout_s: portfolio timeout in seconds
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get timeout
    row = cur.execute("SELECT timeout_s FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
    if not row:
        raise ValueError(f"Portfolio {portfolio_id} not found")
    timeout_s = row[0]
    
    # Load all runs
    runs = cur.execute("""
        SELECT 
            i.file_path,
            c.name as config_name,
            r.solver,
            r.status,
            r.runtime_ms,
            i.logic
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c ON r.config_id = c.id
        WHERE r.portfolio_id = ?
    """, (portfolio_id,)).fetchall()
    
    conn.close()
    
    data = {}
    for file_path, config_name, solver, status, runtime_ms, logic in runs:
        solver_config = f"{solver}::{config_name}"
        if file_path not in data:
            data[file_path] = {'logic': logic, 'runs': {}}
        data[file_path]['runs'][solver_config] = (status, runtime_ms)
    
    return data, timeout_s


def compute_par2_score(status: str, runtime_ms: int, timeout_s: int) -> float:
    """Compute PAR-2 score for a single run."""
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2 * timeout_s  # PAR-2 penalty


def find_vbs_selection(benchmark_data: dict, timeout_s: int) -> Tuple[str, float]:
    """Find the virtual best solver-config for a benchmark."""
    best_sc = None
    best_score = float('inf')
    
    for solver_config, (status, runtime_ms) in benchmark_data['runs'].items():
        score = compute_par2_score(status, runtime_ms, timeout_s)
        if score < best_score:
            best_score = score
            best_sc = solver_config
    
    return best_sc, best_score


def find_sbs(data: dict, timeout_s: int) -> str:
    """Find the single best solver-config across all benchmarks."""
    total_scores = defaultdict(float)
    
    for benchmark_path, benchmark_data in data.items():
        for solver_config, (status, runtime_ms) in benchmark_data['runs'].items():
            score = compute_par2_score(status, runtime_ms, timeout_s)
            total_scores[solver_config] += score
    
    return min(total_scores, key=total_scores.get)


def load_machsmt_decisions(decisions_csv: str) -> Dict[str, str]:
    """Load MachSMT decisions from CSV."""
    decisions = {}
    with open(decisions_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            decisions[row['benchmark']] = row['predicted_solver_config']
    return decisions


def evaluate_decisions(
    data: dict, 
    timeout_s: int, 
    machsmt_decisions: Dict[str, str] = None
) -> dict:
    """
    Evaluate selection strategies and return comprehensive metrics.
    """
    import random
    random.seed(42)
    
    # Get all solver-configs
    all_solver_configs = set()
    for bd in data.values():
        all_solver_configs.update(bd['runs'].keys())
    all_solver_configs = sorted(all_solver_configs)
    
    # Find SBS
    sbs = find_sbs(data, timeout_s)
    
    # Compute scores for each strategy
    results = {
        'vbs': {'total_par2': 0, 'solved': 0, 'decisions': {}},
        'sbs': {'total_par2': 0, 'solved': 0, 'solver_config': sbs, 'decisions': {}},
        'random': {'total_par2': 0, 'solved': 0, 'decisions': {}},
    }
    
    if machsmt_decisions:
        results['machsmt'] = {'total_par2': 0, 'solved': 0, 'decisions': {}}
    
    per_benchmark_results = []
    
    for benchmark_path, benchmark_data in data.items():
        logic = benchmark_data['logic']
        runs = benchmark_data['runs']
        
        row = {
            'benchmark': benchmark_path,
            'logic': logic,
        }
        
        # VBS
        vbs_sc, vbs_score = find_vbs_selection(benchmark_data, timeout_s)
        vbs_status = runs[vbs_sc][0]
        results['vbs']['total_par2'] += vbs_score
        results['vbs']['solved'] += 1 if vbs_status in ('sat', 'unsat') else 0
        results['vbs']['decisions'][benchmark_path] = vbs_sc
        row['vbs_selection'] = vbs_sc
        row['vbs_par2'] = f"{vbs_score:.3f}"
        
        # SBS
        sbs_status, sbs_runtime = runs.get(sbs, ('unknown', timeout_s * 1000))
        sbs_score = compute_par2_score(sbs_status, sbs_runtime, timeout_s)
        results['sbs']['total_par2'] += sbs_score
        results['sbs']['solved'] += 1 if sbs_status in ('sat', 'unsat') else 0
        results['sbs']['decisions'][benchmark_path] = sbs
        row['sbs_selection'] = sbs
        row['sbs_par2'] = f"{sbs_score:.3f}"
        
        # Random
        rand_sc = random.choice(list(runs.keys()))
        rand_status, rand_runtime = runs[rand_sc]
        rand_score = compute_par2_score(rand_status, rand_runtime, timeout_s)
        results['random']['total_par2'] += rand_score
        results['random']['solved'] += 1 if rand_status in ('sat', 'unsat') else 0
        results['random']['decisions'][benchmark_path] = rand_sc
        row['random_selection'] = rand_sc
        row['random_par2'] = f"{rand_score:.3f}"
        
        # MachSMT
        if machsmt_decisions and benchmark_path in machsmt_decisions:
            mach_sc = machsmt_decisions[benchmark_path]
            if mach_sc in runs:
                mach_status, mach_runtime = runs[mach_sc]
                mach_score = compute_par2_score(mach_status, mach_runtime, timeout_s)
            else:
                # Fallback if config doesn't match logic
                mach_score = 2 * timeout_s
                mach_status = 'unknown'
            
            results['machsmt']['total_par2'] += mach_score
            results['machsmt']['solved'] += 1 if mach_status in ('sat', 'unsat') else 0
            results['machsmt']['decisions'][benchmark_path] = mach_sc
            row['machsmt_selection'] = mach_sc
            row['machsmt_par2'] = f"{mach_score:.3f}"
            
            # Did MachSMT match VBS?
            row['machsmt_matched_vbs'] = 'yes' if mach_sc == vbs_sc else 'no'
            row['machsmt_regret'] = f"{mach_score - vbs_score:.3f}"
        
        per_benchmark_results.append(row)
    
    return {
        'summary': results,
        'per_benchmark': per_benchmark_results,
        'solver_configs': all_solver_configs,
        'num_benchmarks': len(data),
    }


def write_decision_table(evaluation: dict, output_path: str):
    """Write the per-benchmark decision table."""
    if not evaluation['per_benchmark']:
        return
    
    fieldnames = list(evaluation['per_benchmark'][0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluation['per_benchmark']:
            writer.writerow(row)
    
    print(f"Wrote decision table to {output_path}")


def write_summary_table(evaluation: dict, output_path: str):
    """Write the summary metrics table."""
    summary = evaluation['summary']
    n = evaluation['num_benchmarks']
    
    rows = []
    for strategy, metrics in summary.items():
        rows.append({
            'strategy': strategy.upper(),
            'total_par2': f"{metrics['total_par2']:.2f}",
            'avg_par2': f"{metrics['total_par2'] / n:.3f}" if n > 0 else "N/A",
            'solved': metrics['solved'],
            'solved_pct': f"{metrics['solved'] / n * 100:.1f}%" if n > 0 else "N/A",
        })
        if strategy == 'sbs':
            rows[-1]['notes'] = f"Uses: {metrics.get('solver_config', 'N/A')}"
    
    fieldnames = ['strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct', 'notes']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"Wrote summary to {output_path}")
    
    # Also print to console
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<12} {'Total PAR-2':>12} {'Avg PAR-2':>12} {'Solved':>10} {'%':>8}")
    print("-" * 70)
    for row in rows:
        print(f"{row['strategy']:<12} {row['total_par2']:>12} {row['avg_par2']:>12} {row['solved']:>10} {row['solved_pct']:>8}")
    print("=" * 70)
    
    # Compute improvement metrics if MachSMT is present
    if 'machsmt' in summary:
        vbs_par2 = summary['vbs']['total_par2']
        sbs_par2 = summary['sbs']['total_par2']
        mach_par2 = summary['machsmt']['total_par2']
        
        # Closeness to VBS (0% = same as SBS, 100% = same as VBS)
        if sbs_par2 != vbs_par2:
            closeness = (sbs_par2 - mach_par2) / (sbs_par2 - vbs_par2) * 100
        else:
            closeness = 100.0
        
        print(f"\nMachSMT Performance:")
        print(f"  - PAR-2 improvement over SBS: {sbs_par2 - mach_par2:.2f} ({(sbs_par2 - mach_par2) / sbs_par2 * 100:.1f}%)")
        print(f"  - Closeness to VBS: {closeness:.1f}%")
        print(f"  - VBS matches: {sum(1 for b, s in summary['machsmt']['decisions'].items() if s == summary['vbs']['decisions'].get(b))} / {n}")


def write_selection_distribution(evaluation: dict, output_path: str):
    """Write a table showing which solver-configs were selected by each strategy."""
    summary = evaluation['summary']
    solver_configs = evaluation['solver_configs']
    
    # Count selections per solver-config per strategy
    counts = {strategy: defaultdict(int) for strategy in summary}
    
    for strategy, metrics in summary.items():
        for benchmark, selection in metrics['decisions'].items():
            counts[strategy][selection] += 1
    
    fieldnames = ['solver_config'] + list(summary.keys())
    rows = []
    for sc in solver_configs:
        row = {'solver_config': sc}
        for strategy in summary:
            count = counts[strategy][sc]
            pct = count / evaluation['num_benchmarks'] * 100 if evaluation['num_benchmarks'] > 0 else 0
            row[strategy] = f"{count} ({pct:.1f}%)"
        rows.append(row)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"Wrote selection distribution to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MachSMT solver-config decisions"
    )
    parser.add_argument(
        "--db", 
        default="benchmark-suite/temp_benchmarks/db/results.sqlite",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--portfolio-id", 
        type=int, 
        default=1,
        help="Portfolio ID"
    )
    parser.add_argument(
        "--decisions-csv",
        default="machsmt_results/decisions.csv",
        help="Path to MachSMT decisions CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="machsmt_results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MachSMT Decision Evaluator")
    print("=" * 60)
    
    # Load data
    print("\nLoading run data from database...")
    data, timeout_s = load_run_data(args.db, args.portfolio_id)
    print(f"Loaded {len(data)} benchmarks with timeout={timeout_s}s")
    
    if not data:
        print("ERROR: No run data found for this portfolio.")
        return 1
    
    # Load MachSMT decisions if available
    machsmt_decisions = None
    if os.path.exists(args.decisions_csv):
        print(f"\nLoading MachSMT decisions from {args.decisions_csv}...")
        machsmt_decisions = load_machsmt_decisions(args.decisions_csv)
        print(f"Loaded {len(machsmt_decisions)} decisions")
    else:
        print(f"\nNote: No MachSMT decisions file found at {args.decisions_csv}")
        print("Will evaluate VBS, SBS, and Random only.")
    
    # Evaluate
    print("\nEvaluating strategies...")
    evaluation = evaluate_decisions(data, timeout_s, machsmt_decisions)
    
    # Write outputs
    write_decision_table(evaluation, str(output_dir / "full_decision_table.csv"))
    write_summary_table(evaluation, str(output_dir / "evaluation_summary.csv"))
    write_selection_distribution(evaluation, str(output_dir / "selection_distribution.csv"))
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
