#!/usr/bin/env python3
"""
MachSMT Family-Aware K-Fold Cross-Validation Evaluator

Performs proper k-fold CV with family-aware splits:
- Instances from the same family are always in the same fold
- For each fold: train MachSMT on k-1 folds, predict on held-out fold
- SBS/CSBS baselines determined from training folds only
- VBS is the theoretical ceiling on test fold
- Results aggregated across all folds with mean ± std

Optimisation: all benchmark features are extracted ONCE, then reused
across folds via MachSMT's selector.train()/predict() API.

Metrics: solved%, PAR-2, ΔPAR-2 vs CSBS, VBS-disagreement %, tail analysis
"""

import argparse
import csv
import os
import sqlite3
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

# Increase thread stack size BEFORE MachSMT creates ThreadPool workers
# (prevents C-level stack overflow on deeply nested .smt2 files)
threading.stack_size(64 * 1024 * 1024)  # 64 MB
sys.setrecursionlimit(50000)

# Add MachSMT to path - import lazily after clearing sys.argv
MACHSMT_PATH = Path(__file__).parent / "artifacts" / "machsmt" / "MachSMT"
sys.path.insert(0, str(MACHSMT_PATH))


def load_run_data(db_path: str, portfolio_id: int) -> Tuple[dict, int]:
    """Load all run data from the database."""
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()

    row = cur.execute(
        "SELECT timeout_s FROM portfolios WHERE id = ?", (portfolio_id,)
    ).fetchone()
    if not row:
        raise ValueError(f"Portfolio {portfolio_id} not found")
    timeout_s = row[0]

    runs = cur.execute("""
        SELECT
            i.file_path, c.name as config_name, r.solver,
            r.status, r.runtime_ms, i.logic, i.family
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c ON r.config_id = c.id
        WHERE r.portfolio_id = ?
    """, (portfolio_id,)).fetchall()

    conn.close()

    data = {}
    for file_path, config_name, solver, status, runtime_ms, logic, family in runs:
        solver_config = f"{solver}::{config_name}"
        if file_path not in data:
            data[file_path] = {'logic': logic, 'family': family, 'runs': {}}
        data[file_path]['runs'][solver_config] = (status, runtime_ms)

    return data, timeout_s


def compute_par2(status: str, runtime_ms: int, timeout_s: int) -> float:
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2 * timeout_s


def find_vbs(benchmark_data: dict, timeout_s: int) -> Tuple[str, float]:
    """Find virtual best solver-config for a single benchmark."""
    best_sc, best_score = None, float('inf')
    for sc, (status, runtime_ms) in benchmark_data['runs'].items():
        score = compute_par2(status, runtime_ms, timeout_s)
        if score < best_score:
            best_score = score
            best_sc = sc
    return best_sc, best_score


def find_sbs_on_subset(data: dict, paths: List[str], timeout_s: int) -> str:
    """Find single best solver-config on a subset of benchmarks."""
    total_scores = defaultdict(float)
    for path in paths:
        if path not in data:
            continue
        for sc, (status, runtime_ms) in data[path]['runs'].items():
            total_scores[sc] += compute_par2(status, runtime_ms, timeout_s)
    if not total_scores:
        return None
    return min(total_scores, key=total_scores.get)


def find_csbs_on_subset(data: dict, paths: List[str], timeout_s: int) -> Dict[str, str]:
    """Find combined SBS (best per-logic) on a subset of benchmarks."""
    by_logic = defaultdict(list)
    for path in paths:
        if path in data:
            by_logic[data[path]['logic']].append(path)

    csbs = {}
    for logic, lpaths in by_logic.items():
        total_scores = defaultdict(float)
        for path in lpaths:
            for sc, (status, runtime_ms) in data[path]['runs'].items():
                total_scores[sc] += compute_par2(status, runtime_ms, timeout_s)
        if total_scores:
            csbs[logic] = min(total_scores, key=total_scores.get)
    return csbs


def build_machsmt_dbs(data: dict, timeout_s: int, output_dir: Path, selector_type: str = "EHM"):
    """
    Build per-logic MachSMT databases ONCE (parse all benchmarks).
    Returns {logic: (mach, path_to_benchmark_obj)}.
    """
    from machsmt import MachSMT
    from machsmt.config.config import CONFIG_OBJ

    CONFIG_OBJ.cores = min(os.cpu_count() or 1, 4)
    CONFIG_OBJ.min_datapoints = 3

    # Enable PWC selector if requested
    if selector_type == "PWC":
        CONFIG_OBJ.pwc = True

    # Group by logic
    by_logic = defaultdict(list)
    for path, bd in data.items():
        by_logic[bd['logic']].append(path)

    result = {}

    for logic in sorted(by_logic.keys()):
        paths = by_logic[logic]
        print(f"\n  Parsing {logic} ({len(paths)} benchmarks)...")

        # Create full CSV for this logic
        csv_path = output_dir / f"full_training_{logic}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['benchmark', 'solver', 'score'])
            for path in paths:
                for sc, (status, runtime_ms) in data[path]['runs'].items():
                    score = compute_par2(status, runtime_ms, timeout_s)
                    writer.writerow([path, sc, score])

        # Parse benchmarks (feature extraction happens here)
        mach = MachSMT(str(csv_path), train_on_init=False)

        # Build path -> benchmark object lookup
        benchmarks = mach.db.get_benchmarks()
        path_to_bench = {b.get_path(): b for b in benchmarks}

        print(f"    Parsed {len(benchmarks)} benchmarks, "
              f"{len(mach.db.get_solvers())} solver-configs")

        result[logic] = (mach, path_to_bench)

    return result


def predict_fold_machsmt(
    machsmt_dbs: dict, data: dict,
    train_paths: List[str], test_paths: List[str],
    selector_type: str = "EHM",
) -> Dict[str, str]:
    """
    Train per-logic selectors on train_paths, predict on test_paths.
    Uses pre-parsed benchmark objects (no re-parsing).
    selector_type: 'EHM' (runtime prediction) or 'PWC' (pairwise comparison).
    Returns {benchmark_path: predicted_solver_config}.
    """
    # Group train/test by logic
    train_by_logic = defaultdict(list)
    test_by_logic = defaultdict(list)
    for path in train_paths:
        if path in data:
            train_by_logic[data[path]['logic']].append(path)
    for path in test_paths:
        if path in data:
            test_by_logic[data[path]['logic']].append(path)

    predictions = {}

    for logic in sorted(train_by_logic.keys()):
        test_logic_paths = test_by_logic.get(logic, [])
        if not test_logic_paths:
            continue

        mach, path_to_bench = machsmt_dbs[logic]

        # Get benchmark objects for train and test
        train_benchmarks = [path_to_bench[p] for p in train_by_logic[logic]
                           if p in path_to_bench]
        test_benchmarks = [path_to_bench[p] for p in test_logic_paths
                          if p in path_to_bench]

        if not train_benchmarks or not test_benchmarks:
            continue

        # Train selector on training benchmarks
        selector = mach.selectors[selector_type]
        selector.train(train_benchmarks)

        # Predict on test benchmarks
        preds = selector.predict(test_benchmarks)

        for bench, pred_solver in zip(test_benchmarks, preds):
            predictions[bench.get_path()] = pred_solver.get_name()

    return predictions


def evaluate_fold(
    data: dict, train_paths: List[str], test_paths: List[str],
    machsmt_preds: Dict[str, str], timeout_s: int
) -> dict:
    """Evaluate all strategies on a single fold's test set."""
    import random
    random.seed(42)

    sbs = find_sbs_on_subset(data, train_paths, timeout_s)
    csbs = find_csbs_on_subset(data, train_paths, timeout_s)

    strategies = ['vbs', 'machsmt', 'csbs', 'sbs', 'random']
    results = {s: {'par2': 0.0, 'solved': 0, 'total': 0} for s in strategies}
    per_benchmark = []

    for path in test_paths:
        if path not in data:
            continue
        bd = data[path]
        logic = bd['logic']
        runs = bd['runs']

        for s in strategies:
            results[s]['total'] += 1

        # VBS
        vbs_sc, vbs_score = find_vbs(bd, timeout_s)
        vbs_solved = runs[vbs_sc][0] in ('sat', 'unsat')
        results['vbs']['par2'] += vbs_score
        results['vbs']['solved'] += int(vbs_solved)

        # SBS (from training data)
        if sbs and sbs in runs:
            sbs_status, sbs_rt = runs[sbs]
            sbs_score = compute_par2(sbs_status, sbs_rt, timeout_s)
            sbs_solved = sbs_status in ('sat', 'unsat')
        else:
            sbs_score = 2 * timeout_s
            sbs_solved = False
        results['sbs']['par2'] += sbs_score
        results['sbs']['solved'] += int(sbs_solved)

        # CSBS (from training data)
        csbs_sc = csbs.get(logic)
        if csbs_sc and csbs_sc in runs:
            csbs_status, csbs_rt = runs[csbs_sc]
            csbs_score = compute_par2(csbs_status, csbs_rt, timeout_s)
            csbs_solved = csbs_status in ('sat', 'unsat')
        else:
            csbs_score = 2 * timeout_s
            csbs_solved = False
        results['csbs']['par2'] += csbs_score
        results['csbs']['solved'] += int(csbs_solved)

        # Random
        rand_sc = random.choice(list(runs.keys()))
        rand_status, rand_rt = runs[rand_sc]
        rand_score = compute_par2(rand_status, rand_rt, timeout_s)
        rand_solved = rand_status in ('sat', 'unsat')
        results['random']['par2'] += rand_score
        results['random']['solved'] += int(rand_solved)

        # MachSMT
        mach_sc = machsmt_preds.get(path)
        if mach_sc and mach_sc in runs:
            mach_status, mach_rt = runs[mach_sc]
            mach_score = compute_par2(mach_status, mach_rt, timeout_s)
            mach_solved = mach_status in ('sat', 'unsat')
        else:
            mach_score = 2 * timeout_s
            mach_solved = False
        results['machsmt']['par2'] += mach_score
        results['machsmt']['solved'] += int(mach_solved)

        per_benchmark.append({
            'benchmark': path, 'logic': logic, 'family': bd['family'],
            'vbs_sc': vbs_sc, 'vbs_par2': vbs_score,
            'machsmt_sc': mach_sc, 'machsmt_par2': mach_score,
            'csbs_sc': csbs_sc, 'csbs_par2': csbs_score,
            'sbs_sc': sbs, 'sbs_par2': sbs_score,
            'machsmt_matched_vbs': mach_sc == vbs_sc,
            'machsmt_regret': mach_score - vbs_score,
        })

    return {'summary': results, 'per_benchmark': per_benchmark,
            'sbs_solver': sbs, 'csbs_solvers': csbs}


def main():
    # Parse our args FIRST, then clear sys.argv for MachSMT
    parser = argparse.ArgumentParser(
        description="Family-aware k-fold CV evaluation of MachSMT"
    )
    parser.add_argument("--db", default="/dcs/large/u5573765/db/results.sqlite")
    parser.add_argument("--portfolio-id", type=int, default=7)
    parser.add_argument("--output-dir", default="machsmt_results")
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--selector", default="EHM", choices=["EHM", "PWC"],
                        help="MachSMT selection strategy: EHM (runtime prediction) or PWC (pairwise comparison)")

    args = parser.parse_args()

    # Clear sys.argv to prevent MachSMT's argparse from conflicting
    sys.argv = [sys.argv[0]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MachSMT Family-Aware K-Fold Cross-Validation Evaluator")
    print("=" * 70)
    print(f"Database: {args.db}")
    print(f"Portfolio: {args.portfolio_id}, K-Fold: {args.k_fold}, Selector: {args.selector}")
    print()

    # Load run data
    print("Loading run data...")
    data, timeout_s = load_run_data(args.db, args.portfolio_id)
    all_paths = list(data.keys())
    print(f"Loaded {len(all_paths)} benchmarks, timeout={timeout_s}s")

    logics = sorted(set(d['logic'] for d in data.values()))
    print(f"Logics: {logics}")

    configs = sorted(set(sc for d in data.values() for sc in d['runs']))
    print(f"Configs ({len(configs)}): {configs}")

    # Parse ALL benchmarks ONCE (expensive step — done only once)
    print("\n" + "=" * 70)
    print("PHASE 1: Parsing all benchmarks (one-time feature extraction)")
    print("=" * 70)
    machsmt_dbs = build_machsmt_dbs(data, timeout_s, output_dir, selector_type=args.selector)

    # Build family groups for GroupKFold
    families = np.array([data[p]['family'] for p in all_paths])
    family_names = sorted(set(families))
    fam_to_idx = {f: i for i, f in enumerate(family_names)}
    groups = np.array([fam_to_idx[f] for f in families])

    print(f"\nFamilies ({len(family_names)}): {family_names}")

    np_paths = np.array(all_paths)
    gkf = GroupKFold(n_splits=args.k_fold)

    # Per-fold results
    print("\n" + "=" * 70)
    print("PHASE 2: Family-aware k-fold cross-validation")
    print("=" * 70)

    fold_summaries = []
    all_per_benchmark = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np_paths, groups=groups)):
        train_paths = list(np_paths[train_idx])
        test_paths = list(np_paths[test_idx])

        train_families = sorted(set(families[train_idx]))
        test_families = sorted(set(families[test_idx]))
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/{args.k_fold}: train={len(train_paths)} ({len(train_families)} fam), "
              f"test={len(test_paths)} ({len(test_families)} fam)")
        print(f"  Test families: {test_families}")

        # Train and predict using pre-parsed benchmarks (fast!)
        print(f"  Training {args.selector} selectors on train set...")
        machsmt_preds = predict_fold_machsmt(
            machsmt_dbs, data, train_paths, test_paths,
            selector_type=args.selector,
        )
        print(f"  Got {len(machsmt_preds)} predictions")

        # Evaluate
        fold_result = evaluate_fold(data, train_paths, test_paths, machsmt_preds, timeout_s)
        fold_summaries.append(fold_result['summary'])

        for row in fold_result['per_benchmark']:
            row['fold'] = fold_idx + 1
        all_per_benchmark.extend(fold_result['per_benchmark'])

        # Print fold summary
        s = fold_result['summary']
        n = s['vbs']['total']
        print(f"\n  {'Strategy':<12} {'PAR-2':>10} {'Solved':>8} {'%':>7}")
        print(f"  {'-'*40}")
        for strat in ['vbs', 'machsmt', 'csbs', 'sbs', 'random']:
            m = s[strat]
            pct = m['solved'] / m['total'] * 100 if m['total'] > 0 else 0
            print(f"  {strat.upper():<12} {m['par2']:>10.2f} {m['solved']:>8} {pct:>6.1f}%")

    # Aggregate across folds
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (sum across all folds)")
    print("=" * 70)

    agg = {s: {'par2': 0, 'solved': 0, 'total': 0}
           for s in ['vbs', 'machsmt', 'csbs', 'sbs', 'random']}
    for fs in fold_summaries:
        for strat in agg:
            agg[strat]['par2'] += fs[strat]['par2']
            agg[strat]['solved'] += fs[strat]['solved']
            agg[strat]['total'] += fs[strat]['total']

    total_n = agg['vbs']['total']

    print(f"\n{'Strategy':<12} {'Total PAR-2':>12} {'Avg PAR-2':>10} {'Solved':>8} {'%':>7}")
    print("-" * 55)

    summary_rows = []
    for strat in ['vbs', 'machsmt', 'csbs', 'sbs', 'random']:
        m = agg[strat]
        n = m['total']
        avg = m['par2'] / n if n > 0 else 0
        pct = m['solved'] / n * 100 if n > 0 else 0
        print(f"{strat.upper():<12} {m['par2']:>12.2f} {avg:>10.3f} {m['solved']:>8} {pct:>6.1f}%")
        summary_rows.append({
            'strategy': strat.upper(), 'total_par2': f"{m['par2']:.2f}",
            'avg_par2': f"{avg:.3f}", 'solved': m['solved'],
            'solved_pct': f"{pct:.1f}%", 'total': n,
        })

    # Per-fold stats (mean ± std)
    print(f"\n{'='*70}")
    print("PER-FOLD STATISTICS (mean ± std)")
    print("=" * 70)

    for strat in ['vbs', 'machsmt', 'csbs', 'sbs', 'random']:
        fold_par2s = [fs[strat]['par2'] / fs[strat]['total'] if fs[strat]['total'] > 0 else 0
                      for fs in fold_summaries]
        fold_solveds = [fs[strat]['solved'] / fs[strat]['total'] * 100 if fs[strat]['total'] > 0 else 0
                        for fs in fold_summaries]
        print(f"  {strat.upper():<12} "
              f"Avg PAR-2: {np.mean(fold_par2s):.3f} ± {np.std(fold_par2s):.3f}  "
              f"Solved%: {np.mean(fold_solveds):.1f} ± {np.std(fold_solveds):.1f}")

    # Improvement metrics
    vbs_par2 = agg['vbs']['par2']
    csbs_par2 = agg['csbs']['par2']
    sbs_par2 = agg['sbs']['par2']
    mach_par2 = agg['machsmt']['par2']

    print(f"\nMachSMT Performance:")
    if csbs_par2 > 0:
        print(f"  ΔPAR-2 vs CSBS: {csbs_par2 - mach_par2:.2f} ({(csbs_par2 - mach_par2) / csbs_par2 * 100:.1f}%)")
    if sbs_par2 > 0:
        print(f"  ΔPAR-2 vs SBS:  {sbs_par2 - mach_par2:.2f} ({(sbs_par2 - mach_par2) / sbs_par2 * 100:.1f}%)")
    if csbs_par2 != vbs_par2:
        closeness = (csbs_par2 - mach_par2) / (csbs_par2 - vbs_par2) * 100
        print(f"  Closeness to VBS (vs CSBS): {closeness:.1f}%")

    # VBS disagreement
    vbs_disagree = sum(1 for r in all_per_benchmark if not r['machsmt_matched_vbs'])
    print(f"  VBS disagreement: {vbs_disagree}/{len(all_per_benchmark)} "
          f"({vbs_disagree/len(all_per_benchmark)*100:.1f}%)")

    # Tail analysis (hardest 10%)
    sorted_by_regret = sorted(all_per_benchmark, key=lambda r: r['machsmt_regret'], reverse=True)
    tail_n = max(1, len(sorted_by_regret) // 10)
    tail = sorted_by_regret[:tail_n]
    tail_regret = sum(r['machsmt_regret'] for r in tail)
    print(f"\n  Tail analysis (hardest 10%, n={tail_n}):")
    print(f"    Total regret: {tail_regret:.2f}")
    print(f"    Avg regret: {tail_regret/tail_n:.3f}")
    tail_by_logic = defaultdict(list)
    for r in tail:
        tail_by_logic[r['logic']].append(r)
    for logic in sorted(tail_by_logic.keys()):
        lr = tail_by_logic[logic]
        print(f"    {logic}: {len(lr)} instances, "
              f"avg regret={np.mean([r['machsmt_regret'] for r in lr]):.3f}")

    # Per-logic breakdown
    print(f"\n{'='*70}")
    print("PER-LOGIC BREAKDOWN")
    print("=" * 70)

    for logic in logics:
        logic_rows = [r for r in all_per_benchmark if r['logic'] == logic]
        n = len(logic_rows)
        if n == 0:
            continue

        logic_stats = {}
        for strat, par2_key in [('VBS', 'vbs_par2'), ('MachSMT', 'machsmt_par2'),
                                 ('CSBS', 'csbs_par2'), ('SBS', 'sbs_par2')]:
            total_p2 = sum(r[par2_key] for r in logic_rows)
            logic_stats[strat] = {'par2': total_p2, 'avg': total_p2 / n}

        vbs_matches = sum(1 for r in logic_rows if r['machsmt_matched_vbs'])

        print(f"\n  {logic} ({n} instances):")
        print(f"    {'Strategy':<12} {'Total PAR-2':>12} {'Avg PAR-2':>10}")
        for strat in ['VBS', 'MachSMT', 'CSBS', 'SBS']:
            s = logic_stats[strat]
            print(f"    {strat:<12} {s['par2']:>12.2f} {s['avg']:>10.3f}")
        print(f"    VBS matches: {vbs_matches}/{n} ({vbs_matches/n*100:.1f}%)")

    # Write outputs
    summary_path = output_dir / "kfold_k5_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct', 'total'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved: {summary_path}")

    perfold_path = output_dir / "kfold_per_fold_results.csv"
    with open(perfold_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'strategy', 'par2', 'solved', 'total', 'avg_par2', 'solved_pct'])
        for fold_idx, fs in enumerate(fold_summaries):
            for strat in ['vbs', 'machsmt', 'csbs', 'sbs', 'random']:
                m = fs[strat]
                n = m['total']
                writer.writerow([
                    fold_idx + 1, strat.upper(), f"{m['par2']:.2f}",
                    m['solved'], n,
                    f"{m['par2']/n:.3f}" if n > 0 else "0",
                    f"{m['solved']/n*100:.1f}" if n > 0 else "0",
                ])
    print(f"Saved: {perfold_path}")

    decisions_path = output_dir / "kfold_decisions.csv"
    fieldnames = ['fold', 'benchmark', 'logic', 'family',
                  'vbs_sc', 'vbs_par2', 'machsmt_sc', 'machsmt_par2',
                  'csbs_sc', 'csbs_par2', 'sbs_sc', 'sbs_par2',
                  'machsmt_matched_vbs', 'machsmt_regret']
    with open(decisions_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_per_benchmark)
    print(f"Saved: {decisions_path}")

    per_logic_path = output_dir / "kfold_per_logic_stats.csv"
    with open(per_logic_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['logic', 'strategy', 'total_par2', 'avg_par2', 'n', 'vbs_match_pct'])
        for logic in logics:
            logic_rows = [r for r in all_per_benchmark if r['logic'] == logic]
            n = len(logic_rows)
            if n == 0:
                continue
            vbs_matches = sum(1 for r in logic_rows if r['machsmt_matched_vbs'])
            for strat, p2key in [('VBS', 'vbs_par2'), ('MachSMT', 'machsmt_par2'),
                                  ('CSBS', 'csbs_par2'), ('SBS', 'sbs_par2')]:
                total_p2 = sum(r[p2key] for r in logic_rows)
                vbs_pct = f"{vbs_matches/n*100:.1f}" if strat == 'MachSMT' else ''
                writer.writerow([logic, strat, f"{total_p2:.2f}", f"{total_p2/n:.3f}", n, vbs_pct])
    print(f"Saved: {per_logic_path}")

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
