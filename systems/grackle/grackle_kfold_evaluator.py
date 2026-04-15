#!/usr/bin/env python3
"""
Grackle K-Fold Cross-Validation Evaluator for suite_9k / Portfolio 7

Family-aware 5-fold CV with comprehensive metrics:
- Family-aware splits: no family spans train+val fold
- Baselines (VBS, SBS, CSBS, Random) recomputed from training data only
- PAR-2 = runtime_s if solved, else 2*timeout
- Metrics: solved%, PAR-2, delta-PAR-2 vs CSBS, VBS-disagreement%, tail analysis
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
from typing import Dict, List, Set, Tuple

import numpy as np


def compute_par2(status: str, runtime_ms: int, timeout_s: int) -> float:
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2 * timeout_s


def load_run_data(db_path: str, portfolio_id: int, suite_name: str) -> Tuple[dict, int]:
    """
    Load all run data.
    Returns: (data_dict, timeout_s)
        data_dict: {file_path: {'logic', 'family', 'instance_id',
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


# ---------------------------------------------------------------------------
# Shared k-fold loading (deterministic, identical across all tools)
# ---------------------------------------------------------------------------

def load_shared_folds(folds_path: str, data: dict) -> List[Tuple[List[str], List[str]]]:
    """
    Load shared fold definitions from JSON and assign instances to folds
    based on family membership.

    Args:
        folds_path: Path to shared_folds_k5.json
        data: {file_path: {'logic', 'family', ...}} from load_run_data

    Returns: List of (train_paths, test_paths) tuples
    """
    with open(folds_path) as f:
        fold_data = json.load(f)

    folds = []
    for fold_def in fold_data["folds"]:
        test_families = set(fold_def["test_families"])
        train_families = set(fold_def["train_families"])

        train_paths = []
        test_paths = []
        for path, bdata in data.items():
            family = bdata['family']
            if family in test_families:
                test_paths.append(path)
            elif family in train_families:
                train_paths.append(path)

        folds.append((train_paths, test_paths))

    return folds


def make_family_aware_folds(data: dict, k: int, rng_seed: int) -> List[Tuple[List[str], List[str]]]:
    """
    Fallback: Split benchmarks into k folds such that no family spans train AND test.
    Only used if shared folds file is not available.
    """
    rng = random.Random(rng_seed)

    # Group families per logic
    logic_families = defaultdict(lambda: defaultdict(list))  # logic -> family -> [paths]
    for path, bdata in data.items():
        logic_families[bdata['logic']][bdata['family']].append(path)

    # Assign families to folds per logic (round-robin after shuffle)
    family_fold = {}  # (logic, family) -> fold_idx
    for logic in sorted(logic_families):
        families = sorted(logic_families[logic].keys())
        rng.shuffle(families)
        for i, fam in enumerate(families):
            family_fold[(logic, fam)] = i % k

    # Build fold path lists
    fold_paths = [[] for _ in range(k)]
    for path, bdata in data.items():
        fold_idx = family_fold[(bdata['logic'], bdata['family'])]
        fold_paths[fold_idx].append(path)

    # Build train/test pairs
    folds = []
    for test_fold in range(k):
        test_paths = fold_paths[test_fold]
        train_paths = []
        for i in range(k):
            if i != test_fold:
                train_paths.extend(fold_paths[i])
        folds.append((train_paths, test_paths))

    return folds


# ---------------------------------------------------------------------------
# Greedy set cover (Grackle algorithm)
# ---------------------------------------------------------------------------

def build_solved_sets(data: dict, benchmark_paths: List[str]) -> Dict[int, Set[str]]:
    solved = defaultdict(set)
    for file_path in benchmark_paths:
        if file_path not in data:
            continue
        for config_id, (sc_str, status, runtime_ms) in data[file_path]['runs'].items():
            if status in ('sat', 'unsat'):
                solved[config_id].add(file_path)
    return dict(solved)


def greedy_cover(solved_sets: Dict[int, Set[str]], max_n: int = None) -> List[int]:
    results = {cid: set(problems) for cid, problems in solved_sets.items()}
    cover = []

    while results:
        best = max(results, key=lambda s: len(results[s]))
        if len(results[best]) == 0:
            break
        cover.append(best)
        eaten = frozenset(results[best])
        for s in results:
            results[s].difference_update(eaten)
        results = {s: results[s] for s in results if results[s]}
        if max_n and len(cover) >= max_n:
            break

    return cover


# ---------------------------------------------------------------------------
# Schedule application
# ---------------------------------------------------------------------------

def apply_schedule(
    runs: dict,
    schedule: List[int],
    timeout_s: int,
) -> Tuple[int, float, str]:
    """Returns: (config_id, par2_score, status)"""
    for config_id in schedule:
        if config_id in runs:
            sc_str, status, runtime_ms = runs[config_id]
            if status in ('sat', 'unsat'):
                return (config_id, compute_par2(status, runtime_ms, timeout_s), status)

    for config_id in schedule:
        if config_id in runs:
            sc_str, status, runtime_ms = runs[config_id]
            return (config_id, compute_par2(status, runtime_ms, timeout_s), status)

    return (schedule[0] if schedule else None, 2 * timeout_s, 'unknown')


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def find_sbs_on_subset(data: dict, paths: List[str], timeout_s: int) -> int:
    """Overall single best config_id on a subset."""
    totals = defaultdict(float)
    for p in paths:
        if p not in data:
            continue
        for cid, (sc, st, rt) in data[p]['runs'].items():
            totals[cid] += compute_par2(st, rt, timeout_s)
    return min(totals, key=totals.get) if totals else None


def find_csbs_on_subset(data: dict, paths: List[str], timeout_s: int) -> Dict[str, int]:
    """Per-logic best config_id on a subset."""
    by_logic = defaultdict(list)
    for p in paths:
        if p in data:
            by_logic[data[p]['logic']].append(p)

    csbs = {}
    for logic, lpaths in by_logic.items():
        totals = defaultdict(float)
        for p in lpaths:
            for cid, (sc, st, rt) in data[p]['runs'].items():
                totals[cid] += compute_par2(st, rt, timeout_s)
        if totals:
            csbs[logic] = min(totals, key=totals.get)
    return csbs


def find_vbs(bdata: dict, timeout_s: int) -> Tuple[int, float, str]:
    """VBS for a single benchmark. Returns (config_id, par2, status)."""
    best_cid = None
    best_score = float('inf')
    best_status = 'unknown'
    for cid, (sc, st, rt) in bdata['runs'].items():
        score = compute_par2(st, rt, timeout_s)
        if score < best_score:
            best_score = score
            best_cid = cid
            best_status = st
    return best_cid, best_score, best_status


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    data: dict,
    train_paths: List[str],
    test_paths: List[str],
    timeout_s: int,
    config_ids: List[int],
    rng: random.Random,
) -> dict:
    """Evaluate all strategies on a single fold. Returns per-instance results."""

    # Train greedy cover per logic on training data
    by_logic_train = defaultdict(list)
    for p in train_paths:
        if p in data:
            by_logic_train[data[p]['logic']].append(p)

    schedules = {}
    for logic, paths in by_logic_train.items():
        solved_sets = build_solved_sets(data, paths)
        schedule = greedy_cover(solved_sets)
        schedules[logic] = schedule

    # Baselines from training data
    sbs_cid = find_sbs_on_subset(data, train_paths, timeout_s)
    csbs = find_csbs_on_subset(data, train_paths, timeout_s)

    # Per-instance results
    instances = []
    for path in test_paths:
        if path not in data:
            continue
        bdata = data[path]
        logic = bdata['logic']
        runs = bdata['runs']

        # VBS (oracle on test)
        vbs_cid, vbs_score, vbs_status = find_vbs(bdata, timeout_s)

        # SBS
        if sbs_cid and sbs_cid in runs:
            sc, st, rt = runs[sbs_cid]
            sbs_score = compute_par2(st, rt, timeout_s)
            sbs_status = st
        else:
            sbs_score = 2 * timeout_s
            sbs_status = 'unknown'

        # CSBS
        csbs_cid = csbs.get(logic)
        if csbs_cid and csbs_cid in runs:
            sc, st, rt = runs[csbs_cid]
            csbs_score = compute_par2(st, rt, timeout_s)
            csbs_status = st
        else:
            csbs_score = 2 * timeout_s
            csbs_status = 'unknown'

        # Random (pick a random config)
        rand_cid = rng.choice(config_ids)
        if rand_cid in runs:
            sc, st, rt = runs[rand_cid]
            rand_score = compute_par2(st, rt, timeout_s)
            rand_status = st
        else:
            rand_score = 2 * timeout_s
            rand_status = 'unknown'

        # Grackle
        schedule = schedules.get(logic, [])
        grackle_cid, grackle_score, grackle_status = apply_schedule(runs, schedule, timeout_s)

        instances.append({
            'path': path,
            'logic': logic,
            'vbs_par2': vbs_score,
            'vbs_solved': vbs_status in ('sat', 'unsat'),
            'vbs_cid': vbs_cid,
            'sbs_par2': sbs_score,
            'sbs_solved': sbs_status in ('sat', 'unsat'),
            'csbs_par2': csbs_score,
            'csbs_solved': csbs_status in ('sat', 'unsat'),
            'random_par2': rand_score,
            'random_solved': rand_status in ('sat', 'unsat'),
            'grackle_par2': grackle_score,
            'grackle_solved': grackle_status in ('sat', 'unsat'),
            'grackle_cid': grackle_cid,
            'grackle_matches_vbs': grackle_cid == vbs_cid,
        })

    return instances


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(all_instances: List[dict], timeout_s: int) -> dict:
    """Compute all required metrics from per-instance results."""
    strategies = ['vbs', 'grackle', 'csbs', 'sbs', 'random']
    logics = sorted(set(inst['logic'] for inst in all_instances))

    results = {}

    # Overall metrics
    for strat in strategies:
        par2s = [inst[f'{strat}_par2'] for inst in all_instances]
        solveds = [inst[f'{strat}_solved'] for inst in all_instances]
        results[strat] = {
            'total_par2': sum(par2s),
            'avg_par2': np.mean(par2s),
            'solved': sum(solveds),
            'solved_pct': sum(solveds) / len(all_instances) * 100,
            'total': len(all_instances),
        }

    # Per-logic metrics
    for logic in logics:
        logic_insts = [i for i in all_instances if i['logic'] == logic]
        for strat in strategies:
            par2s = [inst[f'{strat}_par2'] for inst in logic_insts]
            solveds = [inst[f'{strat}_solved'] for inst in logic_insts]
            results[f'{strat}_{logic}'] = {
                'total_par2': sum(par2s),
                'avg_par2': np.mean(par2s),
                'solved': sum(solveds),
                'solved_pct': sum(solveds) / len(logic_insts) * 100,
                'total': len(logic_insts),
            }

    # ΔPAR-2 vs CSBS
    results['delta_par2_vs_csbs'] = results['csbs']['total_par2'] - results['grackle']['total_par2']
    results['delta_par2_vs_csbs_pct'] = (
        results['delta_par2_vs_csbs'] / results['csbs']['total_par2'] * 100
        if results['csbs']['total_par2'] > 0 else 0
    )

    # VBS-disagreement %
    grackle_disagree = sum(1 for i in all_instances if not i['grackle_matches_vbs'])
    results['vbs_disagreement_pct'] = grackle_disagree / len(all_instances) * 100

    # Tail analysis: hardest 10% by VBS runtime
    sorted_by_vbs = sorted(all_instances, key=lambda i: i['vbs_par2'], reverse=True)
    tail_n = max(1, len(sorted_by_vbs) // 10)
    tail = sorted_by_vbs[:tail_n]
    for strat in strategies:
        par2s = [inst[f'{strat}_par2'] for inst in tail]
        results[f'tail10_{strat}_par2'] = sum(par2s)
        results[f'tail10_{strat}_avg_par2'] = np.mean(par2s)

    return results


# ---------------------------------------------------------------------------
# Per-fold metrics (for mean ± std computation)
# ---------------------------------------------------------------------------

def fold_metrics(instances: List[dict], timeout_s: int) -> dict:
    """Compute key metrics for a single fold."""
    strategies = ['vbs', 'grackle', 'csbs', 'sbs', 'random']
    logics = sorted(set(inst['logic'] for inst in instances))
    m = {}

    for strat in strategies:
        par2s = [inst[f'{strat}_par2'] for inst in instances]
        solveds = [inst[f'{strat}_solved'] for inst in instances]
        n = len(instances)
        m[f'{strat}_avg_par2'] = np.mean(par2s)
        m[f'{strat}_solved_pct'] = sum(solveds) / n * 100 if n > 0 else 0

        for logic in logics:
            li = [i for i in instances if i['logic'] == logic]
            lp = [i[f'{strat}_par2'] for i in li]
            ls = [i[f'{strat}_solved'] for i in li]
            ln = len(li)
            m[f'{strat}_{logic}_avg_par2'] = np.mean(lp) if lp else 0
            m[f'{strat}_{logic}_solved_pct'] = sum(ls) / ln * 100 if ln > 0 else 0

    # ΔPAR-2 vs CSBS
    grackle_par2 = sum(i['grackle_par2'] for i in instances)
    csbs_par2 = sum(i['csbs_par2'] for i in instances)
    m['delta_par2_vs_csbs'] = csbs_par2 - grackle_par2

    # VBS-disagreement
    disagree = sum(1 for i in instances if not i['grackle_matches_vbs'])
    m['vbs_disagreement_pct'] = disagree / len(instances) * 100 if instances else 0

    # Tail 10%
    sorted_by_vbs = sorted(instances, key=lambda i: i['vbs_par2'], reverse=True)
    tail_n = max(1, len(sorted_by_vbs) // 10)
    tail = sorted_by_vbs[:tail_n]
    for strat in strategies:
        m[f'tail10_{strat}_avg_par2'] = np.mean([i[f'{strat}_par2'] for i in tail])

    return m


def main():
    parser = argparse.ArgumentParser(
        description="Family-aware K-Fold CV evaluation of Grackle vs baselines (suite_9k)"
    )
    parser.add_argument("--db", default="/dcs/large/u5573765/db/results.sqlite")
    parser.add_argument("--portfolio-id", type=int, default=7)
    parser.add_argument("--suite", default="suite_9k")
    parser.add_argument("--output-dir", default="grackle_results")
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--rng", type=int, default=42)
    parser.add_argument("--shared-folds", default="shared_folds_k5.json",
                        help="Path to shared folds JSON (use 'none' to compute folds internally)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Grackle K-Fold Cross-Validation Evaluator (Family-Aware)")
    print("=" * 70)
    print(f"K-Fold: {args.k_fold}, Random Seed: {args.rng}")
    print(f"Database: {args.db}")
    print(f"Portfolio: {args.portfolio_id}, Suite: {args.suite}")
    print()

    # Load data
    print("Loading run data...")
    data, timeout_s = load_run_data(args.db, args.portfolio_id, args.suite)
    print(f"Loaded {len(data)} benchmarks with timeout={timeout_s}s")

    # Collect all config_ids
    config_ids = set()
    for bdata in data.values():
        config_ids.update(bdata['runs'].keys())
    config_ids = sorted(config_ids)
    print(f"Config IDs: {config_ids}")

    # Config names
    config_names = {}
    for bdata in data.values():
        for cid, (sc, st, rt) in bdata['runs'].items():
            if cid not in config_names:
                config_names[cid] = sc

    # Family-aware splits — prefer shared folds for cross-tool consistency
    shared_folds_path = args.shared_folds
    if shared_folds_path and shared_folds_path != 'none' and os.path.exists(shared_folds_path):
        print(f"\nLoading shared folds from {shared_folds_path}...")
        folds = load_shared_folds(shared_folds_path, data)
        print(f"Loaded {len(folds)} folds from shared definitions")
    else:
        print(f"\nCreating {args.k_fold} family-aware folds (internal)...")
        folds = make_family_aware_folds(data, args.k_fold, args.rng)

    for i, (train, test) in enumerate(folds):
        # Verify no family overlap
        train_families = set((data[p]['logic'], data[p]['family']) for p in train if p in data)
        test_families = set((data[p]['logic'], data[p]['family']) for p in test if p in data)
        overlap = train_families & test_families
        assert not overlap, f"Fold {i}: family overlap: {overlap}"
        print(f"  Fold {i+1}: train={len(train)}, test={len(test)}, "
              f"train_families={len(train_families)}, test_families={len(test_families)}")

    # Run k-fold evaluation
    print(f"\nRunning {args.k_fold}-fold cross-validation...")

    all_instances = []
    per_fold_metrics = []

    for fold_idx, (train_paths, test_paths) in enumerate(folds):
        rng = random.Random(args.rng + fold_idx)
        print(f"\n  Fold {fold_idx + 1}/{args.k_fold}...")
        fold_instances = evaluate_fold(data, train_paths, test_paths, timeout_s, config_ids, rng)
        all_instances.extend(fold_instances)
        fm = fold_metrics(fold_instances, timeout_s)
        fm['fold'] = fold_idx + 1
        per_fold_metrics.append(fm)

        # Print fold summary
        n = len(fold_instances)
        grackle_solved = sum(1 for i in fold_instances if i['grackle_solved'])
        csbs_solved = sum(1 for i in fold_instances if i['csbs_solved'])
        print(f"    Grackle: {grackle_solved}/{n} solved ({grackle_solved/n*100:.1f}%), "
              f"PAR-2={sum(i['grackle_par2'] for i in fold_instances)/n:.3f}")
        print(f"    CSBS:    {csbs_solved}/{n} solved ({csbs_solved/n*100:.1f}%), "
              f"PAR-2={sum(i['csbs_par2'] for i in fold_instances)/n:.3f}")

    # Aggregate
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION RESULTS (AGGREGATED)")
    print("=" * 70)

    agg = aggregate_metrics(all_instances, timeout_s)

    strategies = ['vbs', 'grackle', 'csbs', 'sbs', 'random']
    logics = sorted(set(i['logic'] for i in all_instances))

    print(f"\n{'Strategy':<15} {'Total PAR-2':>12} {'Avg PAR-2':>12} {'Solved':>10} {'%':>8}")
    print("-" * 70)
    for strat in strategies:
        s = agg[strat]
        print(f"{strat.upper():<15} {s['total_par2']:>12.2f} {s['avg_par2']:>12.3f} "
              f"{s['solved']:>10} {s['solved_pct']:>7.1f}%")
    print("=" * 70)

    # Per-logic
    for logic in logics:
        print(f"\n  {logic}:")
        for strat in strategies:
            s = agg[f'{strat}_{logic}']
            print(f"    {strat.upper():<15} PAR-2={s['avg_par2']:>8.3f}  "
                  f"Solved={s['solved']}/{s['total']} ({s['solved_pct']:.1f}%)")

    # Delta metrics
    print(f"\nΔPAR-2 vs CSBS: {agg['delta_par2_vs_csbs']:.2f} "
          f"({agg['delta_par2_vs_csbs_pct']:.1f}% reduction)")
    print(f"VBS disagreement: {agg['vbs_disagreement_pct']:.1f}%")

    # Tail analysis
    print(f"\nTail analysis (hardest 10% by VBS runtime):")
    for strat in strategies:
        print(f"  {strat.upper():<15} avg_par2={agg[f'tail10_{strat}_avg_par2']:.3f}")

    # Mean ± std across folds
    print(f"\nMean ± Std across {args.k_fold} folds:")
    key_metrics = [
        ('grackle_avg_par2', 'Grackle avg PAR-2'),
        ('grackle_solved_pct', 'Grackle solved%'),
        ('csbs_avg_par2', 'CSBS avg PAR-2'),
        ('csbs_solved_pct', 'CSBS solved%'),
        ('delta_par2_vs_csbs', 'ΔPAR-2 vs CSBS (total)'),
        ('vbs_disagreement_pct', 'VBS disagreement%'),
    ]
    for key, label in key_metrics:
        vals = [fm[key] for fm in per_fold_metrics]
        print(f"  {label:<30} {np.mean(vals):>10.3f} ± {np.std(vals):>8.3f}")

    for logic in logics:
        print(f"\n  {logic}:")
        for strat in ['grackle', 'csbs']:
            for metric in ['avg_par2', 'solved_pct']:
                key = f'{strat}_{logic}_{metric}'
                vals = [fm[key] for fm in per_fold_metrics]
                label = f'{strat.upper()} {metric}'
                print(f"    {label:<30} {np.mean(vals):>10.3f} ± {np.std(vals):>8.3f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    # 1. kfold_k5_summary.csv
    summary_rows = []
    for strat in strategies:
        s = agg[strat]
        summary_rows.append({
            'strategy': strat.upper(),
            'total_par2': s['total_par2'],
            'avg_par2': s['avg_par2'],
            'solved': s['solved'],
            'solved_pct': s['solved_pct'],
        })
    summary_path = output_dir / f"kfold_k{args.k_fold}_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct'])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {summary_path}")

    # 2. Per-fold CSV
    perfold_path = output_dir / f"kfold_k{args.k_fold}_perfold.csv"
    if per_fold_metrics:
        fieldnames = sorted(per_fold_metrics[0].keys())
        with open(perfold_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_fold_metrics)
    print(f"Per-fold results saved to: {perfold_path}")

    # 3. Per-logic summary
    perlogic_path = output_dir / f"kfold_k{args.k_fold}_perlogic.csv"
    perlogic_rows = []
    for logic in logics:
        for strat in strategies:
            s = agg[f'{strat}_{logic}']
            perlogic_rows.append({
                'logic': logic,
                'strategy': strat.upper(),
                'total_par2': s['total_par2'],
                'avg_par2': s['avg_par2'],
                'solved': s['solved'],
                'solved_pct': s['solved_pct'],
                'total': s['total'],
            })
    with open(perlogic_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['logic', 'strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct', 'total'])
        writer.writeheader()
        writer.writerows(perlogic_rows)
    print(f"Per-logic results saved to: {perlogic_path}")

    # 4. Tail analysis CSV
    tail_path = output_dir / f"kfold_k{args.k_fold}_tail10.csv"
    tail_rows = []
    for strat in strategies:
        tail_rows.append({
            'strategy': strat.upper(),
            'tail10_total_par2': agg[f'tail10_{strat}_par2'],
            'tail10_avg_par2': agg[f'tail10_{strat}_avg_par2'],
        })
    with open(tail_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'tail10_total_par2', 'tail10_avg_par2'])
        writer.writeheader()
        writer.writerows(tail_rows)
    print(f"Tail analysis saved to: {tail_path}")

    # 5. Mean ± std summary
    meanstd_path = output_dir / f"kfold_k{args.k_fold}_meanstd.csv"
    meanstd_rows = []
    for key, label in key_metrics:
        vals = [fm[key] for fm in per_fold_metrics]
        meanstd_rows.append({'metric': label, 'mean': np.mean(vals), 'std': np.std(vals)})
    for logic in logics:
        for strat in ['grackle', 'csbs', 'vbs']:
            for metric in ['avg_par2', 'solved_pct']:
                key = f'{strat}_{logic}_{metric}'
                vals = [fm[key] for fm in per_fold_metrics]
                meanstd_rows.append({
                    'metric': f'{strat.upper()} {logic} {metric}',
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                })
    with open(meanstd_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'mean', 'std'])
        writer.writeheader()
        writer.writerows(meanstd_rows)
    print(f"Mean±std saved to: {meanstd_path}")

    print("\n" + "=" * 70)
    print("K-Fold evaluation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
