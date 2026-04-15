#!/usr/bin/env python3
"""
Grackle CLI Evaluator — evaluates portfolios invented by the real Grackle tool.

Reads db-trains-cache.json from each of the 6 Grackle run directories,
combines into a unified portfolio per logic, applies greedy set cover,
and evaluates using shared k-fold splits from shared_folds_k5.json.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


TIMEOUT_S = 30  # must match trains.runner.timeout in .fly files


def compute_par2(solved: bool, runtime_s: float, timeout_s: int) -> float:
    if solved:
        return runtime_s
    return 2.0 * timeout_s


def load_grackle_db(db_path: str) -> dict:
    """Load a Grackle db-trains-cache.json file.

    Returns: {conf_name: {instance_path: [quality, runtime, status, resources]}}
    """
    with open(db_path) as f:
        return json.load(f)


def is_solved(result) -> bool:
    """Check if a Grackle result indicates the instance was solved."""
    if result is None:
        return False
    status = result[2] if len(result) > 2 else None
    return status in ("sat", "unsat")


def result_runtime(result) -> float:
    """Safely extract runtime from a result, returning 0.0 for None."""
    if result is None:
        return 0.0
    return result[1]


def load_shared_folds(folds_path: str) -> list:
    """Load shared fold definitions.

    Returns list of dicts with 'test_families', 'train_families',
    'test_paths', 'train_paths'.
    """
    with open(folds_path) as f:
        data = json.load(f)
    return data["folds"]


def get_family_from_path(instance_path: str) -> str:
    """Extract family name from instance path like QF_BV/20210312-Bouvier/file.smt2."""
    parts = instance_path.split("/")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def get_logic_from_path(instance_path: str) -> str:
    """Extract logic from instance path like QF_BV/family/file.smt2."""
    return instance_path.split("/")[0]


def build_portfolio_from_runs(run_dirs: dict) -> dict:
    """Load all Grackle run results and build a unified portfolio.

    Args:
        run_dirs: {(solver, logic): path_to_run_dir}

    Returns:
        all_results: {conf_key: {instance_path: [quality, runtime, status, ...]}}
        where conf_key = "solver:logic:conf_name"
    """
    all_results = {}
    for (solver, logic), run_dir in run_dirs.items():
        db_path = os.path.join(run_dir, "db-trains-cache.json")
        if not os.path.exists(db_path):
            print(f"  WARNING: {db_path} not found, skipping")
            continue
        db = load_grackle_db(db_path)
        print(f"  Loaded {db_path}: {len(db)} configs")
        for conf_name, instances in db.items():
            conf_key = f"{solver}:{logic}:{conf_name}"
            all_results[conf_key] = instances
    return all_results


def build_solved_sets(
    results: dict, instance_paths: List[str]
) -> Dict[str, Set[str]]:
    """Build solved sets: {conf_key: set(solved_instance_paths)}."""
    solved = defaultdict(set)
    path_set = set(instance_paths)
    for conf_key, instances in results.items():
        for inst, result in instances.items():
            if inst in path_set and is_solved(result):
                solved[conf_key].add(inst)
    return dict(solved)


def greedy_cover(
    solved_sets: Dict[str, Set[str]], max_n: int = None
) -> List[Tuple[str, int]]:
    """Greedy set cover. Returns [(conf_key, num_new_solved), ...]."""
    remaining = {k: set(v) for k, v in solved_sets.items()}
    cover = []

    while remaining:
        best = max(remaining, key=lambda k: len(remaining[k]))
        if len(remaining[best]) == 0:
            break
        cover.append((best, len(remaining[best])))
        eaten = frozenset(remaining[best])
        for k in remaining:
            remaining[k].difference_update(eaten)
        remaining = {k: v for k, v in remaining.items() if v}
        if max_n and len(cover) >= max_n:
            break

    return cover


def apply_schedule(
    results: dict,
    schedule: List[str],
    instance_path: str,
    timeout_s: int,
) -> Tuple[str, float, bool]:
    """Apply a schedule to a single instance.

    Returns: (conf_key, par2, solved)
    """
    # First try: find a config that solves it
    for conf_key in schedule:
        if conf_key in results and instance_path in results[conf_key]:
            result = results[conf_key][instance_path]
            if is_solved(result):
                runtime_s = result_runtime(result)
                return (conf_key, compute_par2(True, runtime_s, timeout_s), True)

    # Second try: use the first config that has a result
    for conf_key in schedule:
        if conf_key in results and instance_path in results[conf_key]:
            result = results[conf_key][instance_path]
            runtime_s = result_runtime(result)
            solved = is_solved(result)
            return (conf_key, compute_par2(solved, runtime_s, timeout_s), solved)

    # Fallback: penalty
    return (schedule[0] if schedule else "none", 2.0 * timeout_s, False)


def vbs_for_instance(
    results: dict, instance_path: str, timeout_s: int
) -> Tuple[str, float, bool]:
    """Virtual best solver for a single instance."""
    best_key = None
    best_score = float("inf")
    best_solved = False
    for conf_key, instances in results.items():
        if instance_path in instances:
            result = instances[instance_path]
            solved = is_solved(result)
            score = compute_par2(solved, result_runtime(result), timeout_s)
            if score < best_score:
                best_score = score
                best_key = conf_key
                best_solved = solved
    if best_key is None:
        return ("none", 2.0 * timeout_s, False)
    return (best_key, best_score, best_solved)


def fold_metrics(instances: List[dict], timeout_s: int) -> dict:
    """Compute metrics for a single fold."""
    strategies = ["vbs", "grackle", "csbs", "sbs"]
    logics = sorted(set(i["logic"] for i in instances))
    m = {}

    for strat in strategies:
        par2s = [i[f"{strat}_par2"] for i in instances]
        solveds = [i[f"{strat}_solved"] for i in instances]
        n = len(instances)
        m[f"{strat}_avg_par2"] = np.mean(par2s)
        m[f"{strat}_solved_pct"] = sum(solveds) / n * 100 if n > 0 else 0

        for logic in logics:
            li = [i for i in instances if i["logic"] == logic]
            if not li:
                continue
            lp = [i[f"{strat}_par2"] for i in li]
            ls = [i[f"{strat}_solved"] for i in li]
            ln = len(li)
            m[f"{strat}_{logic}_avg_par2"] = np.mean(lp)
            m[f"{strat}_{logic}_solved_pct"] = sum(ls) / ln * 100

    # Delta vs CSBS
    grackle_par2 = sum(i["grackle_par2"] for i in instances)
    csbs_par2 = sum(i["csbs_par2"] for i in instances)
    m["delta_par2_vs_csbs"] = csbs_par2 - grackle_par2

    # VBS-disagreement
    disagree = sum(1 for i in instances if not i["grackle_matches_vbs"])
    m["vbs_disagreement_pct"] = disagree / len(instances) * 100 if instances else 0

    return m


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Grackle-invented portfolio using shared k-fold splits"
    )
    parser.add_argument(
        "--runs-dir",
        default="grackle_runs",
        help="Directory containing the 6 Grackle run subdirectories",
    )
    parser.add_argument(
        "--folds",
        default="shared_folds_k5.json",
        help="Shared folds JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="grackle_results",
        help="Output directory for CSV results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_S,
        help="Timeout in seconds (must match Grackle run timeout)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = args.timeout
    runs_dir = args.runs_dir

    print("=" * 70)
    print("Grackle CLI Evaluator (Invented Portfolio)")
    print("=" * 70)

    # Define run directories
    run_dirs = {
        ("cvc5", "QF_BV"): os.path.join(runs_dir, "cvc5_qfbv"),
        ("cvc5", "QF_LIA"): os.path.join(runs_dir, "cvc5_qflia"),
        ("cvc5", "QF_NRA"): os.path.join(runs_dir, "cvc5_qfnra"),
        ("z3", "QF_BV"): os.path.join(runs_dir, "z3_qfbv"),
        ("z3", "QF_LIA"): os.path.join(runs_dir, "z3_qflia"),
        ("z3", "QF_NRA"): os.path.join(runs_dir, "z3_qfnra"),
    }

    # Load all Grackle results
    print("\nLoading Grackle run results...")
    all_results = build_portfolio_from_runs(run_dirs)
    if not all_results:
        print("ERROR: No Grackle results found. Have you run fly-grackle.py?")
        return 1
    print(f"Total configs across all runs: {len(all_results)}")

    # Load shared folds
    print(f"\nLoading shared folds from {args.folds}...")
    folds = load_shared_folds(args.folds)
    print(f"Loaded {len(folds)} folds")

    # Collect all instance paths from the results
    all_instance_paths = set()
    for conf_key, instances in all_results.items():
        all_instance_paths.update(instances.keys())
    print(f"Total unique instances in results: {len(all_instance_paths)}")

    # Build family -> instances mapping
    family_instances = defaultdict(set)
    for inst in all_instance_paths:
        family = get_family_from_path(inst)
        family_instances[family].add(inst)

    # Run k-fold evaluation
    print(f"\nRunning {len(folds)}-fold cross-validation...")

    all_fold_instances = []
    per_fold_metrics = []

    for fold_idx, fold in enumerate(folds):
        fold_num = fold["fold"]
        test_families = set(fold["test_families"])
        train_families = set(fold["train_families"])

        # Assign instances to train/test based on family membership
        train_paths = []
        test_paths = []
        for inst in all_instance_paths:
            family = get_family_from_path(inst)
            if family in test_families:
                test_paths.append(inst)
            elif family in train_families:
                train_paths.append(inst)

        print(f"\n  Fold {fold_num}: train={len(train_paths)}, test={len(test_paths)}")

        # Group train/test by logic
        train_by_logic = defaultdict(list)
        test_by_logic = defaultdict(list)
        for p in train_paths:
            train_by_logic[get_logic_from_path(p)].append(p)
        for p in test_paths:
            test_by_logic[get_logic_from_path(p)].append(p)

        # Train greedy cover per logic on training data
        schedules = {}
        for logic in sorted(train_by_logic.keys()):
            solved_sets = build_solved_sets(all_results, train_by_logic[logic])
            cover = greedy_cover(solved_sets)
            schedules[logic] = [conf_key for conf_key, _ in cover]
            print(f"    {logic}: greedy cover = {len(schedules[logic])} configs")

        # Find SBS on training data (best single config across all logics)
        sbs_totals = defaultdict(float)
        for p in train_paths:
            for conf_key, instances in all_results.items():
                if p in instances:
                    result = instances[p]
                    solved = is_solved(result)
                    sbs_totals[conf_key] += compute_par2(solved, result_runtime(result), timeout_s)
        sbs_key = min(sbs_totals, key=sbs_totals.get) if sbs_totals else None

        # Find CSBS on training data (best single config per logic)
        csbs = {}
        for logic, lpaths in train_by_logic.items():
            logic_totals = defaultdict(float)
            for p in lpaths:
                for conf_key, instances in all_results.items():
                    if p in instances:
                        result = instances[p]
                        solved = is_solved(result)
                        logic_totals[conf_key] += compute_par2(
                            solved, result_runtime(result), timeout_s
                        )
            if logic_totals:
                csbs[logic] = min(logic_totals, key=logic_totals.get)

        # Evaluate on test data
        fold_instances = []
        for path in test_paths:
            logic = get_logic_from_path(path)

            # VBS
            vbs_key, vbs_score, vbs_solved = vbs_for_instance(
                all_results, path, timeout_s
            )

            # Grackle schedule
            schedule = schedules.get(logic, [])
            grackle_key, grackle_score, grackle_solved = apply_schedule(
                all_results, schedule, path, timeout_s
            )

            # SBS
            if sbs_key and sbs_key in all_results and path in all_results[sbs_key]:
                result = all_results[sbs_key][path]
                sbs_solved = is_solved(result)
                sbs_score = compute_par2(sbs_solved, result_runtime(result), timeout_s)
            else:
                sbs_score = 2.0 * timeout_s
                sbs_solved = False

            # CSBS
            csbs_key = csbs.get(logic)
            if (
                csbs_key
                and csbs_key in all_results
                and path in all_results[csbs_key]
            ):
                result = all_results[csbs_key][path]
                csbs_solved = is_solved(result)
                csbs_score = compute_par2(csbs_solved, result_runtime(result), timeout_s)
            else:
                csbs_score = 2.0 * timeout_s
                csbs_solved = False

            fold_instances.append(
                {
                    "path": path,
                    "logic": logic,
                    "vbs_par2": vbs_score,
                    "vbs_solved": vbs_solved,
                    "vbs_key": vbs_key,
                    "sbs_par2": sbs_score,
                    "sbs_solved": sbs_solved,
                    "csbs_par2": csbs_score,
                    "csbs_solved": csbs_solved,
                    "grackle_par2": grackle_score,
                    "grackle_solved": grackle_solved,
                    "grackle_key": grackle_key,
                    "grackle_matches_vbs": grackle_key == vbs_key,
                }
            )

        all_fold_instances.extend(fold_instances)
        fm = fold_metrics(fold_instances, timeout_s)
        fm["fold"] = fold_num
        per_fold_metrics.append(fm)

        # Print fold summary
        n = len(fold_instances)
        grackle_solved_n = sum(1 for i in fold_instances if i["grackle_solved"])
        csbs_solved_n = sum(1 for i in fold_instances if i["csbs_solved"])
        print(
            f"    Grackle: {grackle_solved_n}/{n} solved "
            f"({grackle_solved_n/n*100:.1f}%), "
            f"PAR-2={sum(i['grackle_par2'] for i in fold_instances)/n:.3f}"
        )
        print(
            f"    CSBS:    {csbs_solved_n}/{n} solved "
            f"({csbs_solved_n/n*100:.1f}%), "
            f"PAR-2={sum(i['csbs_par2'] for i in fold_instances)/n:.3f}"
        )

    # Aggregate results
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION RESULTS (AGGREGATED)")
    print("=" * 70)

    strategies = ["vbs", "grackle", "csbs", "sbs"]
    logics = sorted(set(i["logic"] for i in all_fold_instances))
    total = len(all_fold_instances)

    print(
        f"\n{'Strategy':<15} {'Total PAR-2':>12} {'Avg PAR-2':>12} "
        f"{'Solved':>10} {'%':>8}"
    )
    print("-" * 70)
    for strat in strategies:
        par2s = [i[f"{strat}_par2"] for i in all_fold_instances]
        solveds = [i[f"{strat}_solved"] for i in all_fold_instances]
        total_par2 = sum(par2s)
        avg_par2 = np.mean(par2s)
        solved_n = sum(solveds)
        solved_pct = solved_n / total * 100
        print(
            f"{strat.upper():<15} {total_par2:>12.2f} {avg_par2:>12.3f} "
            f"{solved_n:>10} {solved_pct:>7.1f}%"
        )
    print("=" * 70)

    # Per-logic
    for logic in logics:
        print(f"\n  {logic}:")
        logic_insts = [i for i in all_fold_instances if i["logic"] == logic]
        for strat in strategies:
            par2s = [i[f"{strat}_par2"] for i in logic_insts]
            solveds = [i[f"{strat}_solved"] for i in logic_insts]
            n = len(logic_insts)
            print(
                f"    {strat.upper():<15} PAR-2={np.mean(par2s):>8.3f}  "
                f"Solved={sum(solveds)}/{n} ({sum(solveds)/n*100:.1f}%)"
            )

    # Delta metrics
    grackle_total = sum(i["grackle_par2"] for i in all_fold_instances)
    csbs_total = sum(i["csbs_par2"] for i in all_fold_instances)
    delta = csbs_total - grackle_total
    delta_pct = delta / csbs_total * 100 if csbs_total > 0 else 0
    print(f"\nΔPAR-2 vs CSBS: {delta:.2f} ({delta_pct:.1f}% reduction)")

    disagree = sum(1 for i in all_fold_instances if not i["grackle_matches_vbs"])
    print(f"VBS disagreement: {disagree/total*100:.1f}%")

    # Mean ± std across folds
    print(f"\nMean ± Std across {len(folds)} folds:")
    key_metrics = [
        ("grackle_avg_par2", "Grackle avg PAR-2"),
        ("grackle_solved_pct", "Grackle solved%"),
        ("csbs_avg_par2", "CSBS avg PAR-2"),
        ("csbs_solved_pct", "CSBS solved%"),
        ("delta_par2_vs_csbs", "ΔPAR-2 vs CSBS (total)"),
        ("vbs_disagreement_pct", "VBS disagreement%"),
    ]
    for key, label in key_metrics:
        vals = [fm[key] for fm in per_fold_metrics]
        print(f"  {label:<30} {np.mean(vals):>10.3f} ± {np.std(vals):>8.3f}")

    # Save results
    # 1. Summary CSV
    summary_rows = []
    for strat in strategies:
        par2s = [i[f"{strat}_par2"] for i in all_fold_instances]
        solveds = [i[f"{strat}_solved"] for i in all_fold_instances]
        summary_rows.append(
            {
                "strategy": strat.upper(),
                "total_par2": sum(par2s),
                "avg_par2": np.mean(par2s),
                "solved": sum(solveds),
                "solved_pct": sum(solveds) / total * 100,
            }
        )
    summary_path = output_dir / "cli_kfold_k5_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "total_par2", "avg_par2", "solved", "solved_pct"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {summary_path}")

    # 2. Per-fold CSV
    perfold_path = output_dir / "cli_kfold_k5_perfold.csv"
    if per_fold_metrics:
        fieldnames = sorted(per_fold_metrics[0].keys())
        with open(perfold_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_fold_metrics)
    print(f"Per-fold results saved to: {perfold_path}")

    # 3. Per-logic CSV
    perlogic_path = output_dir / "cli_kfold_k5_perlogic.csv"
    perlogic_rows = []
    for logic in logics:
        logic_insts = [i for i in all_fold_instances if i["logic"] == logic]
        for strat in strategies:
            par2s = [i[f"{strat}_par2"] for i in logic_insts]
            solveds = [i[f"{strat}_solved"] for i in logic_insts]
            n = len(logic_insts)
            perlogic_rows.append(
                {
                    "logic": logic,
                    "strategy": strat.upper(),
                    "total_par2": sum(par2s),
                    "avg_par2": np.mean(par2s),
                    "solved": sum(solveds),
                    "solved_pct": sum(solveds) / n * 100,
                    "total": n,
                }
            )
    with open(perlogic_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "logic",
                "strategy",
                "total_par2",
                "avg_par2",
                "solved",
                "solved_pct",
                "total",
            ],
        )
        writer.writeheader()
        writer.writerows(perlogic_rows)
    print(f"Per-logic results saved to: {perlogic_path}")

    # 4. Mean±std CSV
    meanstd_path = output_dir / "cli_kfold_k5_meanstd.csv"
    meanstd_rows = []
    for key, label in key_metrics:
        vals = [fm[key] for fm in per_fold_metrics]
        meanstd_rows.append(
            {"metric": label, "mean": np.mean(vals), "std": np.std(vals)}
        )
    for logic in logics:
        for strat in ["grackle", "csbs", "vbs"]:
            for metric in ["avg_par2", "solved_pct"]:
                key = f"{strat}_{logic}_{metric}"
                vals = [fm[key] for fm in per_fold_metrics if key in fm]
                if vals:
                    meanstd_rows.append(
                        {
                            "metric": f"{strat.upper()} {logic} {metric}",
                            "mean": np.mean(vals),
                            "std": np.std(vals),
                        }
                    )
    with open(meanstd_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(meanstd_rows)
    print(f"Mean±std saved to: {meanstd_path}")

    print("\n" + "=" * 70)
    print("CLI Evaluation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
