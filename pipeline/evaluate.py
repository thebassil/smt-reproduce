"""Cross-validation and evaluation for ablation experiments.

Provides:
  - cross_validate(): k-fold CV loop using SystemPipeline
  - evaluate_decisions(): compute metrics from decisions + cost_matrix
  - compute_vbs() / compute_sbs(): baseline references
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from pipeline.types import Decision, FoldResult

log = logging.getLogger(__name__)


def cross_validate(
    pipeline,  # SystemPipeline (not typed to avoid circular import)
    instance_ids: List[str],
    cost_matrix: np.ndarray,
    config_names: List[str],
    n_folds: int,
    seed: int,
    timeout_s: float,
) -> List[FoldResult]:
    """Run k-fold cross-validation.

    Args:
        pipeline: SystemPipeline instance (featuriser + model + policy)
        instance_ids: file paths, length n_instances
        cost_matrix: shape (n_instances, n_configs), PAR-2 values
        config_names: length n_configs
        n_folds: number of CV folds
        seed: random seed for fold splitting
        timeout_s: budget passed to policy.decide()

    Returns:
        List of FoldResult, one per fold.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ids_arr = np.array(instance_ids)
    fold_results: List[FoldResult] = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(ids_arr)):
        log.info("Fold %d/%d: train=%d test=%d", fold_id, n_folds, len(train_idx), len(test_idx))

        train_paths = ids_arr[train_idx].tolist()
        test_paths = ids_arr[test_idx].tolist()
        train_costs = cost_matrix[train_idx]

        t0 = time.perf_counter()
        train_metrics = pipeline.train(train_paths, train_costs, config_names)
        train_time = time.perf_counter() - t0
        log.info("  Training: %.1fs, metrics=%s", train_time, train_metrics)

        t0 = time.perf_counter()
        decisions = pipeline.predict(test_paths, timeout_s)
        predict_time = time.perf_counter() - t0
        log.info("  Prediction: %.1fs, %d decisions", predict_time, len(decisions))

        test_costs = cost_matrix[test_idx]
        metrics = evaluate_decisions(decisions, test_costs, config_names, timeout_s)
        metrics["train_time_s"] = train_time
        metrics["predict_time_s"] = predict_time
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)

        fold_results.append(FoldResult(
            fold_id=fold_id,
            decisions=decisions,
            metrics=metrics,
        ))
        log.info("  Metrics: PAR-2=%.2f, solved=%.1f%%, VBS_gap=%.4f",
                 metrics["par2"], metrics["solved_pct"], metrics["vbs_gap"])

    return fold_results


def evaluate_decisions(
    decisions: List[Decision],
    cost_matrix: np.ndarray,
    config_names: List[str],
    timeout_s: float,
) -> Dict[str, float]:
    """Evaluate a set of decisions against a cost matrix.

    Handles all three decision types:
      - select: look up cost of selected_config
      - schedule: simulate sequential execution
      - rank: use first-ranked config as selection

    Returns dict of metrics.
    """
    config_idx = {name: i for i, name in enumerate(config_names)}
    n_instances = cost_matrix.shape[0]
    par2_penalty = 2.0 * timeout_s

    decision_costs = np.full(n_instances, par2_penalty)

    for i, d in enumerate(decisions):
        if i >= n_instances:
            break

        if d.decision_type == "select":
            j = config_idx.get(d.selected_config)
            if j is not None:
                decision_costs[i] = cost_matrix[i, j]

        elif d.decision_type == "schedule" and d.schedule:
            decision_costs[i] = _simulate_schedule(
                d.schedule, cost_matrix[i], config_idx, timeout_s
            )

        elif d.decision_type == "rank" and d.ranking:
            j = config_idx.get(d.ranking[0])
            if j is not None:
                decision_costs[i] = cost_matrix[i, j]

    # VBS and SBS baselines
    vbs_costs = cost_matrix.min(axis=1)
    sbs_idx = cost_matrix.sum(axis=0).argmin()
    sbs_costs = cost_matrix[:, sbs_idx]

    solved = (decision_costs < par2_penalty).sum()
    vbs_solved = (vbs_costs < par2_penalty).sum()
    sbs_solved = (sbs_costs < par2_penalty).sum()

    par2 = float(decision_costs.sum())
    vbs_par2 = float(vbs_costs.sum())
    sbs_par2 = float(sbs_costs.sum())

    # VBS gap: 0 = perfect (matches VBS), 1 = as bad as all-timeout
    vbs_gap = (par2 - vbs_par2) / max(vbs_par2, 1e-9)

    # Closeness to VBS: percentage (higher = better)
    # 100% = matches VBS, 0% = matches SBS or worse
    denom = sbs_par2 - vbs_par2
    if denom > 1e-9:
        closeness_pct = 100.0 * (sbs_par2 - par2) / denom
    else:
        closeness_pct = 100.0 if par2 <= vbs_par2 + 1e-9 else 0.0

    return {
        "par2": par2,
        "solved": int(solved),
        "solved_pct": 100.0 * solved / max(n_instances, 1),
        "vbs_par2": vbs_par2,
        "vbs_solved": int(vbs_solved),
        "sbs_par2": sbs_par2,
        "sbs_solved": int(sbs_solved),
        "vbs_gap": vbs_gap,
        "closeness_to_vbs_pct": closeness_pct,
        "n_instances": n_instances,
    }


def _simulate_schedule(
    schedule: List[Tuple[str, float]],
    instance_costs: np.ndarray,
    config_idx: Dict[str, int],
    timeout_s: float,
) -> float:
    """Simulate sequential schedule execution for one instance.

    Runs each solver in order for its allocated budget. If it solves
    within budget, return actual runtime. Otherwise move to next solver.
    """
    elapsed = 0.0
    par2_penalty = 2.0 * timeout_s

    for config_name, budget in schedule:
        j = config_idx.get(config_name)
        if j is None:
            elapsed += budget
            continue

        actual_cost = instance_costs[j]
        if actual_cost <= budget:
            return elapsed + actual_cost
        else:
            elapsed += budget

    return par2_penalty


def aggregate_folds(fold_results: List[FoldResult]) -> Dict[str, float]:
    """Aggregate metrics across folds (mean and std)."""
    keys = [k for k in fold_results[0].metrics if isinstance(fold_results[0].metrics[k], (int, float))]
    agg = {}

    for k in keys:
        vals = [fr.metrics[k] for fr in fold_results]
        agg[f"mean_{k}"] = float(np.mean(vals))
        agg[f"std_{k}"] = float(np.std(vals))

    agg["n_folds"] = len(fold_results)
    return agg
