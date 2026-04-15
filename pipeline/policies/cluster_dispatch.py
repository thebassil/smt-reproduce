"""Cluster-conditional dispatch — FittablePolicy using K-Means.

Canvas: 62cc4a2ee4d74ac0 | decision_type: schedule
Source: SMTGazer
fit(): K-Means on cost_matrix rows -> per-cluster greedy portfolio.
decide(): assign to nearest cluster, return cluster's schedule.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from pipeline.policies._base import BaseFittablePolicy
from pipeline.types import Decision, FeatureResult, Predictions


class ClusterDispatchPolicy(BaseFittablePolicy):
    """Cluster instances by cost profile, dispatch to per-cluster portfolios."""

    def __init__(self, decision_type: str = "schedule", n_clusters: int = 5,
                 allocation: str = "uniform", seed: int = 42, **kwargs):
        self.n_clusters = n_clusters
        self.allocation = allocation
        self.seed = seed
        self._kmeans: Optional[KMeans] = None
        self._cluster_schedules: Dict[int, List[Tuple[str, float]]] = {}
        self._config_names: List[str] = []

    def fit(self, features: List[FeatureResult], cost_matrix: np.ndarray,
            config_names: List[str]) -> None:
        self._config_names = config_names
        n_instances, n_configs = cost_matrix.shape
        k = min(self.n_clusters, n_instances)

        self._kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        self._kmeans.fit(cost_matrix)

        labels = self._kmeans.labels_
        for c in range(k):
            mask = labels == c
            if not mask.any():
                # Empty cluster: use first config
                self._cluster_schedules[c] = [(config_names[0], 1.0)]
                continue

            cluster_costs = cost_matrix[mask]
            # Greedy portfolio: pick configs that solve the most instances
            schedule = self._greedy_portfolio(cluster_costs, config_names)
            self._cluster_schedules[c] = schedule

    def _greedy_portfolio(self, cluster_costs: np.ndarray,
                          config_names: List[str]) -> List[Tuple[str, float]]:
        """Build greedy portfolio for a cluster."""
        n_instances, n_configs = cluster_costs.shape
        # Pick configs greedily by number of instances where they are fastest
        selected = []
        remaining = set(range(n_instances))

        for _ in range(min(n_configs, max(3, n_configs))):
            if not remaining:
                break
            remaining_arr = np.array(sorted(remaining))
            sub = cluster_costs[remaining_arr]
            wins = np.zeros(n_configs)
            for inst_idx in range(sub.shape[0]):
                best_cfg = np.argmin(sub[inst_idx])
                wins[best_cfg] += 1
            best = int(np.argmax(wins))
            if wins[best] == 0:
                break
            selected.append(best)
            # Remove instances where this config is best
            for inst_idx in list(remaining):
                if np.argmin(cluster_costs[inst_idx]) == best:
                    remaining.discard(inst_idx)

        if not selected:
            selected = [0]

        # Allocation
        n_sel = len(selected)
        if self.allocation == "uniform":
            fracs = [1.0 / n_sel] * n_sel
        else:
            # SMAC-style: proportional to number of instances won
            total_wins = np.zeros(len(selected))
            for idx, cfg in enumerate(selected):
                total_wins[idx] = np.sum(np.argmin(cluster_costs, axis=1) == cfg)
            s = total_wins.sum()
            fracs = (total_wins / s).tolist() if s > 0 else [1.0 / n_sel] * n_sel

        return [(config_names[cfg], frac) for cfg, frac in zip(selected, fracs)]

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        if self._kmeans is None:
            raise RuntimeError("ClusterDispatchPolicy.fit() must be called before decide().")

        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        # Assign each instance to nearest cluster using prediction vector
        labels = self._kmeans.predict(values)

        decisions = []
        for i in range(n_instances):
            cluster = int(labels[i])
            template = self._cluster_schedules.get(cluster,
                                                   [(self._config_names[0], 1.0)])
            # Scale fractions to budget
            schedule = [(name, frac * budget_s) for name, frac in template]
            schedule = self._clip_schedule(schedule, budget_s)
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="schedule",
                schedule=schedule,
            ))
        return decisions
