"""Select / Pairwise Voting — weighted pairwise comparison to select best config.

Canvas: ext_pol_pairwise_vote | decision_type: select
Source: SATzilla2012
FittablePolicy: fit() computes pairwise weights from cost_matrix.
decide(): config with most weighted pairwise wins is selected.
"""
from typing import List, Optional

import numpy as np

from pipeline.policies._base import BaseFittablePolicy
from pipeline.types import Decision, FeatureResult, Predictions


class PairwiseVotingPolicy(BaseFittablePolicy):
    """Select config via weighted pairwise majority voting."""

    def __init__(self, decision_type: str = "select", weighted: bool = True,
                 seed: int = 42, **kwargs):
        self.weighted = weighted
        self.seed = seed
        self._pair_weights: Optional[np.ndarray] = None
        self._config_names: List[str] = []

    def fit(self, features: List[FeatureResult], cost_matrix: np.ndarray,
            config_names: List[str]) -> None:
        self._config_names = config_names
        n_instances, n_configs = cost_matrix.shape

        # For each pair (i, j), weight = mean |cost_i - cost_j| across instances
        weights = np.zeros((n_configs, n_configs))
        for ci in range(n_configs):
            for cj in range(ci + 1, n_configs):
                w = np.mean(np.abs(cost_matrix[:, ci] - cost_matrix[:, cj]))
                weights[ci, cj] = w
                weights[cj, ci] = w

        self._pair_weights = weights

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        if self._pair_weights is None:
            raise RuntimeError("PairwiseVotingPolicy.fit() must be called before decide().")

        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        n_configs = len(predictions.config_names)
        decisions = []
        for i in range(n_instances):
            votes = np.zeros(n_configs)
            row = values[i]

            for ci in range(n_configs):
                for cj in range(ci + 1, n_configs):
                    # Lower score = better for scores/ranking
                    if predictions.output_type == "distribution":
                        # Higher = better
                        winner = ci if row[ci] >= row[cj] else cj
                    else:
                        winner = ci if row[ci] <= row[cj] else cj

                    w = float(self._pair_weights[ci, cj]) if self.weighted else 1.0
                    votes[winner] += w

            best = int(np.argmax(votes))
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="select",
                selected_config=predictions.config_names[best],
            ))
        return decisions
