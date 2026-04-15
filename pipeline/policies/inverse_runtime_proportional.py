"""Schedule / Inverse-Runtime Proportional — allocate budget proportional to 1/score.

Canvas: ext_pol_proportional | decision_type: schedule
Source: FlexFolio 2016
Top-k by score -> allocate budget proportional to 1/score.
"""
from typing import List

import numpy as np

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class InverseRuntimeProportionalPolicy(BasePolicy):
    """Allocate budget to top-k configs proportional to inverse predicted score."""

    def __init__(self, decision_type: str = "schedule", k: int = 3,
                 min_fraction: float = 0.05, **kwargs):
        self.k = k
        self.min_fraction = min_fraction

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        if predictions.output_type != "scores":
            raise ValueError(
                "InverseRuntimeProportionalPolicy requires output_type='scores', "
                f"got '{predictions.output_type}'."
            )

        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        decisions = []
        for i in range(n_instances):
            ranked = self._rank_indices(values[i], predictions.output_type)
            k = min(self.k, len(predictions.config_names))
            top_k = ranked[:k]

            scores = values[i][top_k].astype(np.float64)
            # Avoid division by zero
            scores = np.clip(scores, 1e-12, None)
            inv = 1.0 / scores
            fracs = inv / inv.sum()

            # Apply floor
            fracs = np.maximum(fracs, self.min_fraction)
            fracs = fracs / fracs.sum()

            schedule = [(predictions.config_names[top_k[j]], float(fracs[j] * budget_s))
                        for j in range(k)]
            schedule = self._clip_schedule(schedule, budget_s)

            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="schedule",
                schedule=schedule,
            ))
        return decisions
