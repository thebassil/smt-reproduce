"""Schedule / Prob-Proportional Budget — allocate budget proportional to predicted win prob.

Canvas: ext_pol_prob_proportional | decision_type: schedule
Source: Huberman 1997
Requires output_type="distribution". Floor at min_fraction * budget.
"""
from typing import List, Optional

import numpy as np

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class ProbabilityProportionalPolicy(BasePolicy):
    """Allocate budget proportional to predicted win probability."""

    def __init__(self, decision_type: str = "schedule", k: Optional[int] = None,
                 min_fraction: float = 0.05, **kwargs):
        self.k = k
        self.min_fraction = min_fraction

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        if predictions.output_type != "distribution":
            raise ValueError(
                "ProbabilityProportionalPolicy requires output_type='distribution', "
                f"got '{predictions.output_type}'."
            )

        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        n_configs = len(predictions.config_names)
        decisions = []
        for i in range(n_instances):
            probs = values[i].astype(np.float64)

            if self.k is not None:
                # Use only top-k by probability
                ranked = np.argsort(-probs)
                k = min(self.k, n_configs)
                top_k = ranked[:k]
            else:
                top_k = np.arange(n_configs)
                k = n_configs

            fracs = probs[top_k]
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
