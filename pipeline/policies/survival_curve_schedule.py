"""Schedule / Survival-Curve Schedule — schedule solvers by early hazard.

Canvas: ext_pol_survival_sched | decision_type: schedule
Source: R&S2Survive 2025
Rank solvers by expected early hazard (1-S(t_early)). Risk-affine first
(high early hazard = fast but risky), risk-averse for remainder.
"""
from typing import List

import numpy as np

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class SurvivalCurveSchedulePolicy(BasePolicy):
    """Schedule solvers ordered by early hazard: risky-fast first, safe later."""

    def __init__(self, decision_type: str = "schedule", switch_fraction: float = 0.3,
                 n_solvers: int = 2, **kwargs):
        self.switch_fraction = switch_fraction
        self.n_solvers = n_solvers

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        if predictions.output_type == "ranking":
            raise ValueError(
                "SurvivalCurveSchedulePolicy requires output_type 'scores' or "
                "'distribution', got 'ranking'."
            )

        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        n_configs = len(predictions.config_names)
        k = min(self.n_solvers, n_configs)

        decisions = []
        for i in range(n_instances):
            row = values[i].astype(np.float64)

            if predictions.output_type == "distribution":
                # Higher prob = lower hazard. Early hazard ~ 1 - p.
                hazard = 1.0 - row
            else:
                # scores: higher score = higher hazard (worse solver)
                # Normalize to [0, 1]
                rng_val = row.max() - row.min()
                if rng_val > 0:
                    hazard = (row - row.min()) / rng_val
                else:
                    hazard = np.zeros_like(row)

            # Sort by hazard descending (high hazard = risky-fast first)
            order = np.argsort(-hazard)[:k]

            first_budget = self.switch_fraction * budget_s
            rest_budget = budget_s - first_budget

            if k == 1:
                schedule = [(predictions.config_names[order[0]], budget_s)]
            else:
                schedule = [(predictions.config_names[order[0]], first_budget)]
                per_rest = rest_budget / (k - 1)
                for j in range(1, k):
                    schedule.append((predictions.config_names[order[j]], per_rest))

            schedule = self._clip_schedule(schedule, budget_s)
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="schedule",
                schedule=schedule,
            ))
        return decisions
