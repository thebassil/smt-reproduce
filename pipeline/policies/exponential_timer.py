"""Schedule / exponential_timer — exponentially decaying time allocation.

Canvas: b508b4c3e46e420c | decision_type: schedule
Source: MedleySolver
Solver i gets budget * alpha * (1-alpha)^i, remainder to last.
"""
from typing import List

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class ExponentialTimerPolicy(BasePolicy):
    """Allocate budget with exponential decay across top-k solvers."""

    def __init__(self, decision_type: str = "schedule", alpha: float = 0.5,
                 k: int = 3, **kwargs):
        self.alpha = alpha
        self.k = k

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        if predictions.output_type == "ranking":
            raise ValueError(
                "ExponentialTimerPolicy requires output_type 'scores' or "
                "'distribution', got 'ranking'."
            )

        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        decisions = []
        for i in range(n_instances):
            ranked = self._rank_indices(values[i], predictions.output_type)
            k = min(self.k, len(predictions.config_names))

            schedule = []
            allocated = 0.0
            for j in range(k - 1):
                t = budget_s * self.alpha * ((1 - self.alpha) ** j)
                schedule.append((predictions.config_names[ranked[j]], t))
                allocated += t
            # Remainder to last solver
            schedule.append((predictions.config_names[ranked[k - 1]],
                             budget_s - allocated))

            schedule = self._clip_schedule(schedule, budget_s)
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="schedule",
                schedule=schedule,
            ))
        return decisions
