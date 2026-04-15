"""Schedule / top2_split — split budget between top-k configs by ratio.

Canvas: c1f6411932544fd5 | decision_type: schedule
Source: Framework
Sub-ablations: ratio sweep (0.5, 0.6, 0.7, 0.8) and top3.
"""
from typing import List

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class Top2SplitPolicy(BasePolicy):
    """Split budget between top-k configs proportionally."""

    def __init__(self, decision_type: str = "schedule", ratio: float = 0.6,
                 k: int = 2, **kwargs):
        self.ratio = ratio
        self.k = k

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        decisions = []
        for i in range(n_instances):
            ranked = self._rank_indices(values[i], predictions.output_type)
            k = min(self.k, len(predictions.config_names))

            if k == 1:
                schedule = [(predictions.config_names[ranked[0]], budget_s)]
            elif k == 2:
                schedule = [
                    (predictions.config_names[ranked[0]], self.ratio * budget_s),
                    (predictions.config_names[ranked[1]], (1 - self.ratio) * budget_s),
                ]
            else:
                # For k > 2: first gets ratio, rest split equally
                first_budget = self.ratio * budget_s
                remaining = (1 - self.ratio) * budget_s
                per_rest = remaining / (k - 1)
                schedule = [(predictions.config_names[ranked[0]], first_budget)]
                for j in range(1, k):
                    schedule.append((predictions.config_names[ranked[j]], per_rest))

            schedule = self._clip_schedule(schedule, budget_s)
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="schedule",
                schedule=schedule,
            ))
        return decisions
