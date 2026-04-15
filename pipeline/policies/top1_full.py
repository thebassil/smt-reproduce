"""Schedule / top1_full — single-entry schedule giving entire budget to top config.

Canvas: 8913d13c3b794007 | decision_type: schedule
Source: Framework (degenerate case testing scheduling overhead)
"""
from typing import List

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class Top1FullPolicy(BasePolicy):
    """Allocate the entire budget to the single best-predicted config."""

    def __init__(self, decision_type: str = "schedule", **kwargs):
        pass

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        decisions = []
        for i in range(n_instances):
            ranked = self._rank_indices(values[i], predictions.output_type)
            best = predictions.config_names[ranked[0]]
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="schedule",
                schedule=[(best, budget_s)],
            ))
        return decisions
