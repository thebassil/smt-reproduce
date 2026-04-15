"""Rank (ordered list) — full solver ordering without time allocation.

Canvas: 48622257267c4b25 | decision_type: rank
Source: Framework (for NDCG evaluation)
"""
from typing import List

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class RankPolicy(BasePolicy):
    """Produce a full ranking of configs from best to worst."""

    def __init__(self, decision_type: str = "rank", **kwargs):
        pass

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        decisions = []
        for i in range(n_instances):
            ranked = self._rank_indices(values[i], predictions.output_type)
            ordering = [predictions.config_names[j] for j in ranked]
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="rank",
                ranking=ordering,
            ))
        return decisions
