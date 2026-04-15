"""Schedule / Presolver-Then-Select — run presolver first, then predicted best.

Canvas: ext_pol_presolver | decision_type: schedule
Source: 3S 2011
schedule=[(presolver, frac*budget), (predicted_best, (1-frac)*budget)]
"""
from typing import List

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class PresolverThenSelectPolicy(BasePolicy):
    """Two-phase schedule: presolver gets a fraction, predicted best gets remainder."""

    def __init__(self, decision_type: str = "schedule",
                 presolver_config: str = "z3_baseline_default",
                 fraction: float = 0.1, **kwargs):
        self.presolver_config = presolver_config
        self.fraction = fraction

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        decisions = []
        for i in range(n_instances):
            ranked = self._rank_indices(values[i], predictions.output_type)
            best = predictions.config_names[ranked[0]]

            pre_budget = self.fraction * budget_s
            main_budget = budget_s - pre_budget

            if best == self.presolver_config:
                # Presolver IS the best — give it everything
                schedule = [(best, budget_s)]
            else:
                schedule = [
                    (self.presolver_config, pre_budget),
                    (best, main_budget),
                ]

            schedule = self._clip_schedule(schedule, budget_s)
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="schedule",
                schedule=schedule,
            ))
        return decisions
