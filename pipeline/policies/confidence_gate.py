"""Select / Confidence-Gated Fallback — select top-1 if confident, else fallback.

Canvas: ext_pol_confidence_gate | decision_type: select
Source: Cortes 2016
Gap = (score_2nd - score_1st) / score_2nd normalized to [0,1].
If gap > tau -> top-1. Else -> fallback.
"""
from typing import List

import numpy as np

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class ConfidenceGatePolicy(BasePolicy):
    """Select top-1 config when confident, otherwise fall back to a safe default."""

    def __init__(self, decision_type: str = "select", tau: float = 0.7,
                 fallback_config: str = "z3_baseline_default", **kwargs):
        self.tau = tau
        self.fallback_config = fallback_config

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        decisions = []
        for i in range(n_instances):
            ranked = self._rank_indices(values[i], predictions.output_type)
            best_idx = ranked[0]
            best_name = predictions.config_names[best_idx]

            if len(predictions.config_names) < 2:
                # Only one config — always select it
                decisions.append(Decision(
                    instance_id=predictions.instance_ids[i],
                    decision_type="select",
                    selected_config=best_name,
                    confidence=1.0,
                ))
                continue

            second_idx = ranked[1]
            best_val = values[i][best_idx]
            second_val = values[i][second_idx]

            # Compute gap based on output_type
            if predictions.output_type == "distribution":
                # Higher = better for distribution
                gap = (best_val - second_val) / best_val if best_val > 0 else 0.0
            else:
                # Lower = better for scores/ranking
                gap = (second_val - best_val) / second_val if second_val > 0 else 0.0

            gap = float(np.clip(gap, 0.0, 1.0))

            if gap >= self.tau:
                selected = best_name
            else:
                selected = self.fallback_config

            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="select",
                selected_config=selected,
                confidence=gap,
            ))
        return decisions
