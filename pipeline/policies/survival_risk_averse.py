"""Select / Survival Risk-Averse — minimize expected loss under exponential risk.

Canvas: ext_pol_survival_risk | decision_type: select
Source: Run2Survive 2020
Expects output_type="distribution" encoding discretized survival S(t) per solver
per time bin. Expected loss = sum (1-S(t_i)) * dt * exp(risk * t_i).
Pick solver minimizing loss.
"""
from typing import List

import numpy as np

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class SurvivalRiskAversePolicy(BasePolicy):
    """Select config that minimizes risk-weighted expected loss from survival curves."""

    def __init__(self, decision_type: str = "select", risk_parameter: float = 1.0,
                 **kwargs):
        self.risk_parameter = risk_parameter

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        if predictions.output_type != "distribution":
            raise ValueError(
                "SurvivalRiskAversePolicy requires output_type='distribution', "
                f"got '{predictions.output_type}'."
            )

        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        n_configs = len(predictions.config_names)

        # When predictions are simple distribution (probability of being best),
        # higher prob = lower risk. Compute expected loss as (1-p) * exp(risk * rank).
        decisions = []
        for i in range(n_instances):
            probs = values[i].astype(np.float64)

            # Expected loss per config: (1 - p_j) weighted by risk
            # For single-row distribution, treat prob as survival proxy
            losses = np.zeros(n_configs)
            ranked = np.argsort(-probs)  # best prob first
            for rank_pos, cfg_idx in enumerate(ranked):
                p = probs[cfg_idx]
                # Loss = (1 - p) * exp(risk * normalized_rank)
                norm_rank = rank_pos / max(1, n_configs - 1)
                losses[cfg_idx] = (1.0 - p) * np.exp(self.risk_parameter * norm_rank)

            best = int(np.argmin(losses))
            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="select",
                selected_config=predictions.config_names[best],
                confidence=float(probs[best]),
            ))
        return decisions
