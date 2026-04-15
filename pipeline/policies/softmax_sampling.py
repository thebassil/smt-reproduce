"""Select / Softmax Sampling — sample config from softmax over negated scores.

Canvas: ext_pol_softmax | decision_type: select
Source: GraSS 2024
Negate scores (lower=better -> higher=better), apply softmax with temperature T,
sample from resulting distribution.
"""
from typing import List

import numpy as np
from scipy.special import softmax

from pipeline.policies._base import BasePolicy
from pipeline.types import Decision, Predictions


class SoftmaxSamplingPolicy(BasePolicy):
    """Sample a config from a softmax distribution over prediction scores."""

    def __init__(self, decision_type: str = "select", temperature: float = 1.0,
                 seed: int = 42, **kwargs):
        self.temperature = temperature
        self.seed = seed

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]:
        values = self._get_values(predictions)
        n_instances = values.shape[0]
        if n_instances == 0:
            return []

        rng = np.random.default_rng(self.seed)
        decisions = []
        for i in range(n_instances):
            row = values[i].astype(np.float64)

            if predictions.output_type == "distribution":
                # Already probabilities — use directly as logits
                logits = np.log(np.clip(row, 1e-12, None))
            else:
                # scores/ranking: lower = better, so negate
                logits = -row

            probs = softmax(logits / self.temperature)
            chosen = rng.choice(len(predictions.config_names), p=probs)

            decisions.append(Decision(
                instance_id=predictions.instance_ids[i],
                decision_type="select",
                selected_config=predictions.config_names[chosen],
                confidence=float(probs[chosen]),
            ))
        return decisions
