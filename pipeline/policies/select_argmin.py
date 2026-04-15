"""Select/Argmin policy card (from scratch).

Simplest policy: for each instance, select the solver with the
lowest predicted cost.

Study reference: MachSMT and Sibyl both use argmin selection.
"""

from __future__ import annotations

import random

import numpy as np

from ..types import Decision, Predictions


class SelectArgmin:
    """Argmin solver selection policy.

    For each instance, picks the solver-config with the lowest
    predicted score. Ties broken randomly.

    Hydra config: conf/policy/select_argmin.yaml
    """

    decision_type: str = "select"

    def __init__(self, tie_break: str = "random", seed: int = 42, **kwargs):
        self.tie_break = tie_break
        self._rng = random.Random(seed)

    def decide(
        self, predictions: Predictions, budget_s: float
    ) -> list[Decision]:
        """Select best solver per instance.

        Args:
            predictions: model output with values (n_instances, n_configs)
            budget_s: total budget in seconds (unused by this policy)

        Returns:
            One Decision per instance.
        """
        values = np.asarray(predictions.values)
        config_names = predictions.config_names
        instance_ids = predictions.instance_ids
        decisions = []

        for i, row in enumerate(values):
            min_val = np.min(row)
            min_indices = np.where(row == min_val)[0]

            if len(min_indices) == 1:
                idx = min_indices[0]
            elif self.tie_break == "random":
                idx = self._rng.choice(min_indices)
            else:
                # Default: first occurrence
                idx = min_indices[0]

            decisions.append(
                Decision(
                    instance_id=instance_ids[i],
                    selected_config=config_names[idx],
                    decision_type="select",
                    confidence=None,
                )
            )

        return decisions
