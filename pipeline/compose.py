"""Pipeline composition: assembles featuriser + model + policy into a runnable system.

Hydra-instantiable via ``_target_: pipeline.compose.SystemPipeline``.
"""
from __future__ import annotations

import numpy as np

from pipeline.types import Decision, Featuriser, Model, Policy
from pipeline.validate import validate_pipeline


class SystemPipeline:
    """Composes featuriser + model + policy into a runnable system."""

    def __init__(self, featuriser: Featuriser, model: Model, policy: Policy) -> None:
        validate_pipeline(featuriser, model, policy)
        self.featuriser = featuriser
        self.model = model
        self.policy = policy

    def train(
        self,
        train_paths: list[str],
        cost_matrix: np.ndarray,
        config_names: list[str],
    ) -> dict:
        """Extract features, fit model, and optionally fit policy."""
        features = self.featuriser.extract_batch(train_paths)
        metrics = self.model.fit(features, cost_matrix, config_names)
        # If policy has fit(), call it too (e.g. ClusterDispatch)
        if hasattr(self.policy, "fit") and callable(getattr(self.policy, "fit")):
            self.policy.fit(features, cost_matrix, config_names)  # type: ignore[attr-defined]
        return metrics

    def predict(self, test_paths: list[str], budget_s: float) -> list[Decision]:
        """Extract features, run model predictions, and apply policy."""
        features = self.featuriser.extract_batch(test_paths)
        predictions = self.model.predict(features)
        decisions = self.policy.decide(predictions, budget_s)
        return decisions
