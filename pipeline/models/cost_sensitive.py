"""CostSensitiveCard — cost-sensitive random forest model card."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class CostSensitiveCard(ModelCard):
    """Cost-sensitive RF: classify best config per instance, weighted by cost gap."""

    canvas_id: ClassVar[str] = "c6e12f5df8814e94"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(self, n_estimators: int = 100, cost_metric: str = "par2") -> None:
        self.n_estimators = n_estimators
        self.cost_metric = cost_metric
        self.scaler_: Optional[RobustScaler] = None
        self.clf_: Optional[RandomForestClassifier] = None
        self.config_names_: List[str] = []

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)
        n_instances, n_configs = cost_matrix.shape

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Labels: best config per instance
        y = np.argmin(cost_matrix, axis=1)

        # Sample weights: gap between worst and best cost per instance
        worst_cost = np.max(cost_matrix, axis=1)
        best_cost = np.min(cost_matrix, axis=1)
        sample_weight = worst_cost - best_cost
        # Avoid zero weights
        sample_weight = np.maximum(sample_weight, 1e-8)

        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.clf_.fit(X_scaled, y, sample_weight=sample_weight)

        return {"n_configs": n_configs, "cost_metric": self.cost_metric}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)

        # Predict class probabilities
        proba = self.clf_.predict_proba(X_scaled)  # (n_instances, n_classes_seen)

        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        # Map probabilities to full config space (some classes may be absent)
        full_proba = np.zeros((n_instances, n_configs), dtype=np.float64)
        for i, cls in enumerate(self.clf_.classes_):
            full_proba[:, cls] = proba[:, i]

        # Scores: 1 - probability (lower = better, most probable gets lowest score)
        values = 1.0 - full_proba

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
