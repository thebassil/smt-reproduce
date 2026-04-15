"""AutoFolioCard — AutoFolio-style cost-sensitive RF model card."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class AutoFolioCard(ModelCard):
    """Cost-sensitive RF weighted by margin between best and second-best config."""

    canvas_id: ClassVar[str] = "59f57a4526b44dd6"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        cost_sensitive: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cost_sensitive = cost_sensitive
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

        # Sample weights: margin between best and second-best cost
        sorted_costs = np.sort(cost_matrix, axis=1)
        best_cost = sorted_costs[:, 0]
        second_best_cost = sorted_costs[:, 1] if n_configs > 1 else best_cost
        margin = second_best_cost - best_cost
        # Avoid zero weights
        sample_weight = np.maximum(margin, 1e-8) if self.cost_sensitive else None

        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.clf_.fit(X_scaled, y, sample_weight=sample_weight)

        return {"n_configs": n_configs, "cost_sensitive": self.cost_sensitive}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)

        proba = self.clf_.predict_proba(X_scaled)

        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        # Map probabilities to full config space
        full_proba = np.zeros((n_instances, n_configs), dtype=np.float64)
        for i, cls in enumerate(self.clf_.classes_):
            full_proba[:, cls] = proba[:, i]

        # Scores: 1 - probability (lower = better)
        values = 1.0 - full_proba

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
