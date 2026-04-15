"""SurvivalCard — survival analysis model card."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class SurvivalCard(ModelCard):
    """Per-config survival model: estimate solve probability, normalize to distribution."""

    canvas_id: ClassVar[str] = "b07bb3c175a74fed"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["distribution"]] = "distribution"

    def __init__(self, n_estimators: int = 100, method: str = "rsf") -> None:
        self.n_estimators = n_estimators
        self.method = method
        self.scaler_: Optional[RobustScaler] = None
        self.models_: Dict[int, RandomForestClassifier] = {}
        self.config_names_: List[str] = []
        self.timeout_threshold_: float = 0.0

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

        # Estimate timeout threshold as the maximum cost value
        # (instances at timeout typically have the highest cost)
        self.timeout_threshold_ = float(np.percentile(cost_matrix, 95))

        self.models_ = {}
        for c in range(n_configs):
            costs = cost_matrix[:, c]

            # Binary label: solved (1) if cost < timeout threshold, else timeout (0)
            y_solved = (costs < self.timeout_threshold_).astype(np.int64)

            # Weight by inverse runtime — faster solves are more informative
            # Normalize costs to [0, 1] range for weighting
            max_cost = costs.max()
            if max_cost > 0:
                sample_weight = 1.0 - (costs / max_cost) * 0.5  # range [0.5, 1.0]
            else:
                sample_weight = np.ones(n_instances)

            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1,
            )

            # Need at least 2 classes to train a classifier
            if len(np.unique(y_solved)) < 2:
                # All solved or all timeout — store None, will use constant prob
                self.models_[c] = None  # type: ignore[assignment]
                continue

            clf.fit(X_scaled, y_solved, sample_weight=sample_weight)
            self.models_[c] = clf

        return {"n_configs": n_configs, "method": self.method}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        # Predict solve probability per config
        solve_proba = np.zeros((n_instances, n_configs), dtype=np.float64)

        for c in range(n_configs):
            model = self.models_[c]
            if model is None:
                # Constant probability — use 0.5 as uninformative
                solve_proba[:, c] = 0.5
            else:
                # Probability of class 1 (solved)
                proba = model.predict_proba(X_scaled)
                # Find index of class 1
                class_idx = list(model.classes_).index(1)
                solve_proba[:, c] = proba[:, class_idx]

        # Normalize row-wise to get distribution (sum to 1)
        row_sums = solve_proba.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.maximum(row_sums, 1e-10)
        values = solve_proba / row_sums

        return Predictions(
            values=values,
            output_type="distribution",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
