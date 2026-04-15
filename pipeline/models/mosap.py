"""MOSAPCard — Multi-Objective Solver-Algorithm Portfolio selection."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class MOSAPCard(ModelCard):
    """Multi-objective algorithm selection: jointly optimise solve-rate
    (binary) and runtime, then combine via weighted sum into a single
    score per config."""

    canvas_id: ClassVar[str] = "b697ae928fff4f81"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        solve_weight: float = 0.5,
        runtime_weight: float = 0.5,
    ) -> None:
        self.solve_weight = solve_weight
        self.runtime_weight = runtime_weight

        self.scaler_: Optional[RobustScaler] = None
        self.solve_models_: List[RandomForestClassifier] = []
        self.runtime_models_: List[RandomForestRegressor] = []
        self.config_names_: List[str] = []
        self.timeout_threshold_: float = 0.0

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        n_configs = len(config_names)

        # Timeout threshold: 90th percentile of all costs
        self.timeout_threshold_ = float(np.percentile(cost_matrix, 90))

        # Binary solved indicator: cost < threshold
        solved = (cost_matrix < self.timeout_threshold_).astype(int)

        self.solve_models_ = []
        self.runtime_models_ = []
        for j in range(n_configs):
            # Solve classifier
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            )
            clf.fit(X_scaled, solved[:, j])
            self.solve_models_.append(clf)

            # Runtime regressor
            reg = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            )
            reg.fit(X_scaled, cost_matrix[:, j])
            self.runtime_models_.append(reg)

        return {
            "n_train": X_scaled.shape[0],
            "timeout_threshold": self.timeout_threshold_,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        solve_probs = np.zeros((n_instances, n_configs), dtype=np.float64)
        runtime_preds = np.zeros((n_instances, n_configs), dtype=np.float64)

        for j in range(n_configs):
            solve_probs[:, j] = self.solve_models_[j].predict_proba(X_scaled)[:, 1]
            runtime_preds[:, j] = self.runtime_models_[j].predict(X_scaled)

        # Normalise runtime to [0, 1] for combination (lower is better)
        rt_min = runtime_preds.min(axis=1, keepdims=True)
        rt_max = runtime_preds.max(axis=1, keepdims=True)
        rt_range = np.maximum(rt_max - rt_min, 1e-12)
        runtime_norm = (runtime_preds - rt_min) / rt_range

        # Combine: lower score = better.
        # Higher solve prob is better → use (1 - solve_prob)
        values = (
            self.solve_weight * (1.0 - solve_probs)
            + self.runtime_weight * runtime_norm
        )

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
