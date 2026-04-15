"""QuantileRegressionCard — LightGBM quantile regression model card."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class QuantileRegressionCard(ModelCard):
    """Per-config quantile regression via LightGBM."""

    canvas_id: ClassVar[str] = "9d23fe8aca2d4f31"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(self, quantile: float = 0.1, n_estimators: int = 200) -> None:
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.scaler_: Optional[RobustScaler] = None
        self.models_: Dict[int, object] = {}
        self.config_names_: List[str] = []

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        import lightgbm as lgb

        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)
        n_instances, n_configs = cost_matrix.shape

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.models_ = {}
        for c in range(n_configs):
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=self.quantile,
                n_estimators=self.n_estimators,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X_scaled, cost_matrix[:, c])
            self.models_[c] = model

        return {"n_configs": n_configs, "quantile": self.quantile}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        values = np.zeros((n_instances, n_configs), dtype=np.float64)
        for c in range(n_configs):
            values[:, c] = self.models_[c].predict(X_scaled)

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
