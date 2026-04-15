"""XGBoostCard — XGBoost regression for solver runtime prediction."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class XGBoostCard(ModelCard):
    """Multi-output XGBoost regressor on log-transformed PAR-2 costs."""

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.scaler_: Optional[RobustScaler] = None
        self.model_: Optional[MultiOutputRegressor] = None
        self.config_names_: List[str] = []

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        from xgboost import XGBRegressor

        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        Y = np.log1p(cost_matrix)

        self.model_ = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1,
            )
        )
        self.model_.fit(X_scaled, Y)

        return {"n_train": X_scaled.shape[0], "n_features": X_scaled.shape[1]}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        Y_log = self.model_.predict(X_scaled)
        Y = np.expm1(Y_log)

        return Predictions(
            values=Y,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
