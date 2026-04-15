"""ConformalCard — conformal-prediction algorithm selection."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class ConformalCard(ModelCard):
    """Train a base regressor and use split-conformal calibration to produce
    prediction intervals.  Configs whose predicted cost falls within the
    conformal interval receive equal probability; others are down-weighted.
    The result is a softmax-normalised distribution over configs."""

    canvas_id: ClassVar[str] = "cb5f5a7ce3cf4db6"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["distribution"]] = "distribution"

    def __init__(self, alpha: float = 0.1, base_model: str = "ridge") -> None:
        self.alpha = alpha  # miscoverage rate
        self.base_model = base_model

        self.scaler_: Optional[RobustScaler] = None
        self.models_: List[Ridge] = []
        self.quantile_: float = 0.0
        self.config_names_: List[str] = []

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

        n = X_scaled.shape[0]
        n_configs = len(config_names)
        split = int(n * 0.8)

        X_train, X_cal = X_scaled[:split], X_scaled[split:]
        C_train, C_cal = cost_matrix[:split], cost_matrix[split:]

        # Train one Ridge per config
        self.models_ = []
        for j in range(n_configs):
            m = Ridge(alpha=1.0)
            m.fit(X_train, C_train[:, j])
            self.models_.append(m)

        # Compute nonconformity scores on calibration set
        if X_cal.shape[0] > 0:
            preds_cal = np.column_stack([m.predict(X_cal) for m in self.models_])
            residuals = np.abs(preds_cal - C_cal)
            # Use the per-instance max residual across configs
            nc_scores = residuals.max(axis=1)
            q_level = min((1 - self.alpha) * (1 + 1 / len(nc_scores)), 1.0)
            self.quantile_ = float(np.quantile(nc_scores, q_level))
        else:
            self.quantile_ = float("inf")

        return {
            "n_train": split,
            "n_cal": n - split,
            "quantile": self.quantile_,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        preds = np.column_stack([m.predict(X_scaled) for m in self.models_])

        # For each instance, find the best predicted cost and build the
        # conformal interval around it.  Configs within the interval get
        # equal weight; others get a small residual weight.
        values = np.zeros((n_instances, n_configs), dtype=np.float64)
        for i in range(n_instances):
            best_cost = preds[i].min()
            in_interval = np.abs(preds[i] - best_cost) <= self.quantile_
            logits = np.where(in_interval, 0.0, -10.0)
            # Softmax normalisation
            logits -= logits.max()
            exp_logits = np.exp(logits)
            values[i] = exp_logits / exp_logits.sum()

        return Predictions(
            values=values,
            output_type="distribution",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
