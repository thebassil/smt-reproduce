"""GPRegressionCard — Gaussian Process regression model card."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class GPRegressionCard(ModelCard):
    """Per-config Gaussian Process regression on instance features."""

    canvas_id: ClassVar[str] = "7416a9e6e340481a"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(self, n_inducing: int = 50, kernel: str = "rbf") -> None:
        self.n_inducing = n_inducing
        self.kernel = kernel
        self.scaler_: Optional[RobustScaler] = None
        self.models_: Dict[int, GaussianProcessRegressor] = {}
        self.config_names_: List[str] = []

    def _make_kernel(self):
        if self.kernel == "matern":
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        return ConstantKernel(1.0) * RBF(length_scale=1.0)

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

        # Subsample if dataset is large (GP scales cubically)
        if n_instances > self.n_inducing:
            rng = np.random.RandomState(42)
            inducing_idx = rng.choice(n_instances, self.n_inducing, replace=False)
            X_train = X_scaled[inducing_idx]
            cost_train = cost_matrix[inducing_idx]
        else:
            X_train = X_scaled
            cost_train = cost_matrix

        self.models_ = {}
        for c in range(n_configs):
            gp = GaussianProcessRegressor(
                kernel=self._make_kernel(),
                alpha=1e-2,
                normalize_y=True,
                random_state=42,
            )
            gp.fit(X_train, cost_train[:, c])
            self.models_[c] = gp

        return {"n_configs": n_configs, "n_inducing": X_train.shape[0], "kernel": self.kernel}

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
