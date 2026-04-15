"""CollaborativeFilteringCard — matrix-factorisation algorithm selection."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class CollaborativeFilteringCard(ModelCard):
    """Decompose the cost matrix into instance/config latent factors,
    then predict costs for new instances by mapping features to the
    instance factor space via a Ridge regressor."""

    canvas_id: ClassVar[str] = "d21fe3ad818f4f88"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(self, n_factors: int = 20, method: str = "nmf") -> None:
        self.n_factors = n_factors
        self.method = method  # "nmf" or "svd"

        self.scaler_: Optional[RobustScaler] = None
        self.config_factors_: Optional[np.ndarray] = None  # (n_factors, n_configs)
        self.ridge_: Optional[Ridge] = None
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

        # Ensure non-negative costs for NMF; shift if needed
        C = cost_matrix.copy().astype(np.float64)
        if self.method == "nmf":
            c_min = C.min()
            if c_min < 0:
                C -= c_min  # shift to non-negative
        n_factors = min(self.n_factors, min(C.shape))

        # Decompose cost matrix: C ≈ W @ H
        # W = instance factors (n_instances, n_factors)
        # H = config factors  (n_factors, n_configs)
        if self.method == "nmf":
            model = NMF(n_components=n_factors, max_iter=500, random_state=42)
            W = model.fit_transform(C)
            H = model.components_
        else:
            model = TruncatedSVD(n_components=n_factors, random_state=42)
            W = model.fit_transform(C)
            H = model.components_

        self.config_factors_ = H  # (n_factors, n_configs)

        # Train Ridge: features → instance factors
        self.ridge_ = Ridge(alpha=1.0)
        self.ridge_.fit(X_scaled, W)

        return {
            "n_train": X_scaled.shape[0],
            "n_factors": n_factors,
            "method": self.method,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)

        # Map features to instance factors, then reconstruct costs
        W_pred = self.ridge_.predict(X_scaled)  # (n_instances, n_factors)
        values = W_pred @ self.config_factors_  # (n_instances, n_configs)

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
