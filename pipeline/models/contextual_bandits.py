"""ContextualBanditsCard — LinUCB algorithm selection."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class ContextualBanditsCard(ModelCard):
    """LinUCB contextual bandit.  Maintains per-config A (d x d) and b (d)
    matrices.  During fit, updates are performed for the best config per
    instance.  At prediction time, UCB scores are computed and negated so
    that lower = better (matching the 'scores' contract)."""

    canvas_id: ClassVar[str] = "646b979a1d104083"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(self, alpha: float = 1.0, method: str = "linucb") -> None:
        self.alpha = alpha
        self.method = method

        self.scaler_: Optional[RobustScaler] = None
        self.A_inv_: Dict[int, np.ndarray] = {}
        self.b_: Dict[int, np.ndarray] = {}
        self.config_names_: List[str] = []
        self.d_: int = 0

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

        n_instances, d = X_scaled.shape
        n_configs = len(config_names)
        self.d_ = d

        # Initialise A and b per config
        A = {j: np.eye(d) for j in range(n_configs)}
        self.b_ = {j: np.zeros(d) for j in range(n_configs)}

        # Best config per instance = lowest cost
        best_configs = np.argmin(cost_matrix, axis=1)

        # Reward = negative cost (higher reward for lower cost)
        for i in range(n_instances):
            x = X_scaled[i]
            j = best_configs[i]
            reward = -cost_matrix[i, j]
            A[j] += np.outer(x, x)
            self.b_[j] += reward * x

        # Store A_inv
        self.A_inv_ = {j: np.linalg.inv(A[j]) for j in range(n_configs)}

        return {"n_train": n_instances, "d": d, "n_configs": n_configs}

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        values = np.zeros((n_instances, n_configs), dtype=np.float64)

        for j in range(n_configs):
            A_inv = self.A_inv_[j]
            theta = A_inv @ self.b_[j]  # (d,)
            for i in range(n_instances):
                x = X_scaled[i]
                exploit = x @ theta
                explore = self.alpha * np.sqrt(x @ A_inv @ x)
                ucb = exploit + explore
                # Negate: higher UCB = better config → lower score
                values[i, j] = -ucb

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
