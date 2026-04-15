"""OnlineLearningCard — SGD-based incremental algorithm selection."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class OnlineLearningCard(ModelCard):
    """Per-config SGDRegressor models that support incremental updates
    via partial_fit.  A replay buffer stores recent training examples
    for experience replay during incremental learning."""

    canvas_id: ClassVar[str] = "915b79c9103f453c"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(self, lr: float = 0.01, buffer_size: int = 1000) -> None:
        self.lr = lr
        self.buffer_size = buffer_size

        self.scaler_: Optional[RobustScaler] = None
        self.models_: List[SGDRegressor] = []
        self.config_names_: List[str] = []
        # Replay buffer
        self.buffer_X_: Optional[np.ndarray] = None
        self.buffer_C_: Optional[np.ndarray] = None
        self.buffer_pos_: int = 0
        self.buffer_full_: bool = False

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

        self.models_ = []
        for j in range(n_configs):
            m = SGDRegressor(
                loss="squared_error",
                learning_rate="constant",
                eta0=self.lr,
                random_state=42,
                max_iter=1000,
                tol=1e-4,
            )
            m.fit(X_scaled, cost_matrix[:, j])
            self.models_.append(m)

        # Initialise replay buffer with tail of training data
        n = X_scaled.shape[0]
        buf_n = min(n, self.buffer_size)
        self.buffer_X_ = X_scaled[-buf_n:].copy()
        self.buffer_C_ = cost_matrix[-buf_n:].copy()
        self.buffer_pos_ = 0
        self.buffer_full_ = buf_n == self.buffer_size

        return {"n_train": n, "n_configs": n_configs}

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)

        values = np.column_stack([m.predict(X_scaled) for m in self.models_])

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )

    # ------------------------------------------------------------------
    def partial_fit(
        self,
        features: List[FeatureResult],
        costs: np.ndarray,
    ) -> None:
        """Incrementally update models with new observations.

        Parameters
        ----------
        features : list of FeatureResult
        costs : ndarray of shape (n_new, n_configs)
        """
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)

        for j, m in enumerate(self.models_):
            m.partial_fit(X_scaled, costs[:, j])

        # Update replay buffer (ring buffer)
        n_new = X_scaled.shape[0]
        for i in range(n_new):
            if self.buffer_X_ is None:
                d = X_scaled.shape[1]
                nc = costs.shape[1]
                self.buffer_X_ = np.zeros((self.buffer_size, d))
                self.buffer_C_ = np.zeros((self.buffer_size, nc))
            self.buffer_X_[self.buffer_pos_] = X_scaled[i]
            self.buffer_C_[self.buffer_pos_] = costs[i]
            self.buffer_pos_ = (self.buffer_pos_ + 1) % self.buffer_size
            if self.buffer_pos_ == 0:
                self.buffer_full_ = True
