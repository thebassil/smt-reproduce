"""KNNCard — k-nearest-neighbours algorithm selection."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class KNNCard(ModelCard):
    """Predict solver costs by averaging the cost vectors of k nearest neighbours."""

    canvas_id: ClassVar[str] = "83e4a83629a66a2c"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
        include_failures: bool = True,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.include_failures = include_failures

        self.scaler_: Optional[RobustScaler] = None
        self.nn_: Optional[NearestNeighbors] = None
        self.cost_matrix_: Optional[np.ndarray] = None
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

        self.cost_matrix_ = cost_matrix.copy()

        self.nn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )
        self.nn_.fit(X_scaled)

        return {"n_train": X_scaled.shape[0], "n_neighbors": self.n_neighbors}

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)

        distances, indices = self.nn_.kneighbors(X_scaled)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)
        values = np.zeros((n_instances, n_configs), dtype=np.float64)

        for i in range(n_instances):
            neighbour_costs = self.cost_matrix_[indices[i]]  # (k, n_configs)
            if self.weights == "distance":
                d = distances[i]
                # Avoid division by zero for exact matches
                w = 1.0 / np.maximum(d, 1e-12)
                w /= w.sum()
                values[i] = np.average(neighbour_costs, axis=0, weights=w)
            else:
                values[i] = neighbour_costs.mean(axis=0)

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
