"""InstanceClusteringCard — cluster-based algorithm selection."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class InstanceClusteringCard(ModelCard):
    """Cluster instances, then use per-cluster cost statistics for selection."""

    canvas_id: ClassVar[str] = "38f11d5d8759d9fe"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 4,
        max_k: int = 20,
        assignment: str = "hard",
    ) -> None:
        self.method = method
        self.n_clusters = n_clusters
        self.max_k = max_k
        self.assignment = assignment

        self.scaler_: Optional[RobustScaler] = None
        self.cluster_model_: object = None
        self.cluster_scores_: Optional[np.ndarray] = None  # (n_clusters, n_configs)
        self.config_names_: List[str] = []
        self.n_clusters_: int = 0

    # ------------------------------------------------------------------
    def _fit_xmeans(self, X: np.ndarray) -> object:
        """X-means: iteratively split clusters while BIC improves."""
        best_model = KMeans(n_clusters=2, n_init=10, random_state=42)
        best_model.fit(X)
        best_bic = self._kmeans_bic(X, best_model)

        for k in range(3, self.max_k + 1):
            candidate = KMeans(n_clusters=k, n_init=10, random_state=42)
            candidate.fit(X)
            bic = self._kmeans_bic(X, candidate)
            if bic < best_bic:
                best_bic = bic
                best_model = candidate
            else:
                break  # BIC stopped improving

        return best_model

    @staticmethod
    def _kmeans_bic(X: np.ndarray, model: KMeans) -> float:
        """Approximate BIC for KMeans."""
        n, d = X.shape
        k = model.n_clusters
        labels = model.labels_

        # Compute within-cluster variance
        variances = []
        for c in range(k):
            members = X[labels == c]
            if len(members) > 1:
                variances.append(np.mean(np.var(members, axis=0)))
            else:
                variances.append(1e-6)
        avg_var = np.mean(variances) + 1e-12

        # Log-likelihood approximation
        ll = 0.0
        for c in range(k):
            n_c = np.sum(labels == c)
            if n_c > 0:
                ll += n_c * np.log(n_c / n)
                ll -= n_c * d * 0.5 * np.log(2 * np.pi * avg_var)
                ll -= (n_c - 1) * d * 0.5

        # Number of free parameters: k*d centres + k-1 mixing + 1 variance
        n_params = k * d + k
        bic = -2.0 * ll + n_params * np.log(n)
        return bic

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

        # Fit cluster model
        if self.method == "kmeans":
            self.cluster_model_ = KMeans(
                n_clusters=self.n_clusters, n_init=10, random_state=42
            )
            self.cluster_model_.fit(X_scaled)
            self.n_clusters_ = self.n_clusters
        elif self.method == "gmm":
            self.cluster_model_ = GaussianMixture(
                n_components=self.n_clusters, random_state=42
            )
            self.cluster_model_.fit(X_scaled)
            self.n_clusters_ = self.n_clusters
        elif self.method == "xmeans":
            self.cluster_model_ = self._fit_xmeans(X_scaled)
            self.n_clusters_ = self.cluster_model_.n_clusters
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Compute per-cluster mean costs
        if self.method == "gmm":
            labels = self.cluster_model_.predict(X_scaled)
        else:
            labels = self.cluster_model_.labels_

        self.cluster_scores_ = np.zeros(
            (self.n_clusters_, len(config_names)), dtype=np.float64
        )
        for c in range(self.n_clusters_):
            mask = labels == c
            if mask.any():
                self.cluster_scores_[c] = cost_matrix[mask].mean(axis=0)
            else:
                self.cluster_scores_[c] = cost_matrix.mean(axis=0)

        return {"n_clusters": self.n_clusters_, "method": self.method}

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        if self.assignment == "hard":
            if self.method == "gmm":
                labels = self.cluster_model_.predict(X_scaled)
            else:
                labels = self.cluster_model_.predict(X_scaled)
            values = self.cluster_scores_[labels]
        else:
            # Soft assignment: weight by softmax(-distance)
            if self.method == "gmm":
                # Use responsibility (posterior) as soft weights
                probs = self.cluster_model_.predict_proba(X_scaled)  # (n, k)
            else:
                # Distance to each cluster centre
                centres = self.cluster_model_.cluster_centers_  # (k, d)
                dists = np.linalg.norm(
                    X_scaled[:, None, :] - centres[None, :, :], axis=2
                )  # (n, k)
                neg_d = -dists
                neg_d -= neg_d.max(axis=1, keepdims=True)
                probs = np.exp(neg_d)
                probs /= probs.sum(axis=1, keepdims=True)

            # Weighted combination of cluster scores
            values = probs @ self.cluster_scores_  # (n, n_configs)

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
