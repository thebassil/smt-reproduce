"""LambdaRankCard — LambdaMART-based learning-to-rank model card."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class LambdaRankCard(ModelCard):
    """LambdaMART ranker using LightGBM for algorithm selection."""

    canvas_id: ClassVar[str] = "bb1b8120ef55d14e"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["ranking"]] = "ranking"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        feature_mode: str = "instance_only",
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_mode = feature_mode

        self.scaler_: Optional[RobustScaler] = None
        self.ranker_: Optional[LGBMRanker] = None
        self.config_names_: List[str] = []
        self.n_configs_: int = 0

    # ------------------------------------------------------------------
    def _build_ltr_features(
        self, X_scaled: np.ndarray, n_configs: int
    ) -> np.ndarray:
        """Build learning-to-rank feature matrix.

        Each instance becomes n_configs rows (one per "document").
        """
        n_instances, d = X_scaled.shape

        if self.feature_mode == "instance_only":
            # Repeat each instance vector n_configs times
            return np.repeat(X_scaled, n_configs, axis=0)
        else:
            # instance features + config one-hot
            instance_part = np.repeat(X_scaled, n_configs, axis=0)
            config_onehot = np.tile(np.eye(n_configs), (n_instances, 1))
            return np.hstack([instance_part, config_onehot])

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        self.n_configs_ = len(config_names)
        n_instances = len(features)

        X = self._stack_vectors(features)
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Build LTR features
        X_ltr = self._build_ltr_features(X_scaled, self.n_configs_)

        # Relevance labels: higher = better solver
        # Rank by cost (lower cost → higher rank number → higher relevance)
        relevance = np.zeros(n_instances * self.n_configs_, dtype=np.int32)
        for i in range(n_instances):
            # rankdata gives rank 1 = smallest; we want highest relevance for lowest cost
            ranks = rankdata(cost_matrix[i], method="min")
            rel = self.n_configs_ - ranks  # invert: best → highest label
            relevance[i * self.n_configs_ : (i + 1) * self.n_configs_] = rel.astype(
                np.int32
            )

        # Group sizes: each query (instance) has n_configs docs
        groups = np.full(n_instances, self.n_configs_, dtype=np.int32)

        from lightgbm import LGBMRanker

        self.ranker_ = LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            verbose=-1,
        )
        self.ranker_.fit(X_ltr, relevance, group=groups)

        return {"n_instances": n_instances, "n_configs": self.n_configs_}

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]

        X_ltr = self._build_ltr_features(X_scaled, self.n_configs_)
        raw_scores = self.ranker_.predict(X_ltr)

        # Convert raw scores to ranks (1 = best = highest score)
        values = np.zeros((n_instances, self.n_configs_), dtype=np.int32)
        for i in range(n_instances):
            scores_i = raw_scores[i * self.n_configs_ : (i + 1) * self.n_configs_]
            # Higher score → rank 1 (best), so negate for rankdata
            values[i] = rankdata(-scores_i, method="ordinal").astype(np.int32)

        return Predictions(
            values=values,
            output_type="ranking",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
