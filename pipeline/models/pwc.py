"""PWCCard — Pairwise Classification model card."""
from __future__ import annotations

from itertools import combinations
from typing import ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


def _softmin(x: np.ndarray) -> np.ndarray:
    """Row-wise softmin: exp(-x) / sum(exp(-x))."""
    neg = -x
    neg -= neg.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(neg)
    return e / e.sum(axis=1, keepdims=True)


def _make_classifier(classifier_type: str):
    if classifier_type == "adaboost":
        return AdaBoostClassifier(n_estimators=50)
    elif classifier_type == "rf":
        return RandomForestClassifier(n_estimators=100)
    elif classifier_type == "svm":
        return SVC(probability=True)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")


class PWCCard(ModelCard):
    """Pairwise Classification algorithm-selection model."""

    canvas_id: ClassVar[str] = "afee4de3370a5d2e"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        classifier_type: str = "adaboost",
        aggregation: str = "majority",
        top_k_pairs: Optional[int] = None,
        use_calibrated_probs: bool = False,
    ) -> None:
        self.classifier_type = classifier_type
        self.aggregation = aggregation
        self.top_k_pairs = top_k_pairs
        self.use_calibrated_probs = use_calibrated_probs

        self.scaler_: Optional[RobustScaler] = None
        self.classifiers_: Dict[Tuple[int, int], object] = {}
        self.config_names_: List[str] = []
        self.pairs_: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)
        n_configs = len(config_names)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        all_pairs = list(combinations(range(n_configs), 2))

        # Optionally limit to top-k most discriminative pairs
        if self.top_k_pairs is not None and self.top_k_pairs < len(all_pairs):
            # Rank by variance of cost difference
            diffs = []
            for i, j in all_pairs:
                diffs.append(np.var(cost_matrix[:, i] - cost_matrix[:, j]))
            ranked = np.argsort(diffs)[::-1]
            all_pairs = [all_pairs[r] for r in ranked[: self.top_k_pairs]]

        self.pairs_ = all_pairs
        self.classifiers_ = {}
        n_trained = 0

        for i, j in self.pairs_:
            y = (cost_matrix[:, i] < cost_matrix[:, j]).astype(int)  # 1 if i beats j
            # Skip if no variance (one always dominates)
            if y.min() == y.max():
                self.classifiers_[(i, j)] = int(y[0])  # store constant prediction
                continue
            clf = _make_classifier(self.classifier_type)
            clf.fit(X_scaled, y)
            self.classifiers_[(i, j)] = clf
            n_trained += 1

        return {"n_pairs": len(self.pairs_), "n_trained": n_trained}

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        tallies = np.zeros((n_instances, n_configs), dtype=np.float64)

        for i, j in self.pairs_:
            clf = self.classifiers_[(i, j)]
            if isinstance(clf, int):
                preds = np.full(n_instances, clf)
            else:
                preds = clf.predict(X_scaled)
            # preds==1 means config i wins; preds==0 means config j wins
            wins_i = preds.astype(np.float64)
            wins_j = 1.0 - wins_i
            tallies[:, i] += wins_i
            tallies[:, j] += wins_j

        # Softmin converts win tallies to scores (lower = better)
        scores = _softmin(tallies)

        return Predictions(
            values=scores,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
