"""OrdinalRegressionCard — ordinal regression model card."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class OrdinalRegressionCard(ModelCard):
    """Per-config ordinal regression: discretize costs into rank bins, predict ordinal class."""

    canvas_id: ClassVar[str] = "395ecdb1878943aa"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["ranking"]] = "ranking"

    def __init__(self, n_bins: int = 5, method: str = "ordinal_logistic") -> None:
        self.n_bins = n_bins
        self.method = method
        self.scaler_: Optional[RobustScaler] = None
        self.models_: Dict[int, LogisticRegression] = {}
        self.bin_edges_: Dict[int, np.ndarray] = {}
        self.config_names_: List[str] = []

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

        self.models_ = {}
        self.bin_edges_ = {}

        for c in range(n_configs):
            costs = cost_matrix[:, c]
            # Compute quantile bin edges
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(costs, quantiles)
            # Make edges unique to avoid empty bins
            edges = np.unique(edges)
            self.bin_edges_[c] = edges

            # Digitize costs into bins (bin labels 0..len(edges)-2)
            y_bins = np.digitize(costs, edges[1:-1], right=True)

            n_classes = len(np.unique(y_bins))
            if n_classes < 2:
                # All instances fall in one bin — store a trivial model
                self.models_[c] = None  # type: ignore[assignment]
                continue

            clf = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                multi_class="multinomial",
                random_state=42,
            )
            clf.fit(X_scaled, y_bins)
            self.models_[c] = clf

        return {"n_configs": n_configs, "n_bins": self.n_bins, "method": self.method}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)

        values = np.ones((n_instances, n_configs), dtype=np.int64)

        for c in range(n_configs):
            model = self.models_[c]
            if model is None:
                # Trivial case — all same rank
                values[:, c] = 1
            else:
                # Predicted ordinal class — higher class means higher cost
                # Add 1 so ranks start at 1
                values[:, c] = model.predict(X_scaled).astype(np.int64) + 1

        return Predictions(
            values=values,
            output_type="ranking",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
