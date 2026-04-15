"""Card 2: Static-Expanded Poly+PCA featuriser.

Takes StaticLightBoW(counts) output, applies PolynomialFeatures(degree=2),
then PCA(k=35). Degree-2 on 230 features → ~26K → 35 after PCA.

Has fit_batch() for PCA fitting on training data, and save_state()/load_state()
for persistence.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

from pipeline.types import FeatureResult
from pipeline.featurisers.static_light_bow import StaticLightBoW


class StaticExpanded:
    """Static-Expanded Poly+PCA featuriser (Card 2).

    Implements the Featuriser protocol with additional fit/save/load.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        n_components: int = 35,
        degree: int = 2,
        poly_only: bool = False,
        pca_only: bool = False,
        feature_timeout_s: float = 30.0,
        **kwargs,
    ):
        self.n_components = n_components
        self.degree = degree
        self.poly_only = poly_only
        self.pca_only = pca_only
        self.feature_timeout_s = feature_timeout_s

        self._bow = StaticLightBoW(mode="counts", feature_timeout_s=feature_timeout_s)

        if not poly_only:
            self._pca: Optional[PCA] = None
        if not pca_only:
            self._poly = PolynomialFeatures(degree=degree, include_bias=False)
        self._fitted = False

    @property
    def n_features(self) -> int:
        if self.poly_only:
            # Polynomial features count: C(n+d, d) - 1 for include_bias=False
            n = self._bow.n_features
            return int(self._poly.n_output_features_) if self._fitted else n
        if self.pca_only:
            return self.n_components
        return self.n_components

    def fit_batch(self, instance_paths: List[Union[str, Path]]) -> None:
        """Fit PCA on training data."""
        bow_results = self._bow.extract_batch(instance_paths)
        X = np.stack([r.features for r in bow_results])

        if not self.pca_only:
            X = self._poly.fit_transform(X)

        if not self.poly_only:
            k = min(self.n_components, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=k)
            self._pca.fit(X)

        self._fitted = True

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract expanded features from a single instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        bow_result = self._bow.extract(path)
        x = bow_result.features.reshape(1, -1)

        if not self.pca_only:
            if not self._fitted:
                self._poly.fit(x)
            x = self._poly.transform(x)

        if not self.poly_only and self._pca is not None:
            x = self._pca.transform(x)

        features = x.flatten().astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=features,
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=len(features),
            instance_id=str(path),
            logic=bow_result.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]

    def save_state(self, path: Union[str, Path]) -> None:
        """Save fitted PCA and poly state to disk."""
        state = {
            'n_components': self.n_components,
            'degree': self.degree,
            'poly_only': self.poly_only,
            'pca_only': self.pca_only,
            'fitted': self._fitted,
        }
        if not self.poly_only and self._pca is not None:
            state['pca'] = self._pca
        if not self.pca_only:
            state['poly'] = self._poly
        joblib.dump(state, path)

    def load_state(self, path: Union[str, Path]) -> None:
        """Load fitted PCA and poly state from disk."""
        state = joblib.load(path)
        self.n_components = state['n_components']
        self.degree = state['degree']
        self.poly_only = state['poly_only']
        self.pca_only = state['pca_only']
        self._fitted = state['fitted']
        if 'pca' in state:
            self._pca = state['pca']
        if 'poly' in state:
            self._poly = state['poly']
