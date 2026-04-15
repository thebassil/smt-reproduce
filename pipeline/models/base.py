"""ModelCard ABC — base class for all model cards."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Union

import joblib
import numpy as np

from pipeline.types import FeatureResult, Predictions


class ModelCard(ABC):
    """Abstract base for all model cards. Implements the Model protocol."""

    input_type: ClassVar[Literal["VECTOR", "GRAPH"]]
    output_type: ClassVar[Literal["scores", "ranking", "distribution"]]

    @abstractmethod
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        ...

    @abstractmethod
    def predict(self, features: List[FeatureResult]) -> Predictions:
        ...

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self, Path(path))

    def load(self, path: Union[str, Path]) -> None:
        loaded = joblib.load(Path(path))
        self.__dict__.update(loaded.__dict__)

    # --- helpers for VECTOR cards ---
    @staticmethod
    def _stack_vectors(features: List[FeatureResult]) -> np.ndarray:
        """Stack FeatureResult.features into (n_instances, n_features) array."""
        return np.vstack([fr.features for fr in features])

    @staticmethod
    def _instance_ids(features: List[FeatureResult]) -> List[str]:
        return [fr.instance_id for fr in features]

    @staticmethod
    def _logic_labels(features: List[FeatureResult]) -> List[Optional[str]]:
        return [fr.logic for fr in features]
