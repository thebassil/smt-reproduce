"""Base classes for policy cards."""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from pipeline.types import Decision, FeatureResult, Predictions


class BasePolicy(ABC):
    """Base class satisfying the Policy protocol."""

    @abstractmethod
    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]: ...

    # --- Shared helpers ---

    @staticmethod
    def _get_values(predictions: Predictions) -> np.ndarray:
        """Coerce predictions.values to numpy array."""
        return np.asarray(predictions.values)

    @staticmethod
    def _rank_indices(row: np.ndarray, output_type: str) -> np.ndarray:
        """Return config indices sorted best-first.

        scores: lower=better (argsort ascending)
        distribution: higher=better (argsort descending)
        ranking: values ARE ranks (argsort ascending)
        """
        if output_type == "distribution":
            return np.argsort(-row)
        return np.argsort(row)  # scores or ranking

    @staticmethod
    def _clip_schedule(schedule: list, budget_s: float) -> list:
        """Clip last entry if float rounding causes sum > budget_s."""
        total = sum(t for _, t in schedule)
        if total > budget_s:
            overshoot = total - budget_s
            name, time = schedule[-1]
            schedule[-1] = (name, max(0.0, time - overshoot))
        return schedule


class BaseFittablePolicy(BasePolicy):
    """Base class satisfying the FittablePolicy protocol."""

    @abstractmethod
    def fit(self, features: List[FeatureResult], cost_matrix: np.ndarray,
            config_names: List[str]) -> None: ...
