"""GreedyLogicCard — per-logic oracle baseline (no ML)."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class GreedyLogicCard(ModelCard):
    """Pick the single best config per logic; fall back to global best."""

    canvas_id: ClassVar[str] = "mach_greedy_logic"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(self, fallback: str = "global_best") -> None:
        self.fallback = fallback
        self.logic_best_: Dict[str, int] = {}
        self.global_best_: int = 0
        self.config_names_: List[str] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        logics = self._logic_labels(features)

        # Global best: config with lowest total cost across all instances
        self.global_best_ = int(np.argmin(cost_matrix.sum(axis=0)))

        # Per-logic best
        logic_indices: Dict[str, List[int]] = {}
        for i, lg in enumerate(logics):
            key = lg if lg is not None else "__none__"
            logic_indices.setdefault(key, []).append(i)

        self.logic_best_ = {}
        for lg, idxs in logic_indices.items():
            subset = cost_matrix[idxs, :]
            self.logic_best_[lg] = int(np.argmin(subset.sum(axis=0)))

        return {
            "global_best": self.config_names_[self.global_best_],
            "logic_best": {
                lg: self.config_names_[ci] for lg, ci in self.logic_best_.items()
            },
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        n_configs = len(self.config_names_)
        n_instances = len(features)
        values = np.ones((n_instances, n_configs), dtype=np.float64)

        for i, fr in enumerate(features):
            key = fr.logic if fr.logic is not None else "__none__"
            best = self.logic_best_.get(key, self.global_best_)
            values[i, best] = 0.0

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
