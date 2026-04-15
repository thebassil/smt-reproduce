"""SolverLogicPWCCard — per-logic PWC with pooled fallback."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np

from pipeline.models.base import ModelCard
from pipeline.models.pwc import PWCCard
from pipeline.types import FeatureResult, Predictions


class SolverLogicPWCCard(ModelCard):
    """Train a separate PWC model per logic; fall back to a pooled model."""

    canvas_id: ClassVar[str] = "954b5e535e6a238e"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        fallback: str = "pooled",
        min_instances_per_logic: int = 10,
        classifier_type: str = "adaboost",
    ) -> None:
        self.fallback = fallback
        self.min_instances_per_logic = min_instances_per_logic
        self.classifier_type = classifier_type

        self.logic_models_: Dict[str, PWCCard] = {}
        self.pooled_model_: Optional[PWCCard] = None
        self.config_names_: List[str] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)

        # Group by logic
        logic_groups: Dict[str, List[int]] = {}
        for i, fr in enumerate(features):
            key = fr.logic if fr.logic is not None else "__none__"
            logic_groups.setdefault(key, []).append(i)

        # Train pooled model on all data
        self.pooled_model_ = PWCCard(classifier_type=self.classifier_type)
        pooled_info = self.pooled_model_.fit(features, cost_matrix, config_names)

        # Train per-logic models
        self.logic_models_ = {}
        per_logic_info: Dict[str, dict] = {}
        for lg, idxs in logic_groups.items():
            if len(idxs) < self.min_instances_per_logic:
                continue
            sub_features = [features[i] for i in idxs]
            sub_costs = cost_matrix[idxs, :]
            model = PWCCard(classifier_type=self.classifier_type)
            info = model.fit(sub_features, sub_costs, config_names)
            self.logic_models_[lg] = model
            per_logic_info[lg] = info

        return {
            "pooled": pooled_info,
            "per_logic": per_logic_info,
            "logics_with_own_model": list(self.logic_models_.keys()),
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        n_configs = len(self.config_names_)
        n_instances = len(features)
        values = np.zeros((n_instances, n_configs), dtype=np.float64)

        # Group instances by routing target
        pooled_idxs: List[int] = []
        logic_batches: Dict[str, List[int]] = {}

        for i, fr in enumerate(features):
            key = fr.logic if fr.logic is not None else "__none__"
            if key in self.logic_models_:
                logic_batches.setdefault(key, []).append(i)
            else:
                pooled_idxs.append(i)

        # Predict with per-logic models
        for lg, idxs in logic_batches.items():
            sub_features = [features[i] for i in idxs]
            preds = self.logic_models_[lg].predict(sub_features)
            for local_i, global_i in enumerate(idxs):
                values[global_i] = preds.values[local_i]

        # Predict with pooled model for remaining
        if pooled_idxs:
            sub_features = [features[i] for i in pooled_idxs]
            preds = self.pooled_model_.predict(sub_features)
            for local_i, global_i in enumerate(pooled_idxs):
                values[global_i] = preds.values[local_i]

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
