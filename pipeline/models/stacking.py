"""StackingCard — Stacked generalisation model card."""
from __future__ import annotations

import copy
from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class StackingCard(ModelCard):
    """Stacked generalisation (stacking) for algorithm selection.

    Takes a list of base ModelCard instances and combines their
    out-of-fold predictions with a meta-learner (Ridge or LightGBM).
    If no base models are provided, falls back to a simple Ridge
    regression on the raw feature vectors.
    """

    canvas_id: ClassVar[str] = "1728d7c6fad14531"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        meta_learner: str = "ridge",
        base_models: Optional[List[ModelCard]] = None,
    ) -> None:
        self.meta_learner_type = meta_learner
        self.base_models_template = base_models

        self.scaler_: Optional[RobustScaler] = None
        self.config_names_: List[str] = []
        self.fitted_base_models_: List[ModelCard] = []
        self.meta_learners_: List[object] = []  # one per config column
        self.fallback_mode_: bool = False
        self.n_configs_: int = 0

    def _make_meta_learner(self):
        if self.meta_learner_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.meta_learner_type == "lgbm":
            try:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(n_estimators=100, verbose=-1)
            except ImportError:
                # Fall back to Ridge if LightGBM unavailable
                return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown meta_learner: {self.meta_learner_type}")

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

        # Fallback: no base models, just Ridge on raw features
        if self.base_models_template is None or len(self.base_models_template) == 0:
            self.fallback_mode_ = True
            self.meta_learners_ = []
            for c in range(self.n_configs_):
                reg = self._make_meta_learner()
                reg.fit(X_scaled, cost_matrix[:, c])
                self.meta_learners_.append(reg)
            return {"mode": "fallback", "n_configs": self.n_configs_}

        self.fallback_mode_ = False
        n_base = len(self.base_models_template)
        n_folds = 3
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Collect out-of-fold predictions from each base model
        oof_preds = np.zeros((n_instances, self.n_configs_ * n_base), dtype=np.float64)

        self.fitted_base_models_ = []
        for b_idx, base_template in enumerate(self.base_models_template):
            # Full-data fit for later prediction
            full_model = copy.deepcopy(base_template)
            full_model.fit(features, cost_matrix, config_names)
            self.fitted_base_models_.append(full_model)

            # K-fold OOF predictions
            col_start = b_idx * self.n_configs_
            col_end = col_start + self.n_configs_

            for train_idx, val_idx in kf.split(X_scaled):
                fold_features = [features[i] for i in train_idx]
                fold_cost = cost_matrix[train_idx]
                val_features = [features[i] for i in val_idx]

                fold_model = copy.deepcopy(base_template)
                fold_model.fit(fold_features, fold_cost, config_names)
                fold_preds = fold_model.predict(val_features)

                # fold_preds.values: (n_val, n_configs)
                oof_preds[val_idx, col_start:col_end] = fold_preds.values

        # Train meta-learner on stacked OOF predictions
        self.meta_learners_ = []
        for c in range(self.n_configs_):
            reg = self._make_meta_learner()
            reg.fit(oof_preds, cost_matrix[:, c])
            self.meta_learners_.append(reg)

        return {
            "mode": "stacking",
            "n_base_models": n_base,
            "n_folds": n_folds,
            "meta_feature_dim": oof_preds.shape[1],
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        if self.fallback_mode_:
            X = self._stack_vectors(features)
            X_scaled = self.scaler_.transform(X)
            scores = np.column_stack(
                [reg.predict(X_scaled) for reg in self.meta_learners_]
            )
        else:
            # Get predictions from all fitted base models and stack
            base_preds_list = []
            for base_model in self.fitted_base_models_:
                bp = base_model.predict(features)
                base_preds_list.append(bp.values)

            # Stack: (n, n_configs * n_base_models)
            meta_features = np.hstack(base_preds_list)
            scores = np.column_stack(
                [reg.predict(meta_features) for reg in self.meta_learners_]
            )

        return Predictions(
            values=scores,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
