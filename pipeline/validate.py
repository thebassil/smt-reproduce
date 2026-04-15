"""Pipeline validation utilities.

Checks type compatibility between pipeline components and validates
data flowing through the pipeline.
"""
from __future__ import annotations

import numpy as np

from pipeline.types import (
    Decision,
    FeatureResult,
    Featuriser,
    Model,
    Policy,
    Predictions,
)


def validate_pipeline(featuriser: Featuriser, model: Model, policy: Policy) -> None:
    """Validate that featuriser, model, and policy are compatible.

    Raises TypeError if featuriser.input_type does not match model.input_type
    (e.g. VECTOR featuriser + GRAPH model).
    """
    feat_type = featuriser.input_type
    model_type = model.input_type

    if feat_type != model_type:
        raise TypeError(
            f"Featuriser produces {feat_type} features but model expects "
            f"{model_type} input. Use a matching featuriser/model pair."
        )


def validate_feature_result(fr: FeatureResult) -> None:
    """Validate that a FeatureResult is internally consistent."""
    if fr.feature_type == "VECTOR":
        if not isinstance(fr.features, np.ndarray):
            raise TypeError(
                f"feature_type is VECTOR but features is {type(fr.features).__name__}, "
                f"expected np.ndarray."
            )
        if fr.features.ndim != 1:
            raise ValueError(f"VECTOR features must be 1-D, got shape {fr.features.shape}.")
        if fr.features.shape[0] != fr.n_features:
            raise ValueError(
                f"n_features={fr.n_features} but features array has "
                f"{fr.features.shape[0]} elements."
            )
    elif fr.feature_type == "GRAPH":
        if fr.features is None:
            raise ValueError("GRAPH features must not be None.")


def validate_predictions(pred: Predictions) -> None:
    """Validate that Predictions shape matches metadata."""
    if not isinstance(pred.values, np.ndarray):
        raise TypeError(f"Predictions.values must be np.ndarray, got {type(pred.values).__name__}.")
    if pred.values.ndim != 2:
        raise ValueError(
            f"Predictions.values must be 2-D (n_instances, n_configs), got shape {pred.values.shape}."
        )
    n_instances, n_configs = pred.values.shape
    if n_configs != len(pred.config_names):
        raise ValueError(
            f"values has {n_configs} columns but config_names has {len(pred.config_names)} entries."
        )
    if n_instances != len(pred.instance_ids):
        raise ValueError(
            f"values has {n_instances} rows but instance_ids has {len(pred.instance_ids)} entries."
        )
    if pred.output_type == "distribution":
        row_sums = pred.values.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(
                f"output_type is 'distribution' but rows do not sum to 1. "
                f"Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}."
            )


def validate_decisions(decisions: list[Decision], budget_s: float) -> None:
    """Validate a list of decisions against a budget.

    Checks:
    - "select" decisions have selected_config set
    - "schedule" decisions have schedule set, budgets sum <= budget_s
    - "rank" decisions have ranking set
    """
    for d in decisions:
        if d.decision_type == "select":
            if d.selected_config is None:
                raise ValueError(
                    f"Decision for {d.instance_id}: decision_type='select' "
                    f"but selected_config is None."
                )
        elif d.decision_type == "schedule":
            if d.schedule is None or len(d.schedule) == 0:
                raise ValueError(
                    f"Decision for {d.instance_id}: decision_type='schedule' "
                    f"but schedule is None or empty."
                )
            total = sum(t for _, t in d.schedule)
            if total > budget_s + 1e-9:
                raise ValueError(
                    f"Decision for {d.instance_id}: schedule budgets sum to "
                    f"{total:.3f}s but total budget is {budget_s:.3f}s."
                )
        elif d.decision_type == "rank":
            if d.ranking is None or len(d.ranking) == 0:
                raise ValueError(
                    f"Decision for {d.instance_id}: decision_type='rank' "
                    f"but ranking is None or empty."
                )
