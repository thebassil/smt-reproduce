"""Tests for pipeline validation contracts."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.types import FeatureResult, Predictions, Decision
from pipeline.validate import (
    validate_pipeline,
    validate_feature_result,
    validate_predictions,
    validate_decisions,
)


# ---------------------------------------------------------------------------
# Minimal stub implementations for validation tests
# ---------------------------------------------------------------------------

class StubVectorFeaturiser:
    input_type = "VECTOR"
    def extract(self, path): ...
    def extract_batch(self, paths): ...

class StubGraphFeaturiser:
    input_type = "GRAPH"
    def extract(self, path): ...
    def extract_batch(self, paths): ...

class StubVectorModel:
    input_type = "VECTOR"
    output_type = "scores"
    def fit(self, features, cost_matrix, config_names): ...
    def predict(self, features): ...
    def save(self, path): ...
    def load(self, path): ...

class StubGraphModel:
    input_type = "GRAPH"
    output_type = "scores"
    def fit(self, features, cost_matrix, config_names): ...
    def predict(self, features): ...
    def save(self, path): ...
    def load(self, path): ...

class StubPolicy:
    def decide(self, predictions, budget_s): ...


class TestValidatePipeline:
    def test_matching_types_pass(self):
        """VECTOR featuriser + VECTOR model should pass."""
        validate_pipeline(StubVectorFeaturiser(), StubVectorModel(), StubPolicy())

    def test_graph_types_pass(self):
        """GRAPH featuriser + GRAPH model should pass."""
        validate_pipeline(StubGraphFeaturiser(), StubGraphModel(), StubPolicy())

    def test_mismatched_types_fail(self):
        """VECTOR featuriser + GRAPH model must raise TypeError."""
        with pytest.raises(TypeError):
            validate_pipeline(StubVectorFeaturiser(), StubGraphModel(), StubPolicy())

    def test_reverse_mismatch_fail(self):
        """GRAPH featuriser + VECTOR model must raise TypeError."""
        with pytest.raises(TypeError):
            validate_pipeline(StubGraphFeaturiser(), StubVectorModel(), StubPolicy())


class TestValidateFeatureResult:
    def test_valid_vector(self):
        fr = FeatureResult(
            features=np.zeros(10),
            feature_type="VECTOR",
            wall_time_ms=5.0,
            n_features=10,
            instance_id="test.smt2",
        )
        validate_feature_result(fr)  # should not raise

    def test_negative_wall_time_rejected_by_pydantic(self):
        """Pydantic enforces wall_time_ms >= 0 at construction."""
        with pytest.raises(Exception):
            FeatureResult(
                features=np.zeros(10),
                feature_type="VECTOR",
                wall_time_ms=-1.0,
                n_features=10,
                instance_id="test.smt2",
            )


class TestValidatePredictions:
    def test_valid_predictions(self, config_names, instance_ids):
        pred = Predictions(
            values=np.random.randn(len(instance_ids), len(config_names)),
            output_type="scores",
            config_names=config_names,
            instance_ids=instance_ids,
        )
        validate_predictions(pred)  # should not raise

    def test_shape_mismatch_raises(self, config_names, instance_ids):
        pred = Predictions(
            values=np.random.randn(5, 2),  # wrong shape
            output_type="scores",
            config_names=config_names,
            instance_ids=instance_ids,
        )
        with pytest.raises((ValueError, Exception)):
            validate_predictions(pred)


class TestValidateDecisions:
    def test_valid_select_decisions(self):
        decisions = [
            Decision(instance_id=f"i{i}.smt2", decision_type="select", selected_config="z3_default")
            for i in range(5)
        ]
        validate_decisions(decisions, budget_s=60.0)  # should not raise

    def test_schedule_within_budget(self):
        decisions = [
            Decision(
                instance_id="test.smt2",
                decision_type="schedule",
                schedule=[("z3_default", 30.0), ("cvc5_default", 30.0)],
            )
        ]
        validate_decisions(decisions, budget_s=60.0)  # should not raise
