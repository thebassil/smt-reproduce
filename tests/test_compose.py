"""Tests for SystemPipeline composition and the train/predict contract."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.types import FeatureResult, Predictions, Decision
from pipeline.compose import SystemPipeline


# ---------------------------------------------------------------------------
# Minimal real-ish stubs that follow the protocol contracts
# ---------------------------------------------------------------------------

class FakeFeaturiser:
    input_type = "VECTOR"

    def extract(self, instance_path):
        return FeatureResult(
            features=np.random.randn(10),
            feature_type="VECTOR",
            wall_time_ms=1.0,
            n_features=10,
            instance_id=str(instance_path),
        )

    def extract_batch(self, instance_paths):
        return [self.extract(p) for p in instance_paths]


class FakeModel:
    input_type = "VECTOR"
    output_type = "scores"

    def __init__(self):
        self._config_names = None

    def fit(self, features, cost_matrix, config_names):
        self._config_names = config_names
        return {"status": "ok"}

    def predict(self, features):
        n = len(features)
        k = len(self._config_names)
        return Predictions(
            values=np.random.randn(n, k),
            output_type="scores",
            config_names=self._config_names,
            instance_ids=[f.instance_id for f in features],
        )

    def save(self, path): pass
    def load(self, path): pass


class FakePolicy:
    def decide(self, predictions, budget_s):
        return [
            Decision(
                instance_id=iid,
                decision_type="select",
                selected_config=predictions.config_names[
                    int(np.argmin(predictions.values[i]))
                ],
            )
            for i, iid in enumerate(predictions.instance_ids)
        ]


class TestSystemPipeline:
    def test_construction(self):
        pipe = SystemPipeline(FakeFeaturiser(), FakeModel(), FakePolicy())
        assert pipe.featuriser is not None
        assert pipe.model is not None
        assert pipe.policy is not None

    def test_type_mismatch_raises(self):
        """VECTOR featuriser + GRAPH model should fail on construction."""

        class GraphModel(FakeModel):
            input_type = "GRAPH"

        with pytest.raises(TypeError):
            SystemPipeline(FakeFeaturiser(), GraphModel(), FakePolicy())

    def test_train_returns_dict(self, cost_matrix, config_names):
        pipe = SystemPipeline(FakeFeaturiser(), FakeModel(), FakePolicy())
        paths = [f"/data/inst_{i}.smt2" for i in range(cost_matrix.shape[0])]
        result = pipe.train(paths, cost_matrix, config_names)
        assert isinstance(result, dict)

    def test_predict_returns_decisions(self, cost_matrix, config_names):
        pipe = SystemPipeline(FakeFeaturiser(), FakeModel(), FakePolicy())
        paths = [f"/data/inst_{i}.smt2" for i in range(cost_matrix.shape[0])]
        pipe.train(paths, cost_matrix, config_names)
        decisions = pipe.predict(paths, budget_s=60.0)

        assert isinstance(decisions, list)
        assert len(decisions) == len(paths)
        assert all(isinstance(d, Decision) for d in decisions)
        assert all(d.decision_type == "select" for d in decisions)
        assert all(d.selected_config in config_names for d in decisions)

    def test_end_to_end_pipeline(self, cost_matrix, config_names):
        """Full train → predict → evaluate cycle."""
        from pipeline.evaluate import evaluate_decisions

        pipe = SystemPipeline(FakeFeaturiser(), FakeModel(), FakePolicy())
        paths = [f"/data/inst_{i}.smt2" for i in range(cost_matrix.shape[0])]

        pipe.train(paths, cost_matrix, config_names)
        decisions = pipe.predict(paths, budget_s=60.0)
        metrics = evaluate_decisions(decisions, cost_matrix, config_names, timeout_s=60.0)

        assert metrics["n_instances"] == len(paths)
        assert 0.0 <= metrics["solved_pct"] <= 100.0
        assert metrics["par2"] > 0
