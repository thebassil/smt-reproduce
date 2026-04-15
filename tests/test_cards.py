"""Tests for real card implementations (KNN, SelectArgmin).

These test that the actual ML code follows the pipeline contracts.
Requires: pip install -e ".[train]"
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.types import FeatureResult, Predictions, Decision, Featuriser, Model, Policy


class TestKNNCard:
    @pytest.fixture
    def knn(self):
        from pipeline.models.knn import KNNCard
        return KNNCard(n_neighbors=3)

    def test_protocol_conformance(self, knn):
        assert isinstance(knn, Model)
        assert knn.input_type == "VECTOR"
        assert knn.output_type == "scores"

    def test_fit_predict_cycle(self, knn, feature_results, cost_matrix, config_names):
        train_metrics = knn.fit(feature_results, cost_matrix, config_names)
        assert isinstance(train_metrics, dict)

        preds = knn.predict(feature_results)
        assert isinstance(preds, Predictions)
        assert preds.values.shape == (len(feature_results), len(config_names))
        assert preds.output_type == "scores"
        assert len(preds.instance_ids) == len(feature_results)

    def test_predictions_are_finite(self, knn, feature_results, cost_matrix, config_names):
        knn.fit(feature_results, cost_matrix, config_names)
        preds = knn.predict(feature_results)
        assert np.all(np.isfinite(preds.values))


class TestSelectArgmin:
    @pytest.fixture
    def policy(self):
        from pipeline.policies.select_argmin import SelectArgmin
        return SelectArgmin(seed=42)

    def test_protocol_conformance(self, policy):
        assert isinstance(policy, Policy)

    def test_decides_all_instances(self, policy, predictions):
        decisions = policy.decide(predictions, budget_s=60.0)
        assert len(decisions) == len(predictions.instance_ids)
        assert all(isinstance(d, Decision) for d in decisions)
        assert all(d.decision_type == "select" for d in decisions)

    def test_selects_valid_configs(self, policy, predictions):
        decisions = policy.decide(predictions, budget_s=60.0)
        for d in decisions:
            assert d.selected_config in predictions.config_names

    def test_selects_argmin(self, predictions):
        """With deterministic predictions, argmin should pick the lowest-score config."""
        from pipeline.policies.select_argmin import SelectArgmin
        policy = SelectArgmin(tie_break="first", seed=42)
        decisions = policy.decide(predictions, budget_s=60.0)

        for i, d in enumerate(decisions):
            expected = predictions.config_names[np.argmin(predictions.values[i])]
            assert d.selected_config == expected


class TestKNNWithSelectArgmin:
    """Integration: KNN model -> SelectArgmin policy -> evaluate."""

    def test_full_integration(self, feature_results, cost_matrix, config_names):
        from pipeline.models.knn import KNNCard
        from pipeline.policies.select_argmin import SelectArgmin
        from pipeline.evaluate import evaluate_decisions

        model = KNNCard(n_neighbors=3)
        policy = SelectArgmin(seed=42)

        model.fit(feature_results, cost_matrix, config_names)
        preds = model.predict(feature_results)
        decisions = policy.decide(preds, budget_s=60.0)

        metrics = evaluate_decisions(decisions, cost_matrix, config_names, timeout_s=60.0)
        assert metrics["n_instances"] == len(feature_results)
        assert metrics["par2"] > 0
        assert 0.0 <= metrics["solved_pct"] <= 100.0
