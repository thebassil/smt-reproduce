"""Tests for evaluation metrics and cross-validation."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.types import Decision, FoldResult
from pipeline.evaluate import evaluate_decisions, aggregate_folds


class TestEvaluateDecisions:
    def test_perfect_oracle(self, cost_matrix, config_names):
        """Selecting the true best config per instance should match VBS."""
        best_configs = np.argmin(cost_matrix, axis=1)
        decisions = [
            Decision(
                instance_id=f"inst_{i}",
                decision_type="select",
                selected_config=config_names[best_configs[i]],
            )
            for i in range(len(best_configs))
        ]
        metrics = evaluate_decisions(decisions, cost_matrix, config_names, timeout_s=60.0)

        assert "par2" in metrics
        assert "solved_pct" in metrics
        assert "vbs_par2" in metrics
        # Oracle should equal VBS
        assert abs(metrics["par2"] - metrics["vbs_par2"]) < 1e-6

    def test_worst_case(self, cost_matrix, config_names):
        """Selecting the worst config should produce higher PAR-2 than VBS."""
        worst_configs = np.argmax(cost_matrix, axis=1)
        decisions = [
            Decision(
                instance_id=f"inst_{i}",
                decision_type="select",
                selected_config=config_names[worst_configs[i]],
            )
            for i in range(len(worst_configs))
        ]
        metrics = evaluate_decisions(decisions, cost_matrix, config_names, timeout_s=60.0)

        assert metrics["par2"] >= metrics["vbs_par2"]

    def test_returns_all_expected_keys(self, cost_matrix, config_names):
        decisions = [
            Decision(
                instance_id=f"inst_{i}",
                decision_type="select",
                selected_config=config_names[0],
            )
            for i in range(cost_matrix.shape[0])
        ]
        metrics = evaluate_decisions(decisions, cost_matrix, config_names, timeout_s=60.0)

        expected_keys = {"par2", "solved_pct", "vbs_par2", "sbs_par2", "vbs_gap", "n_instances"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_solved_pct_between_0_and_1(self, cost_matrix, config_names):
        decisions = [
            Decision(
                instance_id=f"inst_{i}",
                decision_type="select",
                selected_config=config_names[0],
            )
            for i in range(cost_matrix.shape[0])
        ]
        metrics = evaluate_decisions(decisions, cost_matrix, config_names, timeout_s=60.0)
        assert 0.0 <= metrics["solved_pct"] <= 100.0


class TestAggregateFolds:
    def test_aggregation(self):
        folds = [
            FoldResult(fold_id=0, decisions=[], metrics={"par2": 100.0, "solved_pct": 0.8}),
            FoldResult(fold_id=1, decisions=[], metrics={"par2": 200.0, "solved_pct": 0.9}),
        ]
        agg = aggregate_folds(folds)

        assert agg["mean_par2"] == pytest.approx(150.0)
        assert agg["mean_solved_pct"] == pytest.approx(0.85)
        assert agg["n_folds"] == 2
        assert "std_par2" in agg

    def test_single_fold(self):
        folds = [
            FoldResult(fold_id=0, decisions=[], metrics={"par2": 42.0}),
        ]
        agg = aggregate_folds(folds)
        assert agg["mean_par2"] == pytest.approx(42.0)
        assert agg["n_folds"] == 1
