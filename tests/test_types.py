"""Tests for pipeline data types and protocol contracts."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.types import (
    Decision,
    FeatureResult,
    FoldResult,
    Predictions,
    Featuriser,
    Model,
    Policy,
)


class TestFeatureResult:
    def test_vector_construction(self):
        fr = FeatureResult(
            features=np.zeros(10),
            feature_type="VECTOR",
            wall_time_ms=5.0,
            n_features=10,
            instance_id="test.smt2",
        )
        assert fr.feature_type == "VECTOR"
        assert fr.n_features == 10

    def test_rejects_invalid_type(self):
        with pytest.raises(Exception):
            FeatureResult(
                features=np.zeros(10),
                feature_type="INVALID",
                wall_time_ms=1.0,
                n_features=10,
                instance_id="test.smt2",
            )


class TestPredictions:
    def test_scores_construction(self, config_names, instance_ids):
        n = len(instance_ids)
        k = len(config_names)
        pred = Predictions(
            values=np.random.randn(n, k),
            output_type="scores",
            config_names=config_names,
            instance_ids=instance_ids,
        )
        assert pred.output_type == "scores"
        assert len(pred.config_names) == k
        assert len(pred.instance_ids) == n


class TestDecision:
    def test_select_decision(self):
        d = Decision(
            instance_id="test.smt2",
            decision_type="select",
            selected_config="z3_default",
            confidence=0.95,
        )
        assert d.decision_type == "select"
        assert d.selected_config == "z3_default"

    def test_schedule_decision(self):
        d = Decision(
            instance_id="test.smt2",
            decision_type="schedule",
            schedule=[("z3_default", 30.0), ("cvc5_default", 30.0)],
        )
        assert d.decision_type == "schedule"
        assert len(d.schedule) == 2
        assert sum(t for _, t in d.schedule) == 60.0

    def test_rank_decision(self):
        d = Decision(
            instance_id="test.smt2",
            decision_type="rank",
            ranking=["z3_default", "cvc5_default", "z3_case_split_2"],
        )
        assert d.decision_type == "rank"
        assert d.ranking[0] == "z3_default"


class TestFoldResult:
    def test_construction(self):
        fr = FoldResult(
            fold_id=0,
            decisions=[
                Decision(instance_id="a.smt2", decision_type="select", selected_config="z3_default")
            ],
            metrics={"par2": 100.0, "solved_pct": 0.85},
        )
        assert fr.fold_id == 0
        assert fr.metrics["solved_pct"] == 0.85
