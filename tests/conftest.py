"""Shared test fixtures for the SMT reproducibility test suite."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pipeline.types import FeatureResult, Predictions, Decision, FoldResult


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_names():
    """4 solver configurations (matches portfolio_baseline_4cfg_v1)."""
    return ["z3_default", "cvc5_default", "z3_case_split_2", "cvc5_eager_cadical"]


@pytest.fixture
def cost_matrix(config_names):
    """Synthetic PAR-2 cost matrix: 20 instances x 4 configs.

    Each row has one clearly best solver to make tests deterministic.
    Timeout penalty is 120.0 (2 * 60s timeout).
    """
    rng = np.random.RandomState(42)
    n_instances, n_configs = 20, len(config_names)
    # Base: moderate runtimes
    matrix = rng.uniform(5.0, 50.0, size=(n_instances, n_configs))
    # Make one config clearly best per instance (low runtime)
    for i in range(n_instances):
        best = i % n_configs
        matrix[i, best] = rng.uniform(0.1, 3.0)
    # Sprinkle some timeouts
    for i in range(0, n_instances, 5):
        worst = (i + 2) % n_configs
        matrix[i, worst] = 120.0  # PAR-2 penalty
    return matrix


@pytest.fixture
def instance_ids():
    """20 synthetic instance paths."""
    return [f"/data/instance_{i:03d}.smt2" for i in range(20)]


@pytest.fixture
def feature_results(instance_ids):
    """Synthetic VECTOR feature results for 20 instances."""
    rng = np.random.RandomState(42)
    return [
        FeatureResult(
            features=rng.randn(10),
            feature_type="VECTOR",
            wall_time_ms=rng.uniform(1.0, 50.0),
            n_features=10,
            instance_id=iid,
            logic="QF_LIA",
        )
        for iid in instance_ids
    ]


@pytest.fixture
def predictions(instance_ids, config_names):
    """Synthetic score predictions: 20 instances x 4 configs."""
    rng = np.random.RandomState(42)
    return Predictions(
        values=rng.uniform(1.0, 60.0, size=(len(instance_ids), len(config_names))),
        output_type="scores",
        config_names=config_names,
        instance_ids=instance_ids,
    )


@pytest.fixture
def db_path():
    """Path to the central database (skip if not available)."""
    paths = [
        Path("/dcs/large/u5573765/db/results.sqlite"),
        Path(__file__).parent.parent / "data" / "results.sqlite",
    ]
    for p in paths:
        if p.exists() and p.stat().st_size > 1000:
            return str(p)
    pytest.skip("Database not available (run `smt download` first)")
