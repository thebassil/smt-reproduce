"""Tests for database loading (requires results.sqlite)."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.db import load_training_data


class TestLoadTrainingData:
    """These tests require the database. Skipped if not available."""

    def test_loads_qf_lia(self, db_path):
        instance_ids, cost_matrix, config_names = load_training_data(
            db_path, portfolio_id=7, logic="QF_LIA"
        )
        assert len(instance_ids) > 0
        assert cost_matrix.ndim == 2
        assert cost_matrix.shape[0] == len(instance_ids)
        assert cost_matrix.shape[1] == len(config_names)
        assert len(config_names) > 0

    def test_cost_matrix_is_positive(self, db_path):
        _, cost_matrix, _ = load_training_data(db_path, portfolio_id=7, logic="QF_LIA")
        assert np.all(cost_matrix > 0)

    def test_cost_matrix_has_par2_penalties(self, db_path):
        """Timeout instances should have cost = 2 * timeout = 120.0."""
        _, cost_matrix, _ = load_training_data(db_path, portfolio_id=7, logic="QF_LIA")
        assert np.any(cost_matrix >= 120.0)

    def test_three_logics_available(self, db_path):
        for logic in ["QF_LIA", "QF_BV", "QF_NRA"]:
            ids, matrix, configs = load_training_data(db_path, portfolio_id=7, logic=logic)
            assert len(ids) > 100, f"Expected >100 instances for {logic}, got {len(ids)}"

    def test_config_names_consistent(self, db_path):
        """All logics should return the same config set for portfolio 7."""
        configs_per_logic = []
        for logic in ["QF_LIA", "QF_BV", "QF_NRA"]:
            _, _, configs = load_training_data(db_path, portfolio_id=7, logic=logic)
            configs_per_logic.append(set(configs))
        # All should be the same set
        assert configs_per_logic[0] == configs_per_logic[1] == configs_per_logic[2]
