"""Tests for the CLI tool itself."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from smt.cli import smt


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(smt, ["--help"])
        assert result.exit_code == 0
        assert "reproduce" in result.output
        assert "download" in result.output
        assert "status" in result.output

    def test_reproduce_help(self, runner):
        result = runner.invoke(smt, ["reproduce", "--help"])
        assert result.exit_code == 0
        assert "systems" in result.output
        assert "subsystem" in result.output
        assert "cross-system" in result.output
        assert "ground-truth" in result.output

    def test_status(self, runner):
        result = runner.invoke(smt, ["status"])
        assert result.exit_code == 0
        assert "Environment" in result.output

    def test_systems_inspect(self, runner):
        result = runner.invoke(smt, ["reproduce", "systems", "--inspect"])
        assert result.exit_code == 0
        assert "System Comparison" in result.output

    def test_systems_inspect_single(self, runner):
        result = runner.invoke(smt, ["reproduce", "systems", "--inspect", "--system", "machsmt"])
        assert result.exit_code == 0
        assert "MACHSMT" in result.output

    def test_subsystem_inspect(self, runner):
        result = runner.invoke(smt, ["reproduce", "subsystem", "--inspect", "--system", "machsmt"])
        assert result.exit_code == 0
        assert "MACHSMT" in result.output

    def test_cross_system_inspect(self, runner):
        result = runner.invoke(smt, ["reproduce", "cross-system", "--inspect"])
        assert result.exit_code == 0
        assert "Cross-System" in result.output

    def test_cross_system_inspect_axis(self, runner):
        result = runner.invoke(smt, ["reproduce", "cross-system", "--inspect", "--axis", "model"])
        assert result.exit_code == 0

    def test_ground_truth_inspect(self, runner):
        result = runner.invoke(smt, ["reproduce", "ground-truth", "--inspect"])
        assert result.exit_code == 0
