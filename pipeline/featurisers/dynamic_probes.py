"""Card 19: Dynamic Probes featuriser.

Runs z3 with a short timeout and -st flag, parses solver statistics from
stderr to produce a ~20-dim VECTOR of runtime features (conflicts, decisions,
propagations, memory, etc.) plus derived ratios.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import parse_file


# Stat keys we look for in z3 -st output
_STAT_KEYS = [
    "conflicts",
    "decisions",
    "propagations",
    "restarts",
    "memory",
    "time",
    "num allocs",
    "added eqs",
    "del clause",
    "mk clause",
]

# Regex: matches lines like "  :conflicts    42"  or  " :memory           2.31"
_STAT_RE = re.compile(r"^\s*:(\S+)\s+([\d.]+)", re.MULTILINE)


def _parse_z3_stats(stderr_text: str) -> Dict[str, float]:
    """Parse z3 -st statistics output into a dict of floats."""
    stats: Dict[str, float] = {}
    for m in _STAT_RE.finditer(stderr_text):
        key = m.group(1).replace("-", " ").replace("_", " ").strip()
        try:
            stats[key] = float(m.group(2))
        except ValueError:
            pass
    return stats


class DynamicProbesFeaturiser:
    """Dynamic Probes featuriser (Card 19).

    Implements the Featuriser protocol with input_type = "VECTOR".
    Runs z3 with -st to collect solver runtime statistics as features.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    # Feature names in output order
    FEATURE_NAMES: ClassVar[List[str]] = [
        "conflicts",
        "decisions",
        "propagations",
        "restarts",
        "memory_mb",
        "time_s",
        "result_sat",
        "num_allocs",
        "added_eqs",
        "del_clause",
        "mk_clause",
        # Ratios
        "conflicts_per_decision",
        "propagations_per_decision",
        "propagations_per_conflict",
        "restarts_per_conflict",
        "del_clause_per_mk_clause",
        "conflicts_per_sec",
        "decisions_per_sec",
        "propagations_per_sec",
    ]

    def __init__(
        self, timeout_s: float = 1.0, solver_cmd: str = "z3", **kwargs
    ) -> None:
        self.timeout_s = timeout_s
        self.solver_cmd = solver_cmd

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract dynamic probe features from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()
        n_feats = len(self.FEATURE_NAMES)

        # Try to get the logic from the file
        logic = None
        try:
            info = parse_file(path)
            logic = info.logic
        except Exception:
            pass

        # Run z3 with stats
        feats = np.zeros(n_feats, dtype=np.float64)
        try:
            cmd = [self.solver_cmd, "-st", "-T:" + str(int(self.timeout_s)), str(path)]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s + 1,
            )
            stats = _parse_z3_stats(proc.stderr + proc.stdout)

            # Base features
            feats[0] = stats.get("conflicts", 0.0)
            feats[1] = stats.get("decisions", 0.0)
            feats[2] = stats.get("propagations", 0.0)
            feats[3] = stats.get("restarts", 0.0)
            feats[4] = stats.get("memory", 0.0)
            feats[5] = stats.get("time", 0.0)

            # Result: sat=1, unsat=0, unknown/timeout=0.5
            stdout_lower = proc.stdout.strip().lower()
            if "unsat" in stdout_lower:
                feats[6] = 0.0
            elif "sat" in stdout_lower:
                feats[6] = 1.0
            else:
                feats[6] = 0.5

            feats[7] = stats.get("num allocs", 0.0)
            feats[8] = stats.get("added eqs", 0.0)
            feats[9] = stats.get("del clause", 0.0)
            feats[10] = stats.get("mk clause", 0.0)

            # Ratios (safe division)
            decisions = feats[1]
            conflicts = feats[0]
            mk_clause = feats[10]
            time_s = feats[5]

            feats[11] = conflicts / decisions if decisions > 0 else 0.0
            feats[12] = feats[2] / decisions if decisions > 0 else 0.0
            feats[13] = feats[2] / conflicts if conflicts > 0 else 0.0
            feats[14] = feats[3] / conflicts if conflicts > 0 else 0.0
            feats[15] = feats[9] / mk_clause if mk_clause > 0 else 0.0
            feats[16] = conflicts / time_s if time_s > 0 else 0.0
            feats[17] = decisions / time_s if time_s > 0 else 0.0
            feats[18] = feats[2] / time_s if time_s > 0 else 0.0

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # z3 not available or timed out — return zeros
            feats[6] = 0.5  # unknown result

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=feats,
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=n_feats,
            instance_id=str(path),
            logic=logic,
        )

    def extract_batch(
        self, instance_paths: List[Union[str, Path]]
    ) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
