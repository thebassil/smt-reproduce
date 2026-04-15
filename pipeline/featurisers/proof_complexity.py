"""Card 11: Proof Complexity featuriser.

Resolution-based complexity features extracted from pseudo-CNF representation.
~15-dimensional VECTOR output covering clause statistics, variable occurrence
distributions, Horn clause fraction, and backbone estimates.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


class ProofComplexity:
    """Proof Complexity featuriser (Card 11).

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    # Feature layout (15 dimensions):
    #  0: n_clauses
    #  1: n_variables
    #  2: clause_variable_ratio
    #  3: min_clause_length
    #  4: max_clause_length (also resolution width estimate)
    #  5: mean_clause_length
    #  6: median_clause_length
    #  7: min_pos_literal_freq
    #  8: max_pos_literal_freq
    #  9: mean_pos_literal_freq
    # 10: min_neg_literal_freq
    # 11: max_neg_literal_freq
    # 12: mean_neg_literal_freq
    # 13: backbone_fraction (unit clause fraction)
    # 14: horn_clause_fraction

    N_FEATURES = 15

    def __init__(self, **kwargs) -> None:
        pass

    @property
    def n_features(self) -> int:
        return self.N_FEATURES

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract proof complexity features from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(self.N_FEATURES, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.N_FEATURES,
                instance_id=str(path),
                logic=None,
            )

        n_clauses = len(clauses)
        n_variables = len(var_to_id)

        features = np.zeros(self.N_FEATURES, dtype=np.float32)

        if n_clauses == 0 or n_variables == 0:
            features[0] = float(n_clauses)
            features[1] = float(n_variables)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=features,
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.N_FEATURES,
                instance_id=str(path),
                logic=info.logic,
            )

        # Clause length statistics
        clause_lengths = np.array([len(c) for c in clauses], dtype=np.float64)

        features[0] = float(n_clauses)
        features[1] = float(n_variables)
        features[2] = float(n_clauses) / float(n_variables)
        features[3] = float(clause_lengths.min())
        features[4] = float(clause_lengths.max())  # resolution width estimate
        features[5] = float(clause_lengths.mean())
        features[6] = float(np.median(clause_lengths))

        # Variable occurrence statistics (positive and negative)
        pos_freq = np.zeros(n_variables, dtype=np.float64)
        neg_freq = np.zeros(n_variables, dtype=np.float64)

        for clause in clauses:
            for lit in clause:
                vid = abs(lit)
                idx = vid - 1  # var IDs are 1-based
                if idx < n_variables:
                    if lit > 0:
                        pos_freq[idx] += 1.0
                    else:
                        neg_freq[idx] += 1.0

        # Positive literal frequency stats
        features[7] = float(pos_freq.min())
        features[8] = float(pos_freq.max())
        features[9] = float(pos_freq.mean())

        # Negative literal frequency stats
        features[10] = float(neg_freq.min())
        features[11] = float(neg_freq.max())
        features[12] = float(neg_freq.mean())

        # Backbone fraction: fraction of unit clauses
        n_unit = sum(1 for c in clauses if len(c) == 1)
        features[13] = float(n_unit) / float(n_clauses)

        # Horn clause fraction: clauses with at most one positive literal
        n_horn = 0
        for clause in clauses:
            n_pos = sum(1 for lit in clause if lit > 0)
            if n_pos <= 1:
                n_horn += 1
        features[14] = float(n_horn) / float(n_clauses)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FeatureResult(
            features=features.astype(np.float32),
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=self.N_FEATURES,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
