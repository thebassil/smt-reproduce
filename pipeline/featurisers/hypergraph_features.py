"""Card 18: Hypergraph Features featuriser.

Computes hyperedge overlap statistics from the clause structure:
hyperedge size stats, Jaccard similarity stats, vertex cover estimate,
hypergraph density, and intersection graph properties.

~10 features, VECTOR output.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file

_MAX_CLAUSES = 50_000
_N_FEATURES = 10


def _greedy_vertex_cover_estimate(clauses_as_sets: list[set[int]]) -> int:
    """Estimate minimum vertex cover size via greedy algorithm.

    Repeatedly pick the variable appearing in the most uncovered clauses
    and remove all clauses containing it.
    """
    uncovered = list(range(len(clauses_as_sets)))
    cover = set()

    while uncovered:
        # Count variable frequency in uncovered clauses
        freq: dict[int, int] = {}
        for idx in uncovered:
            for v in clauses_as_sets[idx]:
                freq[v] = freq.get(v, 0) + 1

        if not freq:
            break

        # Pick most frequent variable
        best_var = max(freq, key=freq.get)
        cover.add(best_var)

        # Remove clauses containing best_var
        uncovered = [idx for idx in uncovered if best_var not in clauses_as_sets[idx]]

    return len(cover)


class HypergraphFeatures:
    """Card 18: Hypergraph Features featuriser.

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(self, **kwargs):
        pass

    @property
    def n_features(self) -> int:
        return _N_FEATURES

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract hypergraph features from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(_N_FEATURES, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=_N_FEATURES,
                instance_id=str(path),
                logic=None,
            )

        n_vars = len(var_to_id)
        n_clauses = len(clauses)

        if n_clauses == 0 or n_clauses > _MAX_CLAUSES:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(_N_FEATURES, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=_N_FEATURES,
                instance_id=str(path),
                logic=getattr(info, 'logic', None),
            )

        # Convert clauses to sets of absolute variable IDs (hyperedges)
        clause_sets = [set(abs(lit) for lit in clause) for clause in clauses]

        # Hyperedge size stats
        sizes = np.array([len(c) for c in clause_sets], dtype=np.float64)
        size_min = float(sizes.min())
        size_max = float(sizes.max())
        size_mean = float(sizes.mean())
        size_std = float(sizes.std())

        # Jaccard similarity stats between clause pairs (sample if too many)
        jaccard_values = []
        if n_clauses <= 500:
            # All pairs
            for i in range(n_clauses):
                for j in range(i + 1, n_clauses):
                    inter = len(clause_sets[i] & clause_sets[j])
                    union = len(clause_sets[i] | clause_sets[j])
                    if union > 0:
                        jaccard_values.append(inter / union)
        elif n_clauses > 1:
            # Random sample of pairs
            rng = np.random.RandomState(42)
            n_samples = min(10000, n_clauses * (n_clauses - 1) // 2)
            for _ in range(n_samples):
                i, j = rng.choice(n_clauses, size=2, replace=False)
                inter = len(clause_sets[i] & clause_sets[j])
                union = len(clause_sets[i] | clause_sets[j])
                if union > 0:
                    jaccard_values.append(inter / union)

        if jaccard_values:
            jacc = np.array(jaccard_values, dtype=np.float64)
            jaccard_mean = float(jacc.mean())
            jaccard_std = float(jacc.std())
        else:
            jaccard_mean = 0.0
            jaccard_std = 0.0

        # Vertex cover number estimate (greedy)
        if n_clauses <= 20000:
            vc_size = float(_greedy_vertex_cover_estimate(clause_sets))
        else:
            vc_size = 0.0

        # Hypergraph density: sum of hyperedge sizes / (n_vars * n_clauses)
        total_incidence = float(sizes.sum())
        if n_vars > 0 and n_clauses > 0:
            hg_density = total_incidence / (n_vars * n_clauses)
        else:
            hg_density = 0.0

        # Intersection graph: count how many clause pairs share at least one variable
        n_sharing_pairs = 0
        if n_clauses <= 500:
            for i in range(n_clauses):
                for j in range(i + 1, n_clauses):
                    if clause_sets[i] & clause_sets[j]:
                        n_sharing_pairs += 1
            total_pairs = n_clauses * (n_clauses - 1) // 2
            sharing_fraction = float(n_sharing_pairs) / max(total_pairs, 1)
        elif n_clauses > 1:
            # Estimate from Jaccard samples — pairs with jaccard > 0
            sharing_fraction = float(np.mean(np.array(jaccard_values) > 0)) if jaccard_values else 0.0
        else:
            sharing_fraction = 0.0

        features = np.array([
            size_min,          # 0: min hyperedge size
            size_max,          # 1: max hyperedge size
            size_mean,         # 2: mean hyperedge size
            size_std,          # 3: std hyperedge size
            jaccard_mean,      # 4: mean Jaccard similarity
            jaccard_std,       # 5: std Jaccard similarity
            vc_size,           # 6: vertex cover estimate
            hg_density,        # 7: hypergraph density
            sharing_fraction,  # 8: fraction of clause pairs sharing variables
            float(n_clauses),  # 9: number of hyperedges (clauses)
        ], dtype=np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FeatureResult(
            features=features,
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=_N_FEATURES,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
