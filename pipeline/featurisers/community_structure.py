"""Card 14: Community Structure featuriser.

Builds a Variable Incidence Graph (VIG) from pseudo-CNF clauses and extracts
community-related features: modularity, community count/size stats,
power-law degree exponent, and connected component count.

~10 features, VECTOR output.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file

_MAX_NODES = 50_000
_N_FEATURES = 10


def _build_vig(clauses: list[list[int]], n_vars: int):
    """Build a Variable Incidence Graph from clauses.

    Nodes are variable IDs (positive ints). An edge connects two variables
    if they co-occur in at least one clause.
    """
    import networkx as nx

    G = nx.Graph()
    # Add all variable nodes
    all_vars = set()
    for clause in clauses:
        for lit in clause:
            all_vars.add(abs(lit))
    G.add_nodes_from(all_vars)

    # Add edges for co-occurrence within clauses
    for clause in clauses:
        vars_in_clause = list(set(abs(lit) for lit in clause))
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                G.add_edge(vars_in_clause[i], vars_in_clause[j])

    return G


def _estimate_power_law_exponent(degrees: np.ndarray) -> float:
    """Estimate power-law exponent via MLE on degree sequence (d >= 1)."""
    d = degrees[degrees >= 1].astype(np.float64)
    if len(d) < 2:
        return 0.0
    # Clauset et al. MLE: alpha = 1 + n / sum(ln(d / d_min))
    d_min = d.min()
    log_ratios = np.log(d / d_min)
    total = log_ratios.sum()
    if total <= 0:
        return 0.0
    alpha = 1.0 + len(d) / total
    return float(alpha)


class CommunityStructure:
    """Card 14: Community Structure featuriser.

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(self, method: str = "greedy", **kwargs):
        self.method = method

    @property
    def n_features(self) -> int:
        return _N_FEATURES

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract community structure features from a single SMT-LIB2 instance."""
        import networkx as nx

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

        # Empty formula or too large → zeros
        if n_vars == 0 or n_vars > _MAX_NODES:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(_N_FEATURES, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=_N_FEATURES,
                instance_id=str(path),
                logic=info.logic,
            )

        G = _build_vig(clauses, n_vars)

        # Degree distribution
        degrees = np.array([d for _, d in G.degree()], dtype=np.float64)

        # Connected components
        n_components = nx.number_connected_components(G)

        # Community detection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if self.method == "greedy":
                    communities = list(nx.community.greedy_modularity_communities(G))
                else:
                    communities = list(nx.community.greedy_modularity_communities(G))
            except Exception:
                communities = [set(G.nodes())]

        n_communities = len(communities)

        # Modularity score
        try:
            modularity = nx.community.modularity(G, communities)
        except Exception:
            modularity = 0.0

        # Community size stats
        sizes = np.array([len(c) for c in communities], dtype=np.float64)
        size_min = float(sizes.min()) if len(sizes) > 0 else 0.0
        size_max = float(sizes.max()) if len(sizes) > 0 else 0.0
        size_mean = float(sizes.mean()) if len(sizes) > 0 else 0.0
        size_std = float(sizes.std()) if len(sizes) > 0 else 0.0

        # Power-law exponent
        pl_exponent = _estimate_power_law_exponent(degrees)

        features = np.array([
            float(modularity),           # 0: modularity score
            float(n_communities),        # 1: number of communities
            size_min,                    # 2: min community size
            size_max,                    # 3: max community size
            size_mean,                   # 4: mean community size
            size_std,                    # 5: std community size
            pl_exponent,                 # 6: power-law exponent
            float(n_components),         # 7: connected components
            float(G.number_of_nodes()),  # 8: graph node count
            float(G.number_of_edges()),  # 9: graph edge count
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
