"""Card 17: Topological Data Analysis (TDA) featuriser.

Computes persistent homology features from the VIG:
0-dim and 1-dim persistence diagrams, Betti numbers at thresholds,
max/mean/total persistence. Falls back to graph-based Betti estimates
when ripser/giotto-tda are not available.

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
from pipeline.featurisers.community_structure import _build_vig

_MAX_NODES = 50_000
_N_FEATURES = 10


def _try_import_ripser():
    """Try to import ripser; return None if not available."""
    try:
        import ripser
        return ripser
    except ImportError:
        return None


def _graph_distance_matrix(G, max_dist: float = None):
    """Compute shortest-path distance matrix for a graph.

    For disconnected components, distance is set to max_dist or diameter + 1.
    """
    import networkx as nx

    nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    # Shortest paths
    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    for source in nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        si = node_idx[source]
        for target, d in lengths.items():
            dist[si, node_idx[target]] = float(d)

    # Replace inf with a finite cap
    finite_mask = np.isfinite(dist)
    if finite_mask.any():
        cap = dist[finite_mask].max() + 1.0
    else:
        cap = 1.0
    if max_dist is not None:
        cap = min(cap, max_dist)
    dist[~finite_mask] = cap

    return dist


def _betti_from_graph(G) -> tuple[int, int]:
    """Compute Betti-0 and Betti-1 from the graph structure.

    Betti-0 = number of connected components.
    Betti-1 = number of independent cycles = |E| - |V| + |components|.
    """
    import networkx as nx

    n_comp = nx.number_connected_components(G)
    betti_0 = n_comp
    betti_1 = G.number_of_edges() - G.number_of_nodes() + n_comp
    return betti_0, max(0, betti_1)


def _persistence_from_distance(dist: np.ndarray, max_dim: int, max_edge_length: float):
    """Compute persistence diagrams using ripser if available, else fallback."""
    ripser = _try_import_ripser()
    if ripser is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ripser.ripser(dist, maxdim=max_dim, thresh=max_edge_length,
                                   distance_matrix=True)
            return result['dgms']
    return None


def _fallback_persistence_features(G, thresholds: list[float]) -> np.ndarray:
    """Compute TDA-like features without ripser, using graph properties."""
    import networkx as nx

    betti_0, betti_1 = _betti_from_graph(G)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Approximate threshold-based component counts by removing edges
    # with "weight" above threshold (all edges have weight 1 in unweighted VIG)
    # So at threshold < 1: n_nodes components, at threshold >= 1: actual components
    components_at_thresholds = []
    for t in thresholds:
        if t < 1.0:
            components_at_thresholds.append(float(n_nodes))
        else:
            components_at_thresholds.append(float(betti_0))

    # Persistence estimates from connected components
    # Each component "born" at 0, "dies" when merged — approximate with degree info
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
    if len(degrees) > 0 and degrees.max() > 0:
        # Normalized degree as proxy for persistence
        norm_deg = degrees / degrees.max()
        max_persistence_0 = 1.0  # maximum possible in normalized space
        mean_persistence_0 = float(1.0 - norm_deg.mean())
    else:
        max_persistence_0 = 0.0
        mean_persistence_0 = 0.0

    # H1 persistence proxy: cycle density
    if n_nodes > 0:
        cycle_density = float(betti_1) / max(n_nodes, 1)
    else:
        cycle_density = 0.0

    features = np.array([
        float(betti_0),                      # 0: Betti-0 (connected components)
        float(betti_1),                      # 1: Betti-1 (independent cycles)
        components_at_thresholds[0],         # 2: components at threshold 0.5
        components_at_thresholds[1],         # 3: components at threshold 1.0
        components_at_thresholds[2],         # 4: components at threshold 1.5
        max_persistence_0,                   # 5: max H0 persistence
        mean_persistence_0,                  # 6: mean H0 persistence
        float(betti_0) * max_persistence_0,  # 7: total H0 persistence estimate
        cycle_density,                       # 8: H1 persistence proxy (cycle density)
        float(n_edges) / max(n_nodes * (n_nodes - 1) / 2, 1),  # 9: density as topological proxy
    ], dtype=np.float32)

    return features


def _persistence_features_from_diagrams(dgms: list, G, thresholds: list[float]) -> np.ndarray:
    """Extract features from persistence diagrams."""
    betti_0, betti_1 = _betti_from_graph(G)

    # H0 diagram
    h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
    h0_finite = h0[np.isfinite(h0[:, 1])] if len(h0) > 0 else np.empty((0, 2))
    h0_pers = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([0.0])

    # H1 diagram
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    h1_finite = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))
    h1_pers = h1_finite[:, 1] - h1_finite[:, 0] if len(h1_finite) > 0 else np.array([0.0])

    # Components at thresholds (from H0 diagram)
    components_at_thresholds = []
    for t in thresholds:
        # Count points that are still alive at threshold t
        if len(h0) > 0:
            alive = np.sum((h0[:, 0] <= t) & ((h0[:, 1] > t) | np.isinf(h0[:, 1])))
            components_at_thresholds.append(float(alive))
        else:
            components_at_thresholds.append(0.0)

    max_h0_pers = float(h0_pers.max()) if len(h0_pers) > 0 else 0.0
    mean_h0_pers = float(h0_pers.mean()) if len(h0_pers) > 0 else 0.0
    total_h0_pers = float(h0_pers.sum())

    max_h1_pers = float(h1_pers.max()) if len(h1_pers) > 0 else 0.0
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = float(n_edges) / max(n_nodes * (n_nodes - 1) / 2, 1)

    features = np.array([
        float(betti_0),                # 0: Betti-0
        float(betti_1),                # 1: Betti-1
        components_at_thresholds[0],   # 2: components at threshold 0.5
        components_at_thresholds[1],   # 3: components at threshold 1.0
        components_at_thresholds[2],   # 4: components at threshold 1.5
        max_h0_pers,                   # 5: max H0 persistence
        mean_h0_pers,                  # 6: mean H0 persistence
        total_h0_pers,                 # 7: total H0 persistence
        max_h1_pers,                   # 8: max H1 persistence
        density,                       # 9: topological density
    ], dtype=np.float32)

    return features


class TDAFeatures:
    """Card 17: TDA Features featuriser.

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(self, max_dim: int = 1, max_edge_length: float = 2.0, **kwargs):
        self.max_dim = max_dim
        self.max_edge_length = max_edge_length

    @property
    def n_features(self) -> int:
        return _N_FEATURES

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract TDA features from a single SMT-LIB2 instance."""
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

        if n_vars == 0 or n_vars > _MAX_NODES:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(_N_FEATURES, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=_N_FEATURES,
                instance_id=str(path),
                logic=getattr(info, 'logic', None),
            )

        G = _build_vig(clauses, n_vars)
        thresholds = [0.5, 1.0, 1.5]

        # Try ripser-based computation for small graphs
        ripser = _try_import_ripser()
        if ripser is not None and G.number_of_nodes() <= 2000:
            try:
                dist = _graph_distance_matrix(G, max_dist=self.max_edge_length)
                dgms = _persistence_from_distance(dist, self.max_dim, self.max_edge_length)
                if dgms is not None:
                    features = _persistence_features_from_diagrams(dgms, G, thresholds)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    return FeatureResult(
                        features=features,
                        feature_type="VECTOR",
                        wall_time_ms=elapsed_ms,
                        n_features=_N_FEATURES,
                        instance_id=str(path),
                        logic=info.logic,
                    )
            except Exception:
                pass

        # Fallback: graph-based Betti number estimates
        features = _fallback_persistence_features(G, thresholds)

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
