"""Card 15: Structural Width featuriser.

Approximates treewidth and related structural width measures from the VIG:
degree-based upper bound, min-degree elimination ordering width, greedy
max clique size, bandwidth estimate, density, and degree stats.

~8 features, VECTOR output.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file
from pipeline.featurisers.community_structure import _build_vig

_MAX_NODES = 50_000
_N_FEATURES = 8


def _min_degree_elimination_width(G) -> int:
    """Greedy min-degree elimination ordering to approximate treewidth.

    At each step, eliminate the node of minimum degree, connecting all
    its neighbors. Track the maximum degree at elimination time.
    """
    import networkx as nx

    H = G.copy()
    width = 0

    while H.number_of_nodes() > 0:
        # Find node with minimum degree
        min_node = min(H.nodes(), key=lambda v: H.degree(v))
        d = H.degree(min_node)
        if d > width:
            width = d
        # Connect all neighbors (fill-in)
        neighbors = list(H.neighbors(min_node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                H.add_edge(neighbors[i], neighbors[j])
        H.remove_node(min_node)

    return width


def _greedy_max_clique_size(G) -> int:
    """Greedy approximation of maximum clique size (lower bound on treewidth + 1)."""
    import networkx as nx

    if G.number_of_nodes() == 0:
        return 0
    # Use networkx's max_weight_clique with weight=1 for small graphs,
    # or greedy approach for larger ones
    try:
        clique, _ = nx.max_weight_clique(G, weight=None)
        return len(clique)
    except Exception:
        # Fallback: find largest clique via greedy
        best = 0
        for node in G.nodes():
            clique = {node}
            candidates = set(G.neighbors(node))
            for c in sorted(candidates, key=lambda v: -G.degree(v)):
                if all(G.has_edge(c, x) for x in clique):
                    clique.add(c)
            if len(clique) > best:
                best = len(clique)
        return best


def _bandwidth_estimate(G) -> float:
    """Estimate graph bandwidth using a BFS ordering from a peripheral node."""
    import networkx as nx

    if G.number_of_nodes() <= 1:
        return 0.0

    # Pick a peripheral node (end of longest shortest path from arbitrary start)
    components = list(nx.connected_components(G))
    max_bw = 0
    for comp in components:
        if len(comp) <= 1:
            continue
        subg = G.subgraph(comp)
        start = next(iter(comp))
        # BFS to find most distant node
        lengths = nx.single_source_shortest_path_length(subg, start)
        far_node = max(lengths, key=lengths.get)
        # BFS ordering from far_node
        ordering = list(nx.bfs_tree(subg, far_node).nodes())
        pos = {v: i for i, v in enumerate(ordering)}
        bw = max(abs(pos[u] - pos[v]) for u, v in subg.edges())
        if bw > max_bw:
            max_bw = bw

    return float(max_bw)


class StructuralWidth:
    """Card 15: Structural Width featuriser.

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(self, method: str = "min_degree", **kwargs):
        self.method = method

    @property
    def n_features(self) -> int:
        return _N_FEATURES

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract structural width features from a single SMT-LIB2 instance."""
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
        degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Degree-based upper bound on treewidth: max_degree
        degree_upper_bound = float(degrees.max()) if len(degrees) > 0 else 0.0

        # Min-degree elimination width
        if n_nodes <= 5000:
            elim_width = float(_min_degree_elimination_width(G))
        else:
            # Too expensive for large graphs; use degree bound
            elim_width = degree_upper_bound

        # Max clique size (lower bound on treewidth + 1)
        if n_nodes <= 5000:
            max_clique = float(_greedy_max_clique_size(G))
        else:
            max_clique = 0.0

        # Bandwidth estimate
        if n_nodes <= 10000:
            bandwidth = _bandwidth_estimate(G)
        else:
            bandwidth = 0.0

        # Graph density
        max_possible = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
        density = float(n_edges) / max_possible if max_possible > 0 else 0.0

        # Degree stats
        max_deg = float(degrees.max()) if len(degrees) > 0 else 0.0
        mean_deg = float(degrees.mean()) if len(degrees) > 0 else 0.0
        median_deg = float(np.median(degrees)) if len(degrees) > 0 else 0.0

        features = np.array([
            degree_upper_bound,  # 0: degree-based treewidth upper bound
            elim_width,          # 1: min-degree elimination width
            max_clique,          # 2: max clique size (tw lower bound)
            bandwidth,           # 3: bandwidth estimate
            density,             # 4: graph density
            max_deg,             # 5: max degree
            mean_deg,            # 6: mean degree
            median_deg,          # 7: median degree
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
