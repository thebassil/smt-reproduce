"""Card 29: Hierarchical Subgraph Pooling graph featuriser.

Start with VCG-like bipartite graph, then add a supernode for each
connected component (found via BFS on the VCG). Each supernode connects
to all nodes in its component.
Node features: 3-dim [is_variable, is_clause, is_supernode].
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Set, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


class SubgraphPoolFeaturiser:
    """Hierarchical Subgraph Pooling featuriser (Card 29).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract hierarchical pooling graph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, 3), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=3,
                instance_id=str(path),
                logic=None,
            )

        n_vars = len(var_to_id)
        n_clauses = len(clauses)

        # Empty case
        if n_vars == 0 and n_clauses == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, 3), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=3,
                instance_id=str(path),
                logic=info.logic,
            )

        # Node layout: [var_0..var_{n-1}, clause_0..clause_{m-1}]
        n_vcg = n_vars + n_clauses

        # Build VCG adjacency list for BFS component detection
        adj: Dict[int, Set[int]] = {i: set() for i in range(n_vcg)}

        # VCG edges
        vcg_src: list[int] = []
        vcg_dst: list[int] = []

        for clause_idx, clause in enumerate(clauses):
            clause_node = n_vars + clause_idx
            seen_vars: set[int] = set()
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                if var_id in seen_vars:
                    continue
                seen_vars.add(var_id)
                var_node = var_id - 1
                adj[var_node].add(clause_node)
                adj[clause_node].add(var_node)
                vcg_src.append(var_node)
                vcg_dst.append(clause_node)
                vcg_src.append(clause_node)
                vcg_dst.append(var_node)

        # BFS to find connected components
        visited: set[int] = set()
        components: list[list[int]] = []

        for start in range(n_vcg):
            if start in visited:
                continue
            component: list[int] = []
            queue = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

        n_supernodes = len(components)
        n_total = n_vcg + n_supernodes

        # Node features: 3-dim [is_variable, is_clause, is_supernode]
        x = torch.zeros((n_total, 3), dtype=torch.float)
        x[:n_vars, 0] = 1.0
        x[n_vars:n_vcg, 1] = 1.0
        x[n_vcg:, 2] = 1.0

        # Build all edges: VCG edges + supernode edges
        src_list: list[int] = list(vcg_src)
        dst_list: list[int] = list(vcg_dst)

        for comp_idx, component in enumerate(components):
            supernode = n_vcg + comp_idx
            for member in component:
                src_list.append(supernode)
                dst_list.append(member)
                src_list.append(member)
                dst_list.append(supernode)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=graph,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=3,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
