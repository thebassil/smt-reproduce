"""Card 32: Clause Hypergraph featuriser for HGNN.

Each clause is a hyperedge over its variables.
Represented as a bipartite graph: variable nodes + hyperedge nodes.
Incidence matrix via edge_index; additionally stores hyperedge_index.
Node features: variable=[1,0], hyperedge=[0,1] (2-dim).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


class HypergraphFeaturiser:
    """Clause Hypergraph featuriser (Card 32).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract clause hypergraph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, 2), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            graph.hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=2,
                instance_id=str(path),
                logic=None,
            )

        n_vars = len(var_to_id)
        n_clauses = len(clauses)

        # Empty case
        if n_vars == 0 and n_clauses == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, 2), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            graph.hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=2,
                instance_id=str(path),
                logic=info.logic,
            )

        # Node layout: [var_0..var_{n-1}, hyperedge_0..hyperedge_{m-1}]
        n_total = n_vars + n_clauses

        # Node features: 2-dim [is_variable, is_hyperedge]
        x = torch.zeros((n_total, 2), dtype=torch.float)
        x[:n_vars, 0] = 1.0
        x[n_vars:, 1] = 1.0

        # Build incidence edges: variable <-> hyperedge (undirected)
        src_list: list[int] = []
        dst_list: list[int] = []
        # hyperedge_index: [variable_indices, hyperedge_indices] (for HGNN)
        he_var_list: list[int] = []
        he_idx_list: list[int] = []

        for clause_idx, clause in enumerate(clauses):
            he_node = n_vars + clause_idx
            seen_vars: set[int] = set()
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                if var_id in seen_vars:
                    continue
                seen_vars.add(var_id)
                var_node = var_id - 1

                # Bipartite edge_index (undirected)
                src_list.append(var_node)
                dst_list.append(he_node)
                src_list.append(he_node)
                dst_list.append(var_node)

                # hyperedge_index: variable -> hyperedge mapping
                he_var_list.append(var_node)
                he_idx_list.append(clause_idx)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        if he_var_list:
            hyperedge_index = torch.tensor([he_var_list, he_idx_list], dtype=torch.long)
        else:
            hyperedge_index = torch.zeros((2, 0), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index)
        graph.hyperedge_index = hyperedge_index
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=graph,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=2,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
