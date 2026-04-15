"""Card 31: Literal-Clause Graph (LCG) featuriser.

Bipartite graph with literal nodes and clause nodes.
Literal nodes: 2 * n_vars (positive and negative for each variable).
Clause nodes: n_clauses.
Edges: literal <-> clause when that literal appears in the clause.
Node features: 3-dim one-hot [is_pos_literal, is_neg_literal, is_clause].
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


class LCGFeaturiser:
    """Literal-Clause Graph featuriser (Card 31).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract LCG graph from a single SMT-LIB2 instance."""
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

        # Node layout: [pos_lit_0, pos_lit_1, ..., neg_lit_0, neg_lit_1, ..., clause_0, ...]
        n_lit_nodes = 2 * n_vars
        n_total = n_lit_nodes + n_clauses

        # Node features: 3-dim one-hot
        x = torch.zeros((n_total, 3), dtype=torch.float)
        x[:n_vars, 0] = 1.0          # positive literals
        x[n_vars:n_lit_nodes, 1] = 1.0  # negative literals
        x[n_lit_nodes:, 2] = 1.0      # clause nodes

        # Build edges: literal <-> clause (undirected)
        src_list: list[int] = []
        dst_list: list[int] = []

        for clause_idx, clause in enumerate(clauses):
            clause_node = n_lit_nodes + clause_idx
            for lit in clause:
                var_id = abs(lit)  # 1-based
                if var_id < 1 or var_id > n_vars:
                    continue
                if lit > 0:
                    lit_node = var_id - 1  # positive literal node
                else:
                    lit_node = n_vars + (var_id - 1)  # negative literal node
                # literal -> clause
                src_list.append(lit_node)
                dst_list.append(clause_node)
                # clause -> literal (undirected)
                src_list.append(clause_node)
                dst_list.append(lit_node)

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
