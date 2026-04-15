"""Card 30: Variable-Clause Graph (VCG) featuriser.

Bipartite graph with variable nodes and clause nodes.
Variable nodes: n_vars.
Clause nodes: n_clauses.
Edges: variable <-> clause when the variable (positive or negative) appears in the clause.
Node features: 2-dim one-hot [is_variable, is_clause].
Loses polarity information compared to LCG.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


class VCGFeaturiser:
    """Variable-Clause Graph featuriser (Card 30).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract VCG graph from a single SMT-LIB2 instance."""
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
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=2,
                instance_id=str(path),
                logic=info.logic,
            )

        # Node layout: [var_0, var_1, ..., clause_0, clause_1, ...]
        n_total = n_vars + n_clauses

        # Node features: 2-dim one-hot
        x = torch.zeros((n_total, 2), dtype=torch.float)
        x[:n_vars, 0] = 1.0      # variable nodes
        x[n_vars:, 1] = 1.0      # clause nodes

        # Build edges: variable <-> clause (undirected), deduplicated per clause
        src_list: list[int] = []
        dst_list: list[int] = []

        for clause_idx, clause in enumerate(clauses):
            clause_node = n_vars + clause_idx
            seen_vars: set[int] = set()
            for lit in clause:
                var_id = abs(lit)  # 1-based
                if var_id < 1 or var_id > n_vars:
                    continue
                if var_id in seen_vars:
                    continue
                seen_vars.add(var_id)
                var_node = var_id - 1
                # variable -> clause
                src_list.append(var_node)
                dst_list.append(clause_node)
                # clause -> variable (undirected)
                src_list.append(clause_node)
                dst_list.append(var_node)

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
            n_features=2,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
