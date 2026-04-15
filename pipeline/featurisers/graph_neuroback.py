"""Card 24: NeuroBack compact graph featuriser.

Compact graph for backtracking-like reasoning with no separate literal nodes.
Variable nodes: 2-dim features [pos_count, neg_count] (occurrence counts).
Clause nodes: 1-dim feature [clause_length].
Edges: variable->clause where variable appears (undirected).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file

# Variable features: 2-dim, clause features: 1-dim
# We pad to a common dimension of 3: var=[pos_count, neg_count, 0], clause=[length, 0, 0]
# Actually, use 3-dim for all: var=[pos_count, neg_count, 0], clause=[clause_len, 0, 1]
# The third dim distinguishes node type.
_FEAT_DIM = 3


class NeuroBackFeaturiser:
    """NeuroBack compact graph featuriser (Card 24).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract NeuroBack graph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, _FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=_FEAT_DIM,
                instance_id=str(path),
                logic=None,
            )

        n_vars = len(var_to_id)
        n_clauses = len(clauses)

        if n_vars == 0 and n_clauses == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, _FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=_FEAT_DIM,
                instance_id=str(path),
                logic=info.logic,
            )

        # Count positive and negative occurrences per variable
        pos_count = [0] * n_vars
        neg_count = [0] * n_vars
        clause_lengths = [0] * n_clauses

        # Build edges and count occurrences
        src_list: list[int] = []
        dst_list: list[int] = []

        for clause_idx, clause in enumerate(clauses):
            clause_lengths[clause_idx] = len(clause)
            seen_vars: set[int] = set()
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                if lit > 0:
                    pos_count[var_id - 1] += 1
                else:
                    neg_count[var_id - 1] += 1
                if var_id not in seen_vars:
                    seen_vars.add(var_id)
                    var_node = var_id - 1
                    clause_node = n_vars + clause_idx
                    # Bidirectional
                    src_list.append(var_node)
                    dst_list.append(clause_node)
                    src_list.append(clause_node)
                    dst_list.append(var_node)

        # Node layout: [var_0, ..., var_{n-1}, clause_0, ..., clause_{m-1}]
        n_total = n_vars + n_clauses
        x = torch.zeros((n_total, _FEAT_DIM), dtype=torch.float)

        # Variable features: [pos_count, neg_count, 0]
        for i in range(n_vars):
            x[i, 0] = float(pos_count[i])
            x[i, 1] = float(neg_count[i])

        # Clause features: [clause_length, 0, 1]  (dim 2 = is_clause flag)
        for i in range(n_clauses):
            x[n_vars + i, 0] = float(clause_lengths[i])
            x[n_vars + i, 2] = 1.0

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
            n_features=_FEAT_DIM,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
