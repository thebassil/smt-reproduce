"""Card 20: LCG* (Literal-Clause Graph with negation edges) featuriser.

Heterogeneous graph extending the standard LCG with explicit negation edges
between complementary literals (pos_x <-> neg_x for each variable x).

Node types: positive_literal, negative_literal, clause
Edge types: "appears_in" (lit<->clause), "negation" (lit<->complement)
Node features: 3-dim one-hot [is_pos_lit, is_neg_lit, is_clause]
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import HeteroData

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


def _empty_hetero(feat_dim: int = 3) -> HeteroData:
    """Return an empty HeteroData with the expected schema."""
    data = HeteroData()
    data["pos_lit"].x = torch.zeros((0, feat_dim), dtype=torch.float)
    data["neg_lit"].x = torch.zeros((0, feat_dim), dtype=torch.float)
    data["clause"].x = torch.zeros((0, feat_dim), dtype=torch.float)
    empty_ei = torch.zeros((2, 0), dtype=torch.long)
    data["pos_lit", "appears_in", "clause"].edge_index = empty_ei.clone()
    data["clause", "appears_in", "pos_lit"].edge_index = empty_ei.clone()
    data["neg_lit", "appears_in", "clause"].edge_index = empty_ei.clone()
    data["clause", "appears_in", "neg_lit"].edge_index = empty_ei.clone()
    data["pos_lit", "negation", "neg_lit"].edge_index = empty_ei.clone()
    data["neg_lit", "negation", "pos_lit"].edge_index = empty_ei.clone()
    return data


class LCGStarFeaturiser:
    """LCG* featuriser (Card 20).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract LCG* heterogeneous graph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()
        feat_dim = 3

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=_empty_hetero(feat_dim),
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=feat_dim,
                instance_id=str(path),
                logic=None,
            )

        n_vars = len(var_to_id)
        n_clauses = len(clauses)

        if n_vars == 0 and n_clauses == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=_empty_hetero(feat_dim),
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=feat_dim,
                instance_id=str(path),
                logic=info.logic,
            )

        # Node features: one-hot per type
        pos_x = torch.zeros((n_vars, feat_dim), dtype=torch.float)
        pos_x[:, 0] = 1.0
        neg_x = torch.zeros((n_vars, feat_dim), dtype=torch.float)
        neg_x[:, 1] = 1.0
        clause_x = torch.zeros((n_clauses, feat_dim), dtype=torch.float)
        clause_x[:, 2] = 1.0

        # Appears-in edges (per polarity, separate lists)
        pos_src: list[int] = []
        pos_dst: list[int] = []
        neg_src: list[int] = []
        neg_dst: list[int] = []

        for clause_idx, clause in enumerate(clauses):
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                var_node = var_id - 1
                if lit > 0:
                    pos_src.append(var_node)
                    pos_dst.append(clause_idx)
                else:
                    neg_src.append(var_node)
                    neg_dst.append(clause_idx)

        # Negation edges: for each variable, pos_lit <-> neg_lit
        neg_edge_a = list(range(n_vars))
        neg_edge_b = list(range(n_vars))

        # Build HeteroData
        data = HeteroData()
        data["pos_lit"].x = pos_x
        data["neg_lit"].x = neg_x
        data["clause"].x = clause_x

        def _ei(src: list[int], dst: list[int]) -> torch.Tensor:
            if src:
                return torch.tensor([src, dst], dtype=torch.long)
            return torch.zeros((2, 0), dtype=torch.long)

        # appears_in edges (bidirectional)
        data["pos_lit", "appears_in", "clause"].edge_index = _ei(pos_src, pos_dst)
        data["clause", "appears_in", "pos_lit"].edge_index = _ei(pos_dst, pos_src)
        data["neg_lit", "appears_in", "clause"].edge_index = _ei(neg_src, neg_dst)
        data["clause", "appears_in", "neg_lit"].edge_index = _ei(neg_dst, neg_src)

        # negation edges (bidirectional)
        data["pos_lit", "negation", "neg_lit"].edge_index = _ei(neg_edge_a, neg_edge_b)
        data["neg_lit", "negation", "pos_lit"].edge_index = _ei(neg_edge_b, neg_edge_a)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=data,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=feat_dim,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
