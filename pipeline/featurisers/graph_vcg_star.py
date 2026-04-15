"""Card 21: VCG* (Variable-Clause Graph with polarity-typed edges) featuriser.

Heterogeneous graph extending VCG with polarity-typed edges.
Node types: variable, clause
Edge types: "positive" (var appears positively), "negative" (var appears negatively)
Node features: 2-dim one-hot [is_variable, is_clause]
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import HeteroData

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


def _empty_hetero(feat_dim: int = 2) -> HeteroData:
    """Return an empty HeteroData with the expected schema."""
    data = HeteroData()
    data["variable"].x = torch.zeros((0, feat_dim), dtype=torch.float)
    data["clause"].x = torch.zeros((0, feat_dim), dtype=torch.float)
    empty_ei = torch.zeros((2, 0), dtype=torch.long)
    data["variable", "positive", "clause"].edge_index = empty_ei.clone()
    data["clause", "positive", "variable"].edge_index = empty_ei.clone()
    data["variable", "negative", "clause"].edge_index = empty_ei.clone()
    data["clause", "negative", "variable"].edge_index = empty_ei.clone()
    return data


class VCGStarFeaturiser:
    """VCG* featuriser (Card 21).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract VCG* heterogeneous graph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()
        feat_dim = 2

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

        # Node features
        var_x = torch.zeros((n_vars, feat_dim), dtype=torch.float)
        var_x[:, 0] = 1.0
        clause_x = torch.zeros((n_clauses, feat_dim), dtype=torch.float)
        clause_x[:, 1] = 1.0

        # Polarity-typed edges
        pos_src: list[int] = []
        pos_dst: list[int] = []
        neg_src: list[int] = []
        neg_dst: list[int] = []

        for clause_idx, clause in enumerate(clauses):
            seen_pos: set[int] = set()
            seen_neg: set[int] = set()
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                var_node = var_id - 1
                if lit > 0 and var_id not in seen_pos:
                    seen_pos.add(var_id)
                    pos_src.append(var_node)
                    pos_dst.append(clause_idx)
                elif lit < 0 and var_id not in seen_neg:
                    seen_neg.add(var_id)
                    neg_src.append(var_node)
                    neg_dst.append(clause_idx)

        def _ei(src: list[int], dst: list[int]) -> torch.Tensor:
            if src:
                return torch.tensor([src, dst], dtype=torch.long)
            return torch.zeros((2, 0), dtype=torch.long)

        data = HeteroData()
        data["variable"].x = var_x
        data["clause"].x = clause_x

        # Bidirectional edges per polarity type
        data["variable", "positive", "clause"].edge_index = _ei(pos_src, pos_dst)
        data["clause", "positive", "variable"].edge_index = _ei(pos_dst, pos_src)
        data["variable", "negative", "clause"].edge_index = _ei(neg_src, neg_dst)
        data["clause", "negative", "variable"].edge_index = _ei(neg_dst, neg_src)

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
