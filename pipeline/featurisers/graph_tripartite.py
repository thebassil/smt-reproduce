"""Card 23: Tripartite Graph featuriser.

Three node types: variable, positive_literal, clause.
Edges: variable->positive_literal, variable->negative_literal, literal->clause.
Optional Laplacian eigenvector positional encoding (PE).

Node features: 3-dim one-hot [is_variable, is_literal, is_clause]
  (+ pe_dim if use_posenc is True).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np
import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


def _laplacian_pe(edge_index: torch.Tensor, n_nodes: int, pe_dim: int) -> torch.Tensor:
    """Compute Laplacian eigenvector positional encoding.

    Returns a (n_nodes, pe_dim) tensor. Falls back to zeros if eigendecomposition
    fails or the graph is too small.
    """
    if n_nodes <= pe_dim or edge_index.size(1) == 0:
        return torch.zeros((n_nodes, pe_dim), dtype=torch.float)

    # Build adjacency as dense (for small-medium graphs; large graphs would need sparse)
    # Clamp to manageable size
    if n_nodes > 10000:
        return torch.zeros((n_nodes, pe_dim), dtype=torch.float)

    try:
        A = torch.zeros((n_nodes, n_nodes), dtype=torch.float)
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0
        # Symmetrize
        A = (A + A.T).clamp(max=1.0)
        # Degree matrix
        deg = A.sum(dim=1)
        D_inv_sqrt = torch.zeros(n_nodes, dtype=torch.float)
        mask = deg > 0
        D_inv_sqrt[mask] = 1.0 / torch.sqrt(deg[mask])
        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        L = torch.eye(n_nodes) - D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # Skip the first eigenvector (constant), take next pe_dim
        pe = eigenvectors[:, 1:pe_dim + 1]
        # Pad if fewer eigenvectors than pe_dim
        if pe.size(1) < pe_dim:
            pad = torch.zeros((n_nodes, pe_dim - pe.size(1)), dtype=torch.float)
            pe = torch.cat([pe, pad], dim=1)
        return pe
    except Exception:
        return torch.zeros((n_nodes, pe_dim), dtype=torch.float)


class TripartiteGraphFeaturiser:
    """Tripartite Graph featuriser (Card 23).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self, use_posenc: bool = True, pe_dim: int = 8) -> None:
        self.use_posenc = use_posenc
        self.pe_dim = pe_dim

    @property
    def _base_dim(self) -> int:
        return 3

    @property
    def _feat_dim(self) -> int:
        return self._base_dim + (self.pe_dim if self.use_posenc else 0)

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract tripartite graph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()
        feat_dim = self._feat_dim

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, feat_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
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
            graph = Data(
                x=torch.zeros((0, feat_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=feat_dim,
                instance_id=str(path),
                logic=info.logic,
            )

        # Node layout:
        #   [0, n_vars)                     -> variable nodes
        #   [n_vars, n_vars + 2*n_vars)     -> literal nodes (pos then neg)
        #   [n_vars + 2*n_vars, ...)         -> clause nodes
        n_lits = 2 * n_vars
        n_total = n_vars + n_lits + n_clauses

        # Base features: 3-dim one-hot
        x_base = torch.zeros((n_total, self._base_dim), dtype=torch.float)
        x_base[:n_vars, 0] = 1.0                             # variable
        x_base[n_vars:n_vars + n_lits, 1] = 1.0             # literal
        x_base[n_vars + n_lits:, 2] = 1.0                   # clause

        # Edges
        src_list: list[int] = []
        dst_list: list[int] = []

        # Variable -> positive literal (var_i -> pos_lit_i)
        for i in range(n_vars):
            pos_lit_node = n_vars + i
            neg_lit_node = n_vars + n_vars + i
            # var -> pos_lit
            src_list.append(i)
            dst_list.append(pos_lit_node)
            # var -> neg_lit
            src_list.append(i)
            dst_list.append(neg_lit_node)

        # Literal -> clause
        clause_offset = n_vars + n_lits
        for clause_idx, clause in enumerate(clauses):
            clause_node = clause_offset + clause_idx
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                if lit > 0:
                    lit_node = n_vars + (var_id - 1)  # pos lit
                else:
                    lit_node = n_vars + n_vars + (var_id - 1)  # neg lit
                src_list.append(lit_node)
                dst_list.append(clause_node)
                # Reverse edge
                src_list.append(clause_node)
                dst_list.append(lit_node)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Positional encoding
        if self.use_posenc:
            pe = _laplacian_pe(edge_index, n_total, self.pe_dim)
            x = torch.cat([x_base, pe], dim=1)
        else:
            x = x_base

        graph = Data(x=x, edge_index=edge_index)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=graph,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=feat_dim,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
