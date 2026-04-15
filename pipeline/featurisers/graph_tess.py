"""Card 26: TESS (CDCL-inspired) graph featuriser.

CDCL-inspired graph with activity-like and LBD-like features:
- Variable nodes: [activity, pos_freq, neg_freq]
  where activity = total occurrence frequency (normalised),
  pos_freq / neg_freq = positive/negative occurrence frequency.
- Clause nodes: [lbd_est, length, is_learned]
  where lbd_est = number of unique variable "groups" in the clause
  (proxy for Literal Block Distance), is_learned = 0 (static formula).
- Edges: variable -> clause for each variable appearing in a clause (bidirectional).

Variable groups for LBD estimation: variables are grouped by their index
modulo sqrt(n_vars), as a proxy for decision-level grouping.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file

_FEAT_DIM = 3  # max of var (3) and clause (3) feature dims


class TESSFeaturiser:
    """TESS (CDCL-inspired) graph featuriser (Card 26).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract TESS graph from a single SMT-LIB2 instance."""
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

        # Count occurrences
        pos_count = [0] * n_vars
        neg_count = [0] * n_vars
        total_count = [0] * n_vars

        for clause in clauses:
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                idx = var_id - 1
                total_count[idx] += 1
                if lit > 0:
                    pos_count[idx] += 1
                else:
                    neg_count[idx] += 1

        # Normalise activity by max count
        max_count = max(total_count) if total_count else 1
        if max_count == 0:
            max_count = 1

        # Group size for LBD estimation
        group_size = max(1, int(math.sqrt(n_vars)))

        # Compute LBD estimate per clause
        lbd_estimates: list[float] = []
        clause_lengths: list[float] = []
        for clause in clauses:
            groups_seen: set[int] = set()
            clen = 0
            for lit in clause:
                var_id = abs(lit)
                if var_id < 1 or var_id > n_vars:
                    continue
                clen += 1
                group = (var_id - 1) // group_size
                groups_seen.add(group)
            lbd_estimates.append(float(len(groups_seen)))
            clause_lengths.append(float(clen))

        # Node layout: [var_0, ..., var_{n-1}, clause_0, ..., clause_{m-1}]
        n_total = n_vars + n_clauses
        x = torch.zeros((n_total, _FEAT_DIM), dtype=torch.float)

        # Variable features: [activity, pos_freq, neg_freq]
        for i in range(n_vars):
            x[i, 0] = total_count[i] / max_count  # normalised activity
            tc = total_count[i] if total_count[i] > 0 else 1
            x[i, 1] = pos_count[i] / tc  # positive frequency
            x[i, 2] = neg_count[i] / tc  # negative frequency

        # Clause features: [lbd_est, length, is_learned=0]
        for i in range(n_clauses):
            x[n_vars + i, 0] = lbd_estimates[i]
            x[n_vars + i, 1] = clause_lengths[i]
            x[n_vars + i, 2] = 0.0  # is_learned

        # Edges: variable <-> clause (bidirectional)
        src_list: list[int] = []
        dst_list: list[int] = []

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
                src_list.append(var_node)
                dst_list.append(clause_node)
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
            n_features=_FEAT_DIM,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
