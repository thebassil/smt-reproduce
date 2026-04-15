"""Card 28: Literal Incidence Graph (LIG) featuriser.

Homogeneous graph with literal nodes (2 per variable: positive and negative).
Two literals share an edge if they co-occur in any clause.
Node features: 1-dim [1.0 for positive literal, 0.0 for negative literal].
Similar to VIG but at literal level, preserving polarity information.
"""
from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


class LIGFeaturiser:
    """Literal Incidence Graph featuriser (Card 28).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract LIG graph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, 1), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=1,
                instance_id=str(path),
                logic=None,
            )

        n_vars = len(var_to_id)

        # Empty case
        if n_vars == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, 1), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=1,
                instance_id=str(path),
                logic=info.logic,
            )

        # Literal node layout:
        #   positive literal for var i (1-based) -> node index i-1
        #   negative literal for var i (1-based) -> node index n_vars + (i-1)
        n_lit_nodes = 2 * n_vars

        # Node features: 1-dim [is_positive]
        x = torch.zeros((n_lit_nodes, 1), dtype=torch.float)
        x[:n_vars, 0] = 1.0  # positive literals

        # Collect co-occurrence edges between literals in the same clause
        edge_set: set[tuple[int, int]] = set()

        for clause in clauses:
            # Collect unique literal node indices for this clause
            lit_nodes: list[int] = []
            seen: set[int] = set()
            for lit in clause:
                var_id = abs(lit)  # 1-based
                if var_id < 1 or var_id > n_vars:
                    continue
                if lit > 0:
                    node = var_id - 1
                else:
                    node = n_vars + (var_id - 1)
                if node not in seen:
                    seen.add(node)
                    lit_nodes.append(node)

            # Add edge for every pair of co-occurring literals
            for u, v in combinations(sorted(lit_nodes), 2):
                edge_set.add((u, v))

        # Build edge_index (undirected: both directions)
        if edge_set:
            src_list: list[int] = []
            dst_list: list[int] = []
            for u, v in edge_set:
                src_list.append(u)
                dst_list.append(v)
                src_list.append(v)
                dst_list.append(u)
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=graph,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=1,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
