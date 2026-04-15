"""Card 27: Variable Incidence Graph (VIG) featuriser.

Homogeneous graph with variable nodes only.
Two variables share an edge if they co-occur in any clause.
Node features: 1-dim (ones; degree can be derived from edge_index).
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


class VIGFeaturiser:
    """Variable Incidence Graph featuriser (Card 27).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract VIG graph from a single SMT-LIB2 instance."""
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

        # Collect co-occurrence edges (undirected, deduplicated)
        edge_set: set[tuple[int, int]] = set()

        for clause in clauses:
            # Get unique variable IDs (0-based) in this clause
            clause_vars: list[int] = []
            seen: set[int] = set()
            for lit in clause:
                var_id = abs(lit)  # 1-based
                if var_id < 1 or var_id > n_vars:
                    continue
                var_node = var_id - 1  # 0-based
                if var_node not in seen:
                    seen.add(var_node)
                    clause_vars.append(var_node)

            # Add edge for every pair of co-occurring variables
            for u, v in combinations(sorted(clause_vars), 2):
                edge_set.add((u, v))

        # Node features: 1-dim ones
        x = torch.ones((n_vars, 1), dtype=torch.float)

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
