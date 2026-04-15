"""Card 13: Random Walk Kernel featuriser.

Random walk visit-frequency features on the Variable Incidence Graph (VIG).
Builds the VIG from pseudo-CNF (variables are nodes, edge if co-occurring
in a clause), then performs random walks from each variable node, hashing
visited node labels into a fixed-size bucket vector.
Default output: 1024-dimensional VECTOR.
"""
from __future__ import annotations

import hashlib
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Set, Tuple, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


def _hash_var_id(var_id: int) -> int:
    """Deterministic hash of a variable ID to a 64-bit unsigned integer."""
    return struct.unpack(
        '<Q',
        hashlib.md5(str(var_id).encode('ascii')).digest()[:8],
    )[0]


class RandomWalkKernel:
    """Random Walk Kernel featuriser (Card 13).

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        n_walks: int = 100,
        walk_length: int = 5,
        bucket_size: int = 1024,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.bucket_size = bucket_size
        self.seed = seed

    @property
    def n_features(self) -> int:
        return self.bucket_size

    def _build_vig(
        self, clauses: List[List[int]], n_variables: int
    ) -> Dict[int, List[int]]:
        """Build Variable Incidence Graph as adjacency list.

        Nodes are variable IDs (1-based). An edge exists between two variables
        if they co-occur in at least one clause.
        """
        adj: Dict[int, Set[int]] = defaultdict(set)

        for clause in clauses:
            # Collect unique variable IDs in this clause
            vars_in_clause = set()
            for lit in clause:
                vars_in_clause.add(abs(lit))

            var_list = list(vars_in_clause)
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    adj[var_list[i]].add(var_list[j])
                    adj[var_list[j]].add(var_list[i])

        # Convert sets to sorted lists for deterministic traversal
        return {v: sorted(neighbors) for v, neighbors in adj.items()}

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract random walk kernel features from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(self.bucket_size, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.bucket_size,
                instance_id=str(path),
                logic=None,
            )

        n_variables = len(var_to_id)
        buckets = np.zeros(self.bucket_size, dtype=np.float64)

        if n_variables == 0 or len(clauses) == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=buckets.astype(np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.bucket_size,
                instance_id=str(path),
                logic=info.logic,
            )

        adj = self._build_vig(clauses, n_variables)
        rng = np.random.RandomState(self.seed)

        # All variable IDs that have at least one neighbor
        all_vars = sorted(adj.keys())
        if not all_vars:
            # No edges — just hash each isolated variable once
            for vid in sorted(var_to_id.values()):
                h = _hash_var_id(vid)
                buckets[h % self.bucket_size] += 1.0
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=buckets.astype(np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.bucket_size,
                instance_id=str(path),
                logic=info.logic,
            )

        # Perform random walks from each variable node
        for start_var in all_vars:
            neighbors = adj.get(start_var)
            if not neighbors:
                # Isolated node — hash it once per walk
                h = _hash_var_id(start_var)
                buckets[h % self.bucket_size] += float(self.n_walks)
                continue

            for _ in range(self.n_walks):
                current = start_var
                for _step in range(self.walk_length):
                    # Hash current node into buckets
                    h = _hash_var_id(current)
                    buckets[h % self.bucket_size] += 1.0
                    # Move to random neighbor
                    nbrs = adj.get(current)
                    if not nbrs:
                        break
                    current = nbrs[rng.randint(len(nbrs))]

                # Hash the final node too
                h = _hash_var_id(current)
                buckets[h % self.bucket_size] += 1.0

        # Normalize by total walk count to get frequencies
        total = buckets.sum()
        if total > 0:
            buckets /= total

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FeatureResult(
            features=buckets.astype(np.float32),
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=self.bucket_size,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
