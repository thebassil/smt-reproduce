"""Card 10: Weisfeiler-Leman kernel featuriser.

Builds a Variable Incidence Graph (VIG) from pseudo-CNF, then runs WL
colour-refinement iterations. Features are hash-bucketed histograms of
node labels after each WL iteration.

Output: wl_depth * bucket_size features (default 3 * 1024 = 3072).
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Set, Tuple, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file


def _build_vig(clauses: List[List[int]]) -> Dict[int, Set[int]]:
    """Build a Variable Incidence Graph from CNF clauses.

    Nodes are absolute variable IDs. An edge connects two variables if they
    co-occur in at least one clause.
    """
    adjacency: Dict[int, Set[int]] = defaultdict(set)

    for clause in clauses:
        # Get unique absolute variable IDs in this clause
        abs_vars = list(set(abs(lit) for lit in clause))
        # Add all pairs as edges
        for i in range(len(abs_vars)):
            vi = abs_vars[i]
            adjacency.setdefault(vi, set())
            for j in range(i + 1, len(abs_vars)):
                vj = abs_vars[j]
                adjacency[vi].add(vj)
                adjacency.setdefault(vj, set())
                adjacency[vj].add(vi)

    return dict(adjacency)


def _hash_label(label: str, bucket_size: int) -> int:
    """Hash a label string into a bucket index."""
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    return int(h, 16) % bucket_size


class WLKernel:
    """Weisfeiler-Leman kernel featuriser (Card 10).

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(self, wl_depth: int = 3, bucket_size: int = 1024, **kwargs):
        self.wl_depth = wl_depth
        self.bucket_size = bucket_size

    @property
    def n_features(self) -> int:
        return self.wl_depth * self.bucket_size

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract WL kernel features from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(self.n_features, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.n_features,
                instance_id=str(path),
                logic=None,
            )

        # Extract pseudo-CNF and build VIG
        clauses, var_to_id = extract_cnf(info.assertions)
        adjacency = _build_vig(clauses)

        # If graph is empty, return zeros
        if not adjacency:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(self.n_features, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.n_features,
                instance_id=str(path),
                logic=info.logic,
            )

        nodes = list(adjacency.keys())

        # Initial labels: each node gets its degree as the label
        labels: Dict[int, str] = {}
        for node in nodes:
            labels[node] = str(len(adjacency.get(node, set())))

        features = np.zeros(self.n_features, dtype=np.float32)

        # WL iterations
        for iteration in range(self.wl_depth):
            # Build histogram for current labels
            offset = iteration * self.bucket_size
            label_counts: Counter = Counter()
            for node in nodes:
                bucket = _hash_label(labels[node], self.bucket_size)
                label_counts[bucket] += 1

            for bucket, count in label_counts.items():
                features[offset + bucket] = count

            # Compute new labels: hash(current_label + sorted neighbor labels)
            new_labels: Dict[int, str] = {}
            for node in nodes:
                neighbor_labels = sorted(
                    labels.get(nb, "0") for nb in adjacency.get(node, set())
                )
                combined = labels[node] + "," + ",".join(neighbor_labels)
                new_labels[node] = hashlib.md5(
                    combined.encode("utf-8")
                ).hexdigest()[:16]

            labels = new_labels

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=features,
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=self.n_features,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
