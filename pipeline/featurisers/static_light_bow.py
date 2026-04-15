"""Card 1: Static-Light Bag-of-Words featuriser.

Counts SMT-LIB keyword occurrences via iterative DFS over tokens.
228 keywords + timeout_flag + file_size = 230 features.

Modes:
  - counts (default): raw keyword frequency counts
  - binary: 0/1 presence flags
  - theory_flags: 12 group flags + 2 meta = 14 features

exclude_groups: zeros out named theory columns (keeps vector length constant).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._keywords import (
    N_KEYWORDS,
    all_keywords,
    excluded_indices,
    group_names,
    keyword_to_index,
)
from pipeline.featurisers._smtlib_parser import parse_file, tokenize


class StaticLightBoW:
    """Static-Light Bag-of-Words featuriser (Card 1).

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        mode: str = "counts",
        exclude_groups: Optional[List[str]] = None,
        feature_timeout_s: float = 30.0,
        **kwargs,
    ):
        self.mode = mode
        self.exclude_groups = exclude_groups or []
        self.feature_timeout_s = feature_timeout_s
        self._kw_to_idx = keyword_to_index()
        self._excluded_idx = excluded_indices(self.exclude_groups) if self.exclude_groups else frozenset()

    @property
    def n_features(self) -> int:
        if self.mode == "theory_flags":
            return len(group_names()) + 2  # group flags + timeout + file_size
        return N_KEYWORDS + 2  # keywords + timeout_flag + file_size

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract BoW features from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()
        timed_out = False

        try:
            info = parse_file(path)
        except Exception:
            # On parse failure, return zeros
            elapsed_ms = (time.perf_counter() - t0) * 1000
            features = np.zeros(self.n_features, dtype=np.float32)
            features[-1] = path.stat().st_size if path.exists() else 0
            return FeatureResult(
                features=features,
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.n_features,
                instance_id=str(path),
                logic=None,
            )

        # Count keywords in raw tokens
        counts = np.zeros(N_KEYWORDS, dtype=np.float32)
        deadline = t0 + self.feature_timeout_s
        for token in info.raw_tokens:
            if time.perf_counter() > deadline:
                timed_out = True
                break
            idx = self._kw_to_idx.get(token)
            if idx is not None:
                counts[idx] += 1

        # Apply mode
        if self.mode == "binary":
            counts = (counts > 0).astype(np.float32)
        elif self.mode == "theory_flags":
            gnames = group_names()
            from pipeline.featurisers._keywords import group_indices
            flags = np.zeros(len(gnames), dtype=np.float32)
            for i, g in enumerate(gnames):
                gidx = group_indices(g)
                flags[i] = 1.0 if any(counts[j] > 0 for j in gidx) else 0.0
            features = np.concatenate([
                flags,
                np.array([float(timed_out), float(info.file_size_bytes)], dtype=np.float32),
            ])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=features,
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.n_features,
                instance_id=str(path),
                logic=info.logic,
            )

        # Zero out excluded groups
        if self._excluded_idx:
            for idx in self._excluded_idx:
                counts[idx] = 0.0

        features = np.concatenate([
            counts,
            np.array([float(timed_out), float(info.file_size_bytes)], dtype=np.float32),
        ])

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
