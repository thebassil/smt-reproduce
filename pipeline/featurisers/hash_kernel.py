"""Card 12: Hash Kernel featuriser.

O(n) feature-hashing approximation of expensive string kernels.
Hashes token bigrams and trigrams from the raw SMT-LIB token stream into
a fixed-size bucket vector with signed hashing.
Default output: 2048-dimensional VECTOR.
"""
from __future__ import annotations

import hashlib
import struct
import time
from pathlib import Path
from typing import ClassVar, List, Literal, Tuple, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import parse_file


def _hash_token(token: str) -> int:
    """Deterministic hash of a token string to a 64-bit unsigned integer."""
    return struct.unpack('<Q', hashlib.md5(token.encode('utf-8', errors='replace')).digest()[:8])[0]


class HashKernel:
    """Hash Kernel featuriser (Card 12).

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        n_features_out: int = 2048,
        ngram_range: Tuple[int, int] = (2, 3),
        **kwargs,
    ) -> None:
        self.n_features_out = n_features_out
        self.ngram_range = ngram_range

    @property
    def n_features(self) -> int:
        return self.n_features_out

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract hash kernel features from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
            tokens = info.raw_tokens
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(self.n_features_out, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.n_features_out,
                instance_id=str(path),
                logic=None,
            )

        buckets = np.zeros(self.n_features_out, dtype=np.float64)
        n_tokens = len(tokens)

        if n_tokens == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=buckets.astype(np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=self.n_features_out,
                instance_id=str(path),
                logic=info.logic,
            )

        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            if n > n_tokens:
                continue
            for i in range(n_tokens - n + 1):
                ngram = '\x00'.join(tokens[i:i + n])
                # Primary hash: determines bucket index
                h = _hash_token(ngram)
                bucket_idx = h % self.n_features_out
                # Sign hash: use a different portion of the hash for sign
                sign = 1.0 if (h >> 32) & 1 else -1.0
                buckets[bucket_idx] += sign

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FeatureResult(
            features=buckets.astype(np.float32),
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=self.n_features_out,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
