"""Card 16: Spectral Features featuriser.

Computes spectral graph features from the VIG Laplacian and adjacency matrix:
top-k eigenvalues of normalized Laplacian, spectral gap (Fiedler value),
spectral radius, algebraic connectivity, and energy.

~12 features, VECTOR output.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import extract_cnf, parse_file
from pipeline.featurisers.community_structure import _build_vig

_MAX_NODES = 50_000
_DENSE_THRESHOLD = 3000  # Use dense eigensolvers below this size


class SpectralFeatures:
    """Card 16: Spectral Features featuriser.

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(self, n_eigenvalues: int = 5, **kwargs):
        self.n_eigenvalues = n_eigenvalues

    @property
    def n_features(self) -> int:
        # top-k eigenvalues (n_eigenvalues) + spectral_gap + spectral_radius
        # + algebraic_connectivity + energy + n_eigenvalues normalized Laplacian eigenvalues
        # Total: n_eigenvalues (norm Lap top-k) + spectral_gap + spectral_radius
        #        + algebraic_connectivity + energy = n_eigenvalues + 4
        # Plus additional: ratio of top eigenvalues, eigenvalue entropy, sum of squares
        # = n_eigenvalues + 7
        return self.n_eigenvalues + 7

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract spectral features from a single SMT-LIB2 instance."""
        import networkx as nx

        path = Path(instance_path)
        t0 = time.perf_counter()
        n_feat = self.n_features

        try:
            info = parse_file(path)
            clauses, var_to_id = extract_cnf(info.assertions)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(n_feat, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=n_feat,
                instance_id=str(path),
                logic=None,
            )

        n_vars = len(var_to_id)

        if n_vars == 0 or n_vars > _MAX_NODES:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(n_feat, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=n_feat,
                instance_id=str(path),
                logic=getattr(info, 'logic', None),
            )

        G = _build_vig(clauses, n_vars)
        n_nodes = G.number_of_nodes()
        k = self.n_eigenvalues

        if n_nodes <= 1:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(n_feat, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=n_feat,
                instance_id=str(path),
                logic=info.logic,
            )

        # Compute eigenvalues
        try:
            if n_nodes <= _DENSE_THRESHOLD:
                # Dense computation
                L_norm = nx.normalized_laplacian_matrix(G).toarray()
                A = nx.adjacency_matrix(G).toarray().astype(np.float64)

                lap_eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))
                adj_eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
            else:
                # Sparse computation
                from scipy.sparse.linalg import eigsh

                L_norm_sparse = nx.normalized_laplacian_matrix(G).astype(np.float64)
                A_sparse = nx.adjacency_matrix(G).astype(np.float64)

                n_eig = min(k + 2, n_nodes - 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Smallest eigenvalues of normalized Laplacian
                    lap_eigenvalues = eigsh(L_norm_sparse, k=n_eig, which='SM',
                                           return_eigenvectors=False)
                    lap_eigenvalues = np.sort(lap_eigenvalues)
                    # Largest eigenvalues of adjacency
                    adj_eigenvalues = eigsh(A_sparse, k=min(k, n_nodes - 1), which='LM',
                                           return_eigenvectors=False)
                    adj_eigenvalues = np.sort(np.abs(adj_eigenvalues))[::-1]
        except Exception:
            # Eigenvalue computation failed
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FeatureResult(
                features=np.zeros(n_feat, dtype=np.float32),
                feature_type="VECTOR",
                wall_time_ms=elapsed_ms,
                n_features=n_feat,
                instance_id=str(path),
                logic=info.logic,
            )

        # Top-k eigenvalues of normalized Laplacian (largest)
        top_k_lap = np.zeros(k, dtype=np.float64)
        sorted_desc = np.sort(lap_eigenvalues)[::-1]
        n_avail = min(k, len(sorted_desc))
        top_k_lap[:n_avail] = sorted_desc[:n_avail]

        # Spectral gap (Fiedler value) = second smallest eigenvalue of Laplacian
        sorted_asc = np.sort(lap_eigenvalues)
        spectral_gap = float(sorted_asc[1]) if len(sorted_asc) > 1 else 0.0

        # Spectral radius = largest eigenvalue of adjacency matrix
        spectral_radius = float(adj_eigenvalues[0]) if len(adj_eigenvalues) > 0 else 0.0

        # Algebraic connectivity = same as Fiedler value for connected graph
        algebraic_connectivity = spectral_gap

        # Energy = sum of absolute eigenvalues of adjacency
        energy = float(np.sum(np.abs(adj_eigenvalues)))

        # Ratio of top-2 eigenvalues
        if len(adj_eigenvalues) >= 2 and adj_eigenvalues[0] > 0:
            eig_ratio = float(adj_eigenvalues[1] / adj_eigenvalues[0])
        else:
            eig_ratio = 0.0

        # Eigenvalue entropy (from normalized Laplacian)
        pos_eigs = lap_eigenvalues[lap_eigenvalues > 1e-10]
        if len(pos_eigs) > 0:
            normed = pos_eigs / pos_eigs.sum()
            eig_entropy = float(-np.sum(normed * np.log(normed + 1e-15)))
        else:
            eig_entropy = 0.0

        # Sum of squared eigenvalues (relates to number of walks of length 2)
        eig_sum_sq = float(np.sum(adj_eigenvalues ** 2))

        features = np.concatenate([
            top_k_lap.astype(np.float32),                    # 0..k-1: top-k norm Lap eigenvalues
            np.array([
                spectral_gap,           # k: spectral gap / Fiedler value
                spectral_radius,        # k+1: spectral radius
                algebraic_connectivity, # k+2: algebraic connectivity
                energy,                 # k+3: energy
                eig_ratio,              # k+4: ratio of top-2 adj eigenvalues
                eig_entropy,            # k+5: eigenvalue entropy
                eig_sum_sq,             # k+6: sum of squared adj eigenvalues
            ], dtype=np.float32),
        ])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FeatureResult(
            features=features,
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=n_feat,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
