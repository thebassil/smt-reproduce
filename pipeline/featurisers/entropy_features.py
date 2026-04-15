"""Card 9: Entropy-based features.

Computes distributional entropy and structural statistics from parsed SMT-LIB
formulas. ~10-dimensional VECTOR output. All operations are O(n) in formula size.

Features (10-dim by default, 11-dim with include_mi=True):
  0: variable frequency entropy
  1: assertion size entropy (by node count)
  2: operator frequency entropy
  3: mean variables per assertion
  4: max variables per assertion
  5: avg tree depth across assertions
  6: total nodes across all assertions
  7: n_vars / n_assertions ratio
  8: n_distinct_operators
  9: file_size_bytes (log1p-scaled)
 10: (optional) variable-operator mutual information proxy
"""
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Set, Union

import numpy as np

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import (
    collect_operators,
    count_nodes,
    parse_file,
    tree_depth,
)


def _entropy(counts: List[int]) -> float:
    """Shannon entropy (base-2) from a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _collect_variables(expr, declared: Optional[Set[str]] = None) -> List[str]:
    """Iteratively collect variable names from a parsed S-expression.

    If declared is provided, only names in that set are counted.
    Otherwise, any non-numeric, non-keyword atom is treated as a variable.
    """
    variables: List[str] = []
    work = [expr]
    _skip = frozenset({
        "true", "false", "and", "or", "not", "=>", "xor", "ite",
        "=", "distinct", "+", "-", "*", "/", "div", "mod", "abs",
        "<", ">", "<=", ">=",
        "bvadd", "bvsub", "bvmul", "bvudiv", "bvsdiv", "bvurem",
        "bvsrem", "bvsmod", "bvneg", "bvand", "bvor", "bvnot",
        "bvxor", "bvshl", "bvlshr", "bvashr", "concat", "extract",
        "zero_extend", "sign_extend", "repeat", "rotate_left",
        "rotate_right", "bvult", "bvslt", "bvugt", "bvsgt",
        "bvule", "bvsle", "bvuge", "bvsge", "bvcomp",
        "select", "store", "forall", "exists", "let", "as",
        "str.++", "str.len", "str.contains", "to_real", "to_int",
        "fp.add", "fp.sub", "fp.mul", "fp.div",
    })
    while work:
        node = work.pop()
        if isinstance(node, str):
            if node.startswith(":") or node.startswith('"') or node.startswith("|"):
                continue
            # Skip numerals
            stripped = node.lstrip("-")
            if stripped and stripped.replace(".", "", 1).isdigit():
                continue
            if node in _skip:
                continue
            if declared is not None:
                if node in declared:
                    variables.append(node)
            else:
                variables.append(node)
        elif isinstance(node, list):
            work.extend(node)
    return variables


class EntropyFeatures:
    """Entropy-based featuriser (Card 9).

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(self, include_mi: bool = True, **kwargs):
        self.include_mi = include_mi

    @property
    def n_features(self) -> int:
        return 11 if self.include_mi else 10

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract entropy features from a single SMT-LIB2 instance."""
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

        declared_set = set(info.declared_vars) if info.declared_vars else None

        # --- Variable frequency entropy ---
        all_var_occurrences: Counter = Counter()
        vars_per_assertion: List[int] = []

        # Also collect per-assertion variable sets for MI proxy
        assertion_var_sets: List[Set[str]] = []

        for assertion in info.assertions:
            avars = _collect_variables(assertion, declared_set)
            all_var_occurrences.update(avars)
            vars_per_assertion.append(len(set(avars)))
            assertion_var_sets.append(set(avars))

        var_freq_entropy = _entropy(list(all_var_occurrences.values()))

        # --- Assertion size entropy (distribution of sizes by node count) ---
        assertion_sizes: List[int] = []
        total_nodes = 0
        for assertion in info.assertions:
            nc = count_nodes(assertion)
            assertion_sizes.append(nc)
            total_nodes += nc

        assertion_size_entropy = _entropy(assertion_sizes)

        # --- Operator frequency entropy ---
        all_op_counts: Counter = Counter()
        for assertion in info.assertions:
            ops = collect_operators(assertion)
            for op, cnt in ops.items():
                all_op_counts[op] += cnt

        op_freq_entropy = _entropy(list(all_op_counts.values()))

        # --- Variable-assertion incidence stats ---
        n_assertions = max(len(info.assertions), 1)
        mean_vars_per_assertion = (
            sum(vars_per_assertion) / n_assertions if vars_per_assertion else 0.0
        )
        max_vars_per_assertion = max(vars_per_assertion) if vars_per_assertion else 0

        # --- Structural stats ---
        depths: List[int] = []
        for assertion in info.assertions:
            depths.append(tree_depth(assertion))
        avg_tree_depth = sum(depths) / n_assertions if depths else 0.0

        n_vars = len(info.declared_vars)
        var_assertion_ratio = n_vars / n_assertions if n_assertions > 0 else 0.0

        n_distinct_operators = len(all_op_counts)

        file_size_log = math.log1p(info.file_size_bytes)

        features = [
            var_freq_entropy,
            assertion_size_entropy,
            op_freq_entropy,
            mean_vars_per_assertion,
            float(max_vars_per_assertion),
            avg_tree_depth,
            float(total_nodes),
            var_assertion_ratio,
            float(n_distinct_operators),
            file_size_log,
        ]

        # --- Optional MI proxy ---
        if self.include_mi:
            # Mutual information proxy: average pairwise variable co-occurrence
            # across assertions, normalized. Approximation: for each variable,
            # count how many assertions it appears in, then compute entropy
            # of that distribution vs uniform.
            var_assertion_counts: Counter = Counter()
            for vset in assertion_var_sets:
                for v in vset:
                    var_assertion_counts[v] += 1

            if var_assertion_counts and n_assertions > 1:
                # Normalized entropy of variable-assertion membership
                freqs = list(var_assertion_counts.values())
                total_memberships = sum(freqs)
                mi_proxy = 0.0
                for f in freqs:
                    p = f / n_assertions  # fraction of assertions containing var
                    if 0 < p < 1:
                        mi_proxy -= p * math.log2(p) + (1 - p) * math.log2(1 - p)
                mi_proxy /= len(freqs)  # average per variable
            else:
                mi_proxy = 0.0

            features.append(mi_proxy)

        feat_array = np.array(features, dtype=np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=feat_array,
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=self.n_features,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
