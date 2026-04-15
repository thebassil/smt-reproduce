"""MachSMT-style syntactic feature extraction (from scratch).

Reimplements MachSMT's feature computation:
  1. Parse SMT-LIB file into S-expression tokens
  2. Count occurrences of 187 grammatical constructs
  3. Append timeout flag + file size → 189 features

Study reference: artifacts/machsmt/MachSMT/machsmt/benchmark/benchmark.py
                 artifacts/machsmt/MachSMT/machsmt/smtlib/constructs.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union

import numpy as np

from ..types import FeatureResult


# ── SMT-LIB grammatical constructs (187 items, matching MachSMT) ────────────

GRAMMATICAL_CONSTRUCTS: list[str] = [
    # KEYWORDS (34)
    "as", "assert", "check-sat", "check-sat-assuming",
    "declare-const", "declare-datatype", "declare-datatypes",
    "declare-fun", "declare-sort", "define-fun", "define-fun-rec",
    "define-funs-rec", "define-sort", "echo", "exit",
    "get-assertions", "get-assignment", "get-assignment",
    "get-info", "get-info", "get-model", "get-option", "get-option",
    "get-proof", "get-unsat-assumptions", "get-unsat-core", "get-value",
    "pop", "push", "reset", "reset-assertions",
    "set-info", "set-logic", "set-option",
    # BINDERS (3)
    "exists", "forall", "let",
    # CORE (11)
    "true", "false", "not", "=>", "and", "or", "xor",
    "=", "distinct", "ite", "Bool",
    # ARRAYS (3)
    "Array", "select", "store",
    # BV (37)
    "BitVec", "concat", "extract",
    "bvnot", "bvand", "bvor", "bvneg", "bvadd", "bvmul",
    "bvudiv", "bvurem", "bvshl", "bvlshr", "bvult",
    "bvnand", "bvnor", "bvxor", "bvxnor", "bvcomp",
    "bvsub", "bvsdiv", "bvsrem", "bvsmod", "bvashr",
    "repeat", "zero_extend", "sign_extend",
    "rotate_left", "rotate_right",
    "bvule", "bvugt", "bvuge", "bvslt", "bvsle", "bvsgt", "bvsge",
    # FP (50)
    "RoundingMode", "FloatingPoint",
    "Float16", "Float32", "Float64", "Float128", "fp",
    "roundNearestTiesToEven", "roundNearestTiesToAway",
    "roundTowardPositive", "roundTowardNegative", "roundTowardZero",
    "RNE", "RNA", "RTP", "RTN", "RTZ",
    "fp.abs", "fp.neg", "fp.add", "fp.sub", "fp.mul", "fp.div",
    "fp.fma", "fp.sqrt", "fp.rem", "fp.roundToIntegral",
    "fp.min", "fp.max",
    "fp.leq", "fp.lt", "fp.geq", "fp.gt", "fp.eq",
    "fp.isNormal", "fp.isSubnormal", "fp.isZero",
    "fp.isInfinite", "fp.isNaN", "fp.isNegative", "fp.isPositive",
    "to_fp", "to_fp_unsigned", "fp.to_ubv", "fp.to_sbv", "fp.to_real",
    # INTS+REAL (15)
    "Int", "-", "+", "*", "div", "mod", "abs",
    "<=", "<", ">=", ">",
    "to_real", "to_int", "is_int", "Real",
    # STRINGS+REGEX (35)
    "String", "RegLan",
    "str.++", "str.len", "str.<", "str.to_re", "str.in_re",
    "re.none", "re.all", "re.allchar", "re.++",
    "re.union", "re.inter", "re.*",
    "str.<=", "str.at", "str.substr",
    "str.prefixof", "str.suffixof", "str.contains",
    "str.indexof", "str.replace", "str.replace_all",
    "str.replace_re", "str.replace_re_all",
    "re.comp", "re.diff", "re.comp", "re.diff",
    "re.opt", "re.range", "re.range", "re.loop", "re.^",
    "str.is_digit", "str.to_code", "str.from_code",
    "str.to_int", "str.from_int",
]

N_CONSTRUCTS = len(GRAMMATICAL_CONSTRUCTS)
# Feature vector: [construct_counts..., timeout_flag, file_size]
N_FEATURES = N_CONSTRUCTS + 2

# Build lookup: token → list of indices (some constructs appear twice)
_KEYWORD_TO_INDICES: dict[str, list[int]] = {}
for _i, _kw in enumerate(GRAMMATICAL_CONSTRUCTS):
    _KEYWORD_TO_INDICES.setdefault(_kw, []).append(_i)


# ── S-Expression Tokenizer (from scratch, matching MachSMT's behavior) ──────

class _SExprTokenizer:
    """Streaming S-expression tokenizer for SMT-LIB2 files."""

    def __init__(self, path: Union[str, Path]):
        self._file = open(path, "r")

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        token = self._tokenize()
        if token is None:
            self._file.close()
            raise StopIteration
        return token

    def _tokenize(self):
        exprs: list[list] = []
        cur_expr = None
        cur_quoted: list[str] = []
        cur_comment: list[str] = []
        cur_string: list[str] = []
        cur_token = None
        ws = {" ", "\t", "\n"}

        while True:
            ch = self._file.read(1)
            if not ch:
                break

            # String literals
            if (ch == '"' or cur_string) and not cur_comment:
                cur_string.append(ch)
                if ch == '"' and len(cur_string) > 1:
                    if cur_expr is not None:
                        cur_expr.append("".join(cur_string))
                    cur_string = []
                continue

            # Piped (quoted) symbols
            if ch == "|" or cur_quoted:
                if cur_comment:
                    continue
                cur_quoted.append(ch)
                if ch == "|" and len(cur_quoted) > 1:
                    if cur_expr is None:
                        cur_expr = []
                    cur_expr.append("".join(cur_quoted))
                    cur_quoted = []
                continue

            # Comments
            if ch == ";" or cur_comment:
                cur_comment.append(ch)
                if ch == "\n":
                    comment = "".join(cur_comment)
                    cur_comment = []
                    if cur_expr:
                        cur_expr.append(comment)
                    else:
                        return comment
                continue

            # Open S-expression
            if ch == "(":
                if cur_token is not None:
                    cur_expr.append("".join(cur_token))
                    cur_token = None
                cur_expr = []
                exprs.append(cur_expr)

            # Close S-expression
            elif ch == ")":
                if not exprs:
                    continue
                cur_expr = exprs.pop()
                if cur_token is not None:
                    cur_expr.append("".join(cur_token))
                    cur_token = None
                if exprs:
                    exprs[-1].append(tuple(cur_expr))
                    cur_expr = exprs[-1]
                else:
                    return tuple(cur_expr)

            elif cur_token is None and ch not in ws:
                cur_token = [ch]
            elif cur_token and ch in ws:
                token = "".join(cur_token)
                if cur_expr is not None:
                    cur_expr.append(token)
                else:
                    return token
                cur_token = None
            elif cur_token is not None:
                cur_token.append(ch)

        return None

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass


# ── Feature counting ────────────────────────────────────────────────────────

def _count_constructs(
    sexprs: list,
    features: list[int],
    timeout_s: float = 10.0,
) -> None:
    """Count grammatical construct occurrences via iterative traversal.

    Sets features[-2] = 1 if extraction times out.
    """
    deadline = time.monotonic() + timeout_s
    stack = list(sexprs)

    while stack:
        if time.monotonic() > deadline:
            features[-2] = 1
            return
        cur = stack.pop()
        if isinstance(cur, tuple):
            stack.extend(cur)
        elif isinstance(cur, str):
            indices = _KEYWORD_TO_INDICES.get(cur)
            if indices:
                for idx in indices:
                    features[idx] += 1


# ── Public API ──────────────────────────────────────────────────────────────

class MachSMT162:
    """MachSMT-style syntactic featuriser (from scratch).

    Produces 189 features per instance:
      - 187 grammatical construct counts
      - 1 timeout flag (1 if extraction timed out)
      - 1 file size in bytes

    Hydra config: conf/featuriser/machsmt_162.yaml
    """

    input_type: str = "VECTOR"
    output_type: str = "VECTOR"

    def __init__(self, feature_timeout_s: float = 10.0, **kwargs):
        self.feature_timeout_s = feature_timeout_s

    def extract(self, path: Union[str, Path]) -> FeatureResult:
        """Extract features from a single SMT-LIB instance."""
        path = Path(path)
        t0 = time.perf_counter()

        features = [0] * N_FEATURES

        # File size (last element)
        try:
            features[-1] = float(os.path.getsize(path))
        except OSError:
            features[-1] = 0.0

        # Parse and count
        try:
            tokens = list(_SExprTokenizer(path))
            _count_constructs(tokens, features, timeout_s=self.feature_timeout_s)
        except Exception:
            # On parse failure, return zero counts with timeout flag
            features[-2] = 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return FeatureResult(
            features=np.array(features, dtype=np.float64),
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=N_FEATURES,
            instance_id=str(path),
        )

    def extract_batch(self, paths: list[Union[str, Path]]) -> list[FeatureResult]:
        """Extract features from a batch of instances."""
        return [self.extract(p) for p in paths]
