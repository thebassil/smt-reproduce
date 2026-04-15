"""SMT-LIB keyword vocabulary for bag-of-words features.

204 unique keywords, organized into theory groups.
MachSMT uses 210 keywords but has 6 duplicates; we deduplicate.
Helpers: all_keywords(), keywords_excluding(groups), keyword_to_index().
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Theory-grouped keyword definitions
# ---------------------------------------------------------------------------

GROUPS: Dict[str, Tuple[str, ...]] = {
    # Core SMT-LIB commands and control
    "core": (
        "set-logic", "set-info", "set-option", "declare-fun", "declare-const",
        "declare-sort", "define-fun", "define-sort", "define-fun-rec",
        "declare-datatype", "declare-datatypes",
        "assert", "check-sat", "check-sat-assuming",
        "get-model", "get-value", "get-proof", "get-unsat-core",
        "get-unsat-assumptions", "get-info", "get-option", "get-assertions",
        "push", "pop", "reset", "reset-assertions", "exit", "echo",
    ),
    # Boolean / propositional
    "bool": (
        "true", "false", "Bool",
        "and", "or", "not", "=>", "xor", "ite",
        "=", "distinct",
    ),
    # Integer arithmetic (LIA / NIA)
    "int_arith": (
        "Int",
        "+", "-", "*", "div", "mod", "abs",
        "<", ">", "<=", ">=",
    ),
    # Real arithmetic (LRA / NRA)
    "real_arith": (
        "Real",
        "/", "to_real", "to_int", "is_int",
    ),
    # Mixed integer-real
    "mixed_arith": (
        "RoundingMode",
        "roundNearestTiesToEven", "roundNearestTiesToAway",
        "roundTowardPositive", "roundTowardNegative", "roundTowardZero",
        "RNE", "RNA", "RTP", "RTN", "RTZ",
    ),
    # Bit-vectors (QF_BV)
    "bitvec": (
        "BitVec", "_",
        "bvadd", "bvsub", "bvmul", "bvudiv", "bvsdiv", "bvurem", "bvsrem", "bvsmod",
        "bvneg", "bvand", "bvor", "bvnot", "bvxor", "bvnand", "bvnor", "bvxnor",
        "bvshl", "bvlshr", "bvashr",
        "bvult", "bvslt", "bvugt", "bvsgt", "bvule", "bvsle", "bvuge", "bvsge",
        "concat", "extract", "repeat", "zero_extend", "sign_extend",
        "rotate_left", "rotate_right",
        "bv2nat", "nat2bv",
        "bvcomp",
    ),
    # Arrays (QF_AX, QF_ABV, etc.)
    "array": (
        "Array", "select", "store", "const",
    ),
    # Uninterpreted functions / sorts
    "uf": (
        "forall", "exists", "let", "match", "!",
        "as", "par", "NUMERAL", "DECIMAL", "STRING",
    ),
    # Strings and sequences
    "string": (
        "String", "str.++", "str.len", "str.at", "str.substr",
        "str.contains", "str.prefixof", "str.suffixof",
        "str.indexof", "str.replace", "str.replace_all",
        "str.to_int", "str.from_int", "str.to_re", "str.in_re",
        "str.<", "str.<=",
        "re.none", "re.all", "re.allchar",
        "re.++", "re.union", "re.inter", "re.comp",
        "re.*", "re.+", "re.opt", "re.range", "re.loop",
        "Seq", "seq.empty", "seq.unit", "seq.++", "seq.len",
        "seq.extract", "seq.contains", "seq.at", "seq.indexof",
        "seq.replace", "seq.prefixof", "seq.suffixof",
        "seq.nth",
    ),
    # Floating-point (QF_FP)
    "fp": (
        "FloatingPoint", "Float16", "Float32", "Float64", "Float128",
        "fp", "fp.abs", "fp.neg", "fp.add", "fp.sub", "fp.mul", "fp.div",
        "fp.fma", "fp.sqrt", "fp.rem", "fp.roundToIntegral",
        "fp.min", "fp.max",
        "fp.leq", "fp.lt", "fp.geq", "fp.gt", "fp.eq",
        "fp.isNormal", "fp.isSubnormal", "fp.isZero",
        "fp.isInfinite", "fp.isNaN", "fp.isNegative", "fp.isPositive",
        "to_fp", "fp.to_ubv", "fp.to_sbv", "fp.to_real",
        "+oo", "-oo", "+zero", "-zero", "NaN",
    ),
    # Datatypes / algebraic
    "datatype": (
        "Tuple", "pair", "first", "second",
        "List", "nil", "cons", "head", "tail", "insert",
        "is", "match",
    ),
    # Set theory
    "set": (
        "Set", "singleton", "union", "intersection", "setminus",
        "member", "subset", "complement",
        "set.empty", "set.singleton", "set.union", "set.inter",
        "set.minus", "set.member", "set.subset", "set.complement",
        "set.card",
    ),
}

# ---------------------------------------------------------------------------
# Derived structures (built once at import time)
# ---------------------------------------------------------------------------

def _build_keyword_list() -> Tuple[Tuple[str, ...], Dict[str, int], Dict[str, str]]:
    """Build deduplicated keyword list, index map, and group membership."""
    seen: Set[str] = set()
    keywords: List[str] = []
    kw_to_group: Dict[str, str] = {}
    for group_name, group_kws in GROUPS.items():
        for kw in group_kws:
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)
                kw_to_group[kw] = group_name
    kw_tuple = tuple(keywords)
    kw_to_idx = {kw: i for i, kw in enumerate(kw_tuple)}
    return kw_tuple, kw_to_idx, kw_to_group


_KEYWORDS, _KW_TO_IDX, _KW_TO_GROUP = _build_keyword_list()

N_KEYWORDS = len(_KEYWORDS)  # should be 204


def all_keywords() -> Tuple[str, ...]:
    """Return all 204 unique keywords in canonical order."""
    return _KEYWORDS


def keyword_to_index() -> Dict[str, int]:
    """Return mapping from keyword string to its index in the feature vector."""
    return dict(_KW_TO_IDX)


def keyword_to_group() -> Dict[str, str]:
    """Return mapping from keyword string to its theory group name."""
    return dict(_KW_TO_GROUP)


def group_names() -> Tuple[str, ...]:
    """Return all theory group names in definition order."""
    return tuple(GROUPS.keys())


def keywords_for_group(group: str) -> Tuple[str, ...]:
    """Return keywords belonging to a specific theory group."""
    if group not in GROUPS:
        raise ValueError(f"Unknown group '{group}'. Valid groups: {list(GROUPS.keys())}")
    return GROUPS[group]


def keywords_excluding(groups: List[str]) -> Tuple[str, ...]:
    """Return keywords NOT in any of the listed groups."""
    excluded: Set[str] = set()
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"Unknown group '{g}'. Valid groups: {list(GROUPS.keys())}")
        excluded.update(GROUPS[g])
    return tuple(kw for kw in _KEYWORDS if kw not in excluded)


def group_indices(group: str) -> Tuple[int, ...]:
    """Return the indices (into the 204-dim vector) for keywords in a group."""
    kws = keywords_for_group(group)
    return tuple(_KW_TO_IDX[kw] for kw in kws if kw in _KW_TO_IDX)


def excluded_indices(groups: List[str]) -> FrozenSet[int]:
    """Return the set of indices to zero out when excluding groups."""
    indices: Set[int] = set()
    for g in groups:
        indices.update(group_indices(g))
    return frozenset(indices)
