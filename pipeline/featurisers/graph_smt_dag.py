"""Card 22: SMT Expression DAG featuriser (Sibyl-style).

AST-based expression DAG where nodes are subexpressions and edges are
parent->child relationships.  Identical subexpressions are deduplicated
(DAG, not tree).  Node features: one-hot operator type encoding using
the top-30 most common operators plus an "other" bucket (31-dim).

All walks are iterative (stack-based) to handle deeply nested formulas.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import parse_file

# Top-30 SMT-LIB operators (sorted roughly by frequency across logics)
_TOP_OPS: List[str] = [
    "and", "or", "not", "=>", "=", "ite",
    "+", "-", "*", "/",
    "<=", ">=", "<", ">",
    "bvadd", "bvsub", "bvmul", "bvand", "bvor", "bvxor",
    "bvnot", "bvshl", "bvlshr", "bvashr",
    "concat", "extract", "bvult", "bvslt",
    "distinct", "let",
]
_OP_TO_IDX: Dict[str, int] = {op: i for i, op in enumerate(_TOP_OPS)}
_OTHER_IDX: int = len(_TOP_OPS)  # index 30
_FEAT_DIM: int = len(_TOP_OPS) + 1  # 31


def _op_index(token: str) -> int:
    """Return the feature index for an operator token."""
    return _OP_TO_IDX.get(token, _OTHER_IDX)


def _build_dag(
    assertions: list,
) -> tuple[list[int], list[int], list[int]]:
    """Build a DAG from parsed assertions. Returns (node_op_indices, src, dst).

    Uses iterative post-order traversal with deduplication.
    """
    node_ops: list[int] = []  # op index per node
    src_list: list[int] = []
    dst_list: list[int] = []
    atom_to_id: Dict[str, int] = {}
    tuple_to_id: Dict[tuple, int] = {}

    for assertion in assertions:
        _process_expr(assertion, node_ops, src_list, dst_list, atom_to_id, tuple_to_id)

    return node_ops, src_list, dst_list


def _process_expr(
    root,
    node_ops: list[int],
    src_list: list[int],
    dst_list: list[int],
    atom_to_id: Dict[str, int],
    tuple_to_id: Dict[tuple, int],
) -> int:
    """Process one expression into the DAG iteratively. Returns the node id of root."""
    # We use a stack-based post-order traversal.
    # Stack frames: [expr, children_pushed: bool, child_ids: list]
    # After all children are resolved, we create/lookup the node.
    # Child node IDs are collected via a separate result mechanism.

    # To avoid recursion: use an explicit stack. When a frame's children
    # are all resolved, we pop it and push its node id to the parent frame.

    _VISIT = 0
    _DONE = 1

    stack: list[list] = [[root, _VISIT, []]]

    while stack:
        frame = stack[-1]
        expr = frame[0]
        phase = frame[1]
        child_ids = frame[2]

        if phase == _DONE:
            # Create/lookup node, pop frame, report to parent
            stack.pop()
            nid = _resolve_node(expr, child_ids, node_ops, src_list, dst_list,
                                atom_to_id, tuple_to_id)
            if stack:
                stack[-1][2].append(nid)
            else:
                return nid
            continue

        # Mark as done (children will be pushed next)
        frame[1] = _DONE

        if isinstance(expr, str):
            # Atom: no children to push
            continue
        elif isinstance(expr, list):
            if len(expr) == 0:
                continue
            elif len(expr) == 1:
                # Single-element list: one child
                stack.append([expr[0], _VISIT, []])
            else:
                # (op child1 child2 ...) — push children (not op) in reverse
                for child in reversed(expr[1:]):
                    stack.append([child, _VISIT, []])

    return len(node_ops) - 1 if node_ops else 0


def _resolve_node(
    expr,
    child_ids: list[int],
    node_ops: list[int],
    src_list: list[int],
    dst_list: list[int],
    atom_to_id: Dict[str, int],
    tuple_to_id: Dict[tuple, int],
) -> int:
    """Create or look up a DAG node. Returns its node id."""
    if isinstance(expr, str):
        if expr in atom_to_id:
            return atom_to_id[expr]
        nid = len(node_ops)
        atom_to_id[expr] = nid
        node_ops.append(_op_index(expr))
        return nid

    if isinstance(expr, list) and len(expr) > 0:
        op = expr[0] if isinstance(expr[0], str) else ""
        dedup_key = (op,) + tuple(child_ids)
        if dedup_key in tuple_to_id:
            return tuple_to_id[dedup_key]
        nid = len(node_ops)
        tuple_to_id[dedup_key] = nid
        node_ops.append(_op_index(op))
        # Add parent->child edges
        for cnid in child_ids:
            src_list.append(nid)
            dst_list.append(cnid)
        return nid

    # Fallback
    nid = len(node_ops)
    node_ops.append(_OTHER_IDX)
    return nid


class SMTDAGFeaturiser:
    """SMT Expression DAG featuriser (Card 22).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract the expression DAG from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, _FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=_FEAT_DIM,
                instance_id=str(path),
                logic=None,
            )

        node_ops, src_list, dst_list = _build_dag(info.assertions)
        n_nodes = len(node_ops)

        if n_nodes == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, _FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=_FEAT_DIM,
                instance_id=str(path),
                logic=info.logic,
            )

        # Build node feature matrix (one-hot)
        x = torch.zeros((n_nodes, _FEAT_DIM), dtype=torch.float)
        for i, op_idx in enumerate(node_ops):
            x[i, op_idx] = 1.0

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=graph,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=_FEAT_DIM,
            instance_id=str(path),
            logic=info.logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
