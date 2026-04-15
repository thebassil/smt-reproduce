"""Card 25: And-Inverter Graph (AIG) featuriser.

AIG representation primarily for QF_BV / propositional formulas.
For other logics, each assertion is treated as a single AND gate with
its top-level subexpressions as inputs.

Nodes: AND gates + input variables.
Edges: AND gate -> input (directed), with inversion bit in edge feature.
Node features: 2-dim [is_and_gate, is_input].
Edge features: 1-dim [is_inverted].

All walks are iterative (stack-based).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Set, Tuple, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import parse_file

_NODE_FEAT_DIM = 2
_EDGE_FEAT_DIM = 1

# Operators that map to AND gates
_AND_OPS = frozenset({"and", "bvand"})
# Operators that indicate inversion
_NOT_OPS = frozenset({"not", "bvnot"})
# Operators treated as OR (= AND of inverted inputs via De Morgan)
_OR_OPS = frozenset({"or", "bvor"})


class AIGFeaturiser:
    """And-Inverter Graph featuriser (Card 25).

    Implements the Featuriser protocol with input_type = "GRAPH".
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(self) -> None:
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract AIG from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            info = parse_file(path)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, _NODE_FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, _EDGE_FEAT_DIM), dtype=torch.float),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=_NODE_FEAT_DIM,
                instance_id=str(path),
                logic=None,
            )

        is_boolean = info.logic is not None and (
            "BV" in info.logic or "PROP" in info.logic.upper() or info.logic == "QF_UF"
        )

        # Build AIG
        node_types: list[int] = []  # 0=AND gate, 1=input
        src_list: list[int] = []
        dst_list: list[int] = []
        inv_list: list[float] = []  # 1.0 if inverted, 0.0 otherwise

        input_to_id: Dict[str, int] = {}

        def _get_input_id(name: str) -> int:
            if name in input_to_id:
                return input_to_id[name]
            nid = len(node_types)
            input_to_id[name] = nid
            node_types.append(1)  # input
            return nid

        def _new_and_gate() -> int:
            nid = len(node_types)
            node_types.append(0)  # AND gate
            return nid

        if is_boolean:
            # Full AIG construction by walking assertions
            for assertion in info.assertions:
                self._build_aig_iterative(
                    assertion, node_types, src_list, dst_list, inv_list,
                    input_to_id, _get_input_id, _new_and_gate,
                )
        else:
            # Non-boolean: treat each assertion as one AND gate over its children
            for assertion in info.assertions:
                if isinstance(assertion, list) and len(assertion) > 1:
                    gate = _new_and_gate()
                    for child in assertion[1:]:
                        child_name = self._expr_label(child)
                        inp = _get_input_id(child_name)
                        src_list.append(gate)
                        dst_list.append(inp)
                        inv_list.append(0.0)
                elif isinstance(assertion, str):
                    _get_input_id(assertion)

        n_nodes = len(node_types)

        if n_nodes == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, _NODE_FEAT_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, _EDGE_FEAT_DIM), dtype=torch.float),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=_NODE_FEAT_DIM,
                instance_id=str(path),
                logic=info.logic,
            )

        # Node features
        x = torch.zeros((n_nodes, _NODE_FEAT_DIM), dtype=torch.float)
        for i, nt in enumerate(node_types):
            x[i, nt] = 1.0

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(inv_list, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, _EDGE_FEAT_DIM), dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FeatureResult(
            features=graph,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=_NODE_FEAT_DIM,
            instance_id=str(path),
            logic=info.logic,
        )

    @staticmethod
    def _build_aig_iterative(
        root,
        node_types: list[int],
        src_list: list[int],
        dst_list: list[int],
        inv_list: list[float],
        input_to_id: Dict[str, int],
        get_input_id,
        new_and_gate,
    ) -> int:
        """Build AIG nodes/edges iteratively. Returns root node id.

        Stack-based post-order traversal.
        """
        _VISIT = 0
        _DONE = 1

        # Stack frames: [expr, phase, child_ids, inverted]
        # inverted tracks whether this expr is under a NOT
        stack: list[list] = [[root, _VISIT, [], False]]

        while stack:
            frame = stack[-1]
            expr, phase, child_ids, inverted = frame[0], frame[1], frame[2], frame[3]

            if phase == _DONE:
                stack.pop()
                nid = -1

                if isinstance(expr, str):
                    nid = get_input_id(expr)
                elif isinstance(expr, list) and len(expr) > 0:
                    op = expr[0] if isinstance(expr[0], str) else ""

                    if op in _NOT_OPS and len(child_ids) == 1:
                        # NOT doesn't create a node; it inverts the child
                        nid = child_ids[0]
                        inverted = not inverted
                    elif op in _AND_OPS and child_ids:
                        gate = new_and_gate()
                        for cnid in child_ids:
                            src_list.append(gate)
                            dst_list.append(cnid)
                            inv_list.append(0.0)
                        nid = gate
                    elif op in _OR_OPS and child_ids:
                        # OR(a,b) = NOT(AND(NOT(a), NOT(b))) via De Morgan
                        gate = new_and_gate()
                        for cnid in child_ids:
                            src_list.append(gate)
                            dst_list.append(cnid)
                            inv_list.append(1.0)  # inverted inputs
                        nid = gate
                        inverted = not inverted
                    elif child_ids:
                        # Other ops: treat as AND gate
                        gate = new_and_gate()
                        for cnid in child_ids:
                            src_list.append(gate)
                            dst_list.append(cnid)
                            inv_list.append(0.0)
                        nid = gate
                    else:
                        nid = get_input_id(op)
                else:
                    nid = get_input_id(str(expr))

                # Report to parent
                if stack:
                    stack[-1][2].append(nid)

                continue

            # VISIT phase
            frame[1] = _DONE

            if isinstance(expr, str):
                continue
            elif isinstance(expr, list) and len(expr) > 1:
                for child in reversed(expr[1:]):
                    stack.append([child, _VISIT, [], False])
            elif isinstance(expr, list) and len(expr) == 1:
                stack.append([expr[0], _VISIT, [], False])

        return len(node_types) - 1 if node_types else 0

    @staticmethod
    def _expr_label(expr) -> str:
        """Get a string label for an expression (for non-boolean fallback)."""
        if isinstance(expr, str):
            return expr
        if isinstance(expr, list) and len(expr) > 0:
            op = expr[0] if isinstance(expr[0], str) else "?"
            return f"({op} ...)"
        return str(expr)

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
