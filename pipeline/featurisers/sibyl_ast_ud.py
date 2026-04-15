"""REF-GNN featuriser: Sibyl AST+UD graph construction.

From-scratch reimplementation of Sibyl's graph builder.
Parses SMT-LIB2 → heterogeneous AST + Use-Def directed graph
→ torch_geometric.data.Data with 67-dim one-hot node features.

Reference (study only, not imported):
  artifacts/sibyl/src/data_handlers/graph-builder.py

Node type vocabulary (67 types, indices 0–66):
  0:FORALL  1:EXISTS  2:AND  3:OR  4:NOT  5:IMPLIES  6:IFF
  7:SYMBOL  8:FUNCTION  9:REAL_CONSTANT  10:BOOL_CONSTANT
  11:INT_CONSTANT  12:STR_CONSTANT  13:PLUS  14:MINUS  15:TIMES
  16:LE  17:LT  18:EQUALS  19:ITE  20:TOREAL
  21:BV_CONSTANT  22:BV_NOT  23:BV_AND  24:BV_OR  25:BV_XOR
  26:BV_CONCAT  27:BV_EXTRACT  28:BV_ULT  29:BV_ULE  30:BV_NEG
  31:BV_ADD  32:BV_SUB  33:BV_MUL  34:BV_UDIV  35:BV_UREM
  36:BV_LSHL  37:BV_LSHR  38:BV_ROL  39:BV_ROR  40:BV_ZEXT
  41:BV_SEXT  42:BV_SLT  43:BV_SLE  44:BV_COMP  45:BV_SDIV
  46:BV_SREM  47:BV_ASHR  48:STR_LENGTH  49:STR_CONCAT
  50:STR_CONTAINS  51:STR_INDEXOF  52:STR_REPLACE  53:STR_SUBSTR
  54:STR_PREFIXOF  55:STR_SUFFIXOF  56:STR_TO_INT  57:INT_TO_STR
  58:STR_CHARAT  59:ARRAY_SELECT  60:ARRAY_STORE  61:ARRAY_VALUE
  62:DIV  63:POW  64:ALGEBRAIC_CONSTANT  65:BV_TONATURAL
  66:SYMBOL_AGGREGATOR (synthetic — shared-symbol hub node)

Edge types (stored in edge_attr):
  0 = AST:      parent → child
  1 = Back-AST: child → parent
  2 = Data:     symbol occurrence → SYMBOL_AGGREGATOR hub
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data import Data

from pipeline.types import FeatureResult

try:
    from pysmt.smtlib.parser import SmtLibParser
    from pysmt.walkers.identitydag import IdentityDagWalker
    from pysmt.operators import ALL_TYPES

    _PYSMT_AVAILABLE = True
except ImportError:
    _PYSMT_AVAILABLE = False

# Node feature dimension: len(ALL_TYPES) + 1 for SYMBOL_AGGREGATOR
_NODE_DIM = 67  # pySMT ALL_TYPES has 66 entries (indices 0-65) + 1 synthetic
_SYMBOL_AGGREGATOR_IDX = 66


class _ASTBuilder(IdentityDagWalker):
    """Walk a pySMT formula DAG and build graph tensors.

    Produces:
      - nodes: list of one-hot vectors (length _NODE_DIM)
      - edges: [sources, targets] list pair
      - edge_attr: list of edge-type ints (0/1/2)
      - symbol_to_nodes: dict mapping symbol node_id → list of node indices
    """

    def __init__(self, env=None, invalidate_memoization=None):
        super().__init__(env, invalidate_memoization)
        self.node_counter: int = 0
        self.nodes: list[list[int]] = []
        self.edges: list[list[int]] = [[], []]
        self.edge_attr: list[int] = []
        self.id_to_counter: dict[int, int] = {}
        self.symbol_to_nodes: dict[int, list[int]] = {}
        self.constant_to_nodes: dict[int, list[int]] = {}

    def walk(self, formula, **kwargs):
        if formula in self.memoization:
            return self.memoization[formula]

        self.node_counter = 0
        self.nodes = []
        self.edges = [[], []]
        self.edge_attr = []
        self.id_to_counter = {}
        self.symbol_to_nodes = {}
        self.constant_to_nodes = {}

        res = self.iter_walk(formula, **kwargs)

        if self.invalidate_memoization:
            self.memoization.clear()

        return res

    def _add_node(self, formula) -> None:
        """Create a one-hot node vector for this formula's operator type."""
        rep = [0] * _NODE_DIM
        rep[formula.node_type()] = 1
        self.nodes.append(rep)

    def _get_node_counter(self, formula, is_parent: bool) -> int:
        """Get or create a node index for a formula sub-expression.

        Symbols and constants always get a fresh node when encountered as
        children (is_parent=False) — this preserves multiple occurrences.
        Operators are deduplicated by node_id.
        """
        nid = formula.node_id()

        if formula.is_symbol() and not is_parent:
            idx = self.node_counter
            if nid in self.symbol_to_nodes:
                self.symbol_to_nodes[nid].append(idx)
            else:
                self.symbol_to_nodes[nid] = [idx]
            self.id_to_counter[nid] = idx
            self.node_counter += 1
            self._add_node(formula)
            return idx

        if formula.is_constant() and not is_parent:
            idx = self.node_counter
            if nid in self.constant_to_nodes:
                self.constant_to_nodes[nid].append(idx)
            else:
                self.constant_to_nodes[nid] = [idx]
            self.id_to_counter[nid] = idx
            self.node_counter += 1
            self._add_node(formula)
            return idx

        if nid not in self.id_to_counter:
            idx = self.node_counter
            self.id_to_counter[nid] = idx
            self.node_counter += 1
            self._add_node(formula)
            return idx

        return self.id_to_counter[nid]

    def _push_with_children_to_stack(self, formula, **kwargs):
        """Override: add children to stack while recording AST + Back-AST edges."""
        self.stack.append((True, formula))

        parent_idx = self._get_node_counter(formula, is_parent=True)

        for child in self._get_children(formula):
            child_idx = self._get_node_counter(child, is_parent=False)

            # AST edge: parent → child (type 0)
            self.edges[0].append(parent_idx)
            self.edges[1].append(child_idx)
            self.edge_attr.append(0)

            # Back-AST edge: child → parent (type 1)
            self.edges[0].append(child_idx)
            self.edges[1].append(parent_idx)
            self.edge_attr.append(1)

            key = self._get_key(child, **kwargs)
            if key not in self.memoization:
                self.stack.append((False, child))


def _build_graph_from_formula(
    formula,
    edge_sets: Sequence[str] = ("AST", "Back-AST", "Data"),
    max_nodes: int = 10_000,
) -> Data:
    """Build a torch_geometric Data object from a pySMT formula.

    Args:
        formula: pySMT formula (result of SmtLibParser.get_last_formula()).
        edge_sets: Which edge types to include.
        max_nodes: If graph exceeds this many nodes, return a truncation-safe
            empty graph rather than risk OOM.

    Returns:
        Data with x=[n_nodes, 67], edge_index=[2, n_edges], edge_attr=[n_edges].
    """
    builder = _ASTBuilder()
    builder.walk(formula)

    nodes = builder.nodes
    edges = builder.edges
    edge_attr = builder.edge_attr

    # Add SYMBOL_AGGREGATOR hub nodes for symbols appearing ≥2 times
    if "Data" in edge_sets:
        for occurrences in builder.symbol_to_nodes.values():
            if len(occurrences) < 2:
                continue
            hub_rep = [0] * _NODE_DIM
            hub_rep[_SYMBOL_AGGREGATOR_IDX] = 1
            nodes.append(hub_rep)
            hub_idx = len(nodes) - 1
            for occ_idx in occurrences:
                edges[0].append(occ_idx)
                edges[1].append(hub_idx)
                edge_attr.append(2)

    if len(nodes) > max_nodes:
        return Data(
            x=torch.zeros((0, _NODE_DIM), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros(0, dtype=torch.float),
        )

    x = torch.tensor(nodes, dtype=torch.float)
    ei = torch.tensor(edges, dtype=torch.long) if edges[0] else torch.zeros((2, 0), dtype=torch.long)
    ea = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros(0, dtype=torch.float)

    # Filter edge sets
    if ei.size(1) > 0:
        keep = torch.zeros(ei.size(1), dtype=torch.bool)
        if "AST" in edge_sets:
            keep |= ea == 0
        if "Back-AST" in edge_sets:
            keep |= ea == 1
        if "Data" in edge_sets:
            keep |= ea == 2
        ei = ei[:, keep]
        ea = ea[keep]

    return Data(x=x, edge_index=ei, edge_attr=ea)


class SibylASTUDFeaturiser:
    """Sibyl AST+UD graph featuriser (REF-GNN reference featuriser).

    Parses SMT-LIB2 files via pySMT, walks the formula DAG, and produces
    a directed heterogeneous graph with 67-dim one-hot node features and
    three edge types (AST, Back-AST, Data).

    Implements the Featuriser protocol.
    """

    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"

    def __init__(
        self,
        edge_sets: Sequence[str] = ("AST", "Back-AST", "Data"),
        max_nodes: int = 10_000,
        **kwargs,
    ) -> None:
        if not _PYSMT_AVAILABLE:
            raise ImportError(
                "SibylASTUDFeaturiser requires pySMT. "
                "Install with: pip install pysmt"
            )
        self.edge_sets = tuple(edge_sets)
        self.max_nodes = max_nodes
        self._parser = SmtLibParser()

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract AST+UD graph from a single SMT-LIB2 instance."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            with open(path, "r") as f:
                script = self._parser.get_script(f)
            formula = script.get_last_formula()
            graph = _build_graph_from_formula(
                formula,
                edge_sets=self.edge_sets,
                max_nodes=self.max_nodes,
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            graph = Data(
                x=torch.zeros((0, _NODE_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros(0, dtype=torch.float),
            )
            return FeatureResult(
                features=graph,
                feature_type="GRAPH",
                wall_time_ms=elapsed_ms,
                n_features=_NODE_DIM,
                instance_id=str(path),
                logic=None,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Try to extract logic from the script
        logic: Optional[str] = None
        try:
            for cmd in script.commands:
                if hasattr(cmd, "name") and cmd.name == "set-logic":
                    logic = str(cmd.args[0])
                    break
        except Exception:
            pass

        return FeatureResult(
            features=graph,
            feature_type="GRAPH",
            wall_time_ms=elapsed_ms,
            n_features=_NODE_DIM,
            instance_id=str(path),
            logic=logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
