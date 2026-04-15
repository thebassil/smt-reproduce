"""Card 5: Tree-LSTM Embedding featuriser.

Builds AST from assertions, runs bottom-up Tree-LSTM (children -> parent
aggregation) using iterative (stack-based) traversal. Root embedding =
instance embedding.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import parse_file


@dataclass
class _ASTNode:
    """Linearised AST node for iterative Tree-LSTM."""
    token_id: int
    children: list[int] = field(default_factory=list)  # indices into flat list
    parent: int = -1


def _build_ast(expr, vocab_size: int, max_nodes: int = 5000) -> list[_ASTNode]:
    """Build a flat AST node list from a parsed S-expression (iterative).

    Caps at max_nodes to keep Tree-LSTM tractable.
    """
    nodes: list[_ASTNode] = []

    # Stack items: (expression, parent_index)
    work: list[tuple] = [(expr, -1)]

    while work:
        if len(nodes) >= max_nodes:
            break

        node_expr, parent_idx = work.pop()

        if isinstance(node_expr, str):
            tid = (hash(node_expr) % (vocab_size - 1)) + 1  # 0 reserved
            new_idx = len(nodes)
            nodes.append(_ASTNode(token_id=tid, parent=parent_idx))
            if parent_idx >= 0:
                nodes[parent_idx].children.append(new_idx)

        elif isinstance(node_expr, list) and len(node_expr) > 0:
            # Create node for the operator (first element)
            op = node_expr[0] if isinstance(node_expr[0], str) else "list"
            tid = (hash(op) % (vocab_size - 1)) + 1
            new_idx = len(nodes)
            nodes.append(_ASTNode(token_id=tid, parent=parent_idx))
            if parent_idx >= 0:
                nodes[parent_idx].children.append(new_idx)

            # Push children in reverse order so they are processed left-to-right
            for child in reversed(node_expr[1:]):
                work.append((child, new_idx))

        else:
            # Empty list or None — create a placeholder
            new_idx = len(nodes)
            nodes.append(_ASTNode(token_id=0, parent=parent_idx))
            if parent_idx >= 0:
                nodes[parent_idx].children.append(new_idx)

    return nodes


class _ChildSumTreeLSTMCell(nn.Module):
    """Child-Sum Tree-LSTM cell."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.W_iou = nn.Linear(input_dim, 3 * hidden_dim, bias=True)
        self.U_iou = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.W_f = nn.Linear(input_dim, hidden_dim, bias=True)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # (input_dim,)
        children_h: torch.Tensor,  # (n_children, hidden_dim)
        children_c: torch.Tensor,  # (n_children, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (h, c) for a single node given its children states."""
        h_sum = children_h.sum(dim=0) if children_h.size(0) > 0 else torch.zeros_like(x[: children_c.size(-1)])

        iou = self.W_iou(x) + self.U_iou(h_sum)
        i, o, u = iou.chunk(3)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        if children_h.size(0) > 0:
            f_gates = torch.sigmoid(
                self.W_f(x).unsqueeze(0) + self.U_f(children_h)
            )  # (n_children, hidden_dim)
            fc = (f_gates * children_c).sum(dim=0)
        else:
            fc = torch.zeros_like(u)

        c = i * u + fc
        h = o * torch.tanh(c)
        return h, c


class TreeLSTMFeaturiser:
    """Tree-LSTM Embedding featuriser (Card 5).

    Implements the Featuriser protocol with input_type = "VECTOR".
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        embed_dim: int = 128,
        vocab_size: int = 4096,
    ) -> None:
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).to(self._device)
        self._cell = _ChildSumTreeLSTMCell(embed_dim, embed_dim).to(self._device)
        self._trained = False

    def _bottom_up_pass(self, nodes: list[_ASTNode]) -> torch.Tensor:
        """Iterative bottom-up Tree-LSTM traversal. Returns root hidden state.

        Pre-computes all token embeddings in a single batch, then processes
        nodes bottom-up. Leaf nodes are batched together for efficiency.
        """
        if not nodes:
            return torch.zeros(self.embed_dim, device=self._device)

        n = len(nodes)
        dim = self.embed_dim

        # Batch embed all token IDs at once
        all_token_ids = torch.tensor(
            [nd.token_id for nd in nodes], dtype=torch.long, device=self._device
        )
        all_x = self._embedding(all_token_ids)  # (n, embed_dim)

        h_states = torch.zeros((n, dim), device=self._device)
        c_states = torch.zeros((n, dim), device=self._device)

        remaining = [len(nd.children) for nd in nodes]

        # Batch-process all leaves at once (no children -> simplified cell)
        leaf_indices = [i for i in range(n) if remaining[i] == 0]
        if leaf_indices:
            leaf_idx_t = torch.tensor(leaf_indices, dtype=torch.long, device=self._device)
            leaf_x = all_x[leaf_idx_t]  # (n_leaves, dim)
            # For leaves: h_sum = 0, fc = 0
            # iou = W_iou(x), i = sigmoid(iou_i), o = sigmoid(iou_o), u = tanh(iou_u)
            iou = self._cell.W_iou(leaf_x)  # (n_leaves, 3*dim)
            i_gate, o_gate, u_gate = iou.chunk(3, dim=-1)
            i_gate = torch.sigmoid(i_gate)
            o_gate = torch.sigmoid(o_gate)
            u_gate = torch.tanh(u_gate)
            c_leaf = i_gate * u_gate
            h_leaf = o_gate * torch.tanh(c_leaf)
            h_states[leaf_idx_t] = h_leaf
            c_states[leaf_idx_t] = c_leaf

        # Propagate readiness to parents
        ready: list[int] = []
        for i in leaf_indices:
            nd = nodes[i]
            if nd.parent >= 0:
                remaining[nd.parent] -= 1
                if remaining[nd.parent] == 0:
                    ready.append(nd.parent)

        # Pre-compute all U_f(h) for every node to avoid recomputation
        # Also pre-compute W_iou(x) and W_f(x) for all nodes at once
        W_iou_all = self._cell.W_iou(all_x)  # (n, 3*dim)
        W_f_all = self._cell.W_f(all_x)      # (n, dim)

        # Build child->parent index for scatter-based h_sum
        child_indices: list[int] = []
        parent_of_child: list[int] = []
        for i, nd in enumerate(nodes):
            for ch in nd.children:
                child_indices.append(ch)
                parent_of_child.append(i)

        # Process internal nodes level by level
        while ready:
            batch_idx = ready
            idx_t = torch.tensor(batch_idx, dtype=torch.long, device=self._device)

            # Compute h_sum and fc for each node in batch
            h_sum_batch = torch.zeros((len(batch_idx), dim), device=self._device)
            fc_batch = torch.zeros((len(batch_idx), dim), device=self._device)

            wf_x_batch = W_f_all[idx_t]  # (B, dim)

            for bi, idx in enumerate(batch_idx):
                ch = nodes[idx].children
                if ch:
                    ch_t = torch.tensor(ch, dtype=torch.long, device=self._device)
                    ch_h = h_states[ch_t]
                    ch_c = c_states[ch_t]
                    h_sum_batch[bi] = ch_h.sum(dim=0)
                    uf_ch = self._cell.U_f(ch_h)
                    f_gates = torch.sigmoid(wf_x_batch[bi] + uf_ch)
                    fc_batch[bi] = (f_gates * ch_c).sum(dim=0)

            # Batch iou + gate computation
            iou = W_iou_all[idx_t] + self._cell.U_iou(h_sum_batch)
            i_g, o_g, u_g = iou.chunk(3, dim=-1)
            c_batch = torch.sigmoid(i_g) * torch.tanh(u_g) + fc_batch
            h_batch = torch.sigmoid(o_g) * torch.tanh(c_batch)
            h_states[idx_t] = h_batch
            c_states[idx_t] = c_batch

            # Collect next level
            next_ready: list[int] = []
            for idx in batch_idx:
                p = nodes[idx].parent
                if p >= 0:
                    remaining[p] -= 1
                    if remaining[p] == 0:
                        next_ready.append(p)
            ready = next_ready

        return h_states[0]

    def fit_batch(self, instance_paths: List[Union[str, Path]]) -> None:
        """Train Tree-LSTM on reconstruction objective."""
        paths = [Path(p) for p in instance_paths]
        if not paths:
            return

        params = list(self._embedding.parameters()) + list(self._cell.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        # Simple self-supervised objective: predict tree size from root embedding
        self._embedding.train()
        self._cell.train()

        # Build a linear head for tree-size prediction
        head = nn.Linear(self.embed_dim, 1).to(self._device)

        for _ in range(5):
            total_loss = torch.tensor(0.0, device=self._device)
            count = 0

            for p in paths:
                try:
                    info = parse_file(p)
                except Exception:
                    continue

                for assertion in info.assertions[:10]:  # Limit to first 10 assertions
                    nodes = _build_ast(assertion, self.vocab_size)
                    if not nodes:
                        continue
                    tree_size = float(len(nodes))

                    root_h = self._bottom_up_pass(nodes)
                    pred = head(root_h).squeeze()
                    target = torch.tensor(tree_size, device=self._device)
                    total_loss = total_loss + (pred - target) ** 2
                    count += 1

                    if count >= 50:  # Cap iterations
                        break
                if count >= 50:
                    break

            if count > 0:
                optimizer.zero_grad()
                (total_loss / count).backward()
                optimizer.step()

        self._trained = True

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract Tree-LSTM embedding by running bottom-up pass on AST."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        logic = None
        try:
            info = parse_file(path)
            logic = info.logic

            self._embedding.eval()
            self._cell.eval()

            # Build combined AST from all assertions (virtual root connecting them)
            # Cap total nodes to keep extraction tractable (each node
            # requires a Tree-LSTM cell forward call)
            _MAX_TOTAL_NODES = 200
            all_nodes: list[_ASTNode] = []

            if info.assertions:
                # Virtual root node
                root_idx = 0
                all_nodes.append(_ASTNode(token_id=(hash("and") % (self.vocab_size - 1)) + 1))

                for assertion in info.assertions:
                    budget = _MAX_TOTAL_NODES - len(all_nodes)
                    if budget <= 0:
                        break
                    offset = len(all_nodes)
                    ast_nodes = _build_ast(assertion, self.vocab_size, max_nodes=budget)
                    if not ast_nodes:
                        continue
                    # Reindex children and parent
                    for nd in ast_nodes:
                        nd.children = [c + offset for c in nd.children]
                        nd.parent = nd.parent + offset if nd.parent >= 0 else root_idx
                    # Attach subtree root to virtual root
                    all_nodes[root_idx].children.append(offset)
                    all_nodes.extend(ast_nodes)

            if not all_nodes:
                all_nodes.append(_ASTNode(token_id=0))

            with torch.no_grad():
                root_h = self._bottom_up_pass(all_nodes)
            features = root_h.cpu().numpy().flatten()

        except Exception:
            features = np.zeros(self.embed_dim, dtype=np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FeatureResult(
            features=features,
            feature_type="VECTOR",
            wall_time_ms=elapsed_ms,
            n_features=self.embed_dim,
            instance_id=str(path),
            logic=logic,
        )

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
