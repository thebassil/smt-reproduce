"""Card 6: AST Path Embedding featuriser (code2vec-style).

Extract leaf-to-leaf paths through the AST (up to max_path_length).
Each path = sequence of operators. Attention-weighted aggregation of
path embeddings. Uses iterative traversal throughout.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import parse_file


def _build_flat_ast(expr, vocab_size: int) -> tuple[list[int], list[list[int]], list[int]]:
    """Build flat AST arrays from a parsed S-expression (iterative).

    Returns:
        token_ids: token hash for each node
        children: list of child indices for each node
        parents: parent index for each node (-1 for root)
    """
    token_ids: list[int] = []
    children_list: list[list[int]] = []
    parents: list[int] = []

    # Stack: (expression, parent_index)
    work: list[tuple] = [(expr, -1)]

    while work:
        node_expr, parent_idx = work.pop()

        if isinstance(node_expr, str):
            tid = (hash(node_expr) % (vocab_size - 1)) + 1
            new_idx = len(token_ids)
            token_ids.append(tid)
            children_list.append([])
            parents.append(parent_idx)
            if parent_idx >= 0:
                children_list[parent_idx].append(new_idx)

        elif isinstance(node_expr, list) and len(node_expr) > 0:
            op = node_expr[0] if isinstance(node_expr[0], str) else "list"
            tid = (hash(op) % (vocab_size - 1)) + 1
            new_idx = len(token_ids)
            token_ids.append(tid)
            children_list.append([])
            parents.append(parent_idx)
            if parent_idx >= 0:
                children_list[parent_idx].append(new_idx)

            for child in reversed(node_expr[1:]):
                work.append((child, new_idx))
        else:
            new_idx = len(token_ids)
            token_ids.append(0)
            children_list.append([])
            parents.append(parent_idx)
            if parent_idx >= 0:
                children_list[parent_idx].append(new_idx)

    return token_ids, children_list, parents


def _find_leaves(children_list: list[list[int]]) -> list[int]:
    """Find all leaf node indices."""
    return [i for i, ch in enumerate(children_list) if not ch]


def _path_to_ancestor(
    node: int, parents: list[int], max_depth: int
) -> list[int]:
    """Trace path from node up to root (or max_depth), returning node indices."""
    path = [node]
    current = node
    for _ in range(max_depth):
        p = parents[current]
        if p < 0:
            break
        path.append(p)
        current = p
    return path


def _extract_leaf_paths(
    token_ids: list[int],
    children_list: list[list[int]],
    parents: list[int],
    max_path_length: int,
    max_paths: int,
) -> list[list[int]]:
    """Extract leaf-to-leaf paths via their LCA (iterative).

    Each path is a sequence of token_ids along the path.
    """
    leaves = _find_leaves(children_list)
    if len(leaves) < 2:
        return []

    # Precompute ancestor paths for each leaf
    half_len = max_path_length // 2
    leaf_ancestors: dict[int, list[int]] = {}
    for leaf in leaves:
        leaf_ancestors[leaf] = _path_to_ancestor(leaf, parents, half_len)

    paths: list[list[int]] = []
    count = 0

    # Generate paths between pairs of leaves
    n_leaves = len(leaves)
    # Sample pairs if too many
    if n_leaves * (n_leaves - 1) // 2 > max_paths * 2:
        import random
        pairs = []
        for _ in range(max_paths * 2):
            i = random.randint(0, n_leaves - 1)
            j = random.randint(0, n_leaves - 1)
            if i != j:
                pairs.append((leaves[i], leaves[j]))
    else:
        pairs = []
        for i in range(n_leaves):
            for j in range(i + 1, n_leaves):
                pairs.append((leaves[i], leaves[j]))

    for leaf_a, leaf_b in pairs:
        if count >= max_paths:
            break

        path_a = leaf_ancestors[leaf_a]
        path_b = leaf_ancestors[leaf_b]

        # Find LCA (lowest common ancestor)
        set_b = set(path_b)
        lca_idx_in_a = -1
        for k, node in enumerate(path_a):
            if node in set_b:
                lca_idx_in_a = k
                break

        if lca_idx_in_a < 0:
            # No common ancestor found within depth limit; skip
            continue

        lca_node = path_a[lca_idx_in_a]
        lca_idx_in_b = path_b.index(lca_node)

        # Build path: leaf_a -> ... -> LCA -> ... -> leaf_b
        up_path = path_a[: lca_idx_in_a + 1]
        down_path = path_b[:lca_idx_in_b][::-1]  # reversed, excluding LCA

        full_path_nodes = up_path + down_path
        if len(full_path_nodes) > max_path_length:
            continue

        path_tokens = [token_ids[n] for n in full_path_nodes]
        paths.append(path_tokens)
        count += 1

    return paths


class _ASTPathEncoder(nn.Module):
    """Attention-weighted aggregation of path embeddings."""

    def __init__(self, embed_dim: int, vocab_size: int, max_path_length: int) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.path_rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.attention = nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, paths_tensor: torch.Tensor, path_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            paths_tensor: (n_paths, max_path_len) token IDs
            path_mask: (n_paths,) boolean, True for valid paths
        Returns:
            embedding: (embed_dim,)
        """
        if paths_tensor.size(0) == 0 or not path_mask.any():
            return torch.zeros(self.embed_dim, device=paths_tensor.device)

        x = self.token_embed(paths_tensor)  # (n_paths, max_len, embed_dim)
        _, h = self.path_rnn(x)  # h: (1, n_paths, embed_dim)
        h = h.squeeze(0)  # (n_paths, embed_dim)

        # Attention weights
        attn_scores = self.attention(h).squeeze(-1)  # (n_paths,)
        # Mask invalid paths
        attn_scores[~path_mask] = float("-inf")
        attn_weights = F.softmax(attn_scores, dim=0)  # (n_paths,)

        # Weighted sum
        embedding = (attn_weights.unsqueeze(-1) * h).sum(dim=0)
        return embedding


class ASTPathFeaturiser:
    """AST Path Embedding featuriser (Card 6, code2vec-style).

    Implements the Featuriser protocol with input_type = "VECTOR".
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        embed_dim: int = 128,
        max_path_length: int = 10,
        max_paths: int = 200,
    ) -> None:
        self.embed_dim = embed_dim
        self.max_path_length = max_path_length
        self.max_paths = max_paths
        self._vocab_size = 4096
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = _ASTPathEncoder(
            embed_dim, self._vocab_size, max_path_length
        ).to(self._device)
        self._trained = False

    def _extract_paths(self, info) -> list[list[int]]:
        """Extract AST paths from all assertions."""
        all_paths: list[list[int]] = []
        for assertion in info.assertions:
            token_ids, children_list, parents = _build_flat_ast(
                assertion, self._vocab_size
            )
            if len(token_ids) < 2:
                continue
            paths = _extract_leaf_paths(
                token_ids,
                children_list,
                parents,
                self.max_path_length,
                max(1, self.max_paths - len(all_paths)),
            )
            all_paths.extend(paths)
            if len(all_paths) >= self.max_paths:
                break
        return all_paths[: self.max_paths]

    def _paths_to_tensor(
        self, paths: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad paths into a batch tensor."""
        if not paths:
            t = torch.zeros((1, self.max_path_length), dtype=torch.long, device=self._device)
            m = torch.zeros(1, dtype=torch.bool, device=self._device)
            return t, m

        n = len(paths)
        max_len = min(max(len(p) for p in paths), self.max_path_length)
        tensor = torch.zeros((n, max_len), dtype=torch.long, device=self._device)
        mask = torch.ones(n, dtype=torch.bool, device=self._device)

        for i, p in enumerate(paths):
            length = min(len(p), max_len)
            tensor[i, :length] = torch.tensor(p[:length], dtype=torch.long)

        return tensor, mask

    def fit_batch(self, instance_paths: List[Union[str, Path]]) -> None:
        """Train encoder on self-supervised objective (path reconstruction)."""
        paths_list = [Path(p) for p in instance_paths]
        if not paths_list:
            return

        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=1e-3)
        self._encoder.train()

        # Self-supervised: predict number of paths from embedding
        head = nn.Linear(self.embed_dim, 1).to(self._device)

        for _ in range(10):
            total_loss = torch.tensor(0.0, device=self._device)
            count = 0

            for p in paths_list:
                try:
                    info = parse_file(p)
                except Exception:
                    continue

                ast_paths = self._extract_paths(info)
                if not ast_paths:
                    continue

                tensor, mask = self._paths_to_tensor(ast_paths)
                emb = self._encoder(tensor, mask)
                pred = head(emb).squeeze()
                target = torch.tensor(float(len(ast_paths)), device=self._device)
                total_loss = total_loss + (pred - target) ** 2
                count += 1

            if count > 0:
                optimizer.zero_grad()
                (total_loss / count).backward()
                optimizer.step()

        self._trained = True

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract code2vec-style embedding."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        logic = None
        try:
            info = parse_file(path)
            logic = info.logic

            ast_paths = self._extract_paths(info)
            tensor, mask = self._paths_to_tensor(ast_paths)

            self._encoder.eval()
            with torch.no_grad():
                embedding = self._encoder(tensor, mask)
            features = embedding.cpu().numpy().flatten()

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
