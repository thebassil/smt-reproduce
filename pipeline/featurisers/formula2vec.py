"""Card 7: Formula2Vec featuriser (word2vec-style).

Sliding window over SMT-LIB token stream. Skip-gram objective on tokens.
Instance embedding = mean of token embeddings.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import tokenize


class _SkipGramModel(nn.Module):
    """Simple skip-gram model for token embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.center_embed.weight)
        nn.init.xavier_uniform_(self.context_embed.weight)

    def forward(
        self, center_ids: torch.Tensor, context_ids: torch.Tensor, neg_ids: torch.Tensor
    ) -> torch.Tensor:
        """Skip-gram with negative sampling loss.

        Args:
            center_ids: (batch,)
            context_ids: (batch,)
            neg_ids: (batch, n_neg)
        """
        center = self.center_embed(center_ids)  # (batch, dim)
        context = self.context_embed(context_ids)  # (batch, dim)
        neg = self.context_embed(neg_ids)  # (batch, n_neg, dim)

        # Positive: dot product
        pos_score = (center * context).sum(dim=-1)  # (batch,)
        pos_loss = F.logsigmoid(pos_score)

        # Negative: dot product
        neg_score = torch.bmm(neg, center.unsqueeze(-1)).squeeze(-1)  # (batch, n_neg)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)  # (batch,)

        return -(pos_loss + neg_loss).mean()


class Formula2VecFeaturiser:
    """Formula2Vec featuriser (Card 7).

    Implements the Featuriser protocol with input_type = "VECTOR".
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        embed_dim: int = 100,
        window_size: int = 5,
        min_count: int = 2,
    ) -> None:
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.min_count = min_count
        self._vocab_size = 2048
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _SkipGramModel(self._vocab_size, embed_dim).to(self._device)
        self._trained = False

    def _hash_token(self, tok: str) -> int:
        """Hash a token to a vocabulary index."""
        return (hash(tok) % (self._vocab_size - 1)) + 1  # 0 reserved

    def fit_batch(self, instance_paths: List[Union[str, Path]]) -> None:
        """Train skip-gram model on token streams from instances."""
        paths = [Path(p) for p in instance_paths]
        if not paths:
            return

        # Collect all token streams and count frequencies
        all_streams: list[list[int]] = []
        freq: Counter = Counter()

        for p in paths:
            try:
                text = p.read_text(errors="replace")
                tokens = tokenize(text)
            except Exception:
                continue

            for tok in tokens:
                freq[tok] += 1

            # Filter by min_count and hash
            stream = []
            for tok in tokens:
                if freq[tok] >= self.min_count:
                    stream.append(self._hash_token(tok))
            if stream:
                all_streams.append(stream)

        if not all_streams:
            return

        # Generate skip-gram pairs
        center_ids: list[int] = []
        context_ids: list[int] = []
        n_neg = 5
        max_pairs = 5000  # Cap training data

        for stream in all_streams:
            for i, cid in enumerate(stream):
                start = max(0, i - self.window_size)
                end = min(len(stream), i + self.window_size + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    center_ids.append(cid)
                    context_ids.append(stream[j])
                    if len(center_ids) >= max_pairs:
                        break
                if len(center_ids) >= max_pairs:
                    break
            if len(center_ids) >= max_pairs:
                break

        if not center_ids:
            return

        # Train
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        center_t = torch.tensor(center_ids, dtype=torch.long, device=self._device)
        context_t = torch.tensor(context_ids, dtype=torch.long, device=self._device)

        batch_size = 1024
        n_samples = len(center_ids)

        for epoch in range(3):
            # Shuffle
            perm = torch.randperm(n_samples, device=self._device)
            center_t = center_t[perm]
            context_t = context_t[perm]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                c_batch = center_t[start:end]
                ctx_batch = context_t[start:end]
                neg_batch = torch.randint(
                    1, self._vocab_size, (end - start, n_neg),
                    device=self._device,
                )

                optimizer.zero_grad()
                loss = self._model(c_batch, ctx_batch, neg_batch)
                loss.backward()
                optimizer.step()

        self._trained = True

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Compute instance embedding as mean of token embeddings."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        logic = None
        try:
            text = path.read_text(errors="replace")
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("(set-logic"):
                    parts = line.replace("(", " ").replace(")", " ").split()
                    if len(parts) >= 2:
                        logic = parts[1]
                    break

            tokens = tokenize(text)
            if not tokens:
                features = np.zeros(self.embed_dim, dtype=np.float32)
            else:
                token_ids = [self._hash_token(tok) for tok in tokens]
                ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=self._device)

                self._model.eval()
                with torch.no_grad():
                    embeddings = self._model.center_embed(ids_tensor)  # (n_tokens, dim)
                    mean_emb = embeddings.mean(dim=0)
                features = mean_emb.cpu().numpy().flatten()

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
