"""Card 3: Transformer Embedding featuriser.

Tokenizes SMT-LIB text into subword-like tokens (split on parens/space, hash
to vocab_size). Small transformer encoder (2 layers, 4 heads, dim=128).
CLS token -> embedding. Supports fit_batch() for training and extract() for
inference.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import tokenize


def _tokenize_file(path: Path, vocab_size: int, max_seq_len: int) -> list[int]:
    """Read and tokenize an SMT-LIB file into integer token IDs."""
    text = path.read_text(errors="replace")
    raw_tokens = tokenize(text)
    # Hash each token to a vocab ID (reserve 0 for PAD, 1 for CLS)
    ids = [1]  # CLS token
    for tok in raw_tokens[: max_seq_len - 1]:
        tid = (hash(tok) % (vocab_size - 2)) + 2
        ids.append(tid)
    return ids


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerEncoder(nn.Module):
    """Small transformer encoder for SMT-LIB token sequences."""

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = _PositionalEncoding(embed_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns CLS embeddings (batch_size, embed_dim)."""
        x = self.embedding(token_ids)
        x = self.pos_enc(x)
        # pad_mask: True where padded
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        return x[:, 0, :]  # CLS token embedding


class TransformerEmbedFeaturiser:
    """Transformer Embedding featuriser (Card 3).

    Implements the Featuriser protocol with input_type = "VECTOR".
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        vocab_size: int = 8192,
        max_seq_len: int = 2048,
    ) -> None:
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _TransformerEncoder(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        ).to(self._device)
        self._trained = False

    def _prepare_batch(
        self, paths: list[Path]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize files and pad into a batch tensor."""
        all_ids: list[list[int]] = []
        for p in paths:
            ids = _tokenize_file(p, self.vocab_size, self.max_seq_len)
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids) if all_ids else 1
        max_len = min(max_len, self.max_seq_len)

        batch = torch.zeros((len(all_ids), max_len), dtype=torch.long)
        mask = torch.ones((len(all_ids), max_len), dtype=torch.bool)

        for i, ids in enumerate(all_ids):
            length = min(len(ids), max_len)
            batch[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            mask[i, :length] = False

        return batch.to(self._device), mask.to(self._device)

    def fit_batch(self, instance_paths: List[Union[str, Path]]) -> None:
        """Train encoder on masked token reconstruction objective."""
        paths = [Path(p) for p in instance_paths]
        if not paths:
            return

        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)

        batch_ids, pad_mask = self._prepare_batch(paths)

        # Masked Language Model objective: mask 15% of non-special tokens
        mlm_target = batch_ids.clone()
        mask_prob = torch.rand_like(batch_ids, dtype=torch.float)
        # Don't mask PAD (0), CLS (1)
        special_mask = (batch_ids <= 1) | pad_mask
        mask_prob[special_mask] = 1.0  # won't be masked
        mlm_mask = mask_prob < 0.15
        batch_ids[mlm_mask] = 0  # replace masked tokens with PAD

        # Reconstruction head
        head = nn.Linear(self.embed_dim, self.vocab_size).to(self._device)

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            x = self._model.embedding(batch_ids)
            x = self._model.pos_enc(x)
            x = self._model.encoder(x, src_key_padding_mask=pad_mask)
            x = self._model.norm(x)
            logits = head(x)  # (batch, seq, vocab)

            if mlm_mask.any():
                loss = nn.functional.cross_entropy(
                    logits[mlm_mask], mlm_target[mlm_mask]
                )
                loss.backward()
                optimizer.step()

        self._trained = True

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Forward pass, return CLS embedding."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        try:
            text = path.read_text(errors="replace")
            logic = None
            # Quick logic extraction
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("(set-logic"):
                    parts = line.replace("(", " ").replace(")", " ").split()
                    if len(parts) >= 2:
                        logic = parts[1]
                    break

            ids = _tokenize_file(path, self.vocab_size, self.max_seq_len)
            length = min(len(ids), self.max_seq_len)
            token_tensor = torch.zeros((1, length), dtype=torch.long, device=self._device)
            token_tensor[0, :length] = torch.tensor(ids[:length], dtype=torch.long)
            pad_mask = torch.zeros((1, length), dtype=torch.bool, device=self._device)

            self._model.eval()
            with torch.no_grad():
                embedding = self._model(token_tensor, pad_mask)
            features = embedding.cpu().numpy().flatten()

        except Exception:
            features = np.zeros(self.embed_dim, dtype=np.float32)
            logic = None

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
