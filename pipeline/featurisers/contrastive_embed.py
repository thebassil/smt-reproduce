"""Card 4: Contrastive Embedding featuriser.

Augmentations: random token dropout, token shuffle, subsequence crop.
Encoder: simple MLP on bag-of-words representation.
InfoNCE contrastive loss during fit_batch.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import tokenize


def _bow_vector(tokens: list[str], vocab_size: int) -> np.ndarray:
    """Build a bag-of-words vector by hashing tokens."""
    vec = np.zeros(vocab_size, dtype=np.float32)
    for tok in tokens:
        idx = hash(tok) % vocab_size
        vec[idx] += 1.0
    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _augment_dropout(tokens: list[str], drop_rate: float = 0.15) -> list[str]:
    """Randomly drop tokens."""
    return [t for t in tokens if random.random() > drop_rate]


def _augment_shuffle(tokens: list[str], window: int = 5) -> list[str]:
    """Locally shuffle tokens within a sliding window."""
    result = list(tokens)
    for i in range(0, len(result) - 1, window):
        end = min(i + window, len(result))
        segment = result[i:end]
        random.shuffle(segment)
        result[i:end] = segment
    return result


def _augment_crop(tokens: list[str], crop_ratio: float = 0.7) -> list[str]:
    """Take a random contiguous subsequence."""
    if len(tokens) < 2:
        return tokens
    crop_len = max(1, int(len(tokens) * crop_ratio))
    start = random.randint(0, len(tokens) - crop_len)
    return tokens[start : start + crop_len]


class _ContrastiveEncoder(nn.Module):
    """Simple MLP encoder on BoW features."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return F.normalize(h, dim=-1)


class ContrastiveEmbedFeaturiser:
    """Contrastive Embedding featuriser (Card 4).

    Implements the Featuriser protocol with input_type = "VECTOR".
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        embed_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        self.embed_dim = embed_dim
        self.temperature = temperature
        self._bow_dim = 2048  # BoW vocabulary size for input
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = _ContrastiveEncoder(self._bow_dim, embed_dim).to(self._device)
        self._trained = False

    def _get_tokens(self, path: Path) -> list[str]:
        """Read and tokenize a file."""
        text = path.read_text(errors="replace")
        return tokenize(text)

    def fit_batch(self, instance_paths: List[Union[str, Path]]) -> None:
        """Train encoder with InfoNCE contrastive loss on augmented pairs."""
        paths = [Path(p) for p in instance_paths]
        if len(paths) < 2:
            return

        self._encoder.train()
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=1e-3)

        # Generate two augmented views for each instance
        views_a: list[np.ndarray] = []
        views_b: list[np.ndarray] = []

        for p in paths:
            try:
                tokens = self._get_tokens(p)
            except Exception:
                tokens = []

            # View A: dropout + shuffle
            aug_a = _augment_shuffle(_augment_dropout(tokens))
            # View B: crop + dropout
            aug_b = _augment_dropout(_augment_crop(tokens))

            views_a.append(_bow_vector(aug_a, self._bow_dim))
            views_b.append(_bow_vector(aug_b, self._bow_dim))

        a_tensor = torch.tensor(np.array(views_a), dtype=torch.float, device=self._device)
        b_tensor = torch.tensor(np.array(views_b), dtype=torch.float, device=self._device)

        # InfoNCE training
        for _ in range(10):
            optimizer.zero_grad()
            z_a = self._encoder(a_tensor)  # (N, embed_dim)
            z_b = self._encoder(b_tensor)  # (N, embed_dim)

            # Similarity matrix
            sim = torch.mm(z_a, z_b.t()) / self.temperature  # (N, N)
            labels = torch.arange(len(paths), device=self._device)
            loss = (
                F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
            ) / 2.0
            loss.backward()
            optimizer.step()

        self._trained = True

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Encode an instance as a fixed-dim embedding."""
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
            bow = _bow_vector(tokens, self._bow_dim)
            bow_tensor = torch.tensor(bow, dtype=torch.float, device=self._device).unsqueeze(0)

            self._encoder.eval()
            with torch.no_grad():
                embedding = self._encoder(bow_tensor)
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
