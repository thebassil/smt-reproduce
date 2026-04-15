"""Card 8: LLM Embedding featuriser.

Uses a pretrained model (CodeBERT or similar) on raw SMT-LIB text.
Truncates to max_length tokens, mean-pools hidden states -> embedding.
Falls back to random projection of BoW if transformers library not available.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, List, Literal, Union

import numpy as np
import torch

from pipeline.types import FeatureResult
from pipeline.featurisers._smtlib_parser import tokenize

# Try to import transformers; fall back gracefully
_HAS_TRANSFORMERS = False
try:
    from transformers import AutoModel, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    pass


class _BoWFallbackEncoder:
    """Random projection of bag-of-words when transformers unavailable."""

    def __init__(self, embed_dim: int, bow_dim: int = 8192, seed: int = 42) -> None:
        self.embed_dim = embed_dim
        self.bow_dim = bow_dim
        rng = np.random.RandomState(seed)
        # Fixed random projection matrix (Gaussian)
        self._proj = rng.randn(bow_dim, embed_dim).astype(np.float32)
        self._proj /= np.sqrt(bow_dim)

    def encode(self, text: str) -> np.ndarray:
        """Encode text via BoW + random projection."""
        tokens = tokenize(text)
        bow = np.zeros(self.bow_dim, dtype=np.float32)
        for tok in tokens:
            idx = hash(tok) % self.bow_dim
            bow[idx] += 1.0
        # L2 normalise BoW
        norm = np.linalg.norm(bow)
        if norm > 0:
            bow /= norm
        embedding = bow @ self._proj
        # L2 normalise output
        norm_out = np.linalg.norm(embedding)
        if norm_out > 0:
            embedding /= norm_out
        return embedding


class LLMEmbedFeaturiser:
    """LLM Embedding featuriser (Card 8).

    Implements the Featuriser protocol with input_type = "VECTOR".
    """

    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._use_transformers = False
        self._model = None
        self._tokenizer = None
        self._fallback = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if _HAS_TRANSFORMERS:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name).to(self._device)
                self._model.eval()
                self._use_transformers = True
                self.embed_dim = self._model.config.hidden_size
            except Exception:
                self._use_transformers = False

        if not self._use_transformers:
            self.embed_dim = 768
            self._fallback = _BoWFallbackEncoder(self.embed_dim)

    def fit_batch(self, instance_paths: List[Union[str, Path]]) -> None:
        """No training needed for pretrained model (or fallback)."""
        pass

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult:
        """Extract embedding via pretrained model or fallback."""
        path = Path(instance_path)
        t0 = time.perf_counter()

        logic = None
        try:
            text = path.read_text(errors="replace")
            for line in text.split("\n"):
                line_s = line.strip()
                if line_s.startswith("(set-logic"):
                    parts = line_s.replace("(", " ").replace(")", " ").split()
                    if len(parts) >= 2:
                        logic = parts[1]
                    break

            if self._use_transformers:
                features = self._extract_transformers(text)
            else:
                features = self._fallback.encode(text)

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

    def _extract_transformers(self, text: str) -> np.ndarray:
        """Extract embedding using the pretrained transformer model."""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pool over all non-padding token positions
            last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = last_hidden.mean(dim=1)

        return pooled.cpu().numpy().flatten()

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Extract features for a batch of instances."""
        return [self.extract(p) for p in instance_paths]
