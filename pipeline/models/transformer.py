"""TransformerCard — Self-attention model card."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

import numpy as np
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TransformerCard(ModelCard):
    """Self-attention over feature dimensions for algorithm selection.

    Each feature dimension is treated as a token (scalar projected to
    d_model).  A TransformerEncoder captures inter-feature interactions.
    A CLS token aggregates the sequence, followed by an MLP head that
    predicts the cost vector.
    """

    canvas_id: ClassVar[str] = "e10febbc28924230"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.scaler_: Optional[RobustScaler] = None
        self.config_names_: List[str] = []
        self.state_: Optional[Dict] = None
        self.in_dim_: int = 0
        self.out_dim_: int = 0

    def _build_model(self, device: "torch.device"):
        """Build the Transformer + MLP head."""
        # Token embedding: project each scalar feature to d_model
        token_embed = nn.Linear(1, self.d_model).to(device)

        # CLS token (learnable)
        cls_token = nn.Parameter(torch.randn(1, 1, self.d_model, device=device) * 0.02)

        # Positional embedding for (n_features + 1) tokens
        pos_embed = nn.Parameter(
            torch.randn(1, self.in_dim_ + 1, self.d_model, device=device) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        ).to(device)

        head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.out_dim_),
        ).to(device)

        return token_embed, cls_token, pos_embed, transformer, head

    def _forward(self, X_t, token_embed, cls_token, pos_embed, transformer, head):
        """Forward pass through the transformer model."""
        batch_size = X_t.shape[0]

        # X_t: (batch, n_features) -> (batch, n_features, 1)
        tokens = token_embed(X_t.unsqueeze(-1))  # (batch, n_features, d_model)

        # Prepend CLS token
        cls_expanded = cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_expanded, tokens], dim=1)  # (batch, n_features+1, d_model)

        # Add positional embeddings
        tokens = tokens + pos_embed

        # Transformer encoder
        encoded = transformer(tokens)  # (batch, n_features+1, d_model)

        # Take CLS token output
        cls_out = encoded[:, 0, :]  # (batch, d_model)

        return head(cls_out)  # (batch, n_configs)

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for TransformerCard but is not installed."
            )

        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.in_dim_ = X_scaled.shape[1]
        self.out_dim_ = len(config_names)

        device = torch.device("cpu")
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        y_t = torch.tensor(cost_matrix, dtype=torch.float32, device=device)

        token_embed, cls_token, pos_embed, transformer, head = self._build_model(device)

        all_params = (
            list(token_embed.parameters())
            + [cls_token, pos_embed]
            + list(transformer.parameters())
            + list(head.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=self.lr)
        loss_fn = nn.MSELoss()

        n = X_t.shape[0]
        epoch_losses = []

        for epoch in range(self.epochs):
            # Shuffle for mini-batching
            perm = torch.randperm(n)
            batch_losses = []

            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                X_batch = X_t[idx]
                y_batch = y_t[idx]

                optimizer.zero_grad()
                preds = self._forward(
                    X_batch, token_embed, cls_token, pos_embed, transformer, head
                )
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        self.state_ = {
            "token_embed": {k: v.cpu() for k, v in token_embed.state_dict().items()},
            "cls_token": cls_token.detach().cpu(),
            "pos_embed": pos_embed.detach().cpu(),
            "transformer": {k: v.cpu() for k, v in transformer.state_dict().items()},
            "head": {k: v.cpu() for k, v in head.state_dict().items()},
        }

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "epochs": self.epochs,
            "n_tokens": self.in_dim_ + 1,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for TransformerCard but is not installed."
            )

        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        device = torch.device("cpu")

        token_embed, cls_token, pos_embed, transformer, head = self._build_model(device)

        token_embed.load_state_dict(self.state_["token_embed"])
        cls_token.data = self.state_["cls_token"].to(device)
        pos_embed.data = self.state_["pos_embed"].to(device)
        transformer.load_state_dict(self.state_["transformer"])
        head.load_state_dict(self.state_["head"])

        token_embed.eval()
        transformer.eval()
        head.eval()

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            scores = self._forward(
                X_t, token_embed, cls_token, pos_embed, transformer, head
            )

        return Predictions(
            values=scores.numpy(),
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
