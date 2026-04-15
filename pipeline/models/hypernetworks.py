"""HypernetworkCard — Hypernetwork-based model card."""
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


class HypernetworkCard(ModelCard):
    """Hypernetwork generates config-specific prediction heads.

    A shared feature encoder maps instance features to a hidden
    representation.  A hypernetwork takes a learned config embedding
    and produces the weights for a small prediction head that maps
    the hidden representation to a scalar cost prediction for that
    config.
    """

    canvas_id: ClassVar[str] = "1100fcb76ebe4c8b"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        hidden_dim: int = 64,
        config_embed_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.config_embed_dim = config_embed_dim
        self.lr = lr
        self.epochs = epochs

        self.scaler_: Optional[RobustScaler] = None
        self.config_names_: List[str] = []
        self.state_: Optional[Dict] = None
        self.in_dim_: int = 0
        self.n_configs_: int = 0

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for HypernetworkCard but is not installed."
            )

        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.in_dim_ = X_scaled.shape[1]
        self.n_configs_ = len(config_names)

        device = torch.device("cpu")
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        y_t = torch.tensor(cost_matrix, dtype=torch.float32, device=device)

        # Shared feature encoder
        encoder = nn.Sequential(
            nn.Linear(self.in_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(device)

        # Config embeddings (one per config)
        config_embeds = nn.Embedding(self.n_configs_, self.config_embed_dim).to(device)

        # Hypernetwork: config_embed -> weights for prediction head
        # Prediction head: hidden_dim -> 1 (linear layer)
        # So hypernetwork outputs hidden_dim + 1 parameters (weight + bias)
        head_param_dim = self.hidden_dim + 1
        hypernet = nn.Sequential(
            nn.Linear(self.config_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, head_param_dim),
        ).to(device)

        all_params = (
            list(encoder.parameters())
            + list(config_embeds.parameters())
            + list(hypernet.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=self.lr)
        loss_fn = nn.MSELoss()

        config_idx = torch.arange(self.n_configs_, device=device)
        epoch_losses = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            h = encoder(X_t)  # (n, hidden_dim)

            # For each config, generate head and predict
            preds = []
            embeds = config_embeds(config_idx)  # (n_configs, embed_dim)
            head_params = hypernet(embeds)  # (n_configs, hidden_dim + 1)

            for c in range(self.n_configs_):
                w = head_params[c, : self.hidden_dim]  # (hidden_dim,)
                b = head_params[c, self.hidden_dim]  # scalar
                pred_c = h @ w + b  # (n,)
                preds.append(pred_c)

            preds = torch.stack(preds, dim=1)  # (n, n_configs)
            loss = loss_fn(preds, y_t)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        self.state_ = {
            "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
            "config_embeds": {k: v.cpu() for k, v in config_embeds.state_dict().items()},
            "hypernet": {k: v.cpu() for k, v in hypernet.state_dict().items()},
        }

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "epochs": self.epochs,
            "n_configs": self.n_configs_,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for HypernetworkCard but is not installed."
            )

        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        device = torch.device("cpu")

        encoder = nn.Sequential(
            nn.Linear(self.in_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(device)
        config_embeds = nn.Embedding(self.n_configs_, self.config_embed_dim).to(device)
        head_param_dim = self.hidden_dim + 1
        hypernet = nn.Sequential(
            nn.Linear(self.config_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, head_param_dim),
        ).to(device)

        encoder.load_state_dict(self.state_["encoder"])
        config_embeds.load_state_dict(self.state_["config_embeds"])
        hypernet.load_state_dict(self.state_["hypernet"])

        encoder.eval()
        hypernet.eval()

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        config_idx = torch.arange(self.n_configs_, device=device)

        with torch.no_grad():
            h = encoder(X_t)
            embeds = config_embeds(config_idx)
            head_params = hypernet(embeds)

            preds = []
            for c in range(self.n_configs_):
                w = head_params[c, : self.hidden_dim]
                b = head_params[c, self.hidden_dim]
                pred_c = h @ w + b
                preds.append(pred_c)
            scores = torch.stack(preds, dim=1)

        return Predictions(
            values=scores.numpy(),
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
