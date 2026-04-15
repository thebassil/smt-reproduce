"""NeuralProcessCard — Conditional Neural Process model card."""
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


class NeuralProcessCard(ModelCard):
    """Conditional Neural Process for algorithm selection.

    A context encoder aggregates a context set into a latent
    representation.  A decoder combines the latent with target
    features to predict cost distributions.  Training uses the
    ELBO objective (reconstruction + KL divergence).

    Output is a distribution (probabilities summing to 1 per
    instance) where higher probability indicates the config is
    more likely to be the best choice.
    """

    canvas_id: ClassVar[str] = "66647216a1ff468e"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["distribution"]] = "distribution"

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs

        self.scaler_: Optional[RobustScaler] = None
        self.config_names_: List[str] = []
        self.state_: Optional[Dict] = None
        self.in_dim_: int = 0
        self.out_dim_: int = 0
        # Store training data as context set for prediction
        self.context_X_: Optional[np.ndarray] = None
        self.context_y_: Optional[np.ndarray] = None

    def _build_modules(self, device: "torch.device"):
        """Build encoder, latent projections, and decoder."""
        # Context encoder: maps (x, y) pairs to hidden representations
        context_encoder = nn.Sequential(
            nn.Linear(self.in_dim_ + self.out_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(device)

        # Latent projections (for variational inference)
        mu_proj = nn.Linear(self.hidden_dim, self.latent_dim).to(device)
        logvar_proj = nn.Linear(self.hidden_dim, self.latent_dim).to(device)

        # Decoder: maps (target_x, latent) to output distribution
        decoder = nn.Sequential(
            nn.Linear(self.in_dim_ + self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim_),
        ).to(device)

        return context_encoder, mu_proj, logvar_proj, decoder

    def _encode_context(self, X_ctx, y_ctx, context_encoder, mu_proj, logvar_proj):
        """Encode context set into latent distribution parameters."""
        # Concatenate features and targets
        ctx_input = torch.cat([X_ctx, y_ctx], dim=-1)  # (n_ctx, in_dim + out_dim)
        h = context_encoder(ctx_input)  # (n_ctx, hidden_dim)

        # Aggregate via mean pooling
        h_agg = h.mean(dim=0, keepdim=True)  # (1, hidden_dim)

        mu = mu_proj(h_agg)  # (1, latent_dim)
        logvar = logvar_proj(h_agg)  # (1, latent_dim)

        return mu, logvar

    @staticmethod
    def _reparameterize(mu, logvar):
        """Sample from N(mu, sigma) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_divergence(mu, logvar):
        """KL(q(z|context) || p(z)) where p(z) = N(0, I)."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for NeuralProcessCard but is not installed."
            )

        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.in_dim_ = X_scaled.shape[1]
        self.out_dim_ = len(config_names)

        # Store context for prediction
        self.context_X_ = X_scaled.copy()
        self.context_y_ = cost_matrix.copy()

        device = torch.device("cpu")
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        y_t = torch.tensor(cost_matrix, dtype=torch.float32, device=device)

        context_encoder, mu_proj, logvar_proj, decoder = self._build_modules(device)

        all_params = (
            list(context_encoder.parameters())
            + list(mu_proj.parameters())
            + list(logvar_proj.parameters())
            + list(decoder.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=self.lr)
        recon_loss_fn = nn.MSELoss()

        epoch_losses = []
        n = X_t.shape[0]

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Use all training data as context set
            mu, logvar = self._encode_context(
                X_t, y_t, context_encoder, mu_proj, logvar_proj
            )
            z = self._reparameterize(mu, logvar)  # (1, latent_dim)

            # Expand latent to match all target instances
            z_expanded = z.expand(n, -1)  # (n, latent_dim)

            # Decode: predict cost distribution for each instance
            decoder_input = torch.cat([X_t, z_expanded], dim=-1)
            raw_preds = decoder(decoder_input)  # (n, n_configs)

            # Reconstruction loss (on softmin-transformed targets for distribution)
            recon_loss = recon_loss_fn(raw_preds, y_t)
            kl_loss = self._kl_divergence(mu, logvar)

            # ELBO = reconstruction + beta * KL
            beta = min(1.0, epoch / max(self.epochs * 0.3, 1))  # KL annealing
            loss = recon_loss + beta * kl_loss / n
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        self.state_ = {
            "context_encoder": {
                k: v.cpu() for k, v in context_encoder.state_dict().items()
            },
            "mu_proj": {k: v.cpu() for k, v in mu_proj.state_dict().items()},
            "logvar_proj": {k: v.cpu() for k, v in logvar_proj.state_dict().items()},
            "decoder": {k: v.cpu() for k, v in decoder.state_dict().items()},
        }

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "epochs": self.epochs,
            "latent_dim": self.latent_dim,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for NeuralProcessCard but is not installed."
            )

        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        n = X_scaled.shape[0]
        device = torch.device("cpu")

        context_encoder, mu_proj, logvar_proj, decoder = self._build_modules(device)
        context_encoder.load_state_dict(self.state_["context_encoder"])
        mu_proj.load_state_dict(self.state_["mu_proj"])
        logvar_proj.load_state_dict(self.state_["logvar_proj"])
        decoder.load_state_dict(self.state_["decoder"])

        context_encoder.eval()
        mu_proj.eval()
        logvar_proj.eval()
        decoder.eval()

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        ctx_X = torch.tensor(self.context_X_, dtype=torch.float32, device=device)
        ctx_y = torch.tensor(self.context_y_, dtype=torch.float32, device=device)

        with torch.no_grad():
            # Encode context to get latent
            mu, logvar = self._encode_context(
                ctx_X, ctx_y, context_encoder, mu_proj, logvar_proj
            )
            # Use mean (no sampling) for deterministic prediction
            z = mu.expand(n, -1)  # (n, latent_dim)

            decoder_input = torch.cat([X_t, z], dim=-1)
            raw_preds = decoder(decoder_input)  # (n, n_configs)

            # Convert to distribution via softmin (lower cost -> higher prob)
            neg = -raw_preds
            neg = neg - neg.max(dim=1, keepdim=True).values  # numerical stability
            probs = torch.exp(neg) / torch.exp(neg).sum(dim=1, keepdim=True)

        return Predictions(
            values=probs.numpy(),
            output_type="distribution",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
