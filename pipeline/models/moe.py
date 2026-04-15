"""MoECard — Mixture of Experts model card."""
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


class MoECard(ModelCard):
    """Mixture of Experts with gating network for algorithm selection.

    A gating MLP routes instances to expert sub-models via softmax
    weights.  Each expert is an MLP that predicts cost vectors.
    The final prediction is the weighted sum of expert outputs.
    """

    canvas_id: ClassVar[str] = "965bae6f94b04c3f"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        n_experts: int = 4,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs

        self.scaler_: Optional[RobustScaler] = None
        self.config_names_: List[str] = []
        self.state_: Optional[Dict] = None
        self.in_dim_: int = 0
        self.out_dim_: int = 0

    def _build_modules(self, device: "torch.device"):
        """Build gating and expert networks."""
        gate = nn.Sequential(
            nn.Linear(self.in_dim_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_experts),
        ).to(device)

        experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_dim_, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.out_dim_),
                )
                for _ in range(self.n_experts)
            ]
        ).to(device)

        return gate, experts

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for MoECard but is not installed."
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

        gate, experts = self._build_modules(device)

        all_params = list(gate.parameters()) + list(experts.parameters())
        optimizer = torch.optim.Adam(all_params, lr=self.lr)
        loss_fn = nn.MSELoss()

        epoch_losses = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Gating weights: (n, n_experts)
            gate_logits = gate(X_t)
            gate_weights = torch.softmax(gate_logits, dim=1)

            # Expert predictions: (n_experts, n, n_configs)
            expert_preds = torch.stack([exp(X_t) for exp in experts], dim=0)

            # Weighted combination: (n, n_configs)
            # gate_weights: (n, n_experts) -> (n, n_experts, 1)
            weighted = (
                gate_weights.unsqueeze(2) * expert_preds.permute(1, 0, 2)
            ).sum(dim=1)

            loss = loss_fn(weighted, y_t)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        self.state_ = {
            "gate": {k: v.cpu() for k, v in gate.state_dict().items()},
            "experts": {k: v.cpu() for k, v in experts.state_dict().items()},
        }

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "epochs": self.epochs,
            "n_experts": self.n_experts,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for MoECard but is not installed."
            )

        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        device = torch.device("cpu")

        gate, experts = self._build_modules(device)
        gate.load_state_dict(self.state_["gate"])
        experts.load_state_dict(self.state_["experts"])
        gate.eval()
        experts.eval()

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            gate_weights = torch.softmax(gate(X_t), dim=1)
            expert_preds = torch.stack([exp(X_t) for exp in experts], dim=0)
            scores = (
                gate_weights.unsqueeze(2) * expert_preds.permute(1, 0, 2)
            ).sum(dim=1)

        return Predictions(
            values=scores.numpy(),
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
