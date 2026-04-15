"""DeepKernelCard — MLP encoder + GP head model card."""
from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class DeepKernelCard(ModelCard):
    """MLP feature encoder followed by per-config GP regression."""

    canvas_id: ClassVar[str] = "8b4376742d234834"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_inducing: int = 50,
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_inducing = n_inducing
        self.lr = lr
        self.epochs = epochs
        self.scaler_: Optional[RobustScaler] = None
        self.encoders_: Dict[int, object] = {}  # torch modules stored as state dicts
        self.gps_: Dict[int, GaussianProcessRegressor] = {}
        self.input_dim_: int = 0
        self.config_names_: List[str] = []

    def _build_encoder(self, input_dim: int):
        """Build an MLP encoder using torch."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "DeepKernelCard requires PyTorch. Install with: pip install torch"
            )

        layers = []
        in_dim = input_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim
        return nn.Sequential(*layers)

    def _encode(self, encoder, X_tensor):
        """Run encoder in eval mode."""
        import torch

        encoder.eval()
        with torch.no_grad():
            return encoder(X_tensor).numpy()

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "DeepKernelCard requires PyTorch. Install with: pip install torch"
            )

        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)
        n_instances, n_configs = cost_matrix.shape
        self.input_dim_ = X.shape[1]

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.encoders_ = {}
        self.gps_ = {}

        for c in range(n_configs):
            y = cost_matrix[:, c]
            y_tensor = torch.tensor(y, dtype=torch.float32)

            # Build and train MLP encoder via MSE loss
            encoder = self._build_encoder(self.input_dim_)
            head = nn.Linear(self.hidden_dim, 1)

            params = list(encoder.parameters()) + list(head.parameters())
            optimizer = optim.Adam(params, lr=self.lr)
            loss_fn = nn.MSELoss()

            encoder.train()
            head.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                encoded = encoder(X_tensor)
                pred = head(encoded).squeeze(-1)
                loss = loss_fn(pred, y_tensor)
                loss.backward()
                optimizer.step()

            # Encode features for GP fitting
            encoded_np = self._encode(encoder, X_tensor)

            # Subsample for GP if needed
            if n_instances > self.n_inducing:
                rng = np.random.RandomState(42)
                idx = rng.choice(n_instances, self.n_inducing, replace=False)
                gp_X = encoded_np[idx]
                gp_y = y[idx]
            else:
                gp_X = encoded_np
                gp_y = y

            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                alpha=1e-2,
                normalize_y=True,
                random_state=42,
            )
            gp.fit(gp_X, gp_y)

            # Store encoder state dict and GP
            self.encoders_[c] = encoder.state_dict()
            self.gps_[c] = gp

        return {"n_configs": n_configs, "hidden_dim": self.hidden_dim, "epochs": self.epochs}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        import torch

        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        n_instances = X_scaled.shape[0]
        n_configs = len(self.config_names_)
        values = np.zeros((n_instances, n_configs), dtype=np.float64)

        for c in range(n_configs):
            encoder = self._build_encoder(self.input_dim_)
            encoder.load_state_dict(self.encoders_[c])
            encoded_np = self._encode(encoder, X_tensor)
            values[:, c] = self.gps_[c].predict(encoded_np)

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
