"""RLSchedulingCard — REINFORCE policy-gradient algorithm selection."""
from __future__ import annotations

from typing import ClassVar, List, Literal, Optional

import numpy as np
from sklearn.preprocessing import RobustScaler

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions


class RLSchedulingCard(ModelCard):
    """Simple MLP policy trained with REINFORCE.  Maps instance features to
    config scores for time allocation.  Uses PyTorch if available, otherwise
    falls back to a NumPy-only two-layer MLP."""

    canvas_id: ClassVar[str] = "823adef00d044cdd"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        gamma: float = 0.99,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.gamma = gamma

        self.scaler_: Optional[RobustScaler] = None
        self.config_names_: List[str] = []
        self._use_torch: bool = False
        self._policy = None  # torch.nn.Module or None
        # NumPy fallback weights
        self._W1: Optional[np.ndarray] = None
        self._b1: Optional[np.ndarray] = None
        self._W2: Optional[np.ndarray] = None
        self._b2: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _try_import_torch(self):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
            self._use_torch = True
        except ImportError:
            self._use_torch = False

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        n_instances, d = X_scaled.shape
        n_configs = len(config_names)

        self._try_import_torch()

        if self._use_torch:
            return self._fit_torch(X_scaled, cost_matrix, d, n_configs)
        else:
            return self._fit_numpy(X_scaled, cost_matrix, d, n_configs)

    # ------------------------------------------------------------------
    def _fit_torch(
        self,
        X: np.ndarray,
        cost_matrix: np.ndarray,
        d: int,
        n_configs: int,
    ) -> dict:
        import torch
        import torch.nn as nn

        class PolicyNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                return self.fc2(h)

        policy = PolicyNet(d, self.hidden_dim, n_configs)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32)
        costs_t = torch.tensor(cost_matrix, dtype=torch.float32)

        n = X_t.shape[0]
        for _ in range(self.epochs):
            logits = policy(X_t)  # (n, n_configs)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()  # (n,)

            # Reward = negative cost
            rewards = -costs_t[torch.arange(n), actions]
            # Normalise rewards for stability
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            log_probs = dist.log_prob(actions)
            loss = -(log_probs * rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._policy = policy
        return {"n_train": n, "backend": "torch", "epochs": self.epochs}

    # ------------------------------------------------------------------
    def _fit_numpy(
        self,
        X: np.ndarray,
        cost_matrix: np.ndarray,
        d: int,
        n_configs: int,
    ) -> dict:
        rng = np.random.RandomState(42)
        # Xavier init
        self._W1 = rng.randn(d, self.hidden_dim) * np.sqrt(2.0 / d)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, n_configs) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(n_configs)

        n = X.shape[0]
        for _ in range(self.epochs):
            # Forward
            h = X @ self._W1 + self._b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ self._W2 + self._b2

            # Softmax
            logits_shifted = logits - logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits_shifted)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)

            # Sample actions
            actions = np.array([rng.choice(n_configs, p=probs[i]) for i in range(n)])

            # Rewards
            rewards = -cost_matrix[np.arange(n), actions]
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # REINFORCE gradient (simplified batch update)
            # d log pi / d logits  = (one_hot - probs)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(n), actions] = 1.0
            d_logits = (one_hot - probs) * rewards[:, None]  # (n, n_configs)

            # Backprop through softmax -> linear -> relu -> linear
            grad_W2 = h_relu.T @ d_logits / n
            grad_b2 = d_logits.mean(axis=0)
            d_h_relu = d_logits @ self._W2.T
            d_h = d_h_relu * (h > 0).astype(float)
            grad_W1 = X.T @ d_h / n
            grad_b1 = d_h.mean(axis=0)

            # Gradient ascent (maximise reward)
            self._W2 += self.lr * grad_W2
            self._b2 += self.lr * grad_b2
            self._W1 += self.lr * grad_W1
            self._b1 += self.lr * grad_b1

        return {"n_train": n, "backend": "numpy", "epochs": self.epochs}

    # ------------------------------------------------------------------
    def _forward_numpy(self, X: np.ndarray) -> np.ndarray:
        h = X @ self._W1 + self._b1
        h_relu = np.maximum(h, 0)
        return h_relu @ self._W2 + self._b2

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)

        if self._use_torch and self._policy is not None:
            import torch
            self._policy.eval()
            with torch.no_grad():
                logits = self._policy(torch.tensor(X_scaled, dtype=torch.float32))
                logits_np = logits.numpy()
        else:
            logits_np = self._forward_numpy(X_scaled)

        # Negate: policy logits are "preference" (higher = better config).
        # Scores contract: lower = better.
        values = -logits_np

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )
