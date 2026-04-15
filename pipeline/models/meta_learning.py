"""MetaLearningCard — MAML-style meta-learning model card."""
from __future__ import annotations

from collections import defaultdict
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


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class MetaLearningCard(ModelCard):
    """MAML-style meta-learning for algorithm selection.

    A base MLP predicts cost vectors.  Inner loop adapts to per-logic
    tasks; outer loop optimises the initialisation so that a few
    gradient steps on a new logic yield good predictions.
    """

    canvas_id: ClassVar[str] = "13797f69fd4c430e"
    input_type: ClassVar[Literal["VECTOR"]] = "VECTOR"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        hidden_dim: int = 64,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        epochs: int = 50,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.epochs = epochs

        self.scaler_: Optional[RobustScaler] = None
        self.config_names_: List[str] = []
        self.model_state_: Optional[Dict] = None
        self.in_dim_: int = 0
        self.out_dim_: int = 0
        # Store training data for adaptation at predict time
        self.train_X_: Optional[np.ndarray] = None
        self.train_y_: Optional[np.ndarray] = None
        self.train_logics_: Optional[List[Optional[str]]] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for MetaLearningCard but is not installed."
            )

        self.config_names_ = list(config_names)
        X = self._stack_vectors(features)
        logics = self._logic_labels(features)

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.in_dim_ = X_scaled.shape[1]
        self.out_dim_ = len(config_names)

        # Store for adaptation at predict time
        self.train_X_ = X_scaled.copy()
        self.train_y_ = cost_matrix.copy()
        self.train_logics_ = logics

        # Group indices by logic for task construction
        logic_groups: Dict[str, List[int]] = defaultdict(list)
        for idx, lg in enumerate(logics):
            key = lg if lg is not None else "__none__"
            logic_groups[key].append(idx)

        # If only one logic group, split into synthetic tasks
        if len(logic_groups) < 2:
            all_idx = list(range(len(X_scaled)))
            np.random.shuffle(all_idx)
            chunk = max(len(all_idx) // 4, 1)
            logic_groups = {
                f"synthetic_{i}": all_idx[i * chunk : (i + 1) * chunk]
                for i in range(min(4, len(all_idx)))
            }

        device = torch.device("cpu")
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        y_t = torch.tensor(cost_matrix, dtype=torch.float32, device=device)

        model = _build_mlp(self.in_dim_, self.hidden_dim, self.out_dim_).to(device)
        outer_opt = torch.optim.Adam(model.parameters(), lr=self.outer_lr)
        loss_fn = nn.MSELoss()

        task_keys = list(logic_groups.keys())
        epoch_losses = []

        for epoch in range(self.epochs):
            outer_opt.zero_grad()
            meta_loss = torch.tensor(0.0, device=device)

            for task_key in task_keys:
                idx = logic_groups[task_key]
                if len(idx) == 0:
                    continue
                X_task = X_t[idx]
                y_task = y_t[idx]

                # Split task into support / query
                split = max(1, len(idx) // 2)
                X_support, X_query = X_task[:split], X_task[split:]
                y_support, y_query = y_task[:split], y_task[split:]
                if len(X_query) == 0:
                    X_query, y_query = X_support, y_support

                # Inner loop: adapt a copy of parameters
                fast_weights = [p.clone() for p in model.parameters()]
                for _step in range(self.inner_steps):
                    preds = _functional_forward(model, X_support, fast_weights)
                    inner_loss = loss_fn(preds, y_support)
                    grads = torch.autograd.grad(
                        inner_loss, fast_weights, create_graph=True
                    )
                    fast_weights = [
                        w - self.inner_lr * g for w, g in zip(fast_weights, grads)
                    ]

                # Outer loss on query set using adapted weights
                query_preds = _functional_forward(model, X_query, fast_weights)
                task_loss = loss_fn(query_preds, y_query)
                meta_loss = meta_loss + task_loss

            meta_loss = meta_loss / len(task_keys)
            meta_loss.backward()
            outer_opt.step()
            epoch_losses.append(meta_loss.item())

        self.model_state_ = {k: v.cpu() for k, v in model.state_dict().items()}

        return {
            "n_tasks": len(task_keys),
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "epochs": self.epochs,
        }

    # ------------------------------------------------------------------
    def predict(self, features: List[FeatureResult]) -> Predictions:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for MetaLearningCard but is not installed."
            )

        X = self._stack_vectors(features)
        X_scaled = self.scaler_.transform(X)
        device = torch.device("cpu")

        model = _build_mlp(self.in_dim_, self.hidden_dim, self.out_dim_).to(device)
        model.load_state_dict(self.model_state_)

        # Optionally adapt to training context (few inner steps)
        if self.train_X_ is not None:
            loss_fn = nn.MSELoss()
            X_ctx = torch.tensor(self.train_X_, dtype=torch.float32, device=device)
            y_ctx = torch.tensor(self.train_y_, dtype=torch.float32, device=device)

            fast_weights = [p.clone() for p in model.parameters()]
            for _step in range(self.inner_steps):
                preds = _functional_forward(model, X_ctx, fast_weights)
                inner_loss = loss_fn(preds, y_ctx)
                grads = torch.autograd.grad(inner_loss, fast_weights)
                fast_weights = [
                    w - self.inner_lr * g for w, g in zip(fast_weights, grads)
                ]

            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = _functional_forward_no_grad(model, X_t, fast_weights)
        else:
            model.eval()
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = model(X_t)

        return Predictions(
            values=scores.numpy(),
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )


# -- Functional forward helpers ------------------------------------------

def _functional_forward(
    model: "nn.Sequential",
    x: "torch.Tensor",
    weights: list,
) -> "torch.Tensor":
    """Forward pass using explicit weight list (for MAML inner loop)."""
    idx = 0
    out = x
    for layer in model:
        if isinstance(layer, nn.Linear):
            out = torch.nn.functional.linear(out, weights[idx], weights[idx + 1])
            idx += 2
        elif isinstance(layer, nn.ReLU):
            out = torch.nn.functional.relu(out)
    return out


def _functional_forward_no_grad(
    model: "nn.Sequential",
    x: "torch.Tensor",
    weights: list,
) -> "torch.Tensor":
    """Forward pass with detached weights (no gradient tracking)."""
    idx = 0
    out = x
    for layer in model:
        if isinstance(layer, nn.Linear):
            out = torch.nn.functional.linear(
                out, weights[idx].detach(), weights[idx + 1].detach()
            )
            idx += 2
        elif isinstance(layer, nn.ReLU):
            out = torch.nn.functional.relu(out)
    return out
