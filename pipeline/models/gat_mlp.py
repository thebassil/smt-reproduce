"""REF-GNN model: GATv2 + JumpingKnowledge + AttentionalAggregation + MLP.

From-scratch reimplementation of Sibyl's GAT graph classifier for
algorithm selection over SMT solvers.

Reference (study only, not imported):
  artifacts/sibyl/src/networks/gnn.py:97-162         — GAT architecture
  artifacts/sibyl/src/networks/utils/utils.py:75-87  — ModifiedMarginRankingLoss
  artifacts/sibyl/src/networks/utils/utils.py:102-220 — train_model loop
  artifacts/sibyl/src/networks/inference.py           — score negation

Architecture:
  Input: Data(x=[N,67], edge_index=[2,E], edge_attr=[E])
    ├─ n_layers × GATv2Conv(67→67, heads, concat=False, edge_dim=1) + LeakyReLU
    ├─ × n_passes, collect hidden states for JK
    ├─ JumpingKnowledge(mode='cat') → dim = (n_passes+1) × 67
    ├─ AttentionalAggregation(gate_nn) → graph embedding
    ├─ Concat problemType → [fc_dim]
    ├─ FC1 + LeakyReLU + Dropout
    ├─ FC2 + LeakyReLU + Dropout
    └─ FC3 → n_configs scores (negated at inference)

Loss: ModifiedMarginRankingLoss — for all solver pairs (i,j) where rank_i < rank_j:
  loss += max(0, -(score_i - score_j) + margin * |rank_i - rank_j|)
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Union

import numpy as np

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    from torch.nn import MarginRankingLoss
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import (
        AttentionalAggregation,
        GATv2Conv,
        JumpingKnowledge,
    )
except ImportError as e:
    raise ImportError(
        "GATMLPCard requires torch and torch_geometric. "
        "Install with: pip install torch torch_geometric"
    ) from e


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class _ModifiedMarginRankingLoss(nn.Module):
    """Pairwise margin ranking loss with rank-distance-scaled margins.

    For all pairs (i, j) of solvers where rank_i < rank_j (i.e. solver i
    is faster), penalise if score_i is not sufficiently higher than score_j.
    The margin scales linearly with |rank_i - rank_j|.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [batch, n_configs] — model output (higher = better).
            labels: [batch, n_configs] — runtimes (lower = better).
        """
        device = scores.device
        loss = torch.zeros(1, device=device)
        # Sort by label (ascending runtime) → indices of solvers by rank
        indx = labels.argsort(dim=1)
        n_configs = labels.size(1)
        for i, j in itertools.combinations(range(n_configs), 2):
            # i < j in sorted order, so solver at rank i is faster
            loss_fn = MarginRankingLoss(margin=self.margin * abs(i - j))
            s_i = scores.gather(1, indx[:, i].unsqueeze(1))
            s_j = scores.gather(1, indx[:, j].unsqueeze(1))
            target = torch.ones_like(s_j, device=device)
            loss = loss + loss_fn(s_i, s_j, target)
        return loss


# ---------------------------------------------------------------------------
# GATv2 Network
# ---------------------------------------------------------------------------

class _GATNetwork(nn.Module):
    """GATv2 encoder with JK aggregation, attentional pooling, and MLP head."""

    def __init__(
        self,
        input_dim: int = 67,
        hidden_dim: int = 67,
        n_layers: int = 5,
        n_heads: int = 5,
        n_passes: int = 2,
        jk_mode: str = "cat",
        pooling: str = "attention",
        out_dim: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_passes = n_passes
        self.jk_mode = jk_mode

        # GATv2Conv layers — one set shared across passes
        self.gats = nn.ModuleList([
            GATv2Conv(
                in_channels=input_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                heads=n_heads,
                concat=False,
                dropout=0,
                edge_dim=1,
            )
            for i in range(n_layers)
        ])

        # JumpingKnowledge
        if jk_mode == "cat":
            self.jump = JumpingKnowledge(mode="cat", channels=hidden_dim, num_layers=n_passes)
            jk_out = (n_passes + 1) * hidden_dim
        else:
            self.jump = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=n_passes)
            jk_out = hidden_dim

        # fc_input = jk_out + 1 (for problemType scalar)
        fc_input = jk_out + 1

        # Graph-level pooling
        if pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=nn.Sequential(nn.Linear(jk_out, 1), nn.LeakyReLU()),
            )
        elif pooling == "mean":
            from torch_geometric.nn import global_mean_pool
            self.pool = global_mean_pool
        elif pooling == "max":
            from torch_geometric.nn import global_max_pool
            self.pool = global_max_pool
        elif pooling == "add":
            from torch_geometric.nn import global_add_pool
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # MLP head (dropout applied BEFORE each FC, matching Sibyl)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fc_input, fc_input // 2)
        self.fc2 = nn.Linear(fc_input // 2, fc_input // 2)
        self.fc_last = nn.Linear(fc_input // 2, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        problem_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # Collect layer outputs for JK
        xs = [x]
        for gat in self.gats:
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = F.leaky_relu(x)
            xs.append(x)

        # JK needs exactly (n_passes+1) tensors: [initial, pass_1, ..., pass_n]
        # We have n_layers outputs + initial; Sibyl uses n_layers == n_passes
        # Select last (n_passes+1) for JK
        jk_inputs = xs[-(self.n_passes + 1):]
        x = self.jump(jk_inputs)

        # Graph-level pooling
        x = self.pool(x, batch)

        # Handle empty graphs (0 nodes after pooling)
        if x.size(0) == 0:
            n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else problem_type.size(0)
            x = torch.zeros(n_graphs, x.size(1) if x.dim() > 1 else self.gats[-1].out_channels,
                           device=problem_type.device)

        # Concat problemType scalar
        x = torch.cat(
            (x.reshape(x.size(0), -1), problem_type.unsqueeze(1).to(x.device)),
            dim=1,
        )

        # MLP head
        x = self.fc1(self.dropout(x))
        x = F.leaky_relu(x)
        x = self.fc2(self.dropout(x))
        x = F.leaky_relu(x)
        x = self.fc_last(self.dropout(x))

        return x


# ---------------------------------------------------------------------------
# Data weighting
# ---------------------------------------------------------------------------

def _get_weights(cost_matrix: np.ndarray, mode: str) -> Optional[list[float]]:
    """Compute per-instance sampling weights.

    Args:
        cost_matrix: (n_instances, n_configs) array of runtimes.
        mode: 'best' (inverse frequency of best solver) or None.
    """
    if mode == "best":
        best_solver = cost_matrix.argmin(axis=1)
        unique, counts = np.unique(best_solver, return_counts=True)
        weight_map = {s: 1.0 / c for s, c in zip(unique, counts)}
        return [weight_map[s] for s in best_solver]
    return None


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------

class GATMLPCard(ModelCard):
    """GAT + MLP algorithm selection model for graph-structured features.

    REF-GNN reference model. From-scratch reimplementation of Sibyl's GAT.
    """

    canvas_id: ClassVar[str] = "ref_model_gat_mlp"
    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        input_dim: int = 67,
        hidden_dim: int = 67,
        n_layers: int = 5,
        n_heads: int = 5,
        n_passes: int = 2,
        pooling: str = "attention",
        jk_mode: str = "cat",
        dropout: float = 0.0,
        epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "modified_margin_ranking",
        margin: float = 0.1,
        data_weight: Optional[str] = "best",
        batch_size: int = 1,
        scheduler_patience: int = 3,
        early_stop_lr: float = 1e-7,
        device: str = "auto",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_passes = n_passes
        self.pooling = pooling
        self.jk_mode = jk_mode
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_name = loss
        self.margin = margin
        self.data_weight = data_weight
        self.batch_size = batch_size
        self.scheduler_patience = scheduler_patience
        self.early_stop_lr = early_stop_lr
        self.device_str = device

        self.model_: Optional[_GATNetwork] = None
        self.config_names_: List[str] = []

    def _resolve_device(self) -> torch.device:
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def _build_network(self, out_dim: int) -> _GATNetwork:
        return _GATNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_passes=self.n_passes,
            jk_mode=self.jk_mode,
            pooling=self.pooling,
            out_dim=out_dim,
            dropout=self.dropout,
        )

    def _build_loss(self, device: torch.device) -> nn.Module:
        if self.loss_name == "modified_margin_ranking":
            return _ModifiedMarginRankingLoss(margin=self.margin)
        raise ValueError(f"Unknown loss: {self.loss_name}")

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        n_configs = len(config_names)
        device = self._resolve_device()

        # Build network
        self.model_ = self._build_network(n_configs)
        self.model_.to(device)

        # Prepare data: attach labels and problemType to each Data object
        data_list: List[Data] = []
        targets = torch.tensor(cost_matrix, dtype=torch.float32)
        for i, fr in enumerate(features):
            d = fr.features.clone()
            d.y = targets[i]
            d.problem_type = torch.tensor([0.0])  # default; can encode logic
            data_list.append(d)

        # Data weighting
        weights = _get_weights(cost_matrix, self.data_weight)
        if weights is not None:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(weights, num_samples=len(weights))
            loader = DataLoader(data_list, batch_size=self.batch_size, sampler=sampler)
        else:
            loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

        # Loss, optimizer, scheduler
        loss_fn = self._build_loss(device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.scheduler_patience
        )

        # Training loop
        final_loss = 0.0
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                batch = batch.to(device)
                # Extract problemType (defaults to 0 if not set)
                if hasattr(batch, "problem_type"):
                    pt = batch.problem_type[:batch.num_graphs]
                else:
                    pt = torch.zeros(batch.num_graphs, device=device)

                with autocast():
                    scores = self.model_(
                        batch.x, batch.edge_index, batch.edge_attr, pt, batch.batch
                    )
                    labels = batch.y.view(scores.size(0), n_configs)
                    loss = loss_fn(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                self.model_.float()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss
            scheduler.step(avg_loss)

            # Early stopping if LR drops below threshold
            if optimizer.param_groups[0]["lr"] < self.early_stop_lr:
                break

        self.model_.eval()
        self.model_.cpu()
        return {"final_train_loss": final_loss, "epochs_completed": epoch + 1}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = self._resolve_device()
        self.model_.to(device)
        self.model_.eval()

        data_list: List[Data] = []
        for fr in features:
            d = fr.features.clone()
            d.problem_type = torch.tensor([0.0])
            data_list.append(d)

        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if hasattr(batch, "problem_type"):
                    pt = batch.problem_type[:batch.num_graphs]
                else:
                    pt = torch.zeros(batch.num_graphs, device=device)

                with autocast():
                    scores = self.model_(
                        batch.x, batch.edge_index, batch.edge_attr, pt, batch.batch
                    )
                # Negate scores: in Sibyl, lower runtime = better solver,
                # model learns higher score = better solver, so negate to get
                # "predicted runtime" semantics where lower = better.
                all_preds.append((-scores).cpu().numpy())

        self.model_.cpu()
        values = np.concatenate(all_preds, axis=0)

        return Predictions(
            values=values,
            output_type="scores",
            config_names=self.config_names_,
            instance_ids=self._instance_ids(features),
        )

    def save(self, path: Union[str, Path]) -> None:
        if self.model_ is None:
            raise RuntimeError("No model to save. Call fit() first.")
        torch.save(
            {
                "state_dict": self.model_.state_dict(),
                "config_names": self.config_names_,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "n_passes": self.n_passes,
                "pooling": self.pooling,
                "jk_mode": self.jk_mode,
                "dropout": self.dropout,
            },
            str(path),
        )

    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(str(path), weights_only=False)
        self.config_names_ = checkpoint["config_names"]
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.n_layers = checkpoint["n_layers"]
        self.n_heads = checkpoint["n_heads"]
        self.n_passes = checkpoint["n_passes"]
        self.pooling = checkpoint["pooling"]
        self.jk_mode = checkpoint["jk_mode"]
        self.dropout = checkpoint["dropout"]
        self.model_ = self._build_network(len(self.config_names_))
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()
