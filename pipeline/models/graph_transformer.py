"""GraphTransformerCard -- Graph Transformer model card."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Union

import numpy as np

from pipeline.models.base import ModelCard
from pipeline.types import FeatureResult, Predictions

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GlobalAttention, TransformerConv
except ImportError as e:
    raise ImportError(
        "GraphTransformerCard requires torch and torch_geometric. "
        "Install with: pip install torch torch_geometric"
    ) from e


class _GraphTransformerModel(nn.Module):
    """Graph Transformer with global attention pooling and MLP head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        for i in range(n_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            # TransformerConv outputs heads * out_channels when concat=True (default).
            # Use concat=False to keep hidden_dim consistent.
            self.convs.append(
                TransformerConv(dim_in, hidden_dim, heads=n_heads, concat=False, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Global attention pooling: learns a gate per node
        self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Global attention pooling
        x = self.pool(x, batch)
        return self.head(x)


class GraphTransformerCard(ModelCard):
    """Graph Transformer algorithm selection model for graph-structured features."""

    canvas_id: ClassVar[str] = "709a17f3a8e94c63"
    input_type: ClassVar[Literal["GRAPH"]] = "GRAPH"
    output_type: ClassVar[Literal["scores"]] = "scores"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.model_: Optional[_GraphTransformerModel] = None
        self.config_names_: List[str] = []
        self.in_dim_: int = 0

    def _build_model(self, in_dim: int, out_dim: int) -> _GraphTransformerModel:
        return _GraphTransformerModel(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            out_dim=out_dim,
        )

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict:
        self.config_names_ = list(config_names)
        data_list: List[Data] = [fr.features for fr in features]
        self.in_dim_ = data_list[0].x.size(1)
        n_configs = len(config_names)

        self.model_ = self._build_model(self.in_dim_, n_configs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device)

        targets = torch.tensor(cost_matrix, dtype=torch.float32)
        for i, d in enumerate(data_list):
            d.y = targets[i]

        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model_.train()
        final_loss = 0.0
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = self.model_(batch)
                target = batch.y.view(pred.size(0), n_configs)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)

        self.model_.eval()
        self.model_.cpu()
        return {"final_train_loss": final_loss}

    def predict(self, features: List[FeatureResult]) -> Predictions:
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        data_list: List[Data] = [fr.features for fr in features]
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device)
        self.model_.eval()

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = self.model_(batch)
                all_preds.append(pred.cpu().numpy())

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
                "in_dim": self.in_dim_,
                "hidden_dim": self.hidden_dim,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
            },
            str(path),
        )

    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(str(path), weights_only=False)
        self.config_names_ = checkpoint["config_names"]
        self.in_dim_ = checkpoint["in_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.n_heads = checkpoint["n_heads"]
        self.n_layers = checkpoint["n_layers"]
        self.model_ = self._build_model(self.in_dim_, len(self.config_names_))
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()
