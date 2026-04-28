"""Graph Isomorphism Network (GIN) for ADMET prediction.

GIN is a strong baseline for molecular property prediction (Xu et al., ICLR 2019)
that's expressive enough to be useful while remaining simple to read and debug.
We deliberately keep this small (~3 layers, 64-dim) because (a) TDC ADMET tasks
are small and large models overfit hard, and (b) the point of the repo is to
demonstrate federation/privacy mechanics, not to chase SOTA on any single task.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GINPredictor(nn.Module):
    """GIN encoder + MLP readout for molecular property prediction.

    Args:
        node_feature_dim: Dimension of input atom features.
        hidden_dim: Width of GIN and MLP hidden layers.
        num_layers: Number of GIN message-passing layers.
        dropout: Dropout probability applied after pooling.
        is_regression: If True, output a single scalar; if False, output a logit
            for binary classification. Loss/metric selection happens outside this
            module.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        is_regression: bool = True,
    ) -> None:
        super().__init__()
        self.is_regression = is_regression
        self.dropout = dropout

        self.convs = nn.ModuleList()
        in_dim = node_feature_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            in_dim = hidden_dim

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        graph_repr = global_mean_pool(h, batch)
        out = self.readout(graph_repr).squeeze(-1)
        return out

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Return pooled graph embeddings (pre-readout). Useful for the MIA."""
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return global_mean_pool(h, batch)


def get_model_parameters(model: nn.Module) -> list:
    """Extract model parameters as a list of NumPy arrays for Flower.

    Flower transmits parameters as a list of ndarrays; we use the canonical
    state_dict ordering so client and server stay in sync.
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list) -> None:
    """Load parameters (list of NumPy arrays) into a model in-place."""
    state_dict = model.state_dict()
    new_state = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
    model.load_state_dict(new_state, strict=True)
