"""Flower client wrapping a partner's local GNN training loop."""
from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import torch
from torch_geometric.loader import DataLoader

from src.models import (
    GINPredictor,
    evaluate,
    get_model_parameters,
    set_model_parameters,
    train_one_epoch,
)


class ADMETClient(fl.client.NumPyClient):
    """A simulated pharma partner: local data, local training, no raw-data sharing.

    Each client receives the current global model weights at the start of a round,
    trains for `local_epochs` on its private data, and returns the updated weights
    along with the number of training examples (used by the server to weight the
    FedAvg aggregation).
    """

    def __init__(
        self,
        client_id: int,
        train_graphs: list,
        valid_graphs: list,
        node_feature_dim: int,
        is_regression: bool,
        config: dict,
        device: torch.device | None = None,
    ) -> None:
        self.client_id = client_id
        self.is_regression = is_regression
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GINPredictor(
            node_feature_dim=node_feature_dim,
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            is_regression=is_regression,
        ).to(self.device)

        bs = config["training"]["batch_size"]
        # PyG's DataLoader handles batching of variable-sized graphs.
        self.train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
        self.valid_loader = DataLoader(valid_graphs, batch_size=bs, shuffle=False)
        self.num_train = len(train_graphs)
        self.num_valid = len(valid_graphs)

    # --- Flower interface ----------------------------------------------------

    def get_parameters(self, config: dict) -> list:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: list) -> None:
        set_model_parameters(self.model, parameters)

    def fit(self, parameters: list, config: dict) -> tuple[list, int, dict]:
        """Train locally for `local_epochs` and return updated parameters."""
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", self.config["federation"]["local_epochs"]))
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        last_loss = float("nan")
        for _ in range(local_epochs):
            last_loss = train_one_epoch(
                self.model, self.train_loader, optimizer, self.device, self.is_regression
            )
        return (
            self.get_parameters(config={}),
            self.num_train,
            {"client_id": self.client_id, "train_loss": last_loss},
        )

    def evaluate(self, parameters: list, config: dict) -> tuple[float, int, dict]:
        """Evaluate the global model on this client's local validation data."""
        self.set_parameters(parameters)
        metrics = evaluate(self.model, self.valid_loader, self.device, self.is_regression)
        loss = float(metrics["loss"])
        # Flower expects a {str: scalar} dict; cast everything to float to be safe.
        scalar_metrics = {k: float(v) for k, v in metrics.items()}
        scalar_metrics["client_id"] = self.client_id
        return loss, self.num_valid, scalar_metrics


def make_client_fn(
    client_data: list[tuple[list, list]],
    node_feature_dim: int,
    is_regression: bool,
    config: dict,
):
    """Factory returning a `client_fn(cid)` for Flower's simulation runtime.

    Flower's simulator instantiates clients on demand by string ID; we map that
    ID back to one of our pre-partitioned data shards.
    """

    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        train_graphs, valid_graphs = client_data[idx]
        client = ADMETClient(
            client_id=idx,
            train_graphs=train_graphs,
            valid_graphs=valid_graphs,
            node_feature_dim=node_feature_dim,
            is_regression=is_regression,
            config=config,
        )
        return client.to_client()

    return client_fn
