"""Run baseline (non-federated) experiments for comparison.

Two baselines:
  1. Centralised: train on the union of all training data. This is the
     oracle upper bound — what you could achieve if all partners pooled
     their data into one trusted location.
  2. Single-client: train on just one shard (1/N of the data). This is
     the lower bound — what each partner could do alone without federation.

Comparing the federated result to these two tells you (a) how much of the
centralised performance federation recovers and (b) how much each partner
gains by participating.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from src.utils import (
    ATOM_FEATURE_DIM,
    REGRESSION_TASKS,
    load_admet_task,
    partition_data,
    smiles_list_to_graphs,
)
from src.models import GINPredictor, evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline ADMET experiments")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if args.task is not None:
        config["task"] = args.task
    if args.num_clients is not None:
        config["federation"]["num_clients"] = args.num_clients
    if args.seed is not None:
        config["federation"]["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    return config


def train_model(
    train_graphs: list,
    valid_graphs: list,
    test_graphs: list,
    config: dict,
    is_regression: bool,
    total_epochs: int,
    device: torch.device,
) -> dict:
    """Train a model and return test metrics. Used for both baselines.

    `total_epochs` should be set to roughly match the *effective* number of
    epochs the federated run did on each example: num_rounds * local_epochs.
    Otherwise the baselines are unfairly handicapped or boosted compared to
    the federated setup.
    """
    model = GINPredictor(
        node_feature_dim=ATOM_FEATURE_DIM,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        is_regression=is_regression,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    bs = config["training"]["batch_size"]
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=bs, shuffle=False)

    best_valid = float("inf")
    best_state = None
    for epoch in range(total_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, is_regression)
        valid_metrics = evaluate(model, valid_loader, device, is_regression)
        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == total_epochs - 1:
            print(
                f"  epoch {epoch+1}/{total_epochs}: "
                f"train_loss={train_loss:.4f} "
                f"valid_loss={valid_metrics['loss']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, test_loader, device, is_regression)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)

    task = config["task"]
    seed = config["federation"]["seed"]
    num_clients = config["federation"]["num_clients"]
    num_rounds = config["federation"]["num_rounds"]
    local_epochs = config["federation"]["local_epochs"]
    is_regression = task in REGRESSION_TASKS

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match effective compute budget to the federated run: every example is
    # seen in `num_rounds * local_epochs` SGD passes there, so we mirror that
    # here for an apples-to-apples comparison.
    total_epochs = num_rounds * local_epochs

    print(f"\n=== Baseline ADMET training ===")
    print(f"Task: {task} | Total epochs: {total_epochs} | Device: {device}")

    print("\nLoading TDC data...")
    data = load_admet_task(task, seed=seed)

    train_graphs_all = smiles_list_to_graphs(data.train["Drug"].tolist(), data.train["Y"].tolist())
    valid_graphs = smiles_list_to_graphs(data.valid["Drug"].tolist(), data.valid["Y"].tolist())
    test_graphs = smiles_list_to_graphs(data.test["Drug"].tolist(), data.test["Y"].tolist())
    print(f"  train={len(train_graphs_all)} valid={len(valid_graphs)} test={len(test_graphs)}")

    results: dict = {"task": task, "total_epochs": total_epochs}

    # --- Baseline 1: Centralised oracle -------------------------------------
    print("\n[Centralised baseline] Training on all data...")
    central_metrics = train_model(
        train_graphs_all,
        valid_graphs,
        test_graphs,
        config,
        is_regression,
        total_epochs,
        device,
    )
    print(f"  Centralised test metrics: {central_metrics}")
    results["centralised"] = central_metrics

    # --- Baseline 2: Single client -----------------------------------------
    partitions = partition_data(
        data.train,
        num_clients=num_clients,
        strategy=config["federation"]["partition_strategy"],
        seed=seed,
    )
    single_shard = partitions[0]
    single_graphs = smiles_list_to_graphs(
        single_shard["Drug"].tolist(), single_shard["Y"].tolist()
    )
    print(f"\n[Single-client baseline] Training on shard 0 only ({len(single_graphs)} graphs)...")
    single_metrics = train_model(
        single_graphs,
        valid_graphs,
        test_graphs,
        config,
        is_regression,
        total_epochs,
        device,
    )
    print(f"  Single-client test metrics: {single_metrics}")
    results["single_client"] = single_metrics

    # --- Save -------------------------------------------------------------
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"baselines_{task}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline results to {out_path}")


if __name__ == "__main__":
    main()
