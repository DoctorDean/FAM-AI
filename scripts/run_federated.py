"""Run federated training over simulated ADMET partner nodes.

Usage:
    python scripts/run_federated.py
    python scripts/run_federated.py --task BBB_Martins --num_clients 3 --num_rounds 15

This launches Flower's *simulation* runtime: clients are spun up in-process
rather than across real network endpoints. That keeps the demo runnable on a
single laptop while exercising exactly the same training/aggregation code path
as a real deployment.

After training, the final aggregated parameters are saved to
`outputs/federated_final_<task>.pt` and can be loaded by `evaluate_federated.py`
or `run_mia.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import flwr as fl
import torch
import yaml
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch_geometric.loader import DataLoader

from src.client import make_client_fn
from src.data import (
    ATOM_FEATURE_DIM,
    REGRESSION_TASKS,
    load_admet_task,
    partition_data,
    smiles_list_to_graphs,
)
from src.models import GINPredictor, evaluate, get_model_parameters, set_model_parameters
from src.server import make_dp_strategy, make_strategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run federated ADMET training")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--task", type=str, default=None, help="Override task in config")
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--num_rounds", type=int, default=None)
    p.add_argument("--partition", type=str, default=None, choices=["random", "scaffold"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument(
        "--dp",
        action="store_true",
        help="Enable differential privacy (DP-FedAvg with clipping + noise).",
    )
    p.add_argument("--clip_norm", type=float, default=None, help="L2 clip norm for DP")
    p.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="Gaussian noise multiplier for DP (relative to clip_norm)",
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional suffix appended to checkpoint filename (e.g. 'dp', 'baseline').",
    )
    return p.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(path) as f:
        config = yaml.safe_load(f)

    if args.task is not None:
        config["task"] = args.task
    if args.num_clients is not None:
        config["federation"]["num_clients"] = args.num_clients
    if args.num_rounds is not None:
        config["federation"]["num_rounds"] = args.num_rounds
    if args.partition is not None:
        config["federation"]["partition_strategy"] = args.partition
    if args.seed is not None:
        config["federation"]["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.dp:
        config["differential_privacy"]["enabled"] = True
    if args.clip_norm is not None:
        config["differential_privacy"]["clip_norm"] = args.clip_norm
    if args.noise_multiplier is not None:
        config["differential_privacy"]["noise_multiplier"] = args.noise_multiplier

    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)

    task = config["task"]
    seed = config["federation"]["seed"]
    num_clients = config["federation"]["num_clients"]
    num_rounds = config["federation"]["num_rounds"]
    partition_strategy = config["federation"]["partition_strategy"]
    is_regression = task in REGRESSION_TASKS

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Federated ADMET training ===")
    print(f"Task: {task} ({'regression' if is_regression else 'classification'})")
    print(f"Clients: {num_clients} | Rounds: {num_rounds} | Partition: {partition_strategy}")
    print(f"Device: {device}")

    # --- Load and partition the dataset --------------------------------------
    print("\nLoading TDC data...")
    data = load_admet_task(task, seed=seed)
    print(f"  train={len(data.train)} valid={len(data.valid)} test={len(data.test)}")

    partitions = partition_data(
        data.train, num_clients=num_clients, strategy=partition_strategy, seed=seed
    )

    # Each client gets the same validation slice — modelling the realistic case
    # where partners agree on a held-out validation set for round-by-round
    # tracking of the global model.
    valid_graphs = smiles_list_to_graphs(data.valid["Drug"].tolist(), data.valid["Y"].tolist())
    print(f"  validation graphs: {len(valid_graphs)}")

    client_data = []
    for i, shard in enumerate(partitions):
        train_graphs = smiles_list_to_graphs(shard["Drug"].tolist(), shard["Y"].tolist())
        print(f"  Client {i}: {len(train_graphs)} train graphs (from {len(shard)} SMILES)")
        client_data.append((train_graphs, valid_graphs))

    # --- Initialise the global model ----------------------------------------
    initial_model = GINPredictor(
        node_feature_dim=ATOM_FEATURE_DIM,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        is_regression=is_regression,
    )
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    dp_cfg = config.get("differential_privacy", {"enabled": False})
    if dp_cfg.get("enabled", False):
        print(
            f"\nDifferential privacy ENABLED: clip_norm={dp_cfg['clip_norm']}, "
            f"noise_multiplier={dp_cfg['noise_multiplier']}"
        )
        # Compute and print the formal (epsilon, delta)-DP guarantee.
        try:
            from src.privacy import compute_dp_fedavg_epsilon
            budget = compute_dp_fedavg_epsilon(
                noise_multiplier=dp_cfg["noise_multiplier"],
                num_rounds=num_rounds,
                target_delta=1e-5,
                sampling_rate=config["federation"]["fraction_fit"],
            )
            print(f"Privacy budget: {budget}")
        except ImportError:
            print(
                "  (Install dp-accounting to see formal (epsilon, delta) guarantees: "
                "pip install dp-accounting)"
            )
            budget = None
        strategy = make_dp_strategy(
            initial_parameters=initial_parameters,
            clip_norm=dp_cfg["clip_norm"],
            noise_multiplier=dp_cfg["noise_multiplier"],
            fraction_fit=config["federation"]["fraction_fit"],
            fraction_eval=config["federation"]["fraction_eval"],
            min_clients=num_clients,
        )
        default_tag = "dp"
    else:
        budget = None
        strategy = make_strategy(
            initial_parameters=initial_parameters,
            fraction_fit=config["federation"]["fraction_fit"],
            fraction_eval=config["federation"]["fraction_eval"],
            min_clients=num_clients,
        )
        default_tag = ""

    run_tag = args.run_tag if args.run_tag is not None else default_tag
    suffix = f"_{run_tag}" if run_tag else ""

    client_fn = make_client_fn(
        client_data=client_data,
        node_feature_dim=ATOM_FEATURE_DIM,
        is_regression=is_regression,
        config=config,
    )

    # --- Run the simulation -------------------------------------------------
    print(f"\nStarting Flower simulation for {num_rounds} rounds...\n")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    # --- Save the final aggregated model ------------------------------------
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if strategy.latest_parameters is None:
        raise RuntimeError(
            "Strategy did not capture aggregated parameters — did training fail mid-run?"
        )

    final_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)
    final_model = GINPredictor(
        node_feature_dim=ATOM_FEATURE_DIM,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        is_regression=is_regression,
    )
    set_model_parameters(final_model, final_ndarrays)
    final_model.to(device)

    # Evaluate on test set so we have a single reportable number per run.
    test_graphs = smiles_list_to_graphs(data.test["Drug"].tolist(), data.test["Y"].tolist())
    test_loader = DataLoader(test_graphs, batch_size=config["training"]["batch_size"])
    test_metrics = evaluate(final_model, test_loader, device, is_regression)

    print("\n=== Final test metrics (aggregated model) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    ckpt_path = output_dir / f"federated_final_{task}{suffix}.pt"
    torch.save(
        {
            "task": task,
            "config": config,
            "state_dict": final_model.state_dict(),
            "test_metrics": test_metrics,
            "privacy_budget": budget.__dict__ if budget is not None else None,
        },
        ckpt_path,
    )
    print(f"\nSaved final aggregated model to {ckpt_path}")

    history_path = output_dir / f"federated_history_{task}{suffix}.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "task": task,
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "partition_strategy": partition_strategy,
                "differential_privacy": dp_cfg,
                "privacy_budget": budget.__dict__ if budget is not None else None,
                "test_metrics": test_metrics,
                "losses_distributed": history.losses_distributed,
                "metrics_distributed_fit": history.metrics_distributed_fit,
                "metrics_distributed": history.metrics_distributed,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
