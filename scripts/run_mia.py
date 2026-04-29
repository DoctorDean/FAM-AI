"""Run a membership inference attack against a trained ADMET model.

Targets a checkpoint produced by `run_federated.py` (or `run_baselines.py` if
adapted). Reports the attack ROC-AUC — values above 0.5 indicate that an
adversary can distinguish training molecules from non-training ones better
than chance.

Setup of the attack data:
  - "Members": molecules from the original TDC train split. These were used
    to train the target.
  - "Non-members": molecules from the held-out test split. These were NOT
    used to train the target.
  - Shadow pool: TDC validation split. This is disjoint from both members
    and non-members and is used only to train the shadow models, so the
    attack evaluation is honest.

The attack reports a single AUC; a more thorough analysis would also report
TPR at low FPR (which is more meaningful for privacy than overall AUC), but
we keep things simple for the demo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.attacks import run_membership_inference
from src.utils import (
    ATOM_FEATURE_DIM,
    REGRESSION_TASKS,
    load_admet_task,
    smiles_list_to_graphs,
)
from src.models import GINPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run membership inference attack on a trained model")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--task", type=str, default=None)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint (.pt) produced by run_federated.py. "
        "Defaults to outputs/federated_final_<task>.pt.",
    )
    p.add_argument("--num_shadow_models", type=int, default=None)
    p.add_argument("--shadow_epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if args.task is not None:
        config["task"] = args.task
    if args.num_shadow_models is not None:
        config["attack"]["num_shadow_models"] = args.num_shadow_models
    if args.shadow_epochs is not None:
        config["attack"]["shadow_epochs"] = args.shadow_epochs
    if args.seed is not None:
        config["federation"]["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    task = config["task"]
    seed = config["federation"]["seed"]
    is_regression = task in REGRESSION_TASKS

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["output_dir"]) / f"federated_final_{task}.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Run `python scripts/run_federated.py --task {task}` first."
        )

    print(f"\n=== Membership Inference Attack ===")
    print(f"Task: {task}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    # --- Load target model from checkpoint ----------------------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    target_config = ckpt.get("config", config)
    target_model = GINPredictor(
        node_feature_dim=ATOM_FEATURE_DIM,
        hidden_dim=target_config["model"]["hidden_dim"],
        num_layers=target_config["model"]["num_layers"],
        dropout=target_config["model"]["dropout"],
        is_regression=is_regression,
    )
    target_model.load_state_dict(ckpt["state_dict"])
    target_model.to(device)
    target_model.eval()

    # --- Build attack datasets ----------------------------------------------
    print("\nLoading TDC data...")
    data = load_admet_task(task, seed=seed)
    member_graphs = smiles_list_to_graphs(data.train["Drug"].tolist(), data.train["Y"].tolist())
    nonmember_graphs = smiles_list_to_graphs(data.test["Drug"].tolist(), data.test["Y"].tolist())
    shadow_pool = smiles_list_to_graphs(data.valid["Drug"].tolist(), data.valid["Y"].tolist())
    print(f"  members (train): {len(member_graphs)}")
    print(f"  non-members (test): {len(nonmember_graphs)}")
    print(f"  shadow pool (valid): {len(shadow_pool)}")

    # --- Run attack ---------------------------------------------------------
    num_shadow = config["attack"]["num_shadow_models"]
    shadow_epochs = config["attack"]["shadow_epochs"]
    print(f"\nTraining {num_shadow} shadow models for {shadow_epochs} epochs each...")
    result = run_membership_inference(
        target_model=target_model,
        target_train_graphs=member_graphs,
        target_nonmember_graphs=nonmember_graphs,
        shadow_pool_graphs=shadow_pool,
        node_feature_dim=ATOM_FEATURE_DIM,
        is_regression=is_regression,
        config=target_config,
        num_shadow_models=num_shadow,
        shadow_epochs=shadow_epochs,
        attack_model_type=config["attack"]["attack_model"],
        seed=seed,
        device=device,
    )

    print("\n=== Attack result ===")
    print(f"  Attack ROC-AUC:               {result.attack_auc:.4f}")
    print(f"  Mean loss on members:         {result.target_train_loss_mean:.4f}")
    print(f"  Mean loss on non-members:     {result.target_nonmember_loss_mean:.4f}")
    print(f"  Loss gap (member vs non):     "
          f"{result.target_nonmember_loss_mean - result.target_train_loss_mean:+.4f}")
    print(f"  Shadow models trained:        {result.num_shadow_models}")
    print(f"  Attack train / test examples: {result.num_attack_train} / {result.num_attack_test}")

    if result.attack_auc < 0.55:
        verdict = "minor — close to random guessing"
    elif result.attack_auc < 0.65:
        verdict = "moderate — meaningful leakage"
    else:
        verdict = "significant — strong leakage signal"
    print(f"\n  Privacy verdict: {verdict}")
    print("  Reminder: federated learning alone is not privacy-preserving. "
          "Consider DP-SGD, secure aggregation, or noise on updates for real deployments.")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mia_result_{task}.json"
    with open(out_path, "w") as f:
        json.dump(result.__dict__, f, indent=2)
    print(f"\nSaved attack result to {out_path}")


if __name__ == "__main__":
    main()
