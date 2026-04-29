"""Run a Likelihood Ratio Attack (LiRA) against a trained ADMET model.

LiRA is a stronger membership inference attack than the basic Shokri-style one
in `run_mia.py`. It calibrates per-query (rather than population-level) by
training many shadow models with random subsets of the population and asking
"is this loss more consistent with the IN distribution or the OUT distribution
for this specific molecule?"

Usage:
    python scripts/run_lira.py --task Caco2_Wang --num_shadow_models 32

By default it uses the FedAvg-trained model checkpoint from the standard
output location. Pair this with `run_comparison.py` to attack both vanilla
FedAvg and DP-FedAvg with LiRA.

Reports:
  - Attack ROC-AUC (overall — comparable to the Shokri-style attack)
  - TPR @ FPR=0.1% and TPR @ FPR=1% (the privacy-relevant metrics)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.attacks import run_lira
from src.utils import (
    ATOM_FEATURE_DIM,
    REGRESSION_TASKS,
    load_admet_task,
    smiles_list_to_graphs,
)
from src.models import GINPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LiRA membership inference attack")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--task", type=str, default=None)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint (.pt). "
        "Defaults to outputs/federated_final_<task>.pt.",
    )
    p.add_argument("--num_shadow_models", type=int, default=32)
    p.add_argument("--shadow_epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if args.task is not None:
        config["task"] = args.task
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

    print(f"\n=== LiRA Membership Inference Attack ===")
    print(f"Task: {task}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Shadow models: {args.num_shadow_models} ({args.shadow_epochs} epochs each)")

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

    # --- Run LiRA -----------------------------------------------------------
    print(f"\nTraining {args.num_shadow_models} shadow models...")
    result = run_lira(
        target_model=target_model,
        target_train_graphs=member_graphs,
        target_nonmember_graphs=nonmember_graphs,
        shadow_pool_graphs=shadow_pool,
        node_feature_dim=ATOM_FEATURE_DIM,
        is_regression=is_regression,
        config=target_config,
        num_shadow_models=args.num_shadow_models,
        shadow_epochs=args.shadow_epochs,
        seed=seed,
        device=device,
    )

    print("\n=== LiRA Attack Result ===")
    print(f"  Attack ROC-AUC:         {result.attack_auc:.4f}")
    print(f"  TPR @ FPR=0.1%:         {result.tpr_at_fpr_001:.4f}")
    print(f"  TPR @ FPR=1%:           {result.tpr_at_fpr_01:.4f}")
    print(f"  Shadow models:          {result.num_shadow_models}")
    print(f"  Queries:                {result.num_queries}")
    print(f"  Mean mu_in / mu_out:    {result.mu_in_mean:.4f} / {result.mu_out_mean:.4f}")
    print(f"  Mean sigma_in / out:    {result.sigma_in_mean:.4f} / {result.sigma_out_mean:.4f}")

    # Privacy verdict using TPR @ FPR=1% (more meaningful than overall AUC).
    if result.tpr_at_fpr_01 < 0.02:
        verdict = "negligible — attack barely above chance at low FPR"
    elif result.tpr_at_fpr_01 < 0.05:
        verdict = "minor leakage at low FPR"
    elif result.tpr_at_fpr_01 < 0.15:
        verdict = "moderate leakage — attacker reliably wins on some records"
    else:
        verdict = "significant leakage — strong identification of members at low FPR"
    print(f"\n  Privacy verdict (based on TPR @ FPR=1%): {verdict}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"lira_result_{task}.json"
    with open(out_path, "w") as f:
        json.dump(result.__dict__, f, indent=2)
    print(f"\nSaved attack result to {out_path}")


if __name__ == "__main__":
    main()
