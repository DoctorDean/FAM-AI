"""End-to-end privacy/utility comparison: vanilla FedAvg vs DP-FedAvg.

This script orchestrates the headline result of the repo:

  1. Train a vanilla FedAvg model.
  2. Train a DP-FedAvg model with the same hyperparameters + clip + noise.
  3. Run the same membership inference attack against both.
  4. Print a side-by-side table:

        | Setup           | Test metric | Attack AUC |
        |-----------------|-------------|------------|
        | FedAvg          |    ...      |    ...     |
        | DP-FedAvg       |    ...      |    ...     |

The expected story: DP-FedAvg costs you a small amount of utility (test MAE
ticks up, AUC ticks down) and brings the membership attack AUC noticeably
closer to 0.5. That's the headline trade-off practitioners need to internalise.

Usage:
    python scripts/run_comparison.py --task Caco2_Wang
    python scripts/run_comparison.py --task BBB_Martins --noise_multiplier 1.0

Note: this just shells out to the existing scripts so the comparison uses the
same code paths as the standalone runs. That keeps results reproducible from
either entry point and avoids subtle drift between this script and the
individual runners.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare vanilla FedAvg vs DP-FedAvg + MIA")
    p.add_argument("--task", type=str, default="Caco2_Wang")
    p.add_argument("--num_rounds", type=int, default=10)
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--noise_multiplier", type=float, default=0.5)
    p.add_argument("--num_shadow_models", type=int, default=3)
    p.add_argument("--shadow_epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    """Run a subprocess and stream output, raising on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--task", args.task,
        "--num_clients", str(args.num_clients),
        "--num_rounds", str(args.num_rounds),
        "--seed", str(args.seed),
        "--output_dir", args.output_dir,
    ]

    # --- 1. Vanilla FedAvg --------------------------------------------------
    run([sys.executable, "scripts/run_federated.py", *common, "--run_tag", "vanilla"])

    # --- 2. DP-FedAvg -------------------------------------------------------
    run([
        sys.executable, "scripts/run_federated.py",
        *common,
        "--dp",
        "--clip_norm", str(args.clip_norm),
        "--noise_multiplier", str(args.noise_multiplier),
        "--run_tag", "dp",
    ])

    # --- 3. MIA against each ------------------------------------------------
    for tag in ("vanilla", "dp"):
        ckpt = output_dir / f"federated_final_{args.task}_{tag}.pt"
        run([
            sys.executable, "scripts/run_mia.py",
            "--task", args.task,
            "--checkpoint", str(ckpt),
            "--num_shadow_models", str(args.num_shadow_models),
            "--shadow_epochs", str(args.shadow_epochs),
            "--seed", str(args.seed),
            "--output_dir", args.output_dir,
        ])
        # Rename MIA output so vanilla and DP results don't overwrite each other.
        # run_mia.py writes to mia_result_<task>.json by default, so we move it.
        src = output_dir / f"mia_result_{args.task}.json"
        dst = output_dir / f"mia_result_{args.task}_{tag}.json"
        if src.exists():
            src.replace(dst)

    # --- 4. Print the comparison table --------------------------------------
    print("\n" + "=" * 70)
    print(f"  Privacy / utility comparison on {args.task}")
    print("=" * 70)

    rows = []
    for tag in ("vanilla", "dp"):
        history_path = output_dir / f"federated_history_{args.task}_{tag}.json"
        mia_path = output_dir / f"mia_result_{args.task}_{tag}.json"
        if not history_path.exists() or not mia_path.exists():
            print(f"  Missing results for {tag}; skipping.")
            continue
        history = json.loads(history_path.read_text())
        mia = json.loads(mia_path.read_text())
        test_metrics = history.get("test_metrics", {})
        # Pick whichever primary metric the task uses.
        if "mae" in test_metrics:
            metric_name, metric_val = "MAE", test_metrics["mae"]
        elif "auc" in test_metrics:
            metric_name, metric_val = "AUC", test_metrics["auc"]
        else:
            metric_name, metric_val = "loss", test_metrics.get("loss", float("nan"))
        rows.append((tag, metric_name, metric_val, mia["attack_auc"]))

    if rows:
        # Single header derived from the first row.
        _, metric_name, _, _ = rows[0]
        print(f"\n  {'Setup':<14} | Test {metric_name:<7} | Attack AUC")
        print(f"  {'-' * 14}-+-{'-' * 12}-+-{'-' * 10}")
        for tag, _, metric_val, attack_auc in rows:
            label = "FedAvg" if tag == "vanilla" else "DP-FedAvg"
            print(f"  {label:<14} | {metric_val:>12.4f} | {attack_auc:>10.4f}")

    print()
    print("  Reading the result:")
    print("    - Test metric: lower is better for MAE, higher for AUC.")
    print("    - Attack AUC: closer to 0.5 = better privacy.")
    print(
        "    - You should see DP-FedAvg trade a small amount of utility for a "
        "meaningful drop in attack success."
    )
    print()


if __name__ == "__main__":
    main()
