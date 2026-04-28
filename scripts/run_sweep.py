"""Run a full sweep of experiments for the empirical study.

Sweeps over the cross product of:
  - tasks (from a configurable list)
  - DP noise multipliers (including 0 = no DP)
  - seeds (for variance estimates)

For each cell, runs FedAvg training, then both Shokri MIA and LiRA. Writes
all results to a single CSV in the output directory, ready for `plot_results.py`.

Total work for the default sweep (5 tasks × 3 noise levels × 3 seeds × 2
attacks = 90 experiments) is the bulk of the empirical study. On a laptop
CPU this will take many hours; budget overnight for a full run, or scale down
the grid via CLI args for a quick check.

Usage:
    # Quick smoke test (1 task, 2 noise levels, 1 seed, ~30 min on CPU)
    python scripts/run_sweep.py --tasks Caco2_Wang --noise_multipliers 0 1.0 --seeds 42

    # Full sweep (overnight on laptop CPU)
    python scripts/run_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TASKS = [
    "Caco2_Wang",          # regression, ~900 molecules
    "Lipophilicity_AstraZeneca",  # regression, ~4200 molecules
    "Solubility_AqSolDB",  # regression, ~9900 molecules
    "BBB_Martins",         # classification, ~2000 molecules
    "HIA_Hou",             # classification, ~580 molecules
]

DEFAULT_NOISE = [0.0, 0.5, 1.0, 2.0]
DEFAULT_SEEDS = [42, 1337, 2024]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sweep of FedAvg/DP-FedAvg + MIA experiments")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument(
        "--noise_multipliers",
        nargs="+",
        type=float,
        default=DEFAULT_NOISE,
        help="DP noise multipliers to sweep over. 0.0 disables DP.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--num_rounds", type=int, default=10)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--num_shadow_models", type=int, default=16)
    p.add_argument("--shadow_epochs", type=int, default=30)
    p.add_argument(
        "--skip_lira",
        action="store_true",
        help="Skip the LiRA attack (LiRA is the slowest part of the sweep).",
    )
    p.add_argument(
        "--skip_shokri",
        action="store_true",
        help="Skip the Shokri-style MIA attack.",
    )
    p.add_argument("--output_dir", type=str, default="outputs/sweep")
    return p.parse_args()


def run_or_die(cmd: list[str]) -> None:
    """Run a subprocess and stream output, raising on non-zero exit."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def cell_tag(task: str, noise: float, seed: int) -> str:
    """Filename-safe tag for one sweep cell (task, noise level, seed)."""
    nstr = f"{noise:.2f}".replace(".", "p")
    return f"{task}_n{nstr}_s{seed}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    csv_path = output_dir / "sweep_results.csv"
    fieldnames = [
        "task", "noise_multiplier", "seed", "num_rounds", "clip_norm",
        "test_metric_name", "test_metric_value",
        "epsilon", "delta",
        "shokri_attack_auc",
        "lira_attack_auc", "lira_tpr_fpr001", "lira_tpr_fpr01",
        "wall_time_seconds",
    ]

    total_cells = len(args.tasks) * len(args.noise_multipliers) * len(args.seeds)
    print(f"\n=== Sweep: {total_cells} cells "
          f"({len(args.tasks)} tasks × {len(args.noise_multipliers)} noise levels × "
          f"{len(args.seeds)} seeds) ===\n")

    cell_idx = 0
    for task in args.tasks:
        for noise in args.noise_multipliers:
            for seed in args.seeds:
                cell_idx += 1
                tag = cell_tag(task, noise, seed)
                print(f"\n--- [{cell_idx}/{total_cells}] {tag} ---")
                t0 = time.time()

                # 1. Federated training (with or without DP)
                fed_cmd = [
                    sys.executable, "scripts/run_federated.py",
                    "--task", task,
                    "--num_clients", str(args.num_clients),
                    "--num_rounds", str(args.num_rounds),
                    "--seed", str(seed),
                    "--output_dir", str(output_dir),
                    "--run_tag", tag,
                ]
                if noise > 0:
                    fed_cmd += [
                        "--dp",
                        "--clip_norm", str(args.clip_norm),
                        "--noise_multiplier", str(noise),
                    ]
                try:
                    run_or_die(fed_cmd)
                except RuntimeError as e:
                    print(f"  SKIPPING cell due to training failure: {e}")
                    continue

                ckpt = output_dir / f"federated_final_{task}_{tag}.pt"
                history_path = output_dir / f"federated_history_{task}_{tag}.json"
                history = json.loads(history_path.read_text())
                test_metrics = history.get("test_metrics", {})
                if "mae" in test_metrics:
                    metric_name, metric_val = "mae", test_metrics["mae"]
                elif "auc" in test_metrics:
                    metric_name, metric_val = "auc", test_metrics["auc"]
                else:
                    metric_name, metric_val = "loss", test_metrics.get("loss")

                budget = history.get("privacy_budget")

                # 2. Shokri-style MIA
                shokri_auc = None
                if not args.skip_shokri:
                    try:
                        run_or_die([
                            sys.executable, "scripts/run_mia.py",
                            "--task", task,
                            "--checkpoint", str(ckpt),
                            "--num_shadow_models", str(args.num_shadow_models // 2),
                            "--shadow_epochs", str(args.shadow_epochs),
                            "--seed", str(seed),
                            "--output_dir", str(output_dir),
                        ])
                        # run_mia writes to mia_result_<task>.json regardless of tag,
                        # so move it before the next cell overwrites it.
                        src = output_dir / f"mia_result_{task}.json"
                        dst = output_dir / f"mia_result_{tag}.json"
                        if src.exists():
                            src.replace(dst)
                            shokri = json.loads(dst.read_text())
                            shokri_auc = shokri["attack_auc"]
                    except RuntimeError as e:
                        print(f"  Shokri MIA failed: {e}")

                # 3. LiRA
                lira_auc = lira_tpr_001 = lira_tpr_01 = None
                if not args.skip_lira:
                    try:
                        run_or_die([
                            sys.executable, "scripts/run_lira.py",
                            "--task", task,
                            "--checkpoint", str(ckpt),
                            "--num_shadow_models", str(args.num_shadow_models),
                            "--shadow_epochs", str(args.shadow_epochs),
                            "--seed", str(seed),
                            "--output_dir", str(output_dir),
                        ])
                        src = output_dir / f"lira_result_{task}.json"
                        dst = output_dir / f"lira_result_{tag}.json"
                        if src.exists():
                            src.replace(dst)
                            lira = json.loads(dst.read_text())
                            lira_auc = lira["attack_auc"]
                            lira_tpr_001 = lira["tpr_at_fpr_001"]
                            lira_tpr_01 = lira["tpr_at_fpr_01"]
                    except RuntimeError as e:
                        print(f"  LiRA failed: {e}")

                wall = time.time() - t0
                row = {
                    "task": task,
                    "noise_multiplier": noise,
                    "seed": seed,
                    "num_rounds": args.num_rounds,
                    "clip_norm": args.clip_norm,
                    "test_metric_name": metric_name,
                    "test_metric_value": metric_val,
                    "epsilon": budget["epsilon"] if budget else None,
                    "delta": budget["delta"] if budget else None,
                    "shokri_attack_auc": shokri_auc,
                    "lira_attack_auc": lira_auc,
                    "lira_tpr_fpr001": lira_tpr_001,
                    "lira_tpr_fpr01": lira_tpr_01,
                    "wall_time_seconds": round(wall, 1),
                }
                rows.append(row)
                # Write the CSV after every cell so a crash doesn't lose everything.
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows)
                print(f"  Cell complete in {wall:.0f}s. Results -> {csv_path}")

    print(f"\n=== Sweep complete: {len(rows)}/{total_cells} cells succeeded ===")
    print(f"Results: {csv_path}")
    print(f"Next: python scripts/plot_results.py --sweep_csv {csv_path}")


if __name__ == "__main__":
    main()
