"""Generate the figures for RESULTS.md from a sweep CSV.

Produces three plots (PNG + PDF) in the output directory:

  1. privacy_utility_pareto.png — utility (MAE / AUC) vs attack TPR @ FPR=1%,
     one point per (task, noise_multiplier, seed). The Pareto frontier in this
     plot is the headline finding: how much utility you have to give up to
     drop the attack to a given level of (in)effectiveness.

  2. attack_strength_comparison.png — Shokri MIA AUC vs LiRA AUC, one point
     per cell. Shows that LiRA is a strictly stronger attack on every cell,
     justifying the choice to use it as the privacy yardstick.

  3. epsilon_vs_attack.png — formal (epsilon, delta) privacy budget vs LiRA
     TPR @ FPR=1%. Shows that the formal guarantee meaningfully predicts
     real-world attack difficulty (or doesn't, which would itself be
     interesting).

The script also writes a `summary_table.md` with per-task averaged numbers,
ready to drop into the results writeup.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Use a non-interactive backend so this works on headless CI / servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Consistent styling — readable on light and dark GitHub themes.
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "font.size": 11,
    "legend.frameon": False,
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot sweep results")
    p.add_argument("--sweep_csv", type=str, default="outputs/sweep/sweep_results.csv")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write plots. Defaults to the directory of sweep_csv.")
    return p.parse_args()


def load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce numerics; some columns may be missing if --skip_shokri/--skip_lira used.
    for col in ["test_metric_value", "epsilon", "shokri_attack_auc",
                "lira_attack_auc", "lira_tpr_fpr001", "lira_tpr_fpr01"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_privacy_utility_pareto(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot utility vs LiRA TPR @ FPR=1%, with a per-task colour."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_name in zip(axes, ["mae", "auc"]):
        sub = df[df["test_metric_name"] == metric_name].dropna(subset=["lira_tpr_fpr01"])
        if sub.empty:
            ax.text(0.5, 0.5, f"No {metric_name.upper()} tasks in sweep",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{metric_name.upper()} tasks")
            continue

        # Aggregate over seeds: median + IQR per (task, noise).
        agg = sub.groupby(["task", "noise_multiplier"]).agg(
            metric_med=("test_metric_value", "median"),
            metric_lo=("test_metric_value", lambda x: x.quantile(0.25)),
            metric_hi=("test_metric_value", lambda x: x.quantile(0.75)),
            attack_med=("lira_tpr_fpr01", "median"),
            attack_lo=("lira_tpr_fpr01", lambda x: x.quantile(0.25)),
            attack_hi=("lira_tpr_fpr01", lambda x: x.quantile(0.75)),
        ).reset_index()

        tasks = agg["task"].unique()
        # `tab10` is colourblind-safe up to ~8 categories, which is enough for our 5 tasks.
        cmap = plt.get_cmap("tab10")
        for i, task in enumerate(tasks):
            t = agg[agg["task"] == task].sort_values("noise_multiplier")
            colour = cmap(i % 10)
            ax.errorbar(
                t["attack_med"], t["metric_med"],
                xerr=[t["attack_med"] - t["attack_lo"], t["attack_hi"] - t["attack_med"]],
                yerr=[t["metric_med"] - t["metric_lo"], t["metric_hi"] - t["metric_med"]],
                marker="o", capsize=3, label=task, color=colour, linewidth=1.5,
            )
            # Annotate noise level next to each point.
            for _, row in t.iterrows():
                ax.annotate(
                    f"σ={row['noise_multiplier']}",
                    (row["attack_med"], row["metric_med"]),
                    fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points",
                )

        ax.set_xlabel("LiRA attack TPR @ FPR=1%   (lower = better privacy)")
        if metric_name == "mae":
            ax.set_ylabel("Test MAE   (lower = better utility)")
            ax.set_title("Regression tasks: privacy ↔ utility trade-off")
        else:
            ax.set_ylabel("Test ROC-AUC   (higher = better utility)")
            ax.set_title("Classification tasks: privacy ↔ utility trade-off")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Privacy / utility Pareto across tasks and noise levels", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"privacy_utility_pareto.{ext}")
    plt.close(fig)
    print(f"  -> privacy_utility_pareto.{{png,pdf}}")


def plot_attack_strength_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter of Shokri AUC vs LiRA AUC. Most points should be above y=x."""
    sub = df.dropna(subset=["shokri_attack_auc", "lira_attack_auc"])
    if sub.empty:
        return

    fig, ax = plt.subplots()
    ax.scatter(sub["shokri_attack_auc"], sub["lira_attack_auc"],
               alpha=0.6, edgecolor="none")

    # y = x reference line.
    lo = min(sub["shokri_attack_auc"].min(), sub["lira_attack_auc"].min())
    hi = max(sub["shokri_attack_auc"].max(), sub["lira_attack_auc"].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="y = x")

    # Mark the random-guessing point (0.5, 0.5).
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.text(0.51, 0.49, "random", fontsize=8, color="gray")

    ax.set_xlabel("Shokri MIA AUC")
    ax.set_ylabel("LiRA AUC")
    ax.set_title("LiRA vs Shokri attack strength\n(points above y=x: LiRA is stronger)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"attack_strength_comparison.{ext}")
    plt.close(fig)
    print(f"  -> attack_strength_comparison.{{png,pdf}}")


def plot_epsilon_vs_attack(df: pd.DataFrame, output_dir: Path) -> None:
    """Formal epsilon (log-x) vs LiRA TPR @ FPR=1%."""
    sub = df.dropna(subset=["epsilon", "lira_tpr_fpr01"])
    # Drop infinite-epsilon rows (no DP) — we'll mark them separately.
    finite = sub[np.isfinite(sub["epsilon"])]
    inf = sub[~np.isfinite(sub["epsilon"])]

    if finite.empty:
        return

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    for i, task in enumerate(finite["task"].unique()):
        t = finite[finite["task"] == task]
        ax.scatter(t["epsilon"], t["lira_tpr_fpr01"],
                   alpha=0.7, label=task, color=cmap(i % 10))

    if not inf.empty:
        # Plot no-DP runs at the right edge for reference.
        max_eps = finite["epsilon"].max() * 2
        ax.scatter([max_eps] * len(inf), inf["lira_tpr_fpr01"],
                   marker="x", color="black", alpha=0.5, label="ε=∞ (no DP)")

    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε  (lower = stronger formal guarantee)")
    ax.set_ylabel("LiRA TPR @ FPR=1%")
    ax.set_title("Formal DP guarantee vs empirical attack success")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"epsilon_vs_attack.{ext}")
    plt.close(fig)
    print(f"  -> epsilon_vs_attack.{{png,pdf}}")


def write_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Write a per-task summary markdown table aggregated over seeds."""
    lines = [
        "# Sweep Summary",
        "",
        "Aggregated over seeds (median ± IQR/2). σ is the DP noise multiplier;",
        "σ=0 means vanilla FedAvg (no DP). LiRA TPR is at FPR=1%.",
        "",
    ]

    for metric_name in ["mae", "auc"]:
        sub = df[df["test_metric_name"] == metric_name]
        if sub.empty:
            continue
        better = "lower" if metric_name == "mae" else "higher"
        lines.append(f"## {metric_name.upper()} tasks ({better} = better utility)")
        lines.append("")
        lines.append(f"| Task | σ | ε | Test {metric_name.upper()} | Shokri AUC | LiRA AUC | LiRA TPR@1% |")
        lines.append("|---|---|---|---|---|---|---|")

        agg = sub.groupby(["task", "noise_multiplier"]).agg(
            eps=("epsilon", "median"),
            metric_med=("test_metric_value", "median"),
            metric_iqr=("test_metric_value", lambda x: (x.quantile(0.75) - x.quantile(0.25)) / 2),
            shokri_med=("shokri_attack_auc", "median"),
            lira_auc_med=("lira_attack_auc", "median"),
            lira_tpr_med=("lira_tpr_fpr01", "median"),
        ).reset_index()

        for _, r in agg.iterrows():
            eps_str = (
                "∞" if not np.isfinite(r["eps"]) and not np.isnan(r["eps"])
                else "—" if np.isnan(r["eps"])
                else f"{r['eps']:.2f}"
            )
            metric_str = f"{r['metric_med']:.3f} ± {r['metric_iqr']:.3f}"
            shokri_str = f"{r['shokri_med']:.3f}" if not np.isnan(r["shokri_med"]) else "—"
            lira_auc_str = f"{r['lira_auc_med']:.3f}" if not np.isnan(r["lira_auc_med"]) else "—"
            lira_tpr_str = f"{r['lira_tpr_med']:.3f}" if not np.isnan(r["lira_tpr_med"]) else "—"
            lines.append(
                f"| {r['task']} | {r['noise_multiplier']} | {eps_str} | "
                f"{metric_str} | {shokri_str} | {lira_auc_str} | {lira_tpr_str} |"
            )
        lines.append("")

    out = output_dir / "summary_table.md"
    out.write_text("\n".join(lines))
    print(f"  -> summary_table.md")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.sweep_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sweep CSV not found at {csv_path}")
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sweep(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Writing plots to {output_dir}\n")

    plot_privacy_utility_pareto(df, output_dir)
    plot_attack_strength_comparison(df, output_dir)
    plot_epsilon_vs_attack(df, output_dir)
    write_summary_table(df, output_dir)

    print("\nDone. Drop these into RESULTS.md / docs/explainer.md.")


if __name__ == "__main__":
    main()
