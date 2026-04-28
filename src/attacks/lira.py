"""Likelihood Ratio Attack (LiRA) for membership inference.

LiRA (Carlini et al., "Membership Inference Attacks From First Principles",
IEEE S&P 2022) is currently the strongest practical membership inference attack.
The intuition: instead of training one attack classifier on raw loss values,
train many shadow models such that for each candidate molecule we have shadow
models where it WAS in training (IN) and shadow models where it WASN'T (OUT).
Then for a target molecule's loss under the actual target model, compute a
likelihood ratio: is this loss more consistent with the IN distribution or
the OUT distribution?

For a fixed query x, define:
    L_in(x)  = distribution of loss(model, x) over models trained WITH x
    L_out(x) = distribution of loss(model, x) over models trained WITHOUT x

Empirically, after a logit transform, both are approximately Gaussian. So we
estimate (mu_in, sigma_in) and (mu_out, sigma_out) from the shadow models,
and the test is:

    score(x) = log p(loss | x is OUT) - log p(loss | x is IN)

Higher score = more likely the molecule was a member. We report ROC-AUC and
TPR at a low FPR threshold (the metric that actually matters for privacy —
overall AUC can hide attacks that are very good at high-confidence
identification of a few records).

Why this is stronger than the Shokri attack:
- The Shokri attack uses a single feature (loss) for ALL queries simultaneously.
  This works because high-loss-on-average tends to mean non-member, but it
  ignores per-example calibration: some molecules are inherently easier than
  others.
- LiRA calibrates per-query: it asks "is THIS molecule's loss surprising given
  what shadow models trained without it look like?" This is the difference
  between a population-level test and a per-instance test, and it dramatically
  improves the attack at low false positive rates.

Cost: LiRA needs many shadow models (Carlini et al. used 256). For our small
TDC datasets and a demo we use far fewer (~32) — the attack still works, just
with noisier mu/sigma estimates.

Reference:
  Carlini, Chien, Nasr, Song, Terzis, Tramer. "Membership Inference Attacks
  From First Principles." IEEE S&P 2022. https://arxiv.org/abs/2112.03570
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

from src.models import GINPredictor, train_one_epoch


@dataclass
class LiRAResult:
    """Result of a LiRA attack run."""

    attack_auc: float
    tpr_at_fpr_001: float  # True positive rate at FPR = 0.1%
    tpr_at_fpr_01: float   # True positive rate at FPR = 1%
    num_shadow_models: int
    num_queries: int
    # Per-query diagnostics (useful for plotting).
    mu_in_mean: float
    mu_out_mean: float
    sigma_in_mean: float
    sigma_out_mean: float


def _per_example_losses(
    model: nn.Module,
    graphs: list,
    device: torch.device,
    is_regression: bool,
    batch_size: int = 32,
) -> np.ndarray:
    """Per-example loss for each graph (squared error or BCE)."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y.float()
            if is_regression:
                ex_losses = (preds - targets) ** 2
            else:
                ex_losses = nn.functional.binary_cross_entropy_with_logits(
                    preds, targets, reduction="none"
                )
            losses.extend(ex_losses.cpu().tolist())
    return np.array(losses, dtype=np.float64)


def _logit_transform(losses: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply a logit-style stabilising transform.

    Carlini et al. observe that loss distributions from neural networks are
    long-tailed, but after a transform like phi(x) = log(x / (1 - x)) (when
    losses are bounded in [0, 1]) or simply log(x) for regression squared
    errors, they become much more Gaussian — which is what LiRA's parametric
    test assumes. We use log(x + eps) here as a simple, scale-free transform
    that works for both squared error and BCE.
    """
    return np.log(losses + eps)


def _train_shadow_model(
    train_graphs: list,
    node_feature_dim: int,
    is_regression: bool,
    config: dict,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    """Train one shadow model with the same architecture as the target."""
    model = GINPredictor(
        node_feature_dim=node_feature_dim,
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
    loader = DataLoader(train_graphs, batch_size=config["training"]["batch_size"], shuffle=True)
    for _ in range(epochs):
        train_one_epoch(model, loader, optimizer, device, is_regression)
    return model


def run_lira(
    target_model: nn.Module,
    target_train_graphs: list,
    target_nonmember_graphs: list,
    shadow_pool_graphs: list,
    node_feature_dim: int,
    is_regression: bool,
    config: dict,
    num_shadow_models: int = 32,
    shadow_epochs: int = 50,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> LiRAResult:
    """Run LiRA against the target model.

    Args:
        target_model: The trained model under attack.
        target_train_graphs: Members (used to train the target).
        target_nonmember_graphs: Non-members (held out from the target).
        shadow_pool_graphs: Pool from which shadow training sets are drawn.
            Each shadow model gets a *random subset* of this pool so that for
            each query molecule we have a mix of shadow models where it was
            and wasn't trained on. This is the key data structure that makes
            per-query likelihood-ratio testing possible.

            IMPORTANT: query molecules (members and non-members of the target)
            must also be in this pool, so that we can score each query against
            shadow models that did and didn't see it. We add them automatically.
        num_shadow_models: How many shadow models. Carlini et al. used 256;
            we default to 32 for tractability. With fewer models the per-query
            mu/sigma estimates are noisier but the attack still works.
        shadow_epochs: Per-shadow training epochs.
        seed: RNG seed.
        device: Torch device override.
        verbose: Print progress.

    Returns:
        LiRAResult with attack AUC, TPR @ low FPR, and diagnostic statistics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    # Build the union of all graphs we need to score: member queries +
    # non-member queries + shadow pool. We need shadow models trained on
    # subsets of this whole union to get IN/OUT samples for every query.
    all_queries = target_train_graphs + target_nonmember_graphs
    full_pool = all_queries + shadow_pool_graphs
    n_total = len(full_pool)
    n_queries = len(all_queries)

    if num_shadow_models < 8:
        # With fewer than 8 shadow models the IN/OUT counts per query are
        # too small to estimate variance reliably.
        raise ValueError(
            f"LiRA needs at least 8 shadow models for stable variance estimates, "
            f"got {num_shadow_models}."
        )

    # Membership matrix: shape (num_shadow_models, n_total). Entry [i, j] = 1
    # iff shadow model i was trained on graph j. We use Bernoulli(0.5) so each
    # query has roughly num_shadow_models/2 IN observations and num_shadow_models/2
    # OUT observations.
    membership = rng.random(size=(num_shadow_models, n_total)) < 0.5

    # Collect, for each shadow model, the per-graph loss on the FULL pool.
    # Shape: (num_shadow_models, n_total).
    shadow_losses = np.zeros((num_shadow_models, n_total), dtype=np.float64)

    for s in range(num_shadow_models):
        if verbose and (s % max(1, num_shadow_models // 8) == 0):
            print(f"  Training shadow model {s+1}/{num_shadow_models}...")
        in_idx = np.where(membership[s])[0]
        in_graphs = [full_pool[i] for i in in_idx]
        # Edge case: in extremely unlucky draws a shadow ends up with too few
        # graphs to train on. Resample if so.
        if len(in_graphs) < 4:
            membership[s] = rng.random(size=n_total) < 0.5
            in_idx = np.where(membership[s])[0]
            in_graphs = [full_pool[i] for i in in_idx]

        shadow_model = _train_shadow_model(
            in_graphs, node_feature_dim, is_regression, config, device, shadow_epochs
        )
        # Score the WHOLE pool (members + non-members + shadow pool).
        shadow_losses[s] = _per_example_losses(shadow_model, full_pool, device, is_regression)

    # Logit-transform for Gaussian-ish stability.
    transformed = _logit_transform(shadow_losses)

    # For each query, fit Gaussians to IN and OUT loss distributions across
    # shadow models. We only do this for the queries (first n_queries entries).
    mu_in = np.zeros(n_queries)
    sigma_in = np.zeros(n_queries)
    mu_out = np.zeros(n_queries)
    sigma_out = np.zeros(n_queries)
    # Sigma floor — guards against shadow models that all produce nearly identical
    # losses for an "easy" query, which would otherwise give sigma=0 and make the
    # likelihood ratio degenerate.
    SIGMA_FLOOR = 1e-3

    for q in range(n_queries):
        in_mask = membership[:, q]
        in_obs = transformed[in_mask, q]
        out_obs = transformed[~in_mask, q]
        if len(in_obs) >= 2 and len(out_obs) >= 2:
            mu_in[q] = in_obs.mean()
            sigma_in[q] = max(in_obs.std(ddof=1), SIGMA_FLOOR)
            mu_out[q] = out_obs.mean()
            sigma_out[q] = max(out_obs.std(ddof=1), SIGMA_FLOOR)
        else:
            # Fall back to global statistics for queries with too few obs.
            mu_in[q] = transformed[:, q].mean()
            sigma_in[q] = max(transformed[:, q].std(ddof=1), SIGMA_FLOOR)
            mu_out[q] = mu_in[q]
            sigma_out[q] = sigma_in[q]

    # Score the actual target on the queries.
    target_losses = _per_example_losses(target_model, all_queries, device, is_regression)
    target_transformed = _logit_transform(target_losses)

    # Likelihood ratio: log p(loss | OUT) - log p(loss | IN).
    # A higher score means the loss is more consistent with the IN distribution
    # (i.e. lower under IN's mean), so we flip sign so higher = more likely member.
    log_p_in = norm.logpdf(target_transformed, loc=mu_in, scale=sigma_in)
    log_p_out = norm.logpdf(target_transformed, loc=mu_out, scale=sigma_out)
    scores = log_p_in - log_p_out  # Higher = more likely a member.

    # Ground truth: first len(target_train_graphs) queries are members, rest aren't.
    y_true = np.concatenate([
        np.ones(len(target_train_graphs)),
        np.zeros(len(target_nonmember_graphs)),
    ])

    attack_auc = float(roc_auc_score(y_true, scores))

    # TPR at fixed low FPR thresholds. This is the headline privacy metric:
    # overall AUC can be misleading because the attacker mostly cares about
    # high-confidence identifications. Following Carlini et al., we report
    # TPR at FPR = 0.1% and 1%.
    fpr, tpr, _ = roc_curve(y_true, scores)
    tpr_at_001 = float(np.interp(0.001, fpr, tpr))
    tpr_at_01 = float(np.interp(0.01, fpr, tpr))

    return LiRAResult(
        attack_auc=attack_auc,
        tpr_at_fpr_001=tpr_at_001,
        tpr_at_fpr_01=tpr_at_01,
        num_shadow_models=num_shadow_models,
        num_queries=n_queries,
        mu_in_mean=float(mu_in.mean()),
        mu_out_mean=float(mu_out.mean()),
        sigma_in_mean=float(sigma_in.mean()),
        sigma_out_mean=float(sigma_out.mean()),
    )
