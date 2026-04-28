"""Membership inference attack (MIA) against the trained ADMET model.

We implement a simplified Shokri-style shadow-model attack. The intuition:

  1. The target model was trained on some private set D_train. We want to
     decide, for a query molecule x, whether x was in D_train.
  2. We don't have access to D_train, but we have access to a population
     distribution (other ADMET data) and we know the target model's
     architecture and training recipe.
  3. So: train K *shadow models* on disjoint subsets of population data we DO
     control (so we know each one's true membership labels). For each shadow
     model and each molecule, record the model's loss on that molecule. This
     gives us labelled (loss, member?) examples.
  4. Train a small attack classifier on those examples. Apply it to the target
     model's loss on a held-out query set to recover membership signal.

A higher attack AUC means more privacy leakage. AUC = 0.5 is the no-leakage
baseline (random guessing). AUC > 0.5 means an adversary can do better than
chance — even 0.55 is meaningful in a regulated context.

This is intentionally a *simple* attack — black-box, single-feature (loss),
small shadow-model count. A determined adversary with white-box access to
the gradients or intermediate activations would do considerably more damage.
The point is to show that vanilla federated learning leaks, not to bound the
worst case.

References:
  Shokri, Stronati, Song, Shmatikov. "Membership Inference Attacks Against
  Machine Learning Models." IEEE S&P 2017.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from src.models import GINPredictor, evaluate, train_one_epoch


@dataclass
class MIAResult:
    """Result of a membership inference attack run."""

    attack_auc: float
    target_train_loss_mean: float
    target_nonmember_loss_mean: float
    num_shadow_models: int
    num_attack_train: int
    num_attack_test: int


def _per_example_losses(
    model: nn.Module,
    graphs: list,
    device: torch.device,
    is_regression: bool,
    batch_size: int = 32,
) -> np.ndarray:
    """Return per-example loss for each graph in `graphs`.

    For regression we use squared error, for classification BCE. We keep these
    as raw per-example values (no reduction) because the attack classifier
    treats each loss as one feature of one labelled example.
    """
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


def _train_shadow_model(
    train_graphs: list,
    node_feature_dim: int,
    is_regression: bool,
    config: dict,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    """Train a shadow model with the same architecture as the target.

    The shadow model's role is to mimic the target's training dynamics on data
    where we know the membership labels. Architecture must match so that loss
    distributions are comparable.
    """
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


def run_membership_inference(
    target_model: nn.Module,
    target_train_graphs: list,
    target_nonmember_graphs: list,
    shadow_pool_graphs: list,
    node_feature_dim: int,
    is_regression: bool,
    config: dict,
    num_shadow_models: int = 5,
    shadow_epochs: int = 50,
    attack_model_type: str = "gradient_boosting",
    seed: int = 42,
    device: torch.device | None = None,
) -> MIAResult:
    """Run a Shokri-style MIA against `target_model`.

    Args:
        target_model: The trained model under attack.
        target_train_graphs: Molecules that WERE in the target's training set.
            For each one, the attack should ideally output "member".
        target_nonmember_graphs: Molecules that were NOT in the target's
            training set. The attack should ideally output "non-member".
            Best drawn from the same distribution as target_train_graphs (e.g.
            the held-out test set).
        shadow_pool_graphs: A separate pool of molecules used to train shadow
            models. Should NOT overlap with target_train_graphs or
            target_nonmember_graphs to avoid leakage of the evaluation.
        node_feature_dim: Atom feature dimension (must match the target).
        is_regression: Task type, determines loss function.
        config: Top-level config dict (used for model/training hyperparameters).
        num_shadow_models: How many shadow models to train. More = better
            attack signal but linearly more compute.
        shadow_epochs: Epochs to train each shadow model. Should be roughly
            comparable to the target's effective training (rounds * local_epochs
            for the federated case).
        attack_model_type: 'gradient_boosting' (default, generally stronger) or
            'logistic_regression' (faster, more interpretable).
        seed: RNG seed.
        device: Torch device override.

    Returns:
        MIAResult with the attack's ROC-AUC and supporting diagnostics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    # --- Step 1: Train shadow models, collect (loss, is_member) pairs --------
    if len(shadow_pool_graphs) < 2 * num_shadow_models:
        raise ValueError(
            f"Shadow pool has {len(shadow_pool_graphs)} graphs but we need at least "
            f"{2 * num_shadow_models} to give each shadow model a non-trivial in/out split."
        )

    attack_features: list[float] = []
    attack_labels: list[int] = []  # 1 = member, 0 = non-member

    pool_size = len(shadow_pool_graphs)
    indices = np.arange(pool_size)

    for shadow_i in range(num_shadow_models):
        # Half the pool in, half out — the standard symmetric setup.
        rng.shuffle(indices)
        split = pool_size // 2
        in_idx, out_idx = indices[:split], indices[split:]
        in_graphs = [shadow_pool_graphs[i] for i in in_idx]
        out_graphs = [shadow_pool_graphs[i] for i in out_idx]

        shadow_model = _train_shadow_model(
            in_graphs, node_feature_dim, is_regression, config, device, shadow_epochs
        )

        in_losses = _per_example_losses(shadow_model, in_graphs, device, is_regression)
        out_losses = _per_example_losses(shadow_model, out_graphs, device, is_regression)

        attack_features.extend(in_losses.tolist())
        attack_labels.extend([1] * len(in_losses))
        attack_features.extend(out_losses.tolist())
        attack_labels.extend([0] * len(out_losses))

    X_train = np.array(attack_features).reshape(-1, 1)
    y_train = np.array(attack_labels)

    # --- Step 2: Fit the attack classifier on shadow data --------------------
    if attack_model_type == "gradient_boosting":
        attack_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
    elif attack_model_type == "logistic_regression":
        attack_clf = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"Unknown attack_model_type '{attack_model_type}'")
    attack_clf.fit(X_train, y_train)

    # --- Step 3: Apply attack to the actual target -------------------------
    target_member_losses = _per_example_losses(
        target_model, target_train_graphs, device, is_regression
    )
    target_nonmember_losses = _per_example_losses(
        target_model, target_nonmember_graphs, device, is_regression
    )

    X_test = np.concatenate([target_member_losses, target_nonmember_losses]).reshape(-1, 1)
    y_test = np.concatenate(
        [np.ones(len(target_member_losses)), np.zeros(len(target_nonmember_losses))]
    )
    # Use predicted probabilities for AUC, not hard predictions.
    y_score = attack_clf.predict_proba(X_test)[:, 1]
    attack_auc = roc_auc_score(y_test, y_score)

    return MIAResult(
        attack_auc=float(attack_auc),
        target_train_loss_mean=float(target_member_losses.mean()),
        target_nonmember_loss_mean=float(target_nonmember_losses.mean()),
        num_shadow_models=num_shadow_models,
        num_attack_train=len(y_train),
        num_attack_test=len(y_test),
    )
