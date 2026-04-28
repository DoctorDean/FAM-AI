"""Local training and evaluation loops, shared between Flower clients and baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch_geometric.loader import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_regression: bool,
) -> float:
    """Train the model for one epoch. Returns mean loss."""
    model.train()
    loss_fn = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch.batch)
        targets = batch.y.float()
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_examples += batch.num_graphs

    return total_loss / max(n_examples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_regression: bool,
) -> dict[str, float]:
    """Evaluate the model. Returns loss and the task-appropriate primary metric.

    For regression we report MAE (which is what the TDC leaderboard uses for
    most regression tasks). For classification we report ROC-AUC. Loss is
    always reported as well so federated training curves stay comparable
    across clients.
    """
    model.eval()
    loss_fn = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_examples = 0
    all_preds: list[float] = []
    all_targets: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.batch)
        targets = batch.y.float()
        loss = loss_fn(preds, targets)
        total_loss += loss.item() * batch.num_graphs
        n_examples += batch.num_graphs

        if is_regression:
            all_preds.extend(preds.cpu().tolist())
        else:
            all_preds.extend(torch.sigmoid(preds).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    metrics = {"loss": total_loss / max(n_examples, 1)}
    if is_regression:
        metrics["mae"] = mean_absolute_error(all_targets, all_preds)
    else:
        # ROC-AUC undefined when only one class present in eval set — fall back to NaN
        # so callers can detect the issue rather than silently treating 0.5 as real.
        try:
            metrics["auc"] = roc_auc_score(all_targets, all_preds)
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics
