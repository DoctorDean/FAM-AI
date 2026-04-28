"""Flower server-side aggregation strategy for ADMET federation.

We use FedAvg (the standard weighted-average baseline) but wrap it with a small
amount of metric aggregation logic so that round-by-round MAE/AUC across clients
is logged to the console — useful for sanity-checking convergence at a glance.

We also subclass FedAvg with a `latest_parameters` field that captures the most
recent aggregated parameters after every round. This is needed because Flower's
default FedAvg returns history but not final weights; we want to checkpoint and
later run MIA against the final aggregated model.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.strategy import FedAvg


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate per-client metrics into a population-weighted average.

    Each entry is (num_examples, {metric_name: value}). We weight by the number
    of examples each client used so that small partners don't dominate (and
    large partners don't get drowned out).
    """
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    all_keys: set[str] = set()
    for _, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)) and k != "client_id":
                all_keys.add(k)

    for key in all_keys:
        total = 0.0
        weight = 0
        for num_examples, m in metrics:
            if key in m and isinstance(m[key], (int, float)):
                total += float(m[key]) * num_examples
                weight += num_examples
        if weight > 0:
            aggregated[key] = total / weight

    return aggregated


class CheckpointingFedAvg(FedAvg):
    """FedAvg that retains the most recently aggregated parameters.

    After every fit round, `latest_parameters` is updated to the freshly
    averaged global model. The driver script reads this at the end of the run
    to save a checkpoint usable by downstream evaluation and the MIA.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = kwargs.get("initial_parameters")

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_params is not None:
            self.latest_parameters = aggregated_params
        return aggregated_params, aggregated_metrics


def make_strategy(
    initial_parameters: fl.common.Parameters,
    fraction_fit: float = 1.0,
    fraction_eval: float = 1.0,
    min_clients: int = 2,
    on_fit_config: Callable[[int], dict[str, Any]] | None = None,
) -> CheckpointingFedAvg:
    """Build a checkpointing FedAvg strategy with sensible defaults for our small simulation.

    Args:
        initial_parameters: Starting model weights (from a freshly initialised
            `GINPredictor`). Sharing initial parameters across clients is what
            keeps everyone aligned in the first round.
        fraction_fit: Fraction of clients to sample per training round.
        fraction_eval: Fraction of clients to sample per evaluation round.
        min_clients: Minimum number of clients required for fit / eval. With
            only 2-3 partners you almost always want all of them participating.
        on_fit_config: Optional callable producing per-round config sent to
            clients (e.g. to vary `local_epochs` over time).
    """
    return CheckpointingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
    )
