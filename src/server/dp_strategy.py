"""Differentially private FedAvg via update clipping + Gaussian noise.

This implements *central* (server-side) DP at the user (client) level: each
client's update is L2-clipped to a fixed norm, the clipped updates are averaged
on the server, and Gaussian noise calibrated to the clip norm is added to the
average before it becomes the new global model. This is the standard
"DP-FedAvg" recipe from McMahan et al. (2018), "Learning Differentially Private
Recurrent Language Models".

Important caveats for this demo:

  * The privacy guarantee is *central* DP — i.e. it protects against an
    adversary who sees the final aggregated model, but the server still sees
    individual clipped updates in the clear. *Local* DP or *secure aggregation*
    would close that gap.
  * We do not run a tight privacy accountant here. The (epsilon, delta) bound
    you'd report for a real deployment depends on the noise multiplier, the
    number of rounds, the sampling probability, and the user-population size
    via something like the Moments Accountant or RDP accountant (e.g. Opacus,
    tensorflow-privacy). For a 2-3 client demo with all clients participating
    every round, formal user-level DP guarantees are very weak; the value of
    the noise here is mostly *empirical* — it should make the membership
    inference attack harder.
  * Treat this implementation as pedagogical. For real work, use a vetted
    library (Opacus for client-side gradient DP, or one of the federated-DP
    libraries for the server-side recipe).

Reference:
  McMahan, Ramage, Talwar, Zhang. "Learning Differentially Private Recurrent
  Language Models." ICLR 2018. https://arxiv.org/abs/1710.06963
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.server.strategy import CheckpointingFedAvg, weighted_average


def _flatten(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of arrays into a single 1-D vector."""
    return np.concatenate([a.ravel() for a in arrays])


def _l2_norm_of_arrays(arrays: list[np.ndarray]) -> float:
    """L2 norm across a list of arrays treated as one big vector."""
    return float(np.linalg.norm(_flatten(arrays)))


def _scale_arrays(arrays: list[np.ndarray], factor: float) -> list[np.ndarray]:
    """Multiply every array in the list by `factor`."""
    return [a * factor for a in arrays]


def _clip_update(update: list[np.ndarray], clip_norm: float) -> list[np.ndarray]:
    """Clip the L2 norm of an update (across all parameters) to `clip_norm`.

    If the update's norm is already below the bound, return it unchanged.
    Otherwise scale it so the norm equals `clip_norm`. This bounds the
    sensitivity of the per-client contribution, which is the prerequisite for
    calibrating Gaussian noise.
    """
    norm = _l2_norm_of_arrays(update)
    if norm <= clip_norm or norm == 0.0:
        return update
    return _scale_arrays(update, clip_norm / norm)


class DPFedAvg(CheckpointingFedAvg):
    """FedAvg with per-client update clipping and Gaussian noise on the average.

    Args:
        clip_norm: L2 bound applied to each client's parameter update (the
            difference between their post-training weights and the global
            weights they received). Smaller = more privacy, more bias.
        noise_multiplier: Gaussian noise standard deviation, in units of
            `clip_norm / num_sampled_clients`. So a noise_multiplier of 1.0
            adds noise with std == clip_norm/num_clients to the average update.
        **kwargs: Passed through to FedAvg.

    Notes:
        Setting `noise_multiplier=0.0` gives you plain clipping with no noise —
        useful as an ablation to separate the effect of clipping (which alone
        provides some empirical robustness) from the effect of noise.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0, got {clip_norm}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        # Cache the most recent global parameters so we can compute updates
        # (post-training weights minus pre-training weights).
        self._previous_global: list[np.ndarray] | None = (
            parameters_to_ndarrays(kwargs["initial_parameters"])
            if "initial_parameters" in kwargs and kwargs["initial_parameters"] is not None
            else None
        )

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list,
    ) -> tuple[Parameters | None, dict[str, Any]]:
        if not results:
            return None, {}
        if self._previous_global is None:
            self._previous_global = parameters_to_ndarrays(results[0][1].parameters)

        prev = self._previous_global
        n_clients = len(results)

        # Compute and clip each client's update (delta from prev).
        clipped_updates: list[list[np.ndarray]] = []
        total_examples = 0
        for client_proxy, fit_res in results:
            new_params = parameters_to_ndarrays(fit_res.parameters)
            update = [n - p for n, p in zip(new_params, prev)]
            clipped = _clip_update(update, self.clip_norm)
            clipped_updates.append(clipped)
            total_examples += fit_res.num_examples

        # Sum clipped updates element-wise.
        summed_update = [np.zeros_like(arr) for arr in prev]
        for client_update in clipped_updates:
            for i, arr in enumerate(client_update):
                summed_update[i] += arr

        # Add Gaussian noise to the SUM of clipped updates with std = sigma * C.
        if self.noise_multiplier > 0:
            noise_std = self.noise_multiplier * self.clip_norm
            summed_update = [
                arr + np.random.normal(0.0, noise_std, size=arr.shape).astype(arr.dtype)
                for arr in summed_update
            ]

        # Average to get the noisy mean update, then apply to previous global.
        averaged_update = [arr / n_clients for arr in summed_update]
        new_global = [p + u for p, u in zip(prev, averaged_update)]

        # Build aggregated metrics the same way the parent class would.
        aggregated_metrics: dict[str, Any] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)

        aggregated_params = ndarrays_to_parameters(new_global)
        self._previous_global = new_global
        self.latest_parameters = aggregated_params
        return aggregated_params, aggregated_metrics


def make_dp_strategy(
    initial_parameters: fl.common.Parameters,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    fraction_fit: float = 1.0,
    fraction_eval: float = 1.0,
    min_clients: int = 2,
    on_fit_config: Callable[[int], dict[str, Any]] | None = None,
) -> DPFedAvg:
    """Build a DP-FedAvg strategy."""
    return DPFedAvg(
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
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
