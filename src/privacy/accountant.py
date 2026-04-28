"""Renyi Differential Privacy (RDP) accounting for DP-FedAvg.

Converts our training hyperparameters (noise multiplier, number of rounds,
sampling rate) into a standard (epsilon, delta)-DP guarantee at the *client*
(user) level.

Why this matters:
  Our DP-FedAvg implementation adds Gaussian noise but has no opinion about
  what privacy budget that noise actually buys. A reader has to take it on
  faith that "more noise = more privacy" — but how much? The RDP accountant
  closes that gap by giving a concrete, defensible number.

Approach:
  We use Google's `dp_accounting` library, which implements the tight RDP
  composition bounds (Wang et al. 2018, Mironov et al. 2019) used by both
  TensorFlow Privacy and TFF. We treat each round as a Gaussian mechanism
  applied to a sum of clipped client contributions:

      M(D) = sum_clipped(client_updates) + N(0, (z * C)^2 * I)

  where C is the clip norm and z is the noise multiplier. Sensitivity (the
  max change to M from swapping one client's data) is C, so the noise-to-
  sensitivity ratio is z. Composing this mechanism over T rounds gives the
  total RDP, which we then convert to (epsilon, delta)-DP.

Notes on what this does NOT account for:
  * **Subsampling amplification.** If the federation samples a random subset
    of clients per round (rather than using all of them), privacy is amplified
    by the sub-sampling factor. Our default config has fraction_fit=1.0
    (all clients participate every round) so this doesn't apply, but if you
    set fraction_fit < 1.0 you'd want to wrap the GaussianDpEvent in a
    PoissonSampledDpEvent. We expose a `sampling_rate` parameter for this.
  * **Multiple local epochs.** Each round is one application of the Gaussian
    mechanism regardless of how many local epochs each client runs internally,
    because what gets clipped and noised is the round-final update, not the
    per-batch gradients. So local_epochs doesn't appear in the privacy budget
    (only in the utility cost).
  * **The honest-but-curious server assumption.** The DP guarantee here
    protects against an adversary who only sees the *aggregated* model output.
    The server itself sees individual clipped client updates and would need
    secure aggregation to be excluded from the threat model.
  * **Cross-silo limitations.** With only 3 clients and full participation,
    formal user-level DP guarantees with reasonable epsilon are hard to obtain
    — there's no sub-sampling amplification and the population is tiny. This
    is a fundamental property of cross-silo FL, not a flaw in the accountant.
    For a 3-silo deployment to get e.g. eps=8 protection you'd need either
    a much larger noise multiplier (significant utility cost) or a different
    threat model (e.g. silo-level rather than user-level DP).

References:
  Mironov. "Renyi Differential Privacy." CSF 2017.
  Mironov, Talwar, Zhang. "R'enyi Differential Privacy of the Sampled Gaussian Mechanism." 2019.
  https://github.com/google/differential-privacy/tree/main/python/dp_accounting
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrivacyBudget:
    """The (epsilon, delta) privacy guarantee for a DP-FedAvg run."""

    epsilon: float
    delta: float
    noise_multiplier: float
    num_rounds: int
    sampling_rate: float
    accountant: str = "RDP"

    def __str__(self) -> str:
        return (
            f"({self.epsilon:.3f}, {self.delta:.0e})-DP "
            f"[noise_multiplier={self.noise_multiplier}, "
            f"rounds={self.num_rounds}, sampling_rate={self.sampling_rate}, "
            f"accountant={self.accountant}]"
        )


def compute_dp_fedavg_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    target_delta: float = 1e-5,
    sampling_rate: float = 1.0,
) -> PrivacyBudget:
    """Compute the (epsilon, delta)-DP guarantee for a DP-FedAvg training run.

    Args:
        noise_multiplier: Ratio of Gaussian noise stddev to clip norm. (This is
            our DPFedAvg.noise_multiplier directly.) Must be > 0; for noise=0
            the mechanism has no DP guarantee and we return epsilon=infinity.
        num_rounds: Number of FedAvg rounds the noise was applied across.
        target_delta: The delta in (epsilon, delta)-DP. Standard practice is
            to choose delta < 1/N where N is the dataset size; for a few-
            thousand-molecule ADMET task, 1e-5 is a reasonable default.
        sampling_rate: Probability that any given client participates in a
            given round. For our default cross-silo setup with all clients
            participating every round this is 1.0. If you set fraction_fit
            < 1.0 in the federation config, set this to that same value.

    Returns:
        PrivacyBudget with the computed epsilon.

    Raises:
        ImportError: If `dp_accounting` is not installed.
        ValueError: If parameters are out of range.
    """
    if noise_multiplier < 0:
        raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
    if not 0 < target_delta < 1:
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
    if not 0 < sampling_rate <= 1:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate}")

    if noise_multiplier == 0:
        # No noise = no DP guarantee. Returning infinity is the honest answer
        # rather than e.g. silently substituting a finite number.
        return PrivacyBudget(
            epsilon=float("inf"),
            delta=target_delta,
            noise_multiplier=0.0,
            num_rounds=num_rounds,
            sampling_rate=sampling_rate,
        )

    try:
        import dp_accounting
    except ImportError as e:
        raise ImportError(
            "dp_accounting is required for privacy accounting. "
            "Install with: pip install dp-accounting"
        ) from e

    # Build the composed event: T applications of the (sampled) Gaussian mechanism.
    gaussian_event = dp_accounting.GaussianDpEvent(noise_multiplier)
    if sampling_rate < 1.0:
        per_round_event = dp_accounting.PoissonSampledDpEvent(sampling_rate, gaussian_event)
    else:
        per_round_event = gaussian_event
    composed = dp_accounting.SelfComposedDpEvent(per_round_event, num_rounds)

    accountant = dp_accounting.rdp.RdpAccountant()
    accountant.compose(composed)
    epsilon = accountant.get_epsilon(target_delta=target_delta)

    return PrivacyBudget(
        epsilon=float(epsilon),
        delta=target_delta,
        noise_multiplier=noise_multiplier,
        num_rounds=num_rounds,
        sampling_rate=sampling_rate,
    )


def find_noise_multiplier_for_target_epsilon(
    target_epsilon: float,
    num_rounds: int,
    target_delta: float = 1e-5,
    sampling_rate: float = 1.0,
    search_lo: float = 0.1,
    search_hi: float = 100.0,
    tolerance: float = 0.01,
) -> float:
    """Binary-search the smallest noise_multiplier achieving epsilon <= target.

    Useful for the inverse question: "I want eps=8 protection — what noise
    multiplier should I use for my 10-round, full-participation training?"

    Returns:
        Noise multiplier (float). Note that for very small target epsilons
        with large num_rounds, the required noise may be impractically large.
    """
    if target_epsilon <= 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")

    # Sanity check the bracket.
    eps_lo = compute_dp_fedavg_epsilon(search_hi, num_rounds, target_delta, sampling_rate).epsilon
    if eps_lo > target_epsilon:
        raise ValueError(
            f"Even noise_multiplier={search_hi} only achieves eps={eps_lo:.2f}, "
            f"which exceeds target {target_epsilon}. Increase search_hi."
        )
    eps_hi = compute_dp_fedavg_epsilon(search_lo, num_rounds, target_delta, sampling_rate).epsilon
    if eps_hi < target_epsilon:
        # Already meets the target with the minimum noise.
        return search_lo

    # Binary search. Higher noise = lower epsilon, so the function is monotonic
    # decreasing — a standard bisection works.
    lo, hi = search_lo, search_hi
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        eps = compute_dp_fedavg_epsilon(mid, num_rounds, target_delta, sampling_rate).epsilon
        if eps > target_epsilon:
            lo = mid  # need more noise
        else:
            hi = mid  # could use less noise
    return hi
