"""Tests for the RDP privacy accountant."""
from __future__ import annotations

import math

import pytest

dp_accounting = pytest.importorskip("dp_accounting", reason="dp-accounting not installed")

from src.privacy.accountant import (  # noqa: E402
    PrivacyBudget,
    compute_dp_fedavg_epsilon,
    find_noise_multiplier_for_target_epsilon,
)


def test_zero_noise_gives_infinite_epsilon():
    budget = compute_dp_fedavg_epsilon(noise_multiplier=0.0, num_rounds=5)
    assert math.isinf(budget.epsilon)
    assert budget.noise_multiplier == 0.0


def test_more_noise_means_smaller_epsilon():
    """Privacy budget is monotonic in noise multiplier."""
    eps_low_noise = compute_dp_fedavg_epsilon(0.5, num_rounds=10).epsilon
    eps_med_noise = compute_dp_fedavg_epsilon(1.0, num_rounds=10).epsilon
    eps_high_noise = compute_dp_fedavg_epsilon(2.0, num_rounds=10).epsilon
    assert eps_low_noise > eps_med_noise > eps_high_noise


def test_more_rounds_means_larger_epsilon():
    """Privacy budget grows (worsens) with composition."""
    eps_5 = compute_dp_fedavg_epsilon(1.0, num_rounds=5).epsilon
    eps_10 = compute_dp_fedavg_epsilon(1.0, num_rounds=10).epsilon
    eps_50 = compute_dp_fedavg_epsilon(1.0, num_rounds=50).epsilon
    assert eps_5 < eps_10 < eps_50


def test_subsampling_amplifies_privacy():
    """Lower sampling rate should give a stronger (lower epsilon) guarantee."""
    eps_full = compute_dp_fedavg_epsilon(1.0, num_rounds=10, sampling_rate=1.0).epsilon
    eps_half = compute_dp_fedavg_epsilon(1.0, num_rounds=10, sampling_rate=0.5).epsilon
    assert eps_half < eps_full


def test_inverse_search_recovers_target():
    """Searching for noise to hit eps=8, then computing eps from that noise,
    should give back something very close to 8."""
    target_eps = 8.0
    nm = find_noise_multiplier_for_target_epsilon(target_eps, num_rounds=10)
    achieved = compute_dp_fedavg_epsilon(nm, num_rounds=10).epsilon
    # Should be at or just below target.
    assert achieved <= target_eps
    assert achieved > target_eps - 0.5


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compute_dp_fedavg_epsilon(noise_multiplier=-0.1, num_rounds=10)
    with pytest.raises(ValueError):
        compute_dp_fedavg_epsilon(noise_multiplier=1.0, num_rounds=0)
    with pytest.raises(ValueError):
        compute_dp_fedavg_epsilon(noise_multiplier=1.0, num_rounds=10, target_delta=1.5)
    with pytest.raises(ValueError):
        compute_dp_fedavg_epsilon(noise_multiplier=1.0, num_rounds=10, sampling_rate=0)


def test_privacy_budget_str_formatting():
    """The __str__ should include all key fields without crashing."""
    b = PrivacyBudget(
        epsilon=8.0, delta=1e-5, noise_multiplier=2.0,
        num_rounds=10, sampling_rate=1.0,
    )
    s = str(b)
    assert "8" in s
    assert "1e-05" in s
    assert "2" in s
