"""Tests for LiRA helper logic.

The full LiRA integration test (training shadow models) is too slow for CI.
We test the helper logic and trust the integration via the sweep script.
"""
from __future__ import annotations

import numpy as np
import pytest

# These imports are needed by the LiRA module; skip the whole test file
# gracefully if they're not present in the test environment.
torch = pytest.importorskip("torch", reason="torch not installed")
torch_geometric = pytest.importorskip("torch_geometric", reason="torch-geometric not installed")
scipy = pytest.importorskip("scipy", reason="scipy not installed")

from src.attacks.lira import _logit_transform, run_lira  # noqa: E402


def test_logit_transform_handles_zero():
    """Loss values at 0 should not produce -inf after the transform."""
    losses = np.array([0.0, 0.5, 1.0, 100.0])
    result = _logit_transform(losses)
    assert np.all(np.isfinite(result))
    assert np.all(np.diff(result) > 0)  # monotonic in input


def test_logit_transform_monotonic():
    losses = np.linspace(0, 100, 50)
    result = _logit_transform(losses)
    assert np.all(np.diff(result) > 0)


def test_lira_requires_minimum_shadow_models():
    """LiRA should refuse to run with too few shadow models."""
    with pytest.raises(ValueError, match="at least 8 shadow models"):
        run_lira(
            target_model=None,  # type: ignore[arg-type]
            target_train_graphs=[],
            target_nonmember_graphs=[],
            shadow_pool_graphs=[],
            node_feature_dim=10,
            is_regression=True,
            config={},
            num_shadow_models=4,
        )
