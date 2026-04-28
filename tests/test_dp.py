"""Tests for differential privacy mechanism (clipping + noise) on FedAvg updates."""
from __future__ import annotations

import numpy as np
import pytest

from src.server.dp_strategy import _clip_update, _l2_norm_of_arrays, _scale_arrays


def test_clip_update_below_norm_is_unchanged():
    update = [np.array([0.1, 0.2]), np.array([[0.3, 0.4]])]
    clipped = _clip_update(update, clip_norm=10.0)
    for a, b in zip(update, clipped):
        np.testing.assert_array_equal(a, b)


def test_clip_update_above_norm_is_scaled():
    # An update with L2 norm = 5
    update = [np.array([3.0, 4.0])]
    assert _l2_norm_of_arrays(update) == pytest.approx(5.0)

    clipped = _clip_update(update, clip_norm=1.0)
    # After clipping, L2 norm should be exactly 1.
    assert _l2_norm_of_arrays(clipped) == pytest.approx(1.0, rel=1e-6)
    # Direction should be preserved.
    np.testing.assert_allclose(clipped[0], np.array([0.6, 0.8]), atol=1e-6)


def test_clip_update_zero_update_returns_zero():
    update = [np.zeros(5), np.zeros((2, 3))]
    clipped = _clip_update(update, clip_norm=1.0)
    for a in clipped:
        np.testing.assert_array_equal(a, np.zeros_like(a))


def test_l2_norm_treats_arrays_as_one_vector():
    arrays = [np.array([3.0]), np.array([4.0])]
    # Single-vector norm of [3, 4] = 5.
    assert _l2_norm_of_arrays(arrays) == pytest.approx(5.0)


def test_scale_arrays_preserves_shapes():
    arrays = [np.ones(3), np.ones((2, 2))]
    scaled = _scale_arrays(arrays, factor=0.5)
    assert scaled[0].shape == (3,)
    assert scaled[1].shape == (2, 2)
    np.testing.assert_array_equal(scaled[0], np.full(3, 0.5))


def test_clip_update_invariant_to_array_partitioning():
    """Clipping should treat the update as a single flat vector regardless of how
    the parameters are split across array shards."""
    flat = np.array([3.0, 0.0, 4.0, 0.0])
    split_a = [flat]
    split_b = [flat[:2], flat[2:]]
    norm_target = 1.0

    clipped_a = _clip_update(split_a, clip_norm=norm_target)
    clipped_b = _clip_update(split_b, clip_norm=norm_target)

    flat_a = np.concatenate([a.ravel() for a in clipped_a])
    flat_b = np.concatenate([a.ravel() for a in clipped_b])
    np.testing.assert_allclose(flat_a, flat_b, atol=1e-6)
