"""Smoke tests for the GIN model and parameter conversion utilities."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data  # noqa: E402

from src.models.gnn import GINPredictor, get_model_parameters, set_model_parameters  # noqa: E402


def _toy_batch():
    g1 = Data(
        x=torch.randn(3, 8),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        y=torch.tensor([1.0]),
    )
    g2 = Data(
        x=torch.randn(4, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        y=torch.tensor([0.0]),
    )
    return Batch.from_data_list([g1, g2])


def test_gin_forward_shape():
    model = GINPredictor(node_feature_dim=8, hidden_dim=16, num_layers=2)
    batch = _toy_batch()
    out = model(batch.x, batch.edge_index, batch.batch)
    # One scalar per graph in the batch.
    assert out.shape == (2,)


def test_get_set_parameters_roundtrip():
    model_a = GINPredictor(node_feature_dim=8, hidden_dim=16, num_layers=2)
    model_b = GINPredictor(node_feature_dim=8, hidden_dim=16, num_layers=2)

    params_a = get_model_parameters(model_a)
    set_model_parameters(model_b, params_a)
    params_b = get_model_parameters(model_b)

    assert len(params_a) == len(params_b)
    for pa, pb in zip(params_a, params_b):
        np.testing.assert_array_equal(pa, pb)


def test_get_embeddings_shape():
    model = GINPredictor(node_feature_dim=8, hidden_dim=16, num_layers=2)
    batch = _toy_batch()
    emb = model.get_embeddings(batch.x, batch.edge_index, batch.batch)
    assert emb.shape == (2, 16)
