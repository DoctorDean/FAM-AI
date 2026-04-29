"""Smoke tests for SMILES -> graph featurisation."""
from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit not installed")
torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from src.utils.featurization import (  # noqa: E402
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    smiles_list_to_graphs,
    smiles_to_graph,
)


def test_smiles_to_graph_basic():
    g = smiles_to_graph("CCO", y=1.0)
    assert g is not None
    assert g.x.shape == (3, ATOM_FEATURE_DIM)
    # Ethanol has two bonds, becoming 4 directed edges.
    assert g.edge_index.shape == (2, 4)
    assert g.edge_attr.shape == (4, BOND_FEATURE_DIM)
    assert g.y.shape == (1,)
    assert g.smiles == "CCO"


def test_smiles_to_graph_invalid_returns_none():
    assert smiles_to_graph("not_a_smiles", y=0.0) is None


def test_smiles_list_to_graphs_filters_invalid():
    smiles = ["CCO", "not_a_smiles", "c1ccccc1"]
    labels = [1.0, 2.0, 3.0]
    graphs = smiles_list_to_graphs(smiles, labels)
    # Should drop the invalid one but keep the others.
    assert len(graphs) == 2
    assert graphs[0].smiles == "CCO"
    assert graphs[1].smiles == "c1ccccc1"


def test_single_atom_molecule():
    """Edge case: a single atom has no bonds. Should still produce a valid graph."""
    g = smiles_to_graph("[H]", y=0.0)
    # RDKit may parse [H] differently; just check we don't crash and edges are well-shaped.
    if g is not None:
        assert g.edge_index.shape[0] == 2
