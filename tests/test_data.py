"""Smoke tests for the data pipeline.

These tests don't hit TDC's network — they construct small synthetic DataFrames
so they can run on CI without external dependencies. The tests that DO need
RDKit are marked and skipped if RDKit isn't installed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import _random_partition, _scaffold_partition


def _toy_df(n: int = 30) -> pd.DataFrame:
    """Build a small DataFrame of toy SMILES for partitioning tests."""
    smiles_pool = [
        "CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O",
        "CCOC", "CCS", "C1CCCCC1", "c1ccncc1", "CC(C)O",
    ]
    drugs = [smiles_pool[i % len(smiles_pool)] for i in range(n)]
    return pd.DataFrame({"Drug_ID": range(n), "Drug": drugs, "Y": [float(i) for i in range(n)]})


def test_random_partition_covers_all_rows():
    df = _toy_df(30)
    shards = _random_partition(df, num_clients=3, seed=42)
    assert len(shards) == 3
    total = sum(len(s) for s in shards)
    assert total == 30
    # Disjoint:
    all_ids = set()
    for s in shards:
        ids = set(s["Drug_ID"].tolist())
        assert ids.isdisjoint(all_ids), "Shards must be disjoint"
        all_ids.update(ids)


def test_random_partition_balanced():
    df = _toy_df(30)
    shards = _random_partition(df, num_clients=3, seed=42)
    sizes = [len(s) for s in shards]
    # Balanced to within 1 row.
    assert max(sizes) - min(sizes) <= 1


def test_random_partition_deterministic():
    df = _toy_df(30)
    a = _random_partition(df, num_clients=3, seed=42)
    b = _random_partition(df, num_clients=3, seed=42)
    for sa, sb in zip(a, b):
        assert sa["Drug_ID"].tolist() == sb["Drug_ID"].tolist()


@pytest.mark.skipif(
    pytest.importorskip("rdkit", reason="RDKit not installed") is None,
    reason="RDKit not installed",
)
def test_scaffold_partition_covers_all_rows():
    df = _toy_df(30)
    shards = _scaffold_partition(df, num_clients=3, seed=42)
    total = sum(len(s) for s in shards)
    assert total == 30
