"""Load ADMET datasets from the Therapeutics Data Commons (TDC) benchmark group."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# TDC tasks split into regression and classification. Used to pick the right loss/metric.
REGRESSION_TASKS = {
    "Caco2_Wang",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    "PPBR_AZ",
    "VDss_Lombardo",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "LD50_Zhu",
}

CLASSIFICATION_TASKS = {
    "HIA_Hou",
    "Pgp_Broccatelli",
    "Bioavailability_Ma",
    "BBB_Martins",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "CYP2C9_Substrate_CarbonMangels",
    "hERG",
    "AMES",
    "DILI",
    "Skin_Reaction",
    "Carcinogens_Lagunin",
    "ClinTox",
}


@dataclass
class ADMETData:
    """Holds train/valid/test splits for a TDC ADMET task.

    Attributes:
        task: TDC task name (e.g. 'Caco2_Wang').
        is_regression: Whether the task is regression (else binary classification).
        train: DataFrame with columns ['Drug_ID', 'Drug', 'Y'] where Drug is SMILES.
        valid: Same schema as train.
        test: Same schema as train.
    """

    task: str
    is_regression: bool
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_admet_task(task: str, seed: int = 42) -> ADMETData:
    """Load a TDC ADMET task using the official benchmark group split.

    The benchmark group provides a standard scaffold-aware split so results are
    comparable across runs/papers. We use seed=1 for the default benchmark split
    (TDC's convention) and let `seed` control downstream stochasticity.
    """
    if task not in REGRESSION_TASKS | CLASSIFICATION_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Expected one of regression {sorted(REGRESSION_TASKS)} "
            f"or classification {sorted(CLASSIFICATION_TASKS)}."
        )

    # Imported here so that simply importing this module doesn't trigger TDC's
    # network-fetch behaviour at import time.
    from tdc.benchmark_group import admet_group

    group = admet_group(path="data/tdc/")
    benchmark = group.get(task)
    train_val, test = benchmark["train_val"], benchmark["test"]

    # TDC provides a `get_train_valid_split` helper for an inner split. Seed=1 is
    # the canonical benchmark seed; we keep that for comparability.
    train, valid = group.get_train_valid_split(benchmark=task, split_type="default", seed=1)

    is_regression = task in REGRESSION_TASKS

    return ADMETData(
        task=task,
        is_regression=is_regression,
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def partition_data(
    data: pd.DataFrame,
    num_clients: int,
    strategy: str = "random",
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Split a DataFrame across `num_clients` simulated partners.

    Args:
        data: DataFrame with at least a 'Drug' (SMILES) column.
        num_clients: Number of partner shards to produce.
        strategy: 'random' for an i.i.d. split, or 'scaffold' for a Bemis-Murcko
            scaffold-based split that better simulates the realistic case where
            different pharma partners hold molecules from different chemical
            series (and where the federated benefit is largest).
        seed: RNG seed.

    Returns:
        List of `num_clients` DataFrames, disjoint, covering all of `data`.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if len(data) < num_clients:
        raise ValueError(f"Cannot split {len(data)} rows across {num_clients} clients")

    if strategy == "random":
        return _random_partition(data, num_clients, seed)
    elif strategy == "scaffold":
        return _scaffold_partition(data, num_clients, seed)
    else:
        raise ValueError(f"Unknown partition strategy '{strategy}'")


def _random_partition(data: pd.DataFrame, num_clients: int, seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shuffled = data.sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
    # Split by integer index ranges. (We don't use np.array_split on the DataFrame
    # itself because in some numpy/pandas combinations it converts to ndarray and
    # loses the DataFrame interface.)
    index_chunks = np.array_split(np.arange(len(shuffled)), num_clients)
    return [shuffled.iloc[chunk].reset_index(drop=True) for chunk in index_chunks]


def _scaffold_partition(data: pd.DataFrame, num_clients: int, seed: int) -> list[pd.DataFrame]:
    """Partition by Bemis-Murcko scaffold so partners hold different chemical series.

    Each scaffold (a canonical SMILES of the molecule's core ring system) is
    assigned to one client; molecules without a computable scaffold fall into a
    miscellaneous bucket distributed round-robin. This is harder than random
    splitting because the model must generalise across chemistries — exactly the
    situation where federation should help most.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    rng = np.random.default_rng(seed)
    scaffolds: dict[str, list[int]] = {}
    orphans: list[int] = []

    for idx, smiles in enumerate(data["Drug"].tolist()):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                orphans.append(idx)
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds.setdefault(scaffold or "__empty__", []).append(idx)
        except Exception:
            orphans.append(idx)

    # Largest scaffold groups first, distributed greedily to the smallest shard
    # to keep shards roughly balanced.
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)
    shards: list[list[int]] = [[] for _ in range(num_clients)]
    for group in scaffold_groups:
        smallest = min(range(num_clients), key=lambda i: len(shards[i]))
        shards[smallest].extend(group)

    # Distribute orphans round-robin (shuffled) so they don't all land on one shard.
    rng.shuffle(orphans)
    for i, idx in enumerate(orphans):
        shards[i % num_clients].append(idx)

    return [data.iloc[idx_list].reset_index(drop=True) for idx_list in shards]
