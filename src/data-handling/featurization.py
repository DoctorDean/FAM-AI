"""Convert SMILES strings to PyG graphs with simple, well-known atom/bond features."""
from __future__ import annotations

import torch
from torch_geometric.data import Data

# Atom feature vocabularies. Kept small and standard — these are the features
# used in most introductory GNN-for-chemistry tutorials and are sufficient for
# small ADMET tasks. Production work would use richer featurisers (e.g. OGB's).
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "OTHER"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def _one_hot(value: str, choices: list[str]) -> list[int]:
    """One-hot encode `value`, using the last bucket as a catch-all for unseens."""
    if value not in choices:
        value = choices[-1]
    return [int(c == value) for c in choices]


def atom_features(atom) -> list[float]:
    """Return a fixed-length feature vector for an RDKit atom."""
    return (
        _one_hot(atom.GetSymbol(), ATOM_TYPES)
        + _one_hot(str(atom.GetHybridization()).split(".")[-1], HYBRIDIZATIONS)
        + [
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
    )


def bond_features(bond) -> list[float]:
    """Return a fixed-length feature vector for an RDKit bond."""
    return _one_hot(str(bond.GetBondType()).split(".")[-1], BOND_TYPES) + [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


# Dimensions consumers need to know to size the GNN input layers.
ATOM_FEATURE_DIM = len(ATOM_TYPES) + len(HYBRIDIZATIONS) + 5
BOND_FEATURE_DIM = len(BOND_TYPES) + 2


def smiles_to_graph(smiles: str, y: float | int) -> Data | None:
    """Convert a SMILES string + label into a PyG `Data` object.

    Returns None for SMILES that RDKit cannot parse or that have no atoms;
    callers should filter these out. We keep failures silent and observable
    via the return value rather than raising, because TDC datasets occasionally
    contain a few malformed entries and we don't want one bad row to halt training.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        # Add both directions — PyG expects an undirected graph as two directed edges.
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feats, feats]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Single-atom molecules (rare but possible).
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float),
        smiles=smiles,
    )


def smiles_list_to_graphs(smiles: list[str], labels: list[float]) -> list[Data]:
    """Convert parallel lists of SMILES and labels into a list of PyG `Data`.

    Silently drops any SMILES that fail to parse. Returns the surviving graphs.
    """
    graphs = []
    for s, y in zip(smiles, labels):
        g = smiles_to_graph(s, y)
        if g is not None:
            graphs.append(g)
    return graphs
