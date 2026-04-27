"""
data_pipeline.py
Converts Tox21 / ClinTox SMILES → PyTorch Geometric Data objects.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import deepchem as dc
from sklearn.utils.class_weight import compute_class_weight


# ── Atom / Bond featurisers ──────────────────────────────────────────────────

HYBRIDIZATION = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

BOND_TYPE = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]


def atom_features(atom):
    hyb = [int(atom.GetHybridization() == h) for h in HYBRIDIZATION]
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ] + hyb  # dim = 10


def bond_features(bond):
    bt = bond.GetBondType()
    return [int(bt == b) for b in BOND_TYPE]  # dim = 4


# ── SMILES → PyG Data ────────────────────────────────────────────────────────

def smiles_to_data(smiles, labels):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)

    y = torch.tensor(labels, dtype=torch.float).reshape(-1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_dataset(name="tox21"):
    """Returns list of PyG Data objects and per-task class weights."""
    import pandas as pd
    
    data_dir = dc.utils.data_utils.get_data_dir()
    
    if name == "tox21":
        url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
        dataset_file = dc.utils.data_utils.download_url(url, data_dir)
        df = pd.read_csv(dataset_file)
        tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    elif name == "clintox":
        url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
        dataset_file = dc.utils.data_utils.download_url(url, data_dir)
        df = pd.read_csv(dataset_file)
        tasks = ['FDA_APPROVED', 'CT_TOX']
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data_list = []
    for _, row in df.iterrows():
        smiles = row['smiles']
        labels = row[tasks].values
        d = smiles_to_data(smiles, labels)
        if d is not None:
            data_list.append(d)

    # Per-task class weights (ignore NaN)
    y_all = np.array([d.y.numpy() for d in data_list])  # (N, T)
    num_tasks = y_all.shape[1]
    class_weights = []
    for t in range(num_tasks):
        col = y_all[:, t]
        valid = col[~np.isnan(col)].astype(int)
        classes = np.unique(valid)
        if len(classes) < 2:
            class_weights.append(torch.tensor([1.0, 1.0]))
            continue
        w = compute_class_weight("balanced", classes=classes, y=valid)
        class_weights.append(torch.tensor(w, dtype=torch.float))

    return data_list, class_weights, tasks


if __name__ == "__main__":
    for name in ("tox21", "clintox"):
        data, cw, tasks = load_dataset(name)
        print(f"{name}: {len(data)} molecules, {len(tasks)} tasks")
        print(f"  Sample node shape: {data[0].x.shape}")
        print(f"  Sample edge shape: {data[0].edge_index.shape}")
