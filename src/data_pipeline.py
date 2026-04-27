"""
data_pipeline.py
Converts Tox21 / ClinTox SMILES → PyTorch Geometric Data objects.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
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
    
    # Skip molecules with no bonds (isolated atoms)
    if mol.GetNumBonds() == 0:
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
        # Validate edge indices
        assert edge_index.max() < x.size(0), "Invalid edge index detected"
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)

    y = torch.tensor(labels.astype(np.float32), dtype=torch.float).reshape(-1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ── Dataset loaders ──────────────────────────────────────────────────────────

def compute_class_weights(data_list):
    """Compute per-task class weights from a list of PyG Data objects."""
    y_all = np.array([d.y.numpy() for d in data_list])
    if y_all.ndim == 1:
        y_all = y_all.reshape(-1, 1)

    class_weights = []
    for t in range(y_all.shape[1]):
        col = y_all[:, t]
        valid = col[~np.isnan(col)].astype(int)
        classes = np.unique(valid)
        weights = torch.ones(2, dtype=torch.float)
        if len(classes) == 2:
            computed = compute_class_weight("balanced", classes=classes, y=valid)
            for cls, weight in zip(classes, computed):
                weights[int(cls)] = float(weight)
        class_weights.append(weights)

    return class_weights

def load_dataset(name="tox21"):
    """Returns list of PyG Data objects and per-task class weights."""
    import pandas as pd
    import requests
    from pathlib import Path
    
    # Dataset metadata
    DATASETS = {
        "tox21": {
            "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
            "tasks": ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                     'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
        },
        "clintox": {
            "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
            "tasks": ['FDA_APPROVED', 'CT_TOX'],
        },
    }
    
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
    
    config = DATASETS[name]
    
    # Setup cache directory
    cache_dir = Path.home() / ".cache" / "hqgnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}.csv.gz"
    
    # Download if not cached
    if not cache_file.exists():
        print(f"Downloading {name} dataset...")
        response = requests.get(config["url"], timeout=60)
        response.raise_for_status()
        cache_file.write_bytes(response.content)
        print(f"✓ Downloaded to {cache_file}")
    else:
        print(f"✓ Using cached dataset: {cache_file}")
    
    # Load CSV
    df = pd.read_csv(cache_file)
    tasks = config["tasks"]
    
    # Convert to PyG Data objects
    data_list = []
    failed = 0
    for _, row in df.iterrows():
        smiles = row['smiles']
        labels = row[tasks].values
        d = smiles_to_data(smiles, labels)
        if d is not None:
            data_list.append(d)
        else:
            failed += 1
    
    if failed > 0:
        print(f"⚠ Skipped {failed}/{len(df)} invalid SMILES")
    
    # Per-task class weights (ignore NaN)
    class_weights = compute_class_weights(data_list)
    
    return data_list, class_weights, tasks


if __name__ == "__main__":
    for name in ("tox21", "clintox"):
        data, cw, tasks = load_dataset(name)
        print(f"{name}: {len(data)} molecules, {len(tasks)} tasks")
        print(f"  Sample node shape: {data[0].x.shape}")
        print(f"  Sample edge shape: {data[0].edge_index.shape}")
