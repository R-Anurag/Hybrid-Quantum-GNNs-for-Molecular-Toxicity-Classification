"""
run_experiments.py
Runs all 5 model variants on Tox21 and ClinTox and saves a results CSV.
"""
import os, sys, json
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import load_dataset
from models import GCN, HybridQGNN, QuantumOnly
from evaluate import cross_validate

DEVICE = "cpu"
EPOCHS = 10
LR = 1e-3
BATCH = 64
N_FOLDS = 5
IN_CHANNELS = 10  # atom feature dim (see data_pipeline.py)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    data_list, class_weights, tasks = load_dataset(dataset_name)
    num_tasks = len(tasks)
    print(f"Loaded {len(data_list)} molecules, {num_tasks} tasks")

    variants = {
        "Classical GCN": lambda: GCN(IN_CHANNELS, hidden=64, embed_dim=32,
                                      num_tasks=num_tasks),
        "Quantum-only 4-qubit": lambda: QuantumOnly(IN_CHANNELS, n_qubits=4, n_layers=2,
                                                     num_tasks=num_tasks),
        "Hybrid 4-qubit": lambda: HybridQGNN(IN_CHANNELS, gcn_hidden=64, gcn_embed=32,
                                              n_qubits=4, n_layers=2,
                                              num_tasks=num_tasks),
        "Hybrid 8-qubit": lambda: HybridQGNN(IN_CHANNELS, gcn_hidden=64, gcn_embed=32,
                                              n_qubits=8, n_layers=2,
                                              num_tasks=num_tasks),
        "Hybrid 4-qubit + Edge": lambda: HybridQGNN(IN_CHANNELS, gcn_hidden=64, gcn_embed=32,
                                                     n_qubits=4, n_layers=2,
                                                     num_tasks=num_tasks, edge_embed=True),
    }

    rows = []
    for name, model_fn in variants.items():
        print(f"\n>> {name}")
        # Count params
        m = model_fn()
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        del m

        results = cross_validate(
            model_fn, data_list, class_weights,
            n_splits=N_FOLDS, epochs=EPOCHS, lr=LR,
            batch_size=BATCH, device=DEVICE, verbose=False,
        )
        row = {"Model": name, "Dataset": dataset_name, "Params": n_params, **results}
        rows.append(row)
        print(f"  AUC={results['auc_mean']:.4f}±{results['auc_std']:.4f}  "
              f"F1={results['f1_mean']:.4f}±{results['f1_std']:.4f}  "
              f"time/epoch={results['time_mean']:.2f}s")

    return rows


if __name__ == "__main__":
    all_rows = []
    for ds in ("clintox", "tox21"):
        all_rows.extend(run_dataset(ds))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))
