"""
test_models.py
Quick validation that all models work with correct input dimensions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch_geometric.data import Data, Batch
from models import GCN, HybridQGNN, QuantumOnly

# Create dummy molecular graph data
def create_dummy_data(num_graphs=4, num_nodes=10, num_edges=15):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, 10)  # 10 atom features
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 4)  # 4 bond features
        y = torch.randint(0, 2, (12,)).float()  # 12 tasks (Tox21)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(data_list)

def test_model(model_name, model, batch):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    try:
        model.eval()
        with torch.no_grad():
            out = model(batch)
        print(f"✓ Input shape: x={batch.x.shape}, edge_index={batch.edge_index.shape}")
        print(f"✓ Output shape: {out.shape}")
        print(f"✓ Expected: ({batch.num_graphs}, {NUM_TASKS})")
        assert out.shape == (batch.num_graphs, NUM_TASKS), f"Shape mismatch!"
        print(f"✓ {model_name} PASSED")
        return True
    except Exception as e:
        print(f"✗ {model_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    IN_CHANNELS = 10
    NUM_TASKS = 12
    
    batch = create_dummy_data(num_graphs=4)
    
    models = {
        "Classical GCN": GCN(IN_CHANNELS, hidden=64, embed_dim=32, num_tasks=NUM_TASKS),
        "Quantum-only 4-qubit": QuantumOnly(IN_CHANNELS, n_qubits=4, n_layers=2, num_tasks=NUM_TASKS),
        "Hybrid 4-qubit": HybridQGNN(IN_CHANNELS, gcn_hidden=64, gcn_embed=32,
                                     n_qubits=4, n_layers=2, num_tasks=NUM_TASKS),
        "Hybrid 8-qubit": HybridQGNN(IN_CHANNELS, gcn_hidden=64, gcn_embed=32,
                                     n_qubits=8, n_layers=2, num_tasks=NUM_TASKS),
        "Hybrid 4-qubit + Edge": HybridQGNN(IN_CHANNELS, gcn_hidden=64, gcn_embed=32,
                                            n_qubits=4, n_layers=2, num_tasks=NUM_TASKS,
                                            edge_embed=True),
    }
    
    results = {}
    for name, model in models.items():
        results[name] = test_model(name, model, batch)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print(f"{'='*60}")
    
    sys.exit(0 if all_passed else 1)
