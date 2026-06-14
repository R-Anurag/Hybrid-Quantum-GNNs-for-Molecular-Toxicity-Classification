"""Smoke tests for model forward passes."""

from pathlib import Path
import sys

import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import GCN, HybridQGNN, QuantumOnly  # noqa: E402


IN_CHANNELS = 10
NUM_TASKS = 12


def create_dummy_batch(num_graphs=4, num_nodes=10, num_edges=15):
    """Create a synthetic molecular graph batch for model shape checks."""
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, IN_CHANNELS)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 4)
        y = torch.randint(0, 2, (NUM_TASKS,)).float()
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(data_list)


def assert_forward_shape(model, batch):
    model.eval()
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (batch.num_graphs, NUM_TASKS)


def test_classical_gcn_forward_shape():
    batch = create_dummy_batch()
    model = GCN(IN_CHANNELS, hidden=64, embed_dim=32, num_tasks=NUM_TASKS)
    assert_forward_shape(model, batch)


def test_quantum_only_forward_shape():
    batch = create_dummy_batch()
    model = QuantumOnly(IN_CHANNELS, n_qubits=4, n_layers=2, num_tasks=NUM_TASKS)
    assert_forward_shape(model, batch)


def test_hybrid_qgnn_4_qubit_forward_shape():
    batch = create_dummy_batch()
    model = HybridQGNN(
        IN_CHANNELS,
        gcn_hidden=64,
        gcn_embed=32,
        n_qubits=4,
        n_layers=2,
        num_tasks=NUM_TASKS,
    )
    assert_forward_shape(model, batch)


def test_hybrid_qgnn_8_qubit_forward_shape():
    batch = create_dummy_batch()
    model = HybridQGNN(
        IN_CHANNELS,
        gcn_hidden=64,
        gcn_embed=32,
        n_qubits=8,
        n_layers=2,
        num_tasks=NUM_TASKS,
    )
    assert_forward_shape(model, batch)


def test_hybrid_qgnn_edge_forward_shape():
    batch = create_dummy_batch()
    model = HybridQGNN(
        IN_CHANNELS,
        gcn_hidden=64,
        gcn_embed=32,
        n_qubits=4,
        n_layers=2,
        num_tasks=NUM_TASKS,
        edge_embed=True,
    )
    assert_forward_shape(model, batch)
