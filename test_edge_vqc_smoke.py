"""
Smoke test for the edge-embedded HybridQGNN VQC batching path.

This specifically guards against the PennyLane/TorchLayer shape error:
RuntimeError: shape '[64, -1]' is invalid for input of size 7
"""
from pathlib import Path
import sys

import torch
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models import HybridQGNN  # noqa: E402
from train import masked_bce_loss  # noqa: E402


def make_graph(num_nodes=8, num_edges=14, num_tasks=2):
    x = torch.randn(num_nodes, 10)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)
    y = torch.randint(0, 2, (num_tasks,)).float()
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_tasks = 2

    model = HybridQGNN(
        in_channels=10,
        gcn_hidden=64,
        gcn_embed=32,
        n_qubits=4,
        n_layers=2,
        num_tasks=num_tasks,
        edge_embed=True,
    ).to(device)

    class_weights = [torch.tensor([1.0, 1.0]) for _ in range(num_tasks)]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for batch_size in (1, 4, 64):
        batch = Batch.from_data_list(
            [make_graph(num_tasks=num_tasks) for _ in range(batch_size)]
        ).to(device)

        model.train()
        optimizer.zero_grad()
        out = model(batch)

        expected = (batch.num_graphs, num_tasks)
        assert tuple(out.shape) == expected, f"got {tuple(out.shape)}, expected {expected}"

        loss = masked_bce_loss(out, batch.y, class_weights, device)
        loss.backward()
        optimizer.step()

        print(f"PASS batch_size={batch_size}: output={tuple(out.shape)}, loss={loss.item():.4f}")

    print("Edge VQC smoke test passed.")


if __name__ == "__main__":
    main()
