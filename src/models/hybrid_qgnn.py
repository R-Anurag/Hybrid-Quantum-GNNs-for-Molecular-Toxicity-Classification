"""
models/hybrid_qgnn.py
Hybrid GCN + VQC model. Supports standard VQC and quantum edge embedding variant.
"""
import torch
import torch.nn as nn
import pennylane as qml
from torch_geometric.nn import global_mean_pool
from .gcn import GCN


def build_vqc(n_qubits, n_layers, edge_embed=False):
    dev = qml.device("default.qubit", wires=n_qubits)

    if not edge_embed:
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    else:
        # Quantum edge embedding: bond features modulate CRY entangling angles
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights, edge_params):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CRY(edge_params[i], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    if edge_embed:
        weight_shapes["edge_params"] = (n_qubits - 1,)

    return circuit, weight_shapes


class HybridQGNN(nn.Module):
    def __init__(
        self,
        in_channels,
        gcn_hidden=64,
        gcn_embed=32,
        n_qubits=4,
        n_layers=2,
        num_tasks=1,
        dropout=0.3,
        edge_embed=False,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.edge_embed = edge_embed

        # GCN encoder (no classifier head)
        self.gcn = GCN(in_channels, gcn_hidden, gcn_embed, num_tasks=1, dropout=dropout)

        # Project GCN embedding → n_qubits
        self.proj = nn.Linear(gcn_embed, n_qubits)

        # VQC
        circuit, weight_shapes = build_vqc(n_qubits, n_layers, edge_embed)
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Edge feature projector (only for edge_embed variant)
        if edge_embed:
            self.edge_proj = nn.Linear(4, n_qubits - 1)  # bond dim=4

        # MLP classifier
        combined_dim = gcn_embed + n_qubits
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_tasks),
        )

    def forward(self, data):
        # Classical embedding
        emb = self.gcn.encode(data.x, data.edge_index, data.batch)  # (B, gcn_embed)

        # Project to qubit space and normalise to [-π, π]
        q_in = torch.tanh(self.proj(emb)) * 3.14159  # (B, n_qubits)

        # Quantum forward (batched with backprop)
        if self.edge_embed:
            # Mean-pool edge features per graph
            edge_feat = data.edge_attr  # (E, 4)
            edge_batch = data.batch[data.edge_index[0]]  # (E,)
            B = emb.size(0)
            pooled_edge = torch.zeros(B, 4, device=emb.device)
            pooled_edge.scatter_add_(0, edge_batch.unsqueeze(1).expand(-1, 4), edge_feat)
            counts = torch.bincount(edge_batch, minlength=B).float().clamp(min=1).unsqueeze(1)
            pooled_edge = pooled_edge / counts  # (B, 4)
            ep = self.edge_proj(pooled_edge)  # (B, n_qubits-1)
            # Note: edge_embed variant still needs sample-by-sample due to variable args
            q_out = torch.stack([self.vqc(q_in[i], ep[i]) for i in range(B)])
        else:
            q_out = self.vqc(q_in)  # (B, n_qubits)

        combined = torch.cat([emb, q_out], dim=-1)
        return self.classifier(combined)
