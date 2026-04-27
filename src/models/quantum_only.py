"""
models/quantum_only.py
Quantum-readout-only baseline: no GCN, just VQC on pooled atom features.
"""
import torch
import torch.nn as nn
import pennylane as qml
from torch_geometric.nn import global_mean_pool


def build_vqc(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes


class QuantumOnly(nn.Module):
    def __init__(self, in_channels, n_qubits=4, n_layers=2, num_tasks=1, dropout=0.3):
        super().__init__()
        self.n_qubits = n_qubits

        # Project pooled atom features to qubit space
        self.proj = nn.Linear(in_channels, n_qubits)

        # VQC
        circuit, weight_shapes = build_vqc(n_qubits, n_layers)
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classifier from quantum features only
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_tasks),
        )

    def forward(self, data):
        # Mean pool atom features per graph
        x_pooled = global_mean_pool(data.x, data.batch)  # (B, in_channels)

        # Project to qubit space and normalize to [-π, π]
        q_in = torch.tanh(self.proj(x_pooled)) * 3.14159  # (B, n_qubits)

        # Quantum forward
        q_out = self.vqc(q_in)  # (B, n_qubits)

        return self.classifier(q_out)
