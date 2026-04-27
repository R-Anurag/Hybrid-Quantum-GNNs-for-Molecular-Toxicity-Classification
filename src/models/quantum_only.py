"""
models/quantum_only.py
Quantum-readout baseline: GCN encoder + VQC (no classical features in final layer).
"""
import torch
import torch.nn as nn
import pennylane as qml
from torch_geometric.nn import GCNConv, global_mean_pool


def build_vqc(n_qubits, n_layers):
    """Improved VQC with multi-axis rotations and stronger entanglement."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        
        for layer in range(n_layers):
            # Multi-axis rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Stronger entanglement
            if n_qubits <= 6:
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            else:
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return circuit, weight_shapes


class QuantumOnly(nn.Module):
    """
    Quantum-only baseline: Uses GCN to encode graph structure, then feeds
    to quantum circuit. Classifier uses ONLY quantum features (no classical bypass).
    """
    def __init__(self, in_channels, n_qubits=4, n_layers=2, num_tasks=1, dropout=0.3):
        super().__init__()
        self.n_qubits = n_qubits

        # GCN layers to preserve graph structure (unlike mean pooling)
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 16)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Project GCN output to qubit space
        self.proj = nn.Linear(16, n_qubits)

        # Learnable scaling parameters
        self.input_scale = nn.Parameter(torch.tensor(3.14159))
        self.quantum_scale = nn.Parameter(torch.tensor(5.0))

        # VQC
        circuit, weight_shapes = build_vqc(n_qubits, n_layers)
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classifier from quantum features ONLY (no classical bypass)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_tasks),
        )

    def forward(self, data):
        # GCN encoding (preserves graph structure)
        x = self.relu(self.conv1(data.x, data.edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, data.edge_index))
        x = self.dropout(x)
        
        # Pool to graph-level representation
        x_pooled = global_mean_pool(x, data.batch)  # (B, 16)

        # Project to qubit space and normalize with learnable scaling
        q_in = torch.tanh(self.proj(x_pooled)) * self.input_scale  # (B, n_qubits)

        # Quantum forward (batched with backprop)
        q_out = self.vqc(q_in)  # (B, n_qubits)

        # Scale quantum features
        q_out_scaled = q_out * self.quantum_scale

        # Classify using ONLY quantum features
        return self.classifier(q_out_scaled)
