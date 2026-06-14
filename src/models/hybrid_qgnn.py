"""
models/hybrid_qgnn.py
Hybrid GCN and variational quantum circuit model.
"""
import torch
import torch.nn as nn
import pennylane as qml
from .gcn import GCN


def build_vqc(n_qubits, n_layers):
    """Build the variational circuit used by the hybrid model."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
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


def build_vqc_edge(n_qubits, n_layers):
    """Build the edge-conditioned variational circuit."""
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit_edge(inputs, weights):
        node_inputs = inputs[..., :n_qubits]
        edge_angles = inputs[..., n_qubits:]
        
        qml.AngleEmbedding(node_inputs, wires=range(n_qubits), rotation="Y")
        
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            for i in range(n_qubits - 1):
                qml.CRY(edge_angles[..., i], wires=[i, i + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return circuit_edge, weight_shapes


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

        self.gcn = GCN(in_channels, gcn_hidden, gcn_embed, num_tasks=1, dropout=dropout)
        self.proj = nn.Linear(gcn_embed, n_qubits)
        self.input_scale = nn.Parameter(torch.tensor(3.14159))
        self.quantum_scale = nn.Parameter(torch.tensor(10.0))

        if edge_embed:
            circuit, weight_shapes = build_vqc_edge(n_qubits, n_layers)
            self.edge_proj = nn.Linear(4, n_qubits - 1)
        else:
            circuit, weight_shapes = build_vqc(n_qubits, n_layers)
        
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        combined_dim = gcn_embed + n_qubits
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_tasks),
        )

    def forward(self, data):
        emb = self.gcn.encode(data.x, data.edge_index, data.batch)

        q_in = torch.tanh(self.proj(emb)) * self.input_scale

        if self.edge_embed:
            edge_feat = data.edge_attr
            if data.batch is None:
                pooled_edge = edge_feat.mean(dim=0, keepdim=True)
            else:
                edge_batch = data.batch[data.edge_index[0]]
                B = emb.size(0)
                pooled_edge = torch.zeros(B, 4, device=emb.device)
                pooled_edge.scatter_add_(0, edge_batch.unsqueeze(1).expand(-1, 4), edge_feat)
                counts = torch.bincount(edge_batch, minlength=B).float().clamp(min=1).unsqueeze(1)
                pooled_edge = pooled_edge / counts
            ep = torch.tanh(self.edge_proj(pooled_edge)) * self.input_scale
            vqc_input = torch.cat([q_in, ep], dim=-1)
            q_out = self.vqc(vqc_input)
        else:
            q_out = self.vqc(q_in)

        q_out_scaled = q_out * self.quantum_scale
        combined = torch.cat([emb, q_out_scaled], dim=-1)
        return self.classifier(combined)
