"""
models/hybrid_qgnn.py
Hybrid GCN + VQC model. Supports standard VQC and quantum edge embedding variant.
"""
import torch
import torch.nn as nn
import pennylane as qml
from torch_geometric.nn import global_mean_pool
from .gcn import GCN


def build_vqc(n_qubits, n_layers):
    """Standard VQC with improved ansatz: deeper layers and stronger entanglement."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        
        for layer in range(n_layers):
            # Single-qubit rotations on multiple axes
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Stronger entanglement: circular + all-to-all for small n_qubits
            if n_qubits <= 6:
                # All-to-all entanglement for small circuits
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            else:
                # Circular entanglement for larger circuits (more efficient)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return circuit, weight_shapes


def build_vqc_edge(n_qubits, n_layers):
    """VQC with CRY gates controlled by edge features and improved ansatz."""
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit_edge_v2(inputs, weights):  # Renamed to force cache invalidation
        # TorchLayer may pass 2D tensors (batch_size, features) or 1D (features,)
        # Flatten to 1D if needed
        if inputs.ndim == 2:
            inputs = inputs.squeeze(0)  # (1, 7) -> (7,)
        
        # inputs: [node_features (n_qubits) || edge_angles (n_qubits-1)]
        node_inputs = inputs[:n_qubits]  # First n_qubits elements
        edge_angles = inputs[n_qubits:]  # Remaining n_qubits-1 elements
        
        # Manual angle embedding for node features only
        for i in range(n_qubits):
            qml.RY(node_inputs[i], wires=i)
        
        for layer in range(n_layers):
            # Multi-axis rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Edge-controlled entanglement
            for i in range(len(edge_angles)):
                if i < n_qubits - 1:  # Safety check
                    qml.CRY(edge_angles[i], wires=[i, i + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return circuit_edge_v2, weight_shapes


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

        # Learnable scaling for quantum input angles
        self.input_scale = nn.Parameter(torch.tensor(3.14159))
        
        # Learnable scaling for quantum features to balance with classical features
        self.quantum_scale = nn.Parameter(torch.tensor(10.0))

        # VQC
        if edge_embed:
            circuit, weight_shapes = build_vqc_edge(n_qubits, n_layers)
            self.edge_proj = nn.Linear(4, n_qubits - 1)  # bond dim=4
        else:
            circuit, weight_shapes = build_vqc(n_qubits, n_layers)
        
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

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

        # Project to qubit space and normalise with learnable scaling
        q_in = torch.tanh(self.proj(emb)) * self.input_scale  # (B, n_qubits)

        # Quantum forward (batched with backprop)
        if self.edge_embed:
            # Mean-pool edge features per graph
            edge_feat = data.edge_attr  # (E, 4)
            # Handle both single graph (batch=None) and batched graphs
            if data.batch is None:
                # Single graph: mean over all edges
                pooled_edge = edge_feat.mean(dim=0, keepdim=True)  # (1, 4)
            else:
                # Batched graphs: mean per graph
                edge_batch = data.batch[data.edge_index[0]]  # (E,)
                B = emb.size(0)
                pooled_edge = torch.zeros(B, 4, device=emb.device)
                pooled_edge.scatter_add_(0, edge_batch.unsqueeze(1).expand(-1, 4), edge_feat)
                counts = torch.bincount(edge_batch, minlength=B).float().clamp(min=1).unsqueeze(1)
                pooled_edge = pooled_edge / counts  # (B, 4)
            ep = torch.tanh(self.edge_proj(pooled_edge)) * self.input_scale  # (B, n_qubits-1)
            # Concatenate node and edge features for VQC input
            vqc_input = torch.cat([q_in, ep], dim=-1)  # (B, 2*n_qubits-1)
            q_out = self.vqc(vqc_input)  # (B, n_qubits)
        else:
            q_out = self.vqc(q_in)  # (B, n_qubits)

        # Scale quantum features to balance with classical features
        q_out_scaled = q_out * self.quantum_scale
        combined = torch.cat([emb, q_out_scaled], dim=-1)
        return self.classifier(combined)
