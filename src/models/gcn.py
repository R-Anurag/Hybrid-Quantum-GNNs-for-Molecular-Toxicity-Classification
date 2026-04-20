"""
models/gcn.py
3-layer GCN baseline for molecular toxicity classification.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_channels, hidden=64, embed_dim=32, num_tasks=1, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_tasks),
        )

    def encode(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)  # (B, embed_dim)

    def forward(self, data):
        emb = self.encode(data.x, data.edge_index, data.batch)
        return self.classifier(emb)  # (B, num_tasks)
