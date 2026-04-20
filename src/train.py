"""
train.py
Trains a model (GCN or HybridQGNN) on Tox21 or ClinTox.
Supports masked BCE loss for multi-task datasets with missing labels.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np


def masked_bce_loss(pred, target, class_weights, device):
    """BCE loss that ignores NaN labels. class_weights: list of (2,) tensors."""
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0
    for t in range(target.shape[1]):
        col_t = target[:, t]
        col_p = pred[:, t]
        mask = ~torch.isnan(col_t)
        if mask.sum() == 0:
            continue
        y = col_t[mask]
        p = col_p[mask]
        w = class_weights[t].to(device)
        # per-sample weight: w[0] for class 0, w[1] for class 1
        sample_w = torch.where(y == 1, w[1], w[0])
        loss = nn.functional.binary_cross_entropy_with_logits(p, y, weight=sample_w)
        total_loss = total_loss + loss
        count += 1
    return total_loss / max(count, 1)


def train_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = masked_bce_loss(out, batch.y, class_weights, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, class_weights, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = masked_bce_loss(out, batch.y, class_weights, device)
        total_loss += loss.item()
    return total_loss / len(loader)


def train(model, train_data, val_data, class_weights, epochs=50, lr=1e-3,
          batch_size=32, device="cpu", verbose=True):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    best_val, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, class_weights, device)
        val_loss = eval_epoch(model, val_loader, class_weights, device)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={tr_loss:.4f} | val={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model
