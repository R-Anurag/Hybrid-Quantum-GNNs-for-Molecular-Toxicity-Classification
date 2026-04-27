"""
train.py
Trains a model (GCN or HybridQGNN) on Tox21 or ClinTox.
Supports masked BCE loss for multi-task datasets with missing labels.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score


def masked_bce_loss(pred, target, class_weights, device):
    """BCE loss that ignores NaN labels. class_weights: list of (2,) tensors."""
    if target.dim() == 1:
        target = target.reshape(pred.shape[0], -1)
    if pred.dim() == 1:
        pred = pred.unsqueeze(1)
    
    losses = []
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
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            p, y, weight=sample_w, reduction='mean'
        )
        losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return torch.stack(losses).mean()


def train_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = masked_bce_loss(out, batch.y, class_weights, device)
        loss.backward()
        # Gradient clipping to prevent exploding gradients in quantum circuits
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, class_weights, device, return_predictions=False):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = masked_bce_loss(out, batch.y, class_weights, device)
        total_loss += loss.item()

        if return_predictions:
            preds.append(torch.sigmoid(out).cpu().numpy())
            target = batch.y.cpu().numpy().reshape(out.shape[0], -1)
            targets.append(target)

    mean_loss = total_loss / len(loader)
    if return_predictions:
        return mean_loss, np.vstack(preds), np.vstack(targets)
    return mean_loss


def mean_roc_auc(preds, targets):
    """Mean ROC-AUC across valid tasks, ignoring NaN labels."""
    aucs = []
    for t in range(targets.shape[1]):
        col_t = targets[:, t]
        col_p = preds[:, t]
        mask = ~np.isnan(col_t)
        if mask.sum() == 0 or len(np.unique(col_t[mask])) < 2:
            continue
        aucs.append(roc_auc_score(col_t[mask].astype(int), col_p[mask]))
    return float(np.mean(aucs)) if aucs else float("nan")


def train(model, train_data, val_data, class_weights, epochs=100, lr=1e-3,
          batch_size=32, device="cpu", verbose=True, checkpoint_dir=None,
          model_name="model", early_stop_patience=20, weight_decay=1e-4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    best_score, best_state = float("-inf"), None
    best_epoch = 0
    patience_counter = 0
    min_delta = 1e-4
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, class_weights, device)
        val_loss, val_preds, val_targets = eval_epoch(
            model, val_loader, class_weights, device, return_predictions=True
        )
        val_auc = mean_roc_auc(val_preds, val_targets)
        if np.isfinite(val_auc):
            score = val_auc
        elif np.isfinite(val_loss):
            score = -val_loss
        else:
            score = best_score

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        scheduler.step(score)
        
        if best_state is None or score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if checkpoint_dir:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(best_state, os.path.join(checkpoint_dir, f"{model_name}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
        
        if verbose and epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d} | train={tr_loss:.4f} | "
                f"val={val_loss:.4f} | val_auc={val_auc:.4f}"
            )

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model, history
