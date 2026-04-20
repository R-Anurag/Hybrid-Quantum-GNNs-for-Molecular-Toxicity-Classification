"""
evaluate.py
ROC-AUC / F1 evaluation and 5-fold cross-validation for all model variants.
"""
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold

from train import train


# ── Per-batch inference ──────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, data_list, batch_size=32, device="cpu"):
    model.eval()
    loader = DataLoader(data_list, batch_size=batch_size)
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = torch.sigmoid(model(batch))
        preds.append(out.cpu().numpy())
        targets.append(batch.y.cpu().numpy())
    return np.vstack(preds), np.vstack(targets)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets):
    """Returns mean ROC-AUC and macro F1 across tasks (ignores NaN labels)."""
    num_tasks = targets.shape[1]
    aucs, f1s = [], []
    for t in range(num_tasks):
        col_t = targets[:, t]
        col_p = preds[:, t]
        mask = ~np.isnan(col_t)
        if mask.sum() == 0 or len(np.unique(col_t[mask])) < 2:
            continue
        y_true = col_t[mask].astype(int)
        y_prob = col_p[mask]
        y_pred = (y_prob >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_true, y_prob))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    return np.mean(aucs), np.mean(f1s)


# ── 5-fold CV ────────────────────────────────────────────────────────────────

def cross_validate(model_fn, data_list, class_weights, n_splits=5,
                   epochs=50, lr=1e-3, batch_size=32, device="cpu", verbose=True):
    """
    model_fn: callable() → fresh model instance
    Returns dict with mean/std for auc, f1, epoch_time.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(data_list))
    aucs, f1s, times = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), 1):
        print(f"\n── Fold {fold}/{n_splits} ──")
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        # 90/10 train/val split within training fold
        val_size = max(1, int(0.1 * len(train_data)))
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]

        model = model_fn()
        t0 = time.time()
        model = train(model, train_data, val_data, class_weights,
                      epochs=epochs, lr=lr, batch_size=batch_size,
                      device=device, verbose=verbose)
        epoch_time = (time.time() - t0) / epochs

        preds, targets = predict(model, test_data, batch_size=batch_size, device=device)
        auc, f1 = compute_metrics(preds, targets)
        aucs.append(auc)
        f1s.append(f1)
        times.append(epoch_time)
        print(f"  ROC-AUC={auc:.4f}  F1={f1:.4f}  time/epoch={epoch_time:.2f}s")

    return {
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        "f1_mean": np.mean(f1s),  "f1_std": np.std(f1s),
        "time_mean": np.mean(times),
    }
