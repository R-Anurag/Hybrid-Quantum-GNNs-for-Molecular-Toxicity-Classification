"""
evaluate.py
ROC-AUC / F1 evaluation and 5-fold cross-validation for all model variants.
"""
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold

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
        t = batch.y.cpu().numpy()
        targets.append(t.reshape(out.shape[0], -1))
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
                   epochs=50, lr=1e-3, batch_size=32, device="cpu", verbose=True,
                   checkpoint_dir=None, model_name="model", dataset_name="dataset"):
    """
    model_fn: callable() → fresh model instance
    Returns dict with mean/std for auc, f1, epoch_time.
    """
    # Create stratification labels: use first task or composite for multi-task
    y_all = np.array([d.y.numpy() for d in data_list])
    if y_all.ndim == 1:
        y_all = y_all.reshape(-1, 1)
    # Use first task for stratification, handle NaN
    strat_labels = y_all[:, 0]
    strat_labels = np.nan_to_num(strat_labels, nan=-1).astype(int)
    
    # Use StratifiedKFold if we have valid labels
    if len(np.unique(strat_labels[strat_labels >= 0])) > 1:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        indices = np.arange(len(data_list))
        splits = kf.split(indices, strat_labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        indices = np.arange(len(data_list))
        splits = kf.split(indices)
    
    aucs, f1s, times = [], [], []
    histories = []

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n-- Fold {fold}/{n_splits} --")
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        # 90/10 train/val split within training fold
        val_size = max(1, int(0.1 * len(train_data)))
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]

        model = model_fn()
        t0 = time.time()
        ckpt_name = f"{dataset_name}_{model_name}_fold{fold}" if checkpoint_dir else None
        model, history = train(model, train_data, val_data, class_weights,
                               epochs=epochs, lr=lr, batch_size=batch_size,
                               device=device, verbose=verbose,
                               checkpoint_dir=checkpoint_dir, model_name=ckpt_name)
        epoch_time = (time.time() - t0) / epochs

        preds, targets = predict(model, test_data, batch_size=batch_size, device=device)
        auc, f1 = compute_metrics(preds, targets)
        aucs.append(auc)
        f1s.append(f1)
        times.append(epoch_time)
        histories.append(history)
        print(f"  ROC-AUC={auc:.4f}  F1={f1:.4f}  time/epoch={epoch_time:.2f}s")

    return {
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        "f1_mean": np.mean(f1s),  "f1_std": np.std(f1s),
        "time_mean": np.mean(times),
        "histories": histories,
    }
