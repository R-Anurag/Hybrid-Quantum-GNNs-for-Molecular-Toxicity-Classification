"""
evaluate.py
ROC-AUC evaluation and 5-fold cross-validation for all model variants.
"""
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

from data_pipeline import compute_class_weights
from train import mean_roc_auc, train


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
    """Backward-compatible alias returning mean ROC-AUC only."""
    return mean_roc_auc(preds, targets)


def _stratification_labels(data_list):
    y_all = np.array([d.y.numpy() for d in data_list])
    if y_all.ndim == 1:
        y_all = y_all.reshape(-1, 1)
    return np.nan_to_num(y_all[:, 0], nan=-1).astype(int)


def _can_stratify(labels, min_count):
    unique, counts = np.unique(labels, return_counts=True)
    valid_unique = np.unique(labels[labels >= 0])
    return len(unique) > 1 and len(valid_unique) > 1 and counts.min() >= min_count


def _train_val_split(train_idx, strat_labels, val_fraction, random_state):
    train_idx = np.asarray(train_idx)
    val_size = max(1, int(round(val_fraction * len(train_idx))))
    val_size = min(val_size, len(train_idx) - 1)
    labels = strat_labels[train_idx]

    unique, counts = np.unique(labels, return_counts=True)
    can_stratify = (
        len(unique) > 1
        and counts.min() >= 2
        and val_size >= len(unique)
        and (len(train_idx) - val_size) >= len(unique)
    )

    if can_stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=random_state
        )
        fit_pos, val_pos = next(splitter.split(train_idx, labels))
        return train_idx[fit_pos], train_idx[val_pos]

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(train_idx)
    return shuffled[val_size:], shuffled[:val_size]


# ── 5-fold CV ────────────────────────────────────────────────────────────────

def cross_validate(model_fn, data_list, class_weights=None, n_splits=5,
                   epochs=100, lr=1e-3, batch_size=32, device="cpu", verbose=True,
                   checkpoint_dir=None, model_name="model", dataset_name="dataset",
                   early_stop_patience=20, weight_decay=1e-4, val_fraction=0.1,
                   random_state=42):
    """
    model_fn: callable() → fresh model instance
    Returns dict with mean/std for ROC-AUC, actual epochs, and epoch time.

    class_weights is accepted for backward compatibility. Fold-specific weights
    are recomputed from each training split to avoid test-fold leakage.
    """
    # Create stratification labels: use first task or composite for multi-task
    strat_labels = _stratification_labels(data_list)
    
    # Use StratifiedKFold if we have valid labels
    if _can_stratify(strat_labels, n_splits):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        indices = np.arange(len(data_list))
        splits = kf.split(indices, strat_labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        indices = np.arange(len(data_list))
        splits = kf.split(indices)
    
    aucs, times, epochs_run = [], [], []
    histories = []

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n-- Fold {fold}/{n_splits} --")
        fit_idx, val_idx = _train_val_split(
            train_idx, strat_labels, val_fraction, random_state + fold
        )
        train_data = [data_list[i] for i in fit_idx]
        val_data = [data_list[i] for i in val_idx]
        test_data = [data_list[i] for i in test_idx]
        fold_class_weights = compute_class_weights(train_data)

        model = model_fn()
        t0 = time.time()
        ckpt_name = f"{dataset_name}_{model_name}_fold{fold}" if checkpoint_dir else None
        model, history = train(model, train_data, val_data, fold_class_weights,
                               epochs=epochs, lr=lr, batch_size=batch_size,
                               device=device, verbose=verbose,
                               checkpoint_dir=checkpoint_dir, model_name=ckpt_name,
                               early_stop_patience=early_stop_patience,
                               weight_decay=weight_decay)
        actual_epochs = max(1, len(history["train_loss"]))
        epoch_time = (time.time() - t0) / actual_epochs

        preds, targets = predict(model, test_data, batch_size=batch_size, device=device)
        auc = compute_metrics(preds, targets)
        aucs.append(auc)
        times.append(epoch_time)
        epochs_run.append(actual_epochs)
        histories.append(history)
        print(
            f"  ROC-AUC={auc:.4f}  epochs={actual_epochs}  "
            f"time/epoch={epoch_time:.2f}s"
        )

    return {
        "auc_mean": np.nanmean(aucs), "auc_std": np.nanstd(aucs),
        "time_mean": np.mean(times),
        "epochs_mean": np.mean(epochs_run),
        "epochs_std": np.std(epochs_run),
        "histories": histories,
    }
