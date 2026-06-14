"""
Train one model variant on a chosen dataset.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from data_pipeline import load_dataset
from models import GCN, HybridQGNN
from train import train


def main():
    dataset = "clintox"
    model_type = "hybrid"
    n_qubits = 4
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Single Model Training")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_type}")
    if model_type == "hybrid":
        print(f"Qubits: {n_qubits}")
    print(f"Device: {device}")
    print("=" * 70)

    print("\nLoading dataset...")
    data_list, class_weights, tasks = load_dataset(dataset)
    num_tasks = len(tasks)
    print(f"Loaded {len(data_list)} molecules with {num_tasks} tasks")

    split_idx = int(0.8 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    print("\nInitializing model...")
    in_channels = data_list[0].x.shape[1]
    if model_type == "classical":
        model = GCN(
            in_channels=in_channels,
            hidden=64,
            embed_dim=32,
            num_tasks=num_tasks,
            dropout=0.2,
        )
    else:
        model = HybridQGNN(
            in_channels=in_channels,
            gcn_hidden=64,
            gcn_embed=32,
            n_qubits=n_qubits,
            n_layers=2,
            num_tasks=num_tasks,
            dropout=0.2,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    _, history = train(
        model,
        train_data,
        val_data,
        class_weights,
        epochs=epochs,
        lr=learning_rate,
        batch_size=batch_size,
        device=device,
        checkpoint_dir=".",
        model_name="best_model",
    )

    print("\n" + "=" * 70)
    print(f"Training complete. Best epoch: {history['best_epoch']}")
    print("Best model saved to: best_model.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
