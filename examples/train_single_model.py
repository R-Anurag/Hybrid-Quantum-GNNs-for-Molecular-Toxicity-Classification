"""
train_single_model.py

Example script demonstrating how to train a single model variant
on a chosen dataset with custom hyperparameters.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch_geometric.loader import DataLoader
from data_pipeline import load_tox21, load_clintox
from models import GCN, HybridQGNN
from train import train_epoch, validate
import time

def main():
    # Configuration
    DATASET = 'clintox'  # 'tox21' or 'clintox'
    MODEL_TYPE = 'hybrid'  # 'classical' or 'hybrid'
    N_QUBITS = 4  # Only for hybrid models
    EPOCHS = 50
    BATCH_SIZE = 32
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Single Model Training Example")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL_TYPE}")
    if MODEL_TYPE == 'hybrid':
        print(f"Qubits: {N_QUBITS}")
    print(f"Device: {DEVICE}")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    if DATASET == 'tox21':
        data_list, class_weights, num_tasks = load_tox21()
    else:
        data_list, class_weights, num_tasks = load_clintox()
    
    print(f"Loaded {len(data_list)} molecules with {num_tasks} tasks")
    
    # Train/val split (80/20)
    split_idx = int(0.8 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Initialize model
    print("\nInitializing model...")
    in_channels = data_list[0].x.shape[1]
    
    if MODEL_TYPE == 'classical':
        model = GCN(
            in_channels=in_channels,
            hidden=64,
            embed_dim=32,
            num_tasks=num_tasks,
            dropout=0.2
        )
    else:  # hybrid
        model = HybridQGNN(
            in_channels=in_channels,
            gcn_hidden=64,
            gcn_embed=32,
            n_qubits=N_QUBITS,
            n_layers=2,
            num_tasks=num_tasks,
            dropout=0.2
        )
    
    model = model.to(DEVICE)
    class_weights = class_weights.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    best_val_auc = 0.0
    patience_counter = 0
    patience = 20
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_time = train_epoch(
            model, train_loader, optimizer, class_weights, DEVICE
        )
        
        # Validate
        val_loss, val_auc = validate(
            model, val_loader, class_weights, DEVICE
        )
        
        # Scheduler step
        scheduler.step(val_auc)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
            status = "★"
        else:
            patience_counter += 1
            status = " "
        
        # Print progress
        print(f"Epoch {epoch:3d}/{EPOCHS} {status} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Time: {train_time:.2f}s")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    print("\n" + "="*70)
    print(f"Training complete! Best Val AUC: {best_val_auc:.4f}")
    print(f"Best model saved to: best_model.pt")
    print("="*70)

if __name__ == "__main__":
    main()
