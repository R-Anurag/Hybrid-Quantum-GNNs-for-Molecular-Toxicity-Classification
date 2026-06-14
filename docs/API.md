# API Documentation

## Module Overview

### `src.data_pipeline`

Handles conversion of molecular SMILES to graph-structured data.

#### `load_tox21()`
Load Tox21 dataset and return graph data, class weights, and task count.

#### `load_clintox()`
Load ClinTox dataset and return graph data, class weights, and task count.

#### `smiles_to_graph(smiles: str, label: np.ndarray)`
Convert SMILES string to PyTorch Geometric Data object with node/edge features.

---

### `src.models.gcn`

Classical GCN baseline model.

#### `GCN`
3-layer Graph Convolutional Network with global mean pooling and MLP classifier.

**Parameters:**
- `in_channels`: Number of input node features
- `hidden`: Hidden dimension (default: 64)
- `embed_dim`: Graph embedding dimension (default: 32)
- `num_tasks`: Number of prediction tasks
- `dropout`: Dropout probability (default: 0.2)

---

### `src.models.hybrid_qgnn`

Hybrid quantum-classical GNN model.

#### `HybridQGNN`
Combines GCN with variational quantum circuit for quantum feature encoding.

**Parameters:**
- `in_channels`: Number of input node features
- `gcn_hidden`: GCN hidden dimension (default: 64)
- `gcn_embed`: Classical embedding dimension (default: 32)
- `n_qubits`: Number of qubits (default: 4)
- `n_layers`: Quantum circuit layers (default: 2)
- `num_tasks`: Number of prediction tasks
- `dropout`: Dropout probability (default: 0.2)
- `edge_embed`: Use quantum edge embedding (default: False)

---

### `src.train`

Training and validation functions.

#### `train_epoch()`
Train model for one epoch with masked BCE loss and gradient clipping.

#### `validate()`
Validate model and compute loss and ROC-AUC metrics.

---

### `src.evaluate`

Evaluation and cross-validation utilities.

#### `cross_validate()`
Perform k-fold cross-validation and return mean/std metrics.

#### `compute_metrics()`
Compute ROC-AUC and F1-score handling missing labels.

---

## Usage Example

```python
from src.data_pipeline import load_tox21
from src.models import HybridQGNN
from src.evaluate import cross_validate

# Load data
data_list, class_weights, num_tasks = load_tox21()

# Run cross-validation
results = cross_validate(
    model_class=HybridQGNN,
    model_kwargs={'in_channels': 9, 'n_qubits': 4, 'num_tasks': num_tasks},
    data_list=data_list,
    class_weights=class_weights,
    n_splits=5
)

print(f"ROC-AUC: {results['roc_auc_mean']:.3f} ± {results['roc_auc_std']:.3f}")
```
