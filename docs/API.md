# API Reference

## `src.data_pipeline`

Utilities for loading MoleculeNet datasets and converting molecules into graph data.

### `load_tox21()`

Loads the Tox21 dataset and returns graph samples, class weights, and the number of prediction tasks.

### `load_clintox()`

Loads the ClinTox dataset and returns graph samples, class weights, and the number of prediction tasks.

### `smiles_to_graph(smiles, label)`

Converts a SMILES string and label vector into a PyTorch Geometric `Data` object containing atom features, bond features, graph connectivity, and labels.

## `src.models.gcn`

### `GCN`

Classical graph convolutional network used as the baseline model.

Constructor arguments:

- `in_channels`: number of input node features.
- `hidden`: hidden GCN dimension.
- `embed_dim`: graph embedding dimension.
- `num_tasks`: number of output toxicity tasks.
- `dropout`: dropout probability.

Key methods:

- `encode(x, edge_index, batch)`: returns graph-level embeddings.
- `forward(data)`: returns prediction logits.

## `src.models.hybrid_qgnn`

### `HybridQGNN`

Hybrid model combining a GCN encoder with a variational quantum circuit.

Constructor arguments:

- `in_channels`: number of input node features.
- `gcn_hidden`: hidden GCN dimension.
- `gcn_embed`: graph embedding dimension.
- `n_qubits`: number of qubits in the quantum circuit.
- `n_layers`: number of variational circuit layers.
- `num_tasks`: number of output toxicity tasks.
- `dropout`: dropout probability.
- `edge_embed`: whether to include pooled bond features in the quantum branch.

The forward pass returns prediction logits for each task.

## `src.models.quantum_only`

### `QuantumOnly`

Baseline model that uses a GCN encoder and quantum circuit but classifies only from quantum features.

Constructor arguments:

- `in_channels`: number of input node features.
- `n_qubits`: number of qubits in the quantum circuit.
- `n_layers`: number of variational circuit layers.
- `num_tasks`: number of output toxicity tasks.
- `dropout`: dropout probability.

## `src.train`

### `train_epoch(model, loader, optimizer, class_weights, device)`

Runs one training epoch and returns loss and timing information.

### `validate(model, loader, class_weights, device)`

Evaluates a model on a validation loader and returns validation loss and ROC-AUC.

## `src.evaluate`

### `cross_validate(...)`

Runs k-fold cross-validation for a model configuration and returns aggregated metrics.

### `compute_metrics(...)`

Computes evaluation metrics while handling missing labels.

## Example

```python
from src.data_pipeline import load_tox21
from src.evaluate import cross_validate
from src.models import HybridQGNN

data_list, class_weights, num_tasks = load_tox21()

results = cross_validate(
    model_class=HybridQGNN,
    model_kwargs={
        "in_channels": data_list[0].x.shape[1],
        "n_qubits": 4,
        "num_tasks": num_tasks,
    },
    data_list=data_list,
    class_weights=class_weights,
    n_splits=5,
)

print(results)
```
