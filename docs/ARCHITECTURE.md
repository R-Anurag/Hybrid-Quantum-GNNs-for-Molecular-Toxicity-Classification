# Architecture

## System Overview

The project implements a molecular toxicity classification pipeline that compares a classical graph neural network baseline with hybrid quantum-classical variants. Each molecule is converted from a SMILES string into a graph, encoded with graph convolution layers, and classified at the graph level.

## Data Pipeline

`src/data_pipeline.py` is responsible for dataset loading and graph construction.

Main responsibilities:

- Parse SMILES strings with RDKit.
- Build PyTorch Geometric `Data` objects.
- Encode atom-level node features and bond-level edge features.
- Preserve missing labels for masked multi-task training.
- Compute task-level class weights for imbalanced toxicity endpoints.

The resulting graph objects contain:

- `x`: atom feature matrix.
- `edge_index`: graph connectivity.
- `edge_attr`: bond feature matrix.
- `y`: toxicity labels.

## Classical GCN

`src/models/gcn.py` defines the baseline graph model.

```text
node features, edge index
  |
  v
GCNConv -> ReLU -> Dropout
  |
  v
GCNConv -> ReLU -> Dropout
  |
  v
GCNConv -> ReLU
  |
  v
global mean pooling
  |
  v
MLP classifier
```

The final graph embedding dimension is configurable and is used directly for classification in the baseline model.

## Hybrid Quantum GNN

`src/models/hybrid_qgnn.py` extends the GCN encoder with a variational quantum circuit.

```text
graph embedding
  |
  +------------------------+
  |                        |
  v                        v
classical branch           quantum projection
                           |
                           v
                    angle embedding
                           |
                           v
                    variational layers
                           |
                           v
                    Pauli-Z measurements
  |                        |
  +-----------+------------+
              |
              v
       concatenation
              |
              v
       MLP classifier
```

The quantum circuit uses:

- Angle embedding for classical-to-quantum feature encoding.
- Trainable RY and RZ rotations.
- CNOT-based entanglement.
- Pauli-Z expectation measurements as quantum features.

The model includes learnable input and output scaling parameters to balance the magnitude of classical and quantum features during joint optimization.

## Edge-Embedded Variant

The optional edge-embedded hybrid model projects pooled bond features into additional circuit inputs. These edge-derived angles modulate controlled rotations, allowing bond information to influence the quantum branch.

## Quantum-Only Baseline

`src/models/quantum_only.py` preserves a GCN encoder for graph representation but sends only quantum features into the final classifier. This variant helps isolate whether the quantum branch is independently predictive or mainly useful as an auxiliary feature source.

## Training

`src/train.py` provides the shared training and validation loops.

Training features:

- Masked binary cross-entropy for missing labels.
- Per-task class weighting for imbalanced toxicity endpoints.
- AdamW optimization.
- Learning rate scheduling.
- Gradient clipping.
- Early stopping based on validation ROC-AUC.

## Evaluation

`src/evaluate.py` handles metric computation and cross-validation. The primary metric is ROC-AUC, with F1-score, parameter count, and runtime recorded for model comparison.

## Data Flow

```text
SMILES
  |
  v
RDKit molecule
  |
  v
PyTorch Geometric graph
  |
  v
GCN encoder
  |
  v
classical and/or quantum graph features
  |
  v
toxicity prediction logits
```

## Design Considerations

The circuit depth and qubit count are kept modest because simulated quantum layers are computationally expensive and deeper circuits can suffer from weak gradients. The hybrid model is therefore designed as a compact feature augmentation strategy rather than a large standalone quantum architecture.
