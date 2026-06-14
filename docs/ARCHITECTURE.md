# Architecture Documentation

## System Overview

This project implements a hybrid classical-quantum machine learning pipeline for molecular toxicity prediction. The architecture combines Graph Convolutional Networks (GCNs) with Variational Quantum Circuits (VQCs) to leverage both classical graph learning and quantum feature encoding.

## Component Architecture

### 1. Data Pipeline (`src/data_pipeline.py`)

**Purpose:** Convert molecular SMILES strings into graph-structured data suitable for GNN processing.

**Key Components:**
- **SMILES Parser:** Uses RDKit to parse molecular structures
- **Graph Constructor:** Builds node (atom) and edge (bond) features
- **Feature Engineering:**
  - Node features: atomic number, degree, formal charge, hybridization, aromaticity, H count
  - Edge features: bond type (single/double/triple/aromatic)
- **Class Balancing:** Computes per-task class weights for imbalanced datasets
- **Data Validation:** Ensures graph integrity (no isolated nodes, valid edges)

**Input:** SMILES strings from MoleculeNet datasets  
**Output:** PyTorch Geometric `Data` objects with node/edge features and labels

---

### 2. Classical GCN Baseline (`src/models/gcn.py`)

**Purpose:** Strong classical baseline using standard Graph Convolutional Networks.

**Architecture:**
```
Input: Graph (nodes, edges)
  ↓
GCNConv(in → 64) + ReLU + Dropout(0.2)
  ↓
GCNConv(64 → 64) + ReLU + Dropout(0.2)
  ↓
GCNConv(64 → 32) + ReLU
  ↓
Global Mean Pool → graph embedding (dim=32)
  ↓
MLP: Linear(32 → 16) + ReLU + Dropout(0.3)
  ↓
Linear(16 → num_tasks) → Sigmoid
```

**Key Features:**
- 3-layer message passing for sufficient receptive field
- Graph-level pooling aggregates node information
- Task-specific output heads for multi-task learning

**Parameters:** ~50K (depending on input dimension)

---

### 3. Hybrid Quantum GNN (`src/models/hybrid_qgnn.py`)

**Purpose:** Augment classical GCN with quantum feature encoding via variational circuits.

**Architecture:**

#### Classical Component (GCN)
Same as baseline GCN, produces 32-dimensional graph embeddings.

#### Quantum Component (VQC)
```
Graph embedding (32D)
  ↓
Linear projection → n-qubit input (4 or 8 qubits)
  ↓
AngleEmbedding: |ψ⟩ = ⊗ᵢ RY(θᵢ)|0⟩
  ↓
Variational Layers (L=2):
  For each layer:
    - Multi-axis rotations: RY(wᵢ), RZ(wᵢ)
    - All-to-all entanglement: CNOT gates
  ↓
Measurement: Pauli-Z expectation on each qubit
  ↓
Quantum features (dim = n_qubits)
```

#### Hybrid Fusion
```
[Classical embedding (32D) || Quantum features (4D or 8D)]
  ↓
Learnable quantum scaling: q_out × λ
  ↓
MLP: Linear(36/40 → 16) + ReLU + Dropout(0.3)
  ↓
Linear(16 → num_tasks) → Sigmoid
```

**Key Innovations:**
- **Learnable Quantum Scaling:** Parameter `λ` allows network to learn optimal quantum contribution
- **Multi-axis Rotations:** RY + RZ gates increase circuit expressivity
- **All-to-all Entanglement:** Fully connected CNOT structure maximizes quantum correlations
- **Gradient Flow:** Parameter-shift rule enables backpropagation through quantum layer

**Variants:**
- **4-qubit:** Lower computational cost, faster training
- **8-qubit:** Higher capacity, potentially better feature encoding
- **Edge-embedded:** Bond features modulate entanglement gates (CRY instead of CNOT)

**Parameters:** Classical GCN + ~(n_layers × n_qubits × 2) quantum parameters + fusion MLP

---

### 4. Quantum-Only Baseline (`src/models/quantum_only.py`)

**Purpose:** Evaluate quantum layer contribution by using GCN+VQC without classical features in fusion.

**Architecture:**
Same as Hybrid QGNN but final MLP receives *only* quantum features (no classical embedding concatenation).

**Note:** This model still uses GCN for graph processing (necessary to convert molecular graphs to vector representations) but relies solely on quantum features for classification.

---

### 5. Training Loop (`src/train.py`)

**Key Features:**

#### Loss Function
- **Masked Binary Cross-Entropy:** Ignores missing labels (common in Tox21)
- **Class Weighting:** Handles severe class imbalance
- **Per-task Reduction:** Computes loss separately per task, then averages

#### Optimization
- **Optimizer:** AdamW (weight decay regularization)
- **Learning Rate:** 1e-3 (classical), 1e-4 (quantum parameters)
- **Scheduler:** ReduceLROnPlateau (monitors validation ROC-AUC)
- **Gradient Clipping:** max_norm=1.0 prevents quantum gradient explosion

#### Early Stopping
- Patience: 20 epochs without validation improvement
- Metric: ROC-AUC (maximized)
- Checkpoint: Saves best model based on validation performance

#### Training Loop Pseudocode
```
For each epoch:
  For each batch:
    1. Forward pass (quantum circuits evaluated in batch mode)
    2. Compute masked BCE loss
    3. Backward pass (parameter-shift for quantum gradients)
    4. Clip gradients
    5. Optimizer step
  
  Validate on held-out set
  Update learning rate if plateau
  Check early stopping
```

---

### 6. Evaluation (`src/evaluate.py`)

**Metrics:**
- **ROC-AUC:** Per-task and mean (primary metric)
- **F1-Score:** Precision-recall balance
- **Parameter Count:** Model complexity
- **Training Time:** Wall-clock time per epoch

**Cross-Validation:**
- 5-fold stratified CV
- Reports mean ± standard deviation for all metrics
- Ensures robust performance estimates

---

## Data Flow

```
SMILES String
  ↓
[RDKit Molecular Graph]
  ↓
PyTorch Geometric Data
  ↓
[GCN Layers: Message Passing]
  ↓
Graph Embedding (32D)
  ├─────────────────────┐
  ↓                     ↓
Classical Path      Quantum Path
  (direct)          (projection → VQC)
  ↓                     ↓
[Classical Embedding] [Quantum Features]
  └─────────┬───────────┘
            ↓
    [Concatenation]
            ↓
     [MLP Classifier]
            ↓
    Toxicity Predictions
```

---

## Quantum Circuit Details

### Angle Embedding
Encodes classical data into quantum state:
```
|ψ⟩ = ⊗ᵢ RY(xᵢ)|0⟩
```
Each classical feature value rotates a qubit on the Bloch sphere.

### Variational Ansatz
Parameterized quantum circuit:
```
Layer l:
  For qubit i in [0, n):
    RY(wₗ,ᵢ,₀)|i⟩
    RZ(wₗ,ᵢ,₁)|i⟩
  
  For qubits i < j:
    CNOT(control=i, target=j)
```

**Design Rationale:**
- **RY + RZ:** Covers full single-qubit rotation space (SU(2))
- **All-to-all CNOT:** Maximum entanglement, captures multi-feature correlations
- **2 Layers:** Balance between expressivity and trainability (avoids barren plateaus)

### Measurement
Observable: Pauli-Z on each qubit  
Expectation value ∈ [-1, +1] represents quantum feature

---

## Training Considerations

### Quantum-Specific Challenges

1. **Barren Plateaus:** Gradients vanish in deep quantum circuits
   - **Mitigation:** Shallow 2-layer circuits, small qubit counts (4-8)

2. **Gradient Computation:** Quantum circuits are non-differentiable
   - **Solution:** Parameter-shift rule for exact gradients

3. **Slow Simulation:** CPU-based quantum simulators are expensive
   - **Optimization:** Batch processing, efficient circuit caching

4. **Gradient Magnitude Mismatch:** Quantum gradients much smaller than classical
   - **Solution:** Learnable scaling parameter, separate learning rates

### Classical-Quantum Integration

- **Feature Scaling:** Learnable input/output scales normalize quantum contributions
- **Gradient Balancing:** Quantum gradients weighted appropriately
- **Joint Training:** End-to-end optimization with mixed classical-quantum parameters

---

## Performance Considerations

### Computational Complexity

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| GCN Layer | O(E·d) | O(N·d) |
| VQC Forward | O(L·n²·shots) | O(2ⁿ) |
| VQC Gradient | O(L·n²·params·shots) | O(2ⁿ) |

Where:
- E = edges, N = nodes, d = hidden dim
- L = circuit layers, n = qubits
- shots = measurement samples (1 for expectation)

**Bottleneck:** Quantum circuit evaluation (exponential in qubit count)

### Scalability

- **4-qubit VQC:** ~1-2s per epoch on CPU
- **8-qubit VQC:** ~10-20s per epoch on CPU
- **Classical GCN:** ~0.2s per epoch on CPU

For production use on large datasets, consider:
- GPU-accelerated quantum simulators (cuQuantum)
- Circuit batching and caching
- Approximate gradient methods

---

## Extensibility

### Adding New Models

1. Inherit from `torch.nn.Module`
2. Implement `forward(data: Batch) -> Tensor`
3. Register in `src/run_experiments.py`

### Custom Quantum Circuits

1. Define circuit in PennyLane: `@qml.qnode`
2. Wrap with `qml.qnn.TorchLayer`
3. Integrate into model's forward pass

### New Datasets

1. Add loader in `data_pipeline.py`
2. Ensure consistent `Data` object format
3. Update `run_experiments.py` to include new dataset

---

## References

- **GCN:** Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
- **VQC:** Farhi & Neven (2018) - Classification with Quantum Neural Networks on Near Term Processors
- **MoleculeNet:** Wu et al. (2018) - MoleculeNet: A Benchmark for Molecular Machine Learning
- **PennyLane:** Bergholm et al. (2018) - PennyLane: Automatic differentiation of hybrid quantum-classical computations
