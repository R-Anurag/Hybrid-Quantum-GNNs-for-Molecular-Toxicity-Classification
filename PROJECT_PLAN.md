# Hybrid Quantum GNNs for Molecular Toxicity Classification
**A Benchmark on MoleculeNet Datasets comparing classical GNNs with quantum-embedded GNNs using variational quantum circuits**

**Institution:** BMS Institute of Technology and Management, Dept. of CSE  
**Course:** Machine Learning – BCS602 | Academic Year: 2025–2026  
**Guide:** Dr. Nagabhushan SV, Associate Professor  

| USN | Name |
|---|---|
| 1BY23CS014 | Aishwarya J A |
| 1BY23CS026 | Anurag Rai |
| 1BY23CS053 | Dasiga Venkata Ashish Kumar |
| 1BY23CS068 | G Nithish |

---

## 1. Problem Statement

Classical ML models treat molecules as flat feature vectors, losing the graph topology and quantum-mechanical correlations that govern toxicity. This project investigates whether a hybrid classical–quantum pipeline — a Graph Convolutional Network (GCN) feeding into a Variational Quantum Circuit (VQC) — yields measurable improvement over a classical GCN baseline on standard molecular toxicity benchmarks.

---

## 2. Objectives

1. Build a strong classical GCN baseline on Tox21 (multi-task) and ClinTox (binary), using masked BCE loss and class weighting to handle label sparsity and imbalance.
2. Design a hybrid QGNN where GCN graph embeddings are encoded into a 4–8 qubit VQC via angle embedding; Pauli-Z expectations serve as quantum-enhanced features.
3. Benchmark the hybrid model against the classical GCN and a quantum-readout-only baseline across ROC-AUC, F1-score, parameter count, and training time using 5-fold cross-validation.
4. Explore quantum edge embedding — encoding bond-type information into entangling gate parameters — as a novel contribution and assess its marginal impact.

---

## 3. Abstract

Drug discovery demands rapid, accurate prediction of molecular toxicity. Classical ML models struggle to capture the quantum-mechanical nature of molecular interactions. This project builds and benchmarks a Hybrid Quantum Graph Neural Network (QGNN) against classical GCN baselines on two MoleculeNet datasets: **Tox21** (multi-task, 12 toxicity endpoints) and **ClinTox** (binary clinical toxicity).

Molecules are represented as graphs via RDKit (SMILES → graph). A 3-layer GCN extracts graph-level embeddings, which are then encoded into a **4–8 qubit variational quantum circuit** (PennyLane) using angle embedding. RY rotations and CNOT entangling gates process the quantum state; Pauli-Z expectation values are concatenated with classical features and passed to an MLP classifier. Performance is evaluated via **5-fold cross-validation** on ROC-AUC, F1-score, training time, and parameter efficiency.

---

## 4. Datasets

| Dataset | Task | Molecules | Notes |
|---|---|---|---|
| **Tox21** | Multi-task binary classification (12 endpoints) | ~7,831 | Heavy class imbalance; masked BCE required |
| **ClinTox** | Binary classification (clinical trial toxicity) | ~1,478 | Cleaner labels; good for ablation |

Both datasets are loaded via **DeepChem** or **MoleculeNet** loaders. SMILES strings are converted to PyTorch Geometric `Data` objects using RDKit.

**Node features (per atom):** atomic number, degree, formal charge, hybridization, aromaticity, hydrogen count — encoded as a feature vector of dimension ~9.  
**Edge features (per bond):** bond type (single/double/triple/aromatic) — used in the quantum edge embedding extension.

---

## 5. Architecture

### 5.1 Classical Backbone — GCN
```
SMILES → RDKit Graph → PyTorch Geometric Data
    ↓
GCNConv (layer 1, hidden=64) + ReLU
GCNConv (layer 2, hidden=64) + ReLU
GCNConv (layer 3, hidden=32) + ReLU
    ↓
Global Mean Pooling → graph embedding (dim=32)
```

### 5.2 Quantum Feature Encoding — VQC (PennyLane)
```
graph embedding (dim=4 or 8, projected via linear layer)
    ↓
AngleEmbedding → n-qubit quantum state  (n = 4 or 8)
    ↓
Layer 1: RY(θ) on each qubit + CNOT chain entanglement
Layer 2: RY(θ) on each qubit + CNOT chain entanglement
    ↓
Pauli-Z expectation values → quantum features (dim=n)
```

### 5.3 Hybrid Classifier — MLP
```
[classical embedding (32) || quantum features (4 or 8)]
    ↓
Linear(36/40 → 16) + ReLU + Dropout(0.3)
Linear(16 → num_tasks)
    ↓
Sigmoid → Binary Cross-Entropy (class-weighted)
```

### 5.4 Novel Extension — Quantum Edge Embedding
Bond-type features are encoded as additional rotation angles on entangling gates (parameterized CNOT → CRY), allowing bond information to modulate qubit entanglement directly. This is evaluated as an ablation on ClinTox.

---

## 6. Methodology

### Step 1 — Data Pipeline
- Load Tox21 and ClinTox via DeepChem
- Convert SMILES → `torch_geometric.data.Data` using RDKit
- Compute class weights per task for weighted BCE
- Split: 80/10/10 train/val/test; stratified per task

### Step 2 — Classical GCN Baseline
- Train 3-layer GCN with masked BCE (ignore missing labels in Tox21)
- Tune: hidden dim ∈ {64, 128}, dropout ∈ {0.2, 0.3}, lr ∈ {1e-3, 5e-4}
- Evaluate: ROC-AUC (per task + mean), F1, training time

### Step 3 — Hybrid QGNN
- Project GCN embedding to 4D / 8D via trainable linear layer
- Attach PennyLane VQC as a `qml.qnode` wrapped with `TorchLayer`
- Train end-to-end; quantum parameters updated via parameter-shift rule
- Qubit configs: 4-qubit and 8-qubit variants

### Step 4 — Quantum Edge Embedding (Ablation)
- Encode bond-type one-hot into CRY gate angles on entangling layers
- Compare vs. standard VQC on ClinTox

### Step 5 — Evaluation
- 5-fold cross-validation on both datasets
- Metrics: ROC-AUC, F1-score, parameter count, wall-clock training time per epoch
- Baselines: (a) Classical GCN, (b) Quantum-readout-only (no GCN), (c) Hybrid 4-qubit, (d) Hybrid 8-qubit, (e) Hybrid + edge embedding

---

## 7. Tech Stack

| Component | Tool/Library |
|---|---|
| Molecular graphs | RDKit, DeepChem |
| GNN | PyTorch Geometric (`torch_geometric`) |
| Quantum circuits | PennyLane (`pennylane`, `pennylane-torch`) |
| Training | PyTorch |
| Evaluation | scikit-learn |
| Experiment tracking | (optional) Weights & Biases |
| Environment | Python 3.10+, CUDA optional (VQC runs on CPU simulator) |

**Install:**
```bash
pip install torch torch-geometric rdkit-pypi deepchem pennylane pennylane-torch scikit-learn
```

---

## 8. Project Timeline

| Week | Milestone |
|---|---|
| 1 | Data pipeline: SMILES → PyG graphs for Tox21 & ClinTox; verify class distributions |
| 2 | Classical GCN baseline: train, tune, evaluate; establish benchmark numbers |
| 3 | VQC design in PennyLane; integrate as TorchLayer; test forward pass |
| 4 | End-to-end hybrid training (4-qubit) on ClinTox; debug gradient flow |
| 5 | Scale to 8-qubit; run on Tox21; 5-fold CV for all model variants |
| 6 | Quantum edge embedding ablation on ClinTox |
| 7 | Full results table, analysis, plots (ROC curves, training curves) |
| 8 | Report writing, presentation preparation, code cleanup |

---

## 9. Expected Outcomes & Success Criteria

| Metric | Classical GCN (target) | Hybrid QGNN (target) |
|---|---|---|
| Tox21 mean ROC-AUC | ≥ 0.82 | ≥ 0.83 (marginal gain acceptable) |
| ClinTox ROC-AUC | ≥ 0.85 | ≥ 0.87 |
| F1-score (ClinTox) | ≥ 0.75 | ≥ 0.76 |
| Parameter count | ~50K | ~50K + VQC params (< 5% overhead) |

> Note: Given NISQ-era simulator constraints (4–8 qubits), a large quantum advantage is not expected. The primary contribution is a rigorous, reproducible benchmark demonstrating whether quantum feature encoding provides *any* statistically significant gain, and introducing quantum edge embedding as a novel architectural element.

---

## 10. Deliverables

- [ ] Cleaned, reproducible codebase (GitHub repo)
- [ ] Trained model checkpoints for all 5 variants
- [ ] Results table with 5-fold CV mean ± std for all metrics
- [ ] Ablation: quantum edge embedding vs. standard VQC
- [ ] Final report (IEEE format)
- [ ] Presentation slides

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| VQC training is slow on CPU simulator | Use 4-qubit config first; limit Tox21 to a subset for prototyping |
| Gradient vanishing through quantum layer | Monitor parameter-shift gradients; use shallow 2-layer VQC |
| No quantum advantage observed | Frame as a rigorous negative result — still a valid scientific contribution |
| Tox21 label sparsity causes unstable CV folds | Use masked loss; stratify splits per task |
