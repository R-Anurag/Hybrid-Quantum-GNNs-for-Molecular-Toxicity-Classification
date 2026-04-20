# Hybrid Quantum GNNs for Molecular Toxicity Classification

> A benchmark comparing classical GCNs with quantum-embedded GNNs (VQC) on MoleculeNet datasets.

**Institution:** BMS Institute of Technology and Management, Dept. of CSE  
**Course:** Machine Learning – BCS602 | AY 2025–2026  
**Guide:** Dr. Nagabhushan SV, Associate Professor

| USN | Name |
|---|---|
| 1BY23CS014 | Aishwarya J A |
| 1BY23CS026 | Anurag Rai |
| 1BY23CS053 | Dasiga Venkata Ashish Kumar |
| 1BY23CS068 | G Nithish |

---

## Overview

This project investigates whether a hybrid classical–quantum pipeline — a Graph Convolutional Network (GCN) feeding into a Variational Quantum Circuit (VQC) — yields measurable improvement over a classical GCN baseline on standard molecular toxicity benchmarks (**Tox21**, **ClinTox**).

### Architecture

```
SMILES → RDKit Graph → PyTorch Geometric Data
    ↓
3-layer GCN → graph embedding (dim=32)
    ↓
Linear projection → n-qubit angle embedding
    ↓
VQC: RY rotations + CNOT entanglement (2 layers)
    ↓
Pauli-Z expectations → quantum features (dim=n)
    ↓
[classical (32) || quantum (n)] → MLP → prediction
```

### Model Variants Benchmarked

| Variant | Description |
|---|---|
| Classical GCN | 3-layer GCN, no quantum layer |
| Hybrid 4-qubit | GCN + 4-qubit VQC |
| Hybrid 8-qubit | GCN + 8-qubit VQC |
| Hybrid 4-qubit + Edge | GCN + VQC with quantum edge embedding (CRY gates) |

---

## Setup

```bash
git clone https://github.com/<your-username>/hybrid-qgnn-toxicity.git
cd hybrid-qgnn-toxicity
pip install -r requirements.txt
```

> Python 3.10+ recommended. VQC runs on CPU simulator — no quantum hardware needed.

---

## Usage

```bash
cd src
python run_experiments.py
```

Results are saved to `results/results.csv`.

To run the data pipeline standalone:

```bash
python data_pipeline.py
```

---

## Project Structure

```
├── src/
│   ├── data_pipeline.py      # SMILES → PyG graphs, class weights
│   ├── train.py              # Masked BCE training loop
│   ├── evaluate.py           # ROC-AUC, F1, 5-fold CV
│   ├── run_experiments.py    # Main runner
│   └── models/
│       ├── gcn.py            # Classical GCN baseline
│       └── hybrid_qgnn.py    # GCN + PennyLane VQC
├── checkpoints/              # Saved model weights (git-ignored)
├── results/                  # Output CSVs and plots (git-ignored)
├── requirements.txt
└── PROJECT_PLAN.md
```

---

## Evaluation

- 5-fold cross-validation on both datasets
- Metrics: ROC-AUC, F1-score, parameter count, training time per epoch
- Masked BCE loss handles missing labels in Tox21

---

## Tech Stack

| Component | Library |
|---|---|
| Molecular graphs | RDKit, DeepChem |
| GNN | PyTorch Geometric |
| Quantum circuits | PennyLane (`pennylane-torch`) |
| Training | PyTorch |
| Evaluation | scikit-learn |
