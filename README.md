# Hybrid Quantum GNNs for Molecular Toxicity Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44+-green.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive benchmark comparing classical Graph Convolutional Networks (GCNs) with hybrid quantum-classical Graph Neural Networks on molecular toxicity prediction tasks using MoleculeNet datasets.

**Institution:** BMS Institute of Technology and Management, Department of Computer Science and Engineering  
**Course:** Machine Learning – BCS602 | Academic Year 2025–2026  
**Supervised by:** Dr. Nagabhushan SV, Associate Professor

**Team Members:**
| USN | Name |
|-----|------|
| 1BY23CS014 | Aishwarya J A |
| 1BY23CS026 | Anurag Rai |
| 1BY23CS053 | Dasiga Venkata Ashish Kumar |
| 1BY23CS068 | G Nithish |

---

## 🔬 Overview

Drug discovery requires rapid and accurate prediction of molecular toxicity. Traditional machine learning approaches treat molecules as flat feature vectors, losing critical structural information and quantum-mechanical correlations. This project investigates whether integrating **Variational Quantum Circuits (VQCs)** with **Graph Neural Networks** can capture these complex relationships more effectively than classical methods alone.

### Key Research Questions

1. Can quantum feature encoding provide measurable improvements over classical GNN baselines?
2. How does quantum circuit depth (4 vs 8 qubits) affect model performance and computational cost?
3. Does quantum edge embedding (encoding bond information in entanglement gates) enhance predictions?

---

## 🏗️ Architecture

Our hybrid approach combines classical graph learning with quantum feature encoding:

```
SMILES String
    ↓
[RDKit] → Molecular Graph
    ↓
[PyTorch Geometric] → Graph Data Structure
    ↓
┌─────────────────────────────────────┐
│   3-Layer Graph Convolutional Net   │
│   (Message Passing + Aggregation)   │
└─────────────────┬───────────────────┘
                  ↓
         Graph Embedding (32D)
                  ↓
         ┌────────┴────────┐
         ↓                 ↓
   Classical Path    Quantum Path
   (Direct Use)      (VQC Encoding)
         ↓                 ↓
         │      ┌──────────────────────┐
         │      │  Linear Projection   │
         │      │         ↓            │
         │      │  Angle Embedding     │
         │      │         ↓            │
         │      │  Variational Layers: │
         │      │   • RY/RZ Rotations  │
         │      │   • CNOT Entanglement│
         │      │         ↓            │
         │      │  Pauli-Z Measurement │
         │      └──────────┬───────────┘
         │                 ↓
         │        Quantum Features (4D/8D)
         │                 ↓
         │      [Learnable Scaling: λ]
         └─────────┬───────┘
                   ↓
          [Concatenation]
                   ↓
        ┌──────────────────┐
        │   MLP Classifier  │
        │   (2 Hidden Layers)│
        └──────────┬─────────┘
                   ↓
         Toxicity Predictions
```

### Model Variants

| Variant | Description | Parameters |
|---------|-------------|------------|
| **Classical GCN** | 3-layer GCN baseline (no quantum) | ~50K |
| **Hybrid 4-qubit** | GCN + 4-qubit VQC | ~51K |
| **Hybrid 8-qubit** | GCN + 8-qubit VQC | ~52K |
| **Quantum-Only** | GCN encoding → quantum features only | ~48K |

---

## 📊 Datasets

We benchmark on two standard molecular toxicity datasets from **MoleculeNet**:

### Tox21
- **Size:** 7,831 compounds
- **Task:** Multi-task binary classification (12 toxicity endpoints)
- **Challenges:** Severe class imbalance (1-10% positive rate), ~30% missing labels
- **Solution:** Masked Binary Cross-Entropy with per-task class weighting

### ClinTox
- **Size:** 1,478 compounds  
- **Task:** Binary clinical trial toxicity classification
- **Challenges:** Moderate class imbalance (~20% positive)
- **Use Case:** Cleaner dataset for focused ablation studies

**Molecular Representation:**
- **Input:** SMILES strings
- **Node Features:** Atomic number, degree, formal charge, hybridization, aromaticity, hydrogen count
- **Edge Features:** Bond type (single/double/triple/aromatic)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/R-Anurag/Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification.git
cd Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

**System Requirements:**
- Python 3.10 or higher
- 8GB+ RAM recommended
- CPU sufficient (quantum simulation runs on CPU)
- GPU optional (speeds up classical GCN training)

### Running Experiments

**Full Benchmark Suite:**
```bash
cd src
python run_experiments.py
```

This runs 5-fold cross-validation for all model variants on both datasets. Results saved to `results/results.csv`.

**Train Single Model:**
```bash
cd examples
python train_single_model.py
```

**Visualize Results:**
```bash
cd examples
python visualize_results.py
```

---

## 📁 Project Structure

```
Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification/
│
├── src/                          # Source code
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py               # Classical GCN baseline
│   │   ├── hybrid_qgnn.py       # Hybrid quantum-classical model
│   │   └── quantum_only.py      # Quantum-only variant
│   ├── data_pipeline.py         # SMILES → PyG graph conversion
│   ├── train.py                 # Training loop with early stopping
│   ├── evaluate.py              # Cross-validation and metrics
│   └── run_experiments.py       # Main experiment runner
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── unit_tests.py            # Model architecture validation
│
├── examples/                     # Usage examples
│   ├── train_single_model.py    # Single model training demo
│   └── visualize_results.py     # Result visualization
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md          # Detailed system design
│   ├── RESULTS.md               # Experimental results (template)
│   └── API.md                   # Code documentation
│
├── results/                      # Experiment outputs (git-ignored)
├── checkpoints/                  # Saved models (git-ignored)
│
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package installation script
├── .gitignore
├── LICENSE                       # MIT License
├── CITATION.bib                  # Citation information
├── CONTRIBUTING.md               # Contribution guidelines
├── PROJECT_PLAN.md               # Detailed project plan
└── README.md                     # This file
```

---

## 🔬 Methodology

### Training Protocol

1. **Data Preprocessing:**
   - SMILES → RDKit molecular graph
   - Extract node (atom) and edge (bond) features
   - Validate graph integrity (no isolated nodes)

2. **Model Training:**
   - **Optimizer:** AdamW (weight decay = 1e-5)
   - **Learning Rate:** 1e-3 (classical), separate LR for quantum parameters
   - **Batch Size:** 32
   - **Max Epochs:** 100
   - **Early Stopping:** Patience = 20 epochs (based on validation ROC-AUC)
   - **Gradient Clipping:** max_norm = 1.0 (prevents quantum gradient explosion)

3. **Loss Function:**
   - Masked Binary Cross-Entropy (handles missing labels)
   - Per-task class weighting (handles imbalance)

4. **Evaluation:**
   - **Cross-Validation:** 5-fold stratified
   - **Metrics:** ROC-AUC (primary), F1-score, training time, parameter count
   - **Reporting:** Mean ± standard deviation across folds

### Quantum Circuit Design

**Encoding Layer:**
```
AngleEmbedding: |ψ⟩ = ⊗ᵢ RY(xᵢ)|0⟩
```

**Variational Layers (repeated L=2 times):**
```
For each qubit i:
    RY(wᵢ) • RZ(wᵢ)

For all qubit pairs i < j:
    CNOT(control=i, target=j)
```

**Measurement:**
```
Quantum features = [⟨Z₀⟩, ⟨Z₁⟩, ..., ⟨Zₙ₋₁⟩]
```

**Key Design Choices:**
- **Multi-axis rotations (RY + RZ):** Full single-qubit gate coverage
- **All-to-all entanglement:** Captures multi-feature correlations
- **Shallow depth (2 layers):** Avoids barren plateau problem
- **Learnable scaling:** Balances classical and quantum gradient magnitudes

---

## 📈 Evaluation Metrics

- **ROC-AUC:** Primary metric (handles class imbalance well)
- **F1-Score:** Balance of precision and recall
- **Training Time:** Computational efficiency per epoch
- **Parameter Count:** Model complexity

All metrics computed per task, then averaged (with handling for missing labels in Tox21).

---

## 🛠️ Tech Stack

| Component | Library/Framework |
|-----------|------------------|
| **Molecular Processing** | RDKit, DeepChem |
| **Graph Neural Networks** | PyTorch Geometric |
| **Quantum Circuits** | PennyLane |
| **Deep Learning** | PyTorch |
| **Metrics & CV** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## 📚 Documentation

- **[Architecture Documentation](docs/ARCHITECTURE.md)** - Detailed system design and component breakdown
- **[API Documentation](docs/API.md)** - Code usage and function references
- **[Results](docs/RESULTS.md)** - Experimental results and analysis (template)
- **[Project Plan](PROJECT_PLAN.md)** - Comprehensive project roadmap and methodology

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
cd tests
python unit_tests.py
```

---

## 📊 Expected Outcomes

Given the constraints of NISQ-era quantum simulators (4-8 qubits, CPU-based), we expect:

1. **Modest Quantum Advantage:** 1-3% ROC-AUC improvement over classical baseline (if any)
2. **Computational Cost:** 5-10× slower training due to quantum circuit evaluation
3. **Dataset Dependency:** Potential for quantum benefit on smaller, complex datasets (ClinTox) vs. large, noisy datasets (Tox21)

**Primary Contribution:** A rigorous, reproducible benchmark establishing baseline performance for hybrid quantum-classical GNNs in molecular property prediction.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to Contribute:**
- Report bugs or request features (open an issue)
- Implement new quantum circuit ansätze
- Add support for additional datasets
- Improve documentation
- Optimize performance

---

## 📄 Citation

If you use this code or build upon this work, please cite:

```bibtex
@misc{hybridqgnn2025,
  title={Hybrid Quantum Graph Neural Networks for Molecular Toxicity Classification},
  author={Aishwarya, J. A. and Rai, Anurag and Kumar, Dasiga Venkata Ashish and Nithish, G.},
  year={2025},
  institution={BMS Institute of Technology and Management},
  howpublished={\url{https://github.com/R-Anurag/Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification}}
}
```

See [CITATION.bib](CITATION.bib) for full citation information including references.

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Dr. Nagabhushan SV** for project guidance and mentorship
- **BMS Institute of Technology and Management** for institutional support
- **MoleculeNet** for providing benchmark datasets
- The **PennyLane**, **PyTorch Geometric**, and **RDKit** communities for excellent open-source tools

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- Open an issue on GitHub
- Contact the team through BMS Institute of Technology and Management

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Small qubit counts (4-8) limit quantum advantage potential
- CPU simulation creates computational overhead
- No access to quantum hardware for validation

**Future Directions:**
- Scale to 16+ qubits using GPU simulators (cuQuantum)
- Test on actual quantum hardware (IBM Quantum, Rigetti)
- Explore quantum kernel methods
- Apply to larger molecular databases (ChEMBL, ZINC)
- Investigate quantum feature interpretability

---

<div align="center">

**Star ⭐ this repository if you find it useful!**

</div>
