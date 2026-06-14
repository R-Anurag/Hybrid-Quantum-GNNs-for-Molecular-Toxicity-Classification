# Experimental Results

## Overview

This document presents the benchmark results comparing classical GCN baselines with hybrid quantum-classical GNN architectures on molecular toxicity prediction tasks.

---

## Datasets

### Tox21
- **Samples:** 7,831 compounds
- **Tasks:** 12 binary toxicity endpoints
- **Class Distribution:** Highly imbalanced (1-10% positive rate per task)
- **Missing Labels:** ~30% (handled via masked loss)

### ClinTox
- **Samples:** 1,478 compounds
- **Tasks:** 1 binary clinical toxicity classification
- **Class Distribution:** Moderately imbalanced (~20% positive)
- **Missing Labels:** Minimal

---

## Model Variants

| Model | Description | Parameters |
|-------|-------------|------------|
| Classical GCN | 3-layer GCN baseline | ~50K |
| Hybrid 4-qubit | GCN + 4-qubit VQC | ~51K |
| Hybrid 8-qubit | GCN + 8-qubit VQC | ~52K |
| Quantum-Only 4-qubit | GCN encoding, quantum features only | ~48K |

---

## Evaluation Protocol

- **Cross-Validation:** 5-fold stratified
- **Metrics:** ROC-AUC (primary), F1-score, training time
- **Training:** 100 epochs max, early stopping (patience=20)
- **Hardware:** CPU-based quantum simulator (PennyLane default.qubit)

---

## Results

### Tox21 Multi-Task Classification

**ROC-AUC (mean ± std across 12 tasks)**

| Model | Mean ROC-AUC | F1-Score | Training Time/Epoch |
|-------|--------------|----------|---------------------|
| Classical GCN | TBD | TBD | TBD |
| Hybrid 4-qubit | TBD | TBD | TBD |
| Hybrid 8-qubit | TBD | TBD | TBD |
| Quantum-Only 4-qubit | TBD | TBD | TBD |

**Per-Task Performance**

| Task | Classical | Hybrid 4q | Hybrid 8q |
|------|-----------|-----------|-----------|
| NR-AR | TBD | TBD | TBD |
| NR-AR-LBD | TBD | TBD | TBD |
| NR-AhR | TBD | TBD | TBD |
| NR-Aromatase | TBD | TBD | TBD |
| NR-ER | TBD | TBD | TBD |
| NR-ER-LBD | TBD | TBD | TBD |
| NR-PPAR-gamma | TBD | TBD | TBD |
| SR-ARE | TBD | TBD | TBD |
| SR-ATAD5 | TBD | TBD | TBD |
| SR-HSE | TBD | TBD | TBD |
| SR-MMP | TBD | TBD | TBD |
| SR-p53 | TBD | TBD | TBD |

---

### ClinTox Binary Classification

**Overall Performance**

| Model | ROC-AUC | F1-Score | Precision | Recall | Training Time/Epoch |
|-------|---------|----------|-----------|--------|---------------------|
| Classical GCN | TBD | TBD | TBD | TBD | TBD |
| Hybrid 4-qubit | TBD | TBD | TBD | TBD | TBD |
| Hybrid 8-qubit | TBD | TBD | TBD | TBD | TBD |
| Quantum-Only 4-qubit | TBD | TBD | TBD | TBD | TBD |

---

## Analysis

### Key Findings

1. **Quantum Contribution:** [Analysis of whether quantum features provide measurable improvement]

2. **Scalability:** [Comparison of 4-qubit vs 8-qubit performance and computational cost]

3. **Dataset Dependency:** [How quantum advantage varies between Tox21 and ClinTox]

4. **Training Dynamics:** [Convergence speed, stability, gradient flow]

### Visualization

_Placeholder for figures:_
- ROC curves for each model
- Training/validation loss curves
- Per-task performance heatmaps
- Feature importance analysis

### Statistical Significance

_To be added:_ Paired t-tests or Wilcoxon signed-rank tests comparing classical vs. hybrid models across CV folds.

---

## Limitations

1. **Small Qubit Count:** 4-8 qubits limit quantum advantage potential
2. **CPU Simulation:** No access to quantum hardware; simulator overhead dominates
3. **Dataset Size:** Limited training data may not leverage quantum capacity fully
4. **Circuit Depth:** Shallow circuits (2 layers) to avoid barren plateaus

---

## Future Work

- [ ] Scale to 16+ qubits on GPU simulators (cuQuantum)
- [ ] Test on quantum hardware (IBM Quantum, Rigetti)
- [ ] Explore deeper variational ansätze with gradient-aware initialization
- [ ] Apply to larger molecular datasets (e.g., full ChEMBL)
- [ ] Investigate quantum kernel methods as alternative to VQC
- [ ] Quantum feature importance and interpretability analysis

---

## Reproducibility

All results can be reproduced by running:
```bash
cd src
python run_experiments.py
```

Results will be saved to `results/results.csv` with detailed per-fold metrics.

**Environment:**
- Python 3.10+
- PyTorch 2.0+
- PennyLane 0.44+
- See `requirements.txt` for full dependencies

---

## Citation

If you use these results, please cite:
```bibtex
@misc{hybridqgnn2025,
  title={Hybrid Quantum Graph Neural Networks for Molecular Toxicity Classification},
  author={Aishwarya, J. A. and Rai, Anurag and Kumar, Dasiga Venkata Ashish and Nithish, G.},
  year={2025},
  institution={BMS Institute of Technology and Management}
}
```
