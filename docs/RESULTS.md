# Experimental Results

This document records the reporting format for benchmark results. Fill in the tables after running the full experiment suite.

## Evaluation Protocol

- Datasets: Tox21 and ClinTox.
- Validation: 5-fold cross-validation.
- Primary metric: ROC-AUC.
- Secondary metrics: F1-score, training time, and parameter count.
- Training: AdamW optimization with early stopping on validation ROC-AUC.
- Loss: masked binary cross-entropy with class weighting.

## Tox21

| Model | ROC-AUC | F1-score | Parameters | Time per epoch |
| --- | ---: | ---: | ---: | ---: |
| Classical GCN | TBD | TBD | TBD | TBD |
| Hybrid QGNN 4-qubit | TBD | TBD | TBD | TBD |
| Hybrid QGNN 8-qubit | TBD | TBD | TBD | TBD |
| Quantum-only 4-qubit | TBD | TBD | TBD | TBD |

## ClinTox

| Model | ROC-AUC | F1-score | Parameters | Time per epoch |
| --- | ---: | ---: | ---: | ---: |
| Classical GCN | TBD | TBD | TBD | TBD |
| Hybrid QGNN 4-qubit | TBD | TBD | TBD | TBD |
| Hybrid QGNN 8-qubit | TBD | TBD | TBD | TBD |
| Quantum-only 4-qubit | TBD | TBD | TBD | TBD |

## Analysis Checklist

When experiment outputs are available, include:

- Mean and standard deviation across folds.
- Per-task Tox21 ROC-AUC values.
- Training curves for representative runs.
- Runtime comparison across model variants.
- Statistical comparison between the classical and hybrid models.

## Reproducibility

Run the full benchmark with:

```bash
python src/run_experiments.py
```

The experiment runner writes outputs under `results/`, which is ignored by Git to avoid committing generated artifacts.
