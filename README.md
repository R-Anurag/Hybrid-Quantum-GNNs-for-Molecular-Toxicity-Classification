# Hybrid Quantum GNNs for Molecular Toxicity Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44%2B-green.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository benchmarks classical Graph Convolutional Networks against hybrid quantum-classical graph neural networks for molecular toxicity classification. Molecules are represented as graph-structured data derived from SMILES strings, processed with PyTorch Geometric, and evaluated on MoleculeNet toxicity datasets.

**Institution:** BMS Institute of Technology and Management, Department of Computer Science and Engineering  
**Course:** Machine Learning - BCS602, Academic Year 2025-2026  
**Supervisor:** Dr. Nagabhushan SV, Associate Professor

| USN | Name |
| --- | --- |
| 1BY23CS014 | Aishwarya J A |
| 1BY23CS026 | Anurag Rai |
| 1BY23CS053 | Dasiga Venkata Ashish Kumar |
| 1BY23CS068 | G Nithish |

## Overview

Molecular toxicity prediction is an important screening task in computational drug discovery. Conventional graph neural networks learn directly from atoms and bonds, while variational quantum circuits may provide an additional nonlinear feature transformation for graph-level molecular embeddings. This project studies whether that hybrid design improves classification performance, training stability, or model efficiency compared with a classical GCN baseline.

The main research questions are:

1. Do quantum-encoded graph embeddings improve toxicity prediction over a classical GCN baseline?
2. How do 4-qubit and 8-qubit circuits compare in accuracy and computational cost?
3. Does incorporating bond information into quantum entanglement improve performance?

## Architecture

The pipeline converts SMILES strings into molecular graphs, learns graph embeddings with GCN layers, and optionally sends those embeddings through a variational quantum circuit before classification.

```text
SMILES
  |
  v
RDKit molecular graph
  |
  v
PyTorch Geometric Data
  |
  v
3-layer GCN encoder
  |
  +-----------------------------+
  |                             |
  v                             v
Classical graph embedding       Variational quantum circuit
  |                             |
  +--------------+--------------+
                 |
                 v
          MLP classifier
                 |
                 v
        Toxicity predictions
```

Implemented model variants:

| Model | Description |
| --- | --- |
| Classical GCN | Three-layer GCN baseline with global mean pooling |
| Hybrid QGNN 4-qubit | GCN encoder plus 4-qubit variational quantum circuit |
| Hybrid QGNN 8-qubit | GCN encoder plus 8-qubit variational quantum circuit |
| Quantum-only | GCN encoder followed by quantum features without the classical bypass |

## Datasets

The experiments target standard MoleculeNet toxicity benchmarks.

| Dataset | Samples | Tasks | Notes |
| --- | ---: | ---: | --- |
| Tox21 | 7,831 | 12 | Multi-task toxicity prediction with missing labels and class imbalance |
| ClinTox | 1,478 | 1 | Clinical toxicity classification with a smaller task space |

Node features include atomic number, degree, formal charge, hybridization, aromaticity, and hydrogen count. Edge features encode bond type.

## Installation

```bash
git clone https://github.com/R-Anurag/Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification.git
cd Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

For Linux or macOS, activate the environment with `source venv/bin/activate`.

## Usage

Run the full benchmark suite:

```bash
python src/run_experiments.py
```

Train a single model configuration:

```bash
python examples/train_single_model.py
```

Generate plots from saved experiment results:

```bash
python examples/visualize_results.py
```

Run the model smoke tests:

```bash
python -m pytest tests
```

## Project Structure

```text
.
|-- src/
|   |-- data_pipeline.py
|   |-- train.py
|   |-- evaluate.py
|   |-- run_experiments.py
|   `-- models/
|       |-- gcn.py
|       |-- hybrid_qgnn.py
|       `-- quantum_only.py
|-- tests/
|   `-- test_models.py
|-- examples/
|   |-- train_single_model.py
|   `-- visualize_results.py
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- API.md
|   `-- RESULTS.md
|-- requirements.txt
|-- requirements-dev.txt
|-- setup.py
|-- CITATION.bib
|-- CONTRIBUTING.md
`-- LICENSE
```

## Methodology

Training uses masked binary cross-entropy to ignore missing labels and per-task weighting to reduce the effect of class imbalance. Evaluation uses cross-validation with ROC-AUC as the primary metric, along with F1-score, training time, and parameter count.

The quantum circuits use angle embedding, trainable RY/RZ rotations, and entanglement layers. Hybrid models concatenate classical graph embeddings with scaled quantum features before final classification.

## Documentation

Detailed notes are available in:

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Results Template](docs/RESULTS.md)
- [Project Plan](PROJECT_PLAN.md)

## Citation

```bibtex
@misc{hybridqgnn2025,
  title={Hybrid Quantum Graph Neural Networks for Molecular Toxicity Classification},
  author={Aishwarya, J. A. and Rai, Anurag and Kumar, Dasiga Venkata Ashish and Nithish, G.},
  year={2025},
  institution={BMS Institute of Technology and Management}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
