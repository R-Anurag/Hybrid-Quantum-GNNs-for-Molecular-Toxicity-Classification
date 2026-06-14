# Contributing to Hybrid Quantum GNN

We welcome contributions to this research project! This guide outlines the process for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new features
- Keep commits atomic and well-documented

## Pull Request Process

1. Update documentation for any changed functionality
2. Ensure all tests pass
3. Update the README.md if needed
4. Submit a pull request with a clear description of changes

## Research Contributions

For research-related contributions (new quantum circuits, model architectures, etc.):
- Include theoretical justification
- Provide benchmark results comparing to baseline
- Document hyperparameters and training procedures

## Questions?

Open an issue or contact the maintainers.

## Code of Conduct

Be respectful and constructive in all interactions.
