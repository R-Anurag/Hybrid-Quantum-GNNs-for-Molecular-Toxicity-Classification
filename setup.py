"""
Setup script for Hybrid Quantum GNN package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid-qgnn",
    version="1.0.0",
    author="Aishwarya J A, Anurag Rai, Dasiga Venkata Ashish Kumar, G Nithish",
    author_email="",
    description="Hybrid Quantum Graph Neural Networks for Molecular Toxicity Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/R-Anurag/Hybrid-Quantum-GNNs-for-Molecular-Toxicity-Classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-qgnn-train=src.run_experiments:main",
        ],
    },
)
