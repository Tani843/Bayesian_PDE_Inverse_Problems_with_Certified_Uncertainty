---
layout: page
title: "Documentation"
permalink: /documentation/
---

# Documentation

## Installation Guide

### Requirements
- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- Optional: Jupyter for interactive notebooks

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/tani843/Bayesian_PDE_Inverse_Problems_with_Certified_Uncertainty.git
cd Bayesian_PDE_Inverse_Problems_with_Certified_Uncertainty

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## API Reference

### Core Classes

#### `PDESolver`
Base class for finite difference and finite element solvers.

#### `BayesianInference`
Implements MCMC and variational inference algorithms.

#### `UncertaintyQuantification`
Provides certified bounds computation methods.

#### `Visualization`
Publication-quality plotting and analysis tools.

## Tutorials

### Getting Started
- [Basic Usage Tutorial](../notebooks/01_introduction.ipynb)
- [Parameter Estimation Example](../examples/heat_equation_demo.py)
- [Uncertainty Quantification Guide](../docs/uncertainty_tutorial.md)

### Advanced Topics
- Custom PDE implementation
- Multi-parameter estimation
- Parallel computation setup

## Examples Repository

Complete working examples available in the `/examples` directory:
- Heat equation parameter estimation
- Wave equation inverse problems
- Multi-physics coupled systems