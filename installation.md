---
layout: page
title: Installation
permalink: /installation/
---

# Installation Guide

## System Requirements

- Python 3.8 or higher
- NumPy 1.19+
- SciPy 1.6+
- Matplotlib 3.3+
- Required for advanced features: FEniCS, PyTorch

## Installation Methods

### Option 1: pip installation (Recommended)

```bash
pip install bayesian-pde-solver
```

### Option 2: Development installation

```bash
git clone https://github.com/bayesian-pde-solver/bayesian-pde-solver.git
cd bayesian-pde-solver
pip install -e .
```

### Option 3: Conda installation

```bash
conda install -c conda-forge bayesian-pde-solver
```

## Verify Installation

```python
import bayesian_pde_solver as bps
print(f"Version: {bps.__version__}")

# Run basic test
from bayesian_pde_solver import HeatEquationSolver
solver = HeatEquationSolver()
print("✓ Installation successful!")
```

## Optional Dependencies

For enhanced functionality:

```bash
# High-performance finite element methods
pip install fenics

# GPU acceleration
pip install torch

# Advanced visualization
pip install plotly seaborn

# Parallel computing
pip install mpi4py
```

## Troubleshooting

See our [troubleshooting guide](docs/troubleshooting.html) for common installation issues.