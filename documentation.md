---
layout: page
title: Documentation
permalink: /documentation/
---

# Documentation

Comprehensive guides and references for the Bayesian PDE framework.

## Core Documentation

### 📖 User Guides
- **[Mathematical Background](docs/mathematical_background.html)** - Theory and foundations
- **[API Reference](docs/api_reference.html)** - Complete function and class documentation  
- **[Examples Guide](docs/examples_guide.html)** - From basic to advanced applications

### 🔧 Technical Guides  
- **[Troubleshooting](docs/troubleshooting.html)** - Common issues and solutions
- **[Contributing](docs/contributing.html)** - Development and contribution guidelines

## Framework Components

### PDE Solvers
- Finite difference methods for regular grids
- Finite element methods for complex geometries  
- Boundary condition handling
- Multi-physics coupling support

### Bayesian Inference
- MCMC sampling algorithms (Metropolis-Hastings, HMC)
- Variational inference with mean-field approximation
- Convergence diagnostics and chain analysis
- Prior specification and sensitivity analysis

### Uncertainty Quantification  
- Concentration inequalities (Hoeffding, Bernstein, McDiarmid)
- PAC-Bayes bounds (McAllester, Seeger, Catoni)
- Finite-sample guarantees
- Coverage validation and calibration

### Visualization Tools
- PDE solution plotting
- Posterior distribution analysis
- Convergence diagnostics visualization
- Publication-quality figure generation

## Quick Navigation

| Component | Description | Key Functions |
|-----------|-------------|---------------|
| `HeatEquationSolver` | 2D heat conduction | `solve()`, `observe_at_points()` |
| `WaveEquationSolver` | Wave propagation | `solve_time_dependent()` |  
| `BayesianInference` | Parameter estimation | `mcmc_sampling()`, `variational_inference()` |
| `UncertaintyQuantifier` | Certified bounds | `hoeffding_bound()`, `pac_bayes_bound()` |
| `Visualizer` | Plotting utilities | `plot_posterior()`, `plot_solution()` |

## Mathematical Notation

The framework uses consistent mathematical notation throughout:

- **Parameters**: $\boldsymbol{\theta} \in \Theta$ (thermal conductivity, diffusion coefficients)
- **PDE Solutions**: $u(\mathbf{x}, t; \boldsymbol{\theta})$ (temperature, concentration fields)
- **Observations**: $\mathbf{y} = \{y_i\}_{i=1}^n$ (noisy measurements)
- **Posterior**: $p(\boldsymbol{\theta} | \mathbf{y}) \propto p(\mathbf{y} | \boldsymbol{\theta}) p(\boldsymbol{\theta})$

## Code Organization

```
bayesian_pde_solver/
├── pde_solvers/          # PDE solving methods
├── bayesian_inference/   # MCMC and VI algorithms  
├── uncertainty/          # Certified bounds
├── visualization/        # Plotting utilities
└── examples/            # Usage examples
```

## Performance Considerations

- Use sparse matrices for large systems
- Leverage vectorization for batch operations
- Consider GPU acceleration for intensive computations
- Implement adaptive mesh refinement for accuracy

## Research Applications

The framework has been successfully applied to:
- Thermal management in electronics
- Groundwater flow parameter estimation  
- Material property identification
- Biomedical imaging inverse problems
- Climate model calibration