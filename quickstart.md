---
layout: page
title: Quick Start
permalink: /quickstart/
---

# Quick Start Guide

Get up and running with Bayesian PDE inverse problems in 5 minutes.

## Basic Heat Equation Example

```python
import numpy as np
import matplotlib.pyplot as plt
from bayesian_pde_solver import HeatEquationSolver, BayesianInference

# 1. Set up the PDE problem
solver = HeatEquationSolver(
    domain_size=(1.0, 1.0),    # 1m x 1m domain
    grid_size=(50, 50),        # 50x50 grid
    boundary_conditions='dirichlet'
)

# 2. Generate synthetic observations
np.random.seed(42)
true_conductivity = 0.8
n_obs = 15
obs_points = np.random.uniform([0.1, 0.1], [0.9, 0.9], (n_obs, 2))
true_solution = solver.solve(thermal_conductivity=true_conductivity)
observations = solver.observe_at_points(true_solution, obs_points)
observations += np.random.normal(0, 0.05, len(observations))  # Add noise

# 3. Set up Bayesian inference
inference = BayesianInference(
    solver=solver,
    observations=observations,
    observation_points=obs_points,
    prior_bounds=(0.1, 2.0)  # Thermal conductivity bounds
)

# 4. Run MCMC sampling
posterior_samples = inference.mcmc_sampling(
    n_samples=2000,
    burn_in=500,
    algorithm='metropolis-hastings'
)

# 5. Compute certified uncertainty bounds
bounds = inference.certified_bounds(
    confidence=0.95,
    method='hoeffding'
)

# 6. Visualize results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Posterior distribution
axes[0].hist(posterior_samples, bins=50, alpha=0.7, density=True)
axes[0].axvline(true_conductivity, color='red', linestyle='--', 
                label=f'True value: {true_conductivity}')
axes[0].axvline(np.mean(posterior_samples), color='blue', 
                label=f'Posterior mean: {np.mean(posterior_samples):.3f}')
axes[0].fill_between([bounds[0], bounds[1]], 0, axes[0].get_ylim()[1], 
                     alpha=0.3, label='95% Certified bounds')
axes[0].set_xlabel('Thermal Conductivity')
axes[0].set_title('Posterior Distribution')
axes[0].legend()

# Temperature field
estimated_conductivity = np.mean(posterior_samples)
estimated_solution = solver.solve(thermal_conductivity=estimated_conductivity)
im = axes[1].imshow(estimated_solution, extent=[0, 1, 0, 1], origin='lower')
axes[1].scatter(obs_points[:, 0], obs_points[:, 1], c='red', s=50, 
                label='Observations')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Estimated Temperature Field')
plt.colorbar(im, ax=axes[1])
axes[1].legend()

plt.tight_layout()
plt.show()

# 7. Print results
print(f"True thermal conductivity: {true_conductivity:.3f}")
print(f"Estimated conductivity: {np.mean(posterior_samples):.3f} ± {np.std(posterior_samples):.3f}")
print(f"95% Certified bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
print(f"True value within bounds: {bounds[0] <= true_conductivity <= bounds[1]}")
```

## What This Example Shows

1. **PDE Setup**: Define a 2D heat equation with unknown thermal conductivity
2. **Observations**: Simulate noisy temperature measurements at random points
3. **Bayesian Inference**: Use MCMC to sample from the posterior distribution
4. **Uncertainty Quantification**: Compute mathematically certified confidence bounds
5. **Validation**: Verify that true parameter lies within certified bounds

## Next Steps

- Explore [interactive notebooks](notebooks.html) for more detailed tutorials
- Read the [documentation](documentation.html) for advanced features
- Check out [examples](examples.html) for real-world applications

## Key Concepts Covered

- **Forward Problem**: Solving PDEs with known parameters
- **Inverse Problem**: Estimating parameters from observations  
- **Bayesian Approach**: Treating parameters as random variables
- **Certified Bounds**: Mathematically rigorous uncertainty quantification