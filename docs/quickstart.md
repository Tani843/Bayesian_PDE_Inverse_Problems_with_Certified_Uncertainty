# Quick Start Guide

Get up and running with Bayesian PDE inverse problems in 5 minutes!

## Overview

This framework combines PDE solving, Bayesian inference, and uncertainty quantification to solve inverse problems with certified uncertainty bounds. Here's what you can do:

- **Solve PDEs**: Finite difference and finite element methods
- **Bayesian Inference**: MCMC and variational inference for parameter estimation  
- **Uncertainty Quantification**: Certified bounds with mathematical guarantees
- **Visualization**: Publication-quality plots for results analysis

## 5-Minute Example

### Step 1: Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt

from bayesian_pde_solver.pde_solvers import FiniteDifferenceSolver
from bayesian_pde_solver.bayesian_inference import MCMCSampler
from bayesian_pde_solver.uncertainty_quantification import CertifiedBounds
from bayesian_pde_solver.visualization import PDEPlotter, UncertaintyPlotter
```

### Step 2: Set Up the Forward Problem

```python
# Define domain and mesh
domain = (0.0, 1.0, 0.0, 1.0)  # [x_min, x_max, y_min, y_max]
mesh_size = (31, 31)

# Create PDE solver
solver = FiniteDifferenceSolver(
    domain_bounds=domain,
    mesh_size=mesh_size,
    pde_type="elliptic"
)

# Define boundary conditions (Dirichlet: u = 0 on boundary)
boundary_conditions = {
    "left": {"type": "dirichlet", "value": 0.0},
    "right": {"type": "dirichlet", "value": 0.0},
    "top": {"type": "dirichlet", "value": 0.0},
    "bottom": {"type": "dirichlet", "value": 0.0}
}
```

### Step 3: Generate Synthetic Data

```python
# True parameters (unknown in real problems)
true_diffusion = 1.5
true_source_strength = 2.0

# Define true source function
def true_source(x, y):
    return true_source_strength * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

# Solve forward problem with true parameters
true_params = {
    'diffusion': true_diffusion,
    'reaction': 0.0,
    'source': true_source
}

true_solution = solver.solve(true_params, boundary_conditions)

# Generate noisy observations
np.random.seed(42)
n_obs = 20
obs_points = np.random.uniform([0.1, 0.1], [0.9, 0.9], (n_obs, 2))
true_obs_values = solver._interpolate_solution(true_solution, obs_points)
noise_std = 0.02
obs_values = true_obs_values + np.random.normal(0, noise_std, n_obs)

print(f"✅ Generated {n_obs} observations with {noise_std*100}% noise")
```

### Step 4: Define the Inverse Problem

```python
def log_posterior(params):
    """Log posterior probability for Bayesian inference."""
    diffusion, source_strength = params
    
    # Parameter bounds
    if diffusion <= 0 or source_strength <= 0:
        return -np.inf
    
    # Log-normal priors
    log_prior = (
        -0.5 * (np.log(diffusion) - 0)**2 / 0.5**2 +
        -0.5 * (np.log(source_strength) - 0)**2 / 0.5**2
    )
    
    try:
        # Define source function
        def source_func(x, y):
            return source_strength * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
        
        # Solve PDE
        pde_params = {
            'diffusion': diffusion,
            'reaction': 0.0,
            'source': source_func
        }
        
        solution = solver.solve(pde_params, boundary_conditions)
        
        # Compare with observations
        predictions = solver._interpolate_solution(solution, obs_points)
        residuals = obs_values - predictions
        log_likelihood = -0.5 * np.sum(residuals**2) / noise_std**2
        
        return log_prior + log_likelihood
        
    except Exception:
        return -np.inf

print("✅ Defined Bayesian inverse problem")
```

### Step 5: Run Bayesian Inference

```python
# Set up MCMC sampler
sampler = MCMCSampler(
    log_posterior_fn=log_posterior,
    parameter_dim=2,
    sampler_type="metropolis_hastings"
)

# Run MCMC
print("🔗 Running MCMC sampling...")
result = sampler.sample(
    n_samples=3000,
    initial_state=np.array([1.0, 1.0]),
    step_size=0.05,
    adapt_step_size=True
)

# Extract results
samples = result['samples'][600:]  # Remove burn-in
acceptance_rate = result['acceptance_rate']

print(f"✅ MCMC completed with {acceptance_rate:.3f} acceptance rate")
```

### Step 6: Analyze Results

```python
# Compute parameter estimates
param_means = np.mean(samples, axis=0)
param_stds = np.std(samples, axis=0)

print("\\n📊 Results:")
print(f"True diffusion: {true_diffusion:.3f}")
print(f"Estimated diffusion: {param_means[0]:.3f} ± {param_stds[0]:.3f}")
print(f"True source strength: {true_source_strength:.3f}")
print(f"Estimated source strength: {param_means[1]:.3f} ± {param_stds[1]:.3f}")

# Compute relative errors
rel_errors = np.abs(param_means - [true_diffusion, true_source_strength]) / [true_diffusion, true_source_strength]
print(f"Relative errors: {rel_errors[0]:.2%}, {rel_errors[1]:.2%}")
```

### Step 7: Uncertainty Quantification

```python
# Compute certified uncertainty bounds
cert_bounds = CertifiedBounds()

bounds = cert_bounds.compute_parameter_bounds(
    posterior_samples=samples,
    parameter_names=['diffusion', 'source_strength'],
    parameter_bounds=[(0.1, 5.0), (0.1, 5.0)],
    confidence_level=0.95
)

print("\\n🔒 95% Certified Bounds:")
for param_name in ['diffusion', 'source_strength']:
    emp_bounds = bounds[param_name]['empirical_ci']
    conc_bounds = bounds[param_name]['concentration']
    
    print(f"{param_name}:")
    print(f"  Empirical CI: [{emp_bounds['lower']:.3f}, {emp_bounds['upper']:.3f}]")
    print(f"  Concentration bound: [{conc_bounds['lower']:.3f}, {conc_bounds['upper']:.3f}]")
```

### Step 8: Visualization

```python
# Create visualizations
from bayesian_pde_solver.visualization import BayesianPlotter, UncertaintyPlotter

# Set up plotters
pde_plotter = PDEPlotter(style="academic")
bayesian_plotter = BayesianPlotter(style="academic") 
uncertainty_plotter = UncertaintyPlotter(style="academic")

# Plot true solution
x = np.linspace(0, 1, mesh_size[0])
y = np.linspace(0, 1, mesh_size[1])
solution_2d = true_solution.reshape(mesh_size)

fig1 = pde_plotter.plot_solution_field_2d(x, y, solution_2d, title="True PDE Solution")
plt.show()

# Plot MCMC traces
fig2 = bayesian_plotter.plot_traces(
    samples=samples,
    parameter_names=['Diffusion', 'Source Strength'],
    true_values=[true_diffusion, true_source_strength]
)
plt.show()

# Plot posterior distributions
fig3 = bayesian_plotter.plot_posterior_distributions(
    samples=samples,
    parameter_names=['Diffusion', 'Source Strength'],
    true_values=[true_diffusion, true_source_strength]
)
plt.show()

print("✅ Created visualizations")
```

## Complete Working Example

Here's the complete code you can copy and run:

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Import framework
from bayesian_pde_solver.pde_solvers import FiniteDifferenceSolver
from bayesian_pde_solver.bayesian_inference import MCMCSampler
from bayesian_pde_solver.uncertainty_quantification import CertifiedBounds

print("🚀 Bayesian PDE Inverse Problem - Quick Start")
print("=" * 50)

# 1. Set up PDE solver
solver = FiniteDifferenceSolver(
    domain_bounds=(0, 1, 0, 1),
    mesh_size=(21, 21),
    pde_type="elliptic"
)

boundary_conditions = {
    "left": {"type": "dirichlet", "value": 0.0},
    "right": {"type": "dirichlet", "value": 0.0},
    "top": {"type": "dirichlet", "value": 0.0},
    "bottom": {"type": "dirichlet", "value": 0.0}
}

# 2. Generate synthetic data
true_diffusion = 1.5
true_source_strength = 2.0

def true_source(x, y):
    return true_source_strength * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

true_params = {
    'diffusion': true_diffusion,
    'reaction': 0.0,
    'source': true_source
}

true_solution = solver.solve(true_params, boundary_conditions)

# Generate observations
n_obs = 25
obs_points = np.random.uniform([0.1, 0.1], [0.9, 0.9], (n_obs, 2))
true_obs_values = solver._interpolate_solution(true_solution, obs_points)
noise_std = 0.02
obs_values = true_obs_values + np.random.normal(0, noise_std, n_obs)

print(f"✅ Generated {n_obs} noisy observations")

# 3. Define posterior
def log_posterior(params):
    diffusion, source_strength = params
    
    if diffusion <= 0 or source_strength <= 0:
        return -np.inf
    
    log_prior = (
        -0.5 * (np.log(diffusion) - 0)**2 / 0.5**2 +
        -0.5 * (np.log(source_strength) - 0)**2 / 0.5**2
    )
    
    try:
        def source_func(x, y):
            return source_strength * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
        
        pde_params = {
            'diffusion': diffusion,
            'reaction': 0.0,
            'source': source_func
        }
        
        solution = solver.solve(pde_params, boundary_conditions)
        predictions = solver._interpolate_solution(solution, obs_points)
        residuals = obs_values - predictions
        log_likelihood = -0.5 * np.sum(residuals**2) / noise_std**2
        
        return log_prior + log_likelihood
        
    except Exception:
        return -np.inf

# 4. Run MCMC
print("🔗 Running MCMC sampling...")
sampler = MCMCSampler(log_posterior, parameter_dim=2)
result = sampler.sample(
    n_samples=2000,
    initial_state=np.array([1.0, 1.0]),
    step_size=0.05
)

samples = result['samples'][400:]
print(f"✅ MCMC completed with {result['acceptance_rate']:.3f} acceptance rate")

# 5. Analyze results
param_means = np.mean(samples, axis=0)
param_stds = np.std(samples, axis=0)

print("\\n📊 Parameter Estimates:")
print(f"Diffusion: {param_means[0]:.3f} ± {param_stds[0]:.3f} (true: {true_diffusion:.3f})")
print(f"Source: {param_means[1]:.3f} ± {param_stds[1]:.3f} (true: {true_source_strength:.3f})")

rel_errors = np.abs(param_means - [true_diffusion, true_source_strength]) / [true_diffusion, true_source_strength]
print(f"Relative errors: {rel_errors[0]:.2%}, {rel_errors[1]:.2%}")

# 6. Uncertainty quantification
cert_bounds = CertifiedBounds()
bounds = cert_bounds.compute_parameter_bounds(
    posterior_samples=samples,
    parameter_names=['diffusion', 'source_strength'],
    parameter_bounds=[(0.1, 5.0), (0.1, 5.0)],
    confidence_level=0.95
)

print("\\n🔒 95% Uncertainty Bounds:")
for i, param_name in enumerate(['diffusion', 'source_strength']):
    emp_bounds = bounds[param_name]['empirical_ci']
    true_val = [true_diffusion, true_source_strength][i]
    coverage = emp_bounds['lower'] <= true_val <= emp_bounds['upper']
    
    print(f"{param_name}: [{emp_bounds['lower']:.3f}, {emp_bounds['upper']:.3f}] "
          f"(covers true: {'✓' if coverage else '✗'})")

print("\\n🎉 Quick start completed successfully!")
print("\\nNext steps:")
print("- Explore examples/ for more complex problems")
print("- Check docs/ for detailed documentation")
print("- Try notebooks/ for interactive tutorials")
```

## What's Next?

### Explore More Examples
- **Heat Equation**: `examples/heat_equation_inference.py`
- **Reaction-Diffusion**: `examples/reaction_diffusion_system.py`
- **Real Data**: `examples/groundwater_flow.py`

### Interactive Tutorials
- **Jupyter Notebooks**: `notebooks/01_introduction.ipynb`
- **Step-by-step Guides**: `notebooks/02_pde_solvers_demo.ipynb`
- **Advanced Topics**: `notebooks/05_uncertainty_quantification.ipynb`

### Advanced Features
- **Finite Element Methods**: See `docs/finite_elements.md`
- **Variational Inference**: See `docs/variational_inference.md`
- **Custom Priors**: See `docs/custom_priors.md`
- **Large-Scale Problems**: See `docs/scalability.md`

### Getting Help
- **Documentation**: Browse `docs/` directory
- **API Reference**: `docs/api_reference.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **GitHub Issues**: Report bugs and request features

## Performance Tips

### For Faster Inference
```python
# Use smaller mesh for initial exploration
solver = FiniteDifferenceSolver(mesh_size=(15, 15))

# Reduce MCMC samples for testing
result = sampler.sample(n_samples=1000)

# Use variational inference for quick approximation
from bayesian_pde_solver.bayesian_inference import VariationalInference
vi = VariationalInference(log_posterior, parameter_dim=2)
vi_result = vi.optimize(n_iterations=1000)
```

### For Better Accuracy
```python
# Use finer mesh
solver = FiniteDifferenceSolver(mesh_size=(51, 51))

# Run longer MCMC
result = sampler.sample(n_samples=10000)

# Use finite elements for complex geometries
from bayesian_pde_solver.pde_solvers import FiniteElementSolver
fe_solver = FiniteElementSolver(mesh_size=(41, 41))
```

Congratulations! You've successfully run your first Bayesian PDE inverse problem. The framework provides powerful tools for uncertainty quantification with mathematical guarantees - perfect for scientific and engineering applications where reliability is crucial.