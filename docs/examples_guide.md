# Examples Guide

Comprehensive guide to the examples provided with the Bayesian PDE framework.

## Overview

The `examples/` directory contains progressively complex demonstrations of the framework capabilities:

1. **Basic Examples**: Simple problems for learning the framework
2. **Intermediate Examples**: Real-world applications with moderate complexity  
3. **Advanced Examples**: Research-level problems with full uncertainty quantification

## Basic Examples

### `minimal_example.py`

**What it demonstrates**: Core workflow with minimal dependencies

```python
# Key features shown:
- Basic PDE solving (2D Poisson equation)
- Simple MCMC parameter estimation
- Uncertainty visualization
- Requires only NumPy + Matplotlib
```

**When to use**: First-time users, teaching, resource-constrained environments

**Runtime**: ~2 minutes

### `simple_diffusion.py`

**What it demonstrates**: Heat equation parameter estimation

```python
# Problem: Estimate thermal diffusivity
# PDE: ∂u/∂t = α∇²u + f
# Unknown: thermal diffusivity α
# Data: Temperature measurements over time
```

**Key features**:
- Time-dependent PDE
- Parabolic problem setup
- Temporal data integration
- Basic uncertainty bounds

**Runtime**: ~5 minutes

## Intermediate Examples

### `reaction_diffusion_system.py`

**What it demonstrates**: Multi-parameter inference in reaction-diffusion

```python
# Problem: Gray-Scott reaction-diffusion system
# PDEs: ∂u/∂t = D_u∇²u - uv² + F(1-u)
#       ∂v/∂t = D_v∇²v + uv² - (F+k)v
# Unknowns: D_u, D_v, F, k (4 parameters)
# Data: Concentration measurements at multiple times
```

**Key features**:
- Coupled PDE system
- Multi-parameter inference
- Complex spatiotemporal patterns
- Advanced MCMC diagnostics

**Runtime**: ~15 minutes

### `groundwater_flow.py`

**What it demonstrates**: Real-world hydrogeology application

```python
# Problem: Groundwater flow parameter estimation
# PDE: ∇·(K(x,y)∇h) = S ∂h/∂t + Q
# Unknowns: Hydraulic conductivity field K(x,y)
# Data: Hydraulic head measurements from wells
```

**Key features**:
- Heterogeneous material properties
- Real measurement locations
- Finite element discretization
- Geological prior information

**Runtime**: ~10 minutes

### `heat_transfer_optimization.py`

**What it demonstrates**: Engineering design under uncertainty

```python
# Problem: Heat sink design optimization
# PDE: -∇·(k∇T) = q in Ω
# Unknowns: Material properties, boundary heat flux
# Objective: Minimize maximum temperature with uncertainty
```

**Key features**:
- Engineering constraints
- Uncertainty propagation
- Design optimization
- Safety margins with certified bounds

**Runtime**: ~20 minutes

## Advanced Examples

### `complete_workflow_demo.py`

**What it demonstrates**: Full framework capabilities

```python
# Problem: Multi-physics parameter estimation
# PDEs: Coupled flow and transport
# Methods: MCMC + Variational Inference comparison
# UQ: All concentration inequalities and PAC-Bayes bounds
# Visualization: Publication-quality figures
```

**Key features**:
- Complete Bayesian workflow
- Method comparison (MCMC vs VI)
- Comprehensive uncertainty quantification
- Advanced visualization suite
- Performance benchmarking

**Runtime**: ~30 minutes

### `adaptive_mesh_refinement.py`

**What it demonstrates**: Error control and mesh adaptation

```python
# Problem: Adaptive solution of inverse problem
# Features: Goal-oriented error estimation
# Adaptation: Mesh refinement based on posterior uncertainty
# Goal: Balance computational cost vs accuracy
```

**Key features**:
- A posteriori error estimation
- Adaptive mesh refinement
- Goal-oriented adaptation
- Computational efficiency analysis

**Runtime**: ~25 minutes

### `multilevel_monte_carlo.py`

**What it demonstrates**: Advanced sampling for expensive PDEs

```python
# Problem: Reduce computational cost for expensive forward solves
# Method: Multilevel Monte Carlo (MLMC)
# Goal: Achieve target accuracy with minimal cost
```

**Key features**:
- Hierarchy of discretizations
- Variance reduction techniques
- Cost-accuracy tradeoffs
- Parallel implementation

**Runtime**: ~45 minutes

### `real_data_application.py`

**What it demonstrates**: Analysis of experimental data

```python
# Problem: Parameter estimation from laboratory measurements
# Data: Real experimental data with complex noise structure
# Challenges: Model selection, outlier detection, validation
```

**Key features**:
- Real experimental data
- Model uncertainty
- Validation techniques
- Publication-ready analysis

**Runtime**: ~20 minutes

## Example Usage Patterns

### Running Examples

```bash
# Basic usage
cd examples/
python minimal_example.py

# With custom parameters
python simple_diffusion.py --mesh-size 51 --n-samples 10000

# Save results
python reaction_diffusion_system.py --output-dir results/

# Parallel execution
python complete_workflow_demo.py --n-cores 4
```

### Modifying Examples

```python
# Template for custom problems
import numpy as np
from bayesian_pde_solver import *

# 1. Define your PDE problem
def your_pde_solver(parameters):
    # Set up domain, mesh, boundary conditions
    # Solve PDE with given parameters
    # Return solution
    pass

# 2. Define observation model
def generate_observations(true_solution, noise_level):
    # Extract values at measurement locations
    # Add realistic noise
    # Return observations
    pass

# 3. Set up Bayesian inference
def log_posterior(params):
    # Prior + likelihood computation
    # Use your_pde_solver internally
    # Return log probability
    pass

# 4. Run inference and analysis
# Follow patterns from existing examples
```

## Example Outputs

### Generated Files

Each example creates:
```
results/
├── figures/
│   ├── solution_field.png
│   ├── mcmc_traces.png
│   ├── posterior_distributions.png
│   ├── uncertainty_bounds.png
│   └── convergence_analysis.png
├── data/
│   ├── samples.h5
│   ├── summary_statistics.csv
│   └── metadata.json
└── reports/
    ├── summary_report.html
    └── technical_details.pdf
```

### Typical Outputs

**Console Output Example**:
```
🚀 Starting Bayesian PDE Analysis
================================

📊 Problem Setup:
  Domain: [0, 1] × [0, 1]
  Mesh: 51 × 51 points
  Parameters: 2 (diffusion, source_strength)
  Observations: 25 points

🔗 Running MCMC:
  Progress: 100%|████████████| 5000/5000 [02:34<00:00, 32.4it/s]
  Acceptance rate: 0.423

📈 Parameter Estimates:
  Diffusion: 1.487 ± 0.156 (true: 1.500)
  Source: 2.031 ± 0.203 (true: 2.000)
  
🔒 95% Certified Bounds:
  Diffusion: [1.201, 1.773] ✓ (covers true)
  Source: [1.651, 2.411] ✓ (covers true)

✅ Analysis completed successfully!
```

## Customization Guide

### Problem-Specific Modifications

#### 1. Different PDE Types

```python
# Modify solver initialization
if pde_type == "wave_equation":
    solver = WaveEquationSolver(domain, mesh_size)
elif pde_type == "navier_stokes":
    solver = NavierStokesSolver(domain, mesh_size, reynolds_number)
```

#### 2. Custom Boundary Conditions

```python
# Define spatially-varying boundary conditions
def inlet_velocity(x, y, t):
    return 1.0 + 0.1 * np.sin(2 * np.pi * t)

boundary_conditions = {
    "inlet": {"type": "dirichlet", "value": inlet_velocity},
    "outlet": {"type": "neumann", "value": 0.0},
    "walls": {"type": "dirichlet", "value": 0.0}
}
```

#### 3. Complex Observation Models

```python
# Non-Gaussian noise
def log_likelihood_student_t(predictions, observations, nu=3):
    residuals = observations - predictions
    return np.sum(scipy.stats.t.logpdf(residuals, df=nu))

# Measurement operator (e.g., integral measurements)
def measurement_operator(solution, measurement_domains):
    measurements = []
    for domain in measurement_domains:
        # Integrate solution over measurement domain
        measurement = integrate_over_domain(solution, domain)
        measurements.append(measurement)
    return np.array(measurements)
```

#### 4. Advanced Priors

```python
# Hierarchical priors
def log_prior_hierarchical(params):
    # Extract hyperparameters
    hyperparams = params[-2:]
    main_params = params[:-2]
    
    # Hyperprior
    log_hyperprior = scipy.stats.gamma.logpdf(hyperparams, a=2, scale=1)
    
    # Conditional prior
    log_conditional = scipy.stats.multivariate_normal.logpdf(
        main_params, mean=0, cov=np.diag(hyperparams)
    )
    
    return np.sum(log_hyperprior) + log_conditional

# Gaussian process priors for spatial fields
def log_prior_gp(field_values, coordinates, length_scale, variance):
    from sklearn.gaussian_process.kernels import RBF
    
    kernel = variance * RBF(length_scale=length_scale)
    K = kernel(coordinates)
    
    return scipy.stats.multivariate_normal.logpdf(field_values, mean=0, cov=K)
```

### Performance Optimization

#### 1. Faster Forward Solves

```python
# Use compiled solvers for repeated evaluations
@numba.jit(nopython=True)
def fast_pde_solve_2d(diffusion_field, source_field, mesh_size):
    # Optimized finite difference implementation
    pass

# Precompute system matrices
class PrecomputedSolver:
    def __init__(self, domain, mesh_size):
        self.base_matrix = self._build_base_matrix()
        
    def solve(self, diffusion, source):
        # Modify base matrix and solve
        A = diffusion * self.base_matrix
        return spsolve(A, source)
```

#### 2. Parallel MCMC

```python
# Multiple chain parallel sampling
from multiprocessing import Pool

def run_parallel_mcmc(n_chains=4):
    with Pool(n_chains) as pool:
        seeds = [42 + i for i in range(n_chains)]
        results = pool.map(run_single_chain, seeds)
    
    # Combine chains
    all_samples = np.vstack([r['samples'] for r in results])
    return all_samples

def run_single_chain(seed):
    np.random.seed(seed)
    return sampler.sample(n_samples=n_samples_per_chain)
```

#### 3. Memory-Efficient Storage

```python
# Stream large results to disk
import h5py

def streaming_mcmc_to_disk(filename, n_samples, batch_size=1000):
    with h5py.File(filename, 'w') as f:
        # Pre-allocate dataset
        samples_dataset = f.create_dataset(
            'samples', 
            (n_samples, parameter_dim), 
            dtype=np.float64
        )
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_samples = sampler.sample(n_samples=batch_end-batch_start)
            
            # Write directly to disk
            samples_dataset[batch_start:batch_end] = batch_samples['samples']
```

## Best Practices

### 1. Development Workflow

```python
# Start with minimal example
# 1. Get basic PDE solving working
# 2. Add simple 1-parameter inference
# 3. Gradually increase complexity
# 4. Add full uncertainty quantification

# Use progressive mesh refinement
mesh_sizes = [(11, 11), (21, 21), (41, 41)]
for mesh_size in mesh_sizes:
    solver = FiniteDifferenceSolver(domain, mesh_size)
    # Test convergence
```

### 2. Validation Strategy

```python
# Synthetic data validation
def validate_with_synthetic_data():
    # 1. Generate data with known parameters
    # 2. Run inference
    # 3. Check parameter recovery
    # 4. Verify uncertainty calibration
    pass

# Cross-validation for real data
def k_fold_validation(data, k=5):
    # Split data into k folds
    # Train on k-1 folds, test on 1
    # Report predictive performance
    pass
```

### 3. Computational Guidelines

```python
# Resource planning
def estimate_computational_cost():
    # Single PDE solve time
    single_solve_time = time_pde_solve()
    
    # MCMC requirements
    n_samples = 10000
    mcmc_time = n_samples * single_solve_time
    
    # Add overhead for adaptation, diagnostics
    total_time = mcmc_time * 1.5
    
    print(f"Estimated runtime: {total_time/3600:.1f} hours")
    return total_time
```

## Troubleshooting Examples

### Common Issues

1. **Example won't run**: Check dependencies in requirements.txt
2. **Poor parameter recovery**: Increase n_samples, check prior specifications
3. **Slow performance**: Use coarser mesh initially, enable parallel processing
4. **Memory issues**: Reduce mesh size, use streaming storage
5. **Convergence problems**: Check MCMC diagnostics, adjust step sizes

### Getting Help

- Each example includes detailed comments explaining the approach
- Check the troubleshooting guide for common issues
- Modify examples gradually to understand each component
- Use the minimal example as a starting point for custom problems

The examples provide a comprehensive learning path from basic concepts to advanced research applications. Start with the minimal example and progressively work through more complex scenarios as you become familiar with the framework.