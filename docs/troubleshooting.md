# Troubleshooting Guide

Common issues and solutions for the Bayesian PDE Inverse Problems framework.

## Installation Issues

### Import Errors

**Problem**: `ModuleNotFoundError` when importing bayesian_pde_solver

**Solutions**:
```bash
# Check if package is installed
pip list | grep bayesian-pde

# Install in development mode
cd /path/to/Bayesian_PDE_Inverse_Problems_with_Certified_Uncertainty
pip install -e .

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```

**Problem**: `ImportError: No module named 'numpy'`

**Solutions**:
```bash
# Install core dependencies first
pip install numpy scipy matplotlib

# Or use conda
conda install numpy scipy matplotlib

# For Apple M1/M2 Macs
conda install numpy scipy matplotlib -c conda-forge
```

### Dependency Conflicts

**Problem**: Version conflicts between packages

**Solutions**:
```bash
# Create clean environment
conda create -n bayesian-pde-clean python=3.9
conda activate bayesian-pde-clean

# Install step by step
pip install numpy==1.21.0 scipy==1.7.0
pip install matplotlib==3.5.0
pip install -e .

# Check for conflicts
pip check
```

**Problem**: `RuntimeError: BLAS/LAPACK linking issues`

**Solutions**:
```bash
# Install optimized BLAS
conda install mkl mkl-service

# Or use OpenBLAS
conda install openblas

# Set environment variables
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## PDE Solver Issues

### Convergence Problems

**Problem**: `PDESolverError: Linear system did not converge`

**Diagnosis**:
```python
import numpy as np
from scipy.linalg import cond

# Check system condition number
A, b = solver.assemble_system(parameters, boundary_conditions)
condition_number = cond(A.toarray())
print(f"Condition number: {condition_number}")

if condition_number > 1e12:
    print("System is ill-conditioned")
```

**Solutions**:
```python
# 1. Refine mesh
solver = FiniteDifferenceSolver(
    domain_bounds=(0, 1, 0, 1),
    mesh_size=(101, 101),  # Finer mesh
    pde_type="elliptic"
)

# 2. Use iterative solver with preconditioning
from scipy.sparse.linalg import cg, spsolve
from scipy.sparse import diags

# Diagonal preconditioning
M = diags(A.diagonal())
solution, info = cg(A, b, M=M, tol=1e-8)

# 3. Regularize parameters
parameters = {
    'diffusion': max(1e-6, diffusion),  # Avoid zero diffusion
    'reaction': reaction,
    'source': source
}
```

### Boundary Condition Errors

**Problem**: `BoundaryConditionError: Invalid boundary condition type`

**Solutions**:
```python
# Correct boundary condition format
boundary_conditions = {
    "left": {"type": "dirichlet", "value": 0.0},
    "right": {"type": "dirichlet", "value": 1.0},
    "top": {"type": "neumann", "value": lambda x, y: 0.0},
    "bottom": {"type": "dirichlet", "value": 0.0}
}

# For functions, ensure they're callable
def bc_function(x, y):
    return np.sin(np.pi * x)

boundary_conditions = {
    "left": {"type": "dirichlet", "value": bc_function},
    # ...
}
```

### Memory Issues with Large Meshes

**Problem**: `MemoryError` for large problems

**Solutions**:
```python
# 1. Use sparse matrices explicitly
import scipy.sparse as sp

# 2. Reduce mesh size initially
solver = FiniteDifferenceSolver(
    mesh_size=(51, 51),  # Start smaller
    pde_type="elliptic"
)

# 3. Use iterative solvers
from scipy.sparse.linalg import gmres, bicgstab

solution, info = gmres(A, b, tol=1e-6, restart=50)

# 4. Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## MCMC Issues

### Low Acceptance Rates

**Problem**: Acceptance rate < 10%

**Diagnosis**:
```python
result = sampler.sample(n_samples=1000, step_size=0.1)
print(f"Acceptance rate: {result['acceptance_rate']}")

# Check posterior values
log_probs = [log_posterior(sample) for sample in result['samples'][:10]]
print(f"Log posterior range: {min(log_probs)} to {max(log_probs)}")
```

**Solutions**:
```python
# 1. Reduce step size
result = sampler.sample(
    n_samples=5000,
    step_size=0.01,  # Much smaller
    initial_state=good_starting_point
)

# 2. Use adaptive step size
result = sampler.sample(
    n_samples=5000,
    step_size=0.1,
    adapt_step_size=True,
    target_acceptance_rate=0.4
)

# 3. Better initialization
# Find maximum a posteriori (MAP) estimate first
from scipy.optimize import minimize

def neg_log_posterior(params):
    return -log_posterior(params)

map_result = minimize(neg_log_posterior, x0=np.array([1.0, 1.0]))
good_starting_point = map_result.x

result = sampler.sample(
    n_samples=5000,
    initial_state=good_starting_point,
    step_size=0.05
)
```

### High Acceptance Rates

**Problem**: Acceptance rate > 80%

**Solutions**:
```python
# Increase step size for better mixing
result = sampler.sample(
    n_samples=5000,
    step_size=0.5,  # Larger steps
    initial_state=initial_state
)

# Check effective sample size
def effective_sample_size(samples):
    """Compute effective sample size"""
    n = len(samples)
    autocorr = np.correlate(samples - np.mean(samples), 
                           samples - np.mean(samples), mode='full')
    autocorr = autocorr[n-1:]
    autocorr = autocorr / autocorr[0]
    
    # Find first negative value
    first_negative = np.where(autocorr < 0)[0]
    if len(first_negative) > 0:
        cutoff = first_negative[0]
    else:
        cutoff = len(autocorr)
    
    return n / (1 + 2 * np.sum(autocorr[1:cutoff]))

for i in range(samples.shape[1]):
    ess = effective_sample_size(samples[:, i])
    print(f"Parameter {i} ESS: {ess:.1f}")
```

### Posterior Evaluation Errors

**Problem**: `log_posterior` returns NaN or -inf frequently

**Diagnosis**:
```python
def debug_log_posterior(params):
    diffusion, source_strength = params
    print(f"Testing params: {params}")
    
    # Check bounds
    if diffusion <= 0:
        print("Diffusion <= 0")
        return -np.inf
    if source_strength <= 0:
        print("Source strength <= 0")
        return -np.inf
    
    # Check prior
    log_prior = -0.5 * np.sum(params**2)
    print(f"Log prior: {log_prior}")
    
    try:
        # PDE solve
        solution = solver.solve(pde_params, boundary_conditions)
        print(f"Solution range: {np.min(solution)} to {np.max(solution)}")
        
        # Likelihood
        predictions = solver._interpolate_solution(solution, obs_points)
        residuals = obs_values - predictions
        log_likelihood = -0.5 * np.sum(residuals**2) / noise_std**2
        print(f"Log likelihood: {log_likelihood}")
        
        return log_prior + log_likelihood
        
    except Exception as e:
        print(f"PDE solve failed: {e}")
        return -np.inf

# Test with reasonable parameters
test_params = np.array([1.0, 1.0])
debug_result = debug_log_posterior(test_params)
```

**Solutions**:
```python
def robust_log_posterior(params):
    diffusion, source_strength = params
    
    # Robust bounds checking
    if not (0.01 <= diffusion <= 10.0):
        return -np.inf
    if not (0.01 <= source_strength <= 10.0):
        return -np.inf
    
    # Robust prior
    log_prior = (
        -0.5 * (np.log(diffusion) - 0)**2 / 1.0**2 +
        -0.5 * (np.log(source_strength) - 0)**2 / 1.0**2
    )
    
    if not np.isfinite(log_prior):
        return -np.inf
    
    try:
        # Robust PDE parameters
        def source_func(x, y):
            try:
                result = source_strength * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
                return np.clip(result, 0, 1e6)  # Clip extreme values
            except:
                return 0.0
        
        pde_params = {
            'diffusion': diffusion,
            'reaction': 0.0,
            'source': source_func
        }
        
        solution = solver.solve(pde_params, boundary_conditions)
        
        # Check solution validity
        if not np.all(np.isfinite(solution)):
            return -np.inf
        
        predictions = solver._interpolate_solution(solution, obs_points)
        
        if not np.all(np.isfinite(predictions)):
            return -np.inf
        
        residuals = obs_values - predictions
        log_likelihood = -0.5 * np.sum(residuals**2) / noise_std**2
        
        if not np.isfinite(log_likelihood):
            return -np.inf
        
        return log_prior + log_likelihood
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error in log_posterior: {e}")
        return -np.inf
```

## Variational Inference Issues

### ELBO Not Improving

**Problem**: ELBO remains constant or decreases

**Diagnosis**:
```python
# Plot ELBO history
import matplotlib.pyplot as plt

result = vi.optimize(n_iterations=1000)
plt.plot(result['elbo_history'])
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ELBO Convergence')
plt.show()

print(f"Initial ELBO: {result['elbo_history'][0]}")
print(f"Final ELBO: {result['elbo_history'][-1]}")
print(f"Improvement: {result['elbo_history'][-1] - result['elbo_history'][0]}")
```

**Solutions**:
```python
# 1. Adjust learning rate
result = vi.optimize(
    n_iterations=2000,
    learning_rate=0.001,  # Smaller learning rate
    n_samples=100,
    optimizer="adam"
)

# 2. Increase sample size for ELBO estimation
result = vi.optimize(
    n_iterations=1000,
    learning_rate=0.01,
    n_samples=200,  # More samples
    optimizer="adam"
)

# 3. Try different optimizer
result = vi.optimize(
    n_iterations=1000,
    learning_rate=0.05,
    optimizer="adagrad"  # Different optimizer
)

# 4. Better initialization
vi.variational_params['means'] = map_estimate  # Use MAP as starting point
vi.variational_params['log_stds'] = np.log(0.1) * np.ones(parameter_dim)
```

### Gradient Issues

**Problem**: `RuntimeError: Gradient computation failed`

**Solutions**:
```python
# 1. Use finite differences instead of analytical gradients
vi = VariationalInference(
    log_posterior_fn=log_posterior,
    parameter_dim=2,
    vi_type="mean_field"
)

# Reduce finite difference step size
result = vi.optimize(
    n_iterations=1000,
    learning_rate=0.01
)

# 2. Check gradient numerically
def check_gradients(vi, samples):
    analytical_grad = vi.compute_elbo_gradient_analytic(samples)
    numerical_grad = vi.compute_elbo_gradient(samples, h=1e-6)
    
    diff = np.abs(analytical_grad - numerical_grad)
    rel_diff = diff / (np.abs(analytical_grad) + 1e-8)
    
    print(f"Max absolute difference: {np.max(diff)}")
    print(f"Max relative difference: {np.max(rel_diff)}")
    
    return np.max(rel_diff) < 0.01

# Test gradients
test_samples = vi.sample_variational(50)
gradients_ok = check_gradients(vi, test_samples)
```

## Visualization Issues

### Plot Display Problems

**Problem**: Plots don't appear or show blank

**Solutions**:
```python
# 1. Set backend explicitly
import matplotlib
matplotlib.use('Agg')  # For saving only
# matplotlib.use('TkAgg')  # For interactive display
import matplotlib.pyplot as plt

# 2. Force display
plt.show(block=True)

# 3. Save instead of display
fig = plotter.plot_solution_field_2d(x, y, solution)
fig.savefig('solution.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 4. Check GUI backend
import tkinter
try:
    tkinter._test()
    print("GUI available")
except tkinter.TclError:
    print("No GUI available, use Agg backend")
    matplotlib.use('Agg')
```

### Memory Issues with Plots

**Problem**: `MemoryError` when creating many plots

**Solutions**:
```python
# Close figures after saving
fig = plotter.plot_traces(samples, parameter_names)
fig.savefig('traces.png')
plt.close(fig)  # Important!

# Reduce plot resolution
fig = plotter.plot_solution_field_2d(x, y, solution)
fig.savefig('solution.png', dpi=150)  # Lower DPI
plt.close(fig)

# Clear matplotlib state
plt.clf()
plt.cla()
plt.close('all')
```

### LaTeX Rendering Issues

**Problem**: `RuntimeError: LaTeX not found`

**Solutions**:
```python
# Disable LaTeX
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

# Or install LaTeX
# Ubuntu: sudo apt-get install texlive-latex-extra
# macOS: brew install --cask mactex
# Windows: Install MiKTeX

# Use mathtext instead
plt.rcParams['mathtext.fontset'] = 'cm'
```

## Performance Issues

### Slow MCMC Sampling

**Problem**: MCMC takes too long

**Diagnosis**:
```python
import time

# Time individual posterior evaluations
start_time = time.time()
for i in range(100):
    _ = log_posterior(test_params)
avg_time = (time.time() - start_time) / 100
print(f"Average posterior evaluation time: {avg_time:.4f}s")

# Profile the code
import cProfile
cProfile.run('sampler.sample(n_samples=100)', 'mcmc_profile.prof')
```

**Solutions**:
```python
# 1. Use coarser mesh for initial exploration
fast_solver = FiniteDifferenceSolver(
    domain_bounds=(0, 1, 0, 1),
    mesh_size=(21, 21),  # Coarser
    pde_type="elliptic"
)

# 2. Parallel evaluation (if multiple cores)
import multiprocessing as mp

def parallel_log_posterior(params_list):
    with mp.Pool() as pool:
        return pool.map(log_posterior, params_list)

# 3. Use compiled solvers (if available)
# Install with: pip install numba
from numba import jit

@jit(nopython=True)
def fast_residual_computation(predictions, observations):
    return np.sum((predictions - observations)**2)

# 4. Cache PDE solutions for similar parameters
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_solve(diffusion_rounded, source_strength_rounded):
    # Round parameters to enable caching
    diffusion = float(diffusion_rounded)
    source_strength = float(source_strength_rounded)
    # ... solve PDE ...
    return solution

def log_posterior_cached(params):
    diffusion, source_strength = params
    # Round to 3 decimal places for caching
    diff_round = round(diffusion, 3)
    source_round = round(source_strength, 3)
    
    solution = cached_solve(diff_round, source_round)
    # ... compute likelihood ...
```

### Memory Leaks

**Problem**: Memory usage keeps increasing

**Diagnosis**:
```python
import psutil
import os

process = psutil.Process(os.getpid())

def monitor_memory():
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    return memory_mb

# Monitor during MCMC
initial_memory = monitor_memory()
result = sampler.sample(n_samples=1000)
final_memory = monitor_memory()
print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
```

**Solutions**:
```python
# 1. Explicitly delete large objects
del solution, predictions, residuals

# 2. Use context managers for temporary objects
class PDESolver:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up internal state
        self._cleanup()

# 3. Limit sample storage
def streaming_mcmc(n_samples, batch_size=1000):
    all_means = []
    
    for batch_start in range(0, n_samples, batch_size):
        batch_samples = min(batch_size, n_samples - batch_start)
        result = sampler.sample(n_samples=batch_samples)
        
        # Process batch immediately
        batch_mean = np.mean(result['samples'], axis=0)
        all_means.append(batch_mean)
        
        # Delete batch data
        del result
    
    return np.array(all_means)
```

## Getting Help

### Enable Debug Logging

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('bayesian_pde_solver')
logger.debug("Debug logging enabled")
```

### Create Minimal Reproducible Example

```python
# Minimal example for bug reports
import numpy as np
from bayesian_pde_solver.pde_solvers import FiniteDifferenceSolver

# Set random seed for reproducibility
np.random.seed(42)

# Minimal problem setup
solver = FiniteDifferenceSolver(
    domain_bounds=(0, 1, 0, 1),
    mesh_size=(11, 11),
    pde_type="elliptic"
)

parameters = {
    'diffusion': 1.0,
    'reaction': 0.0,
    'source': lambda x, y: 1.0
}

boundary_conditions = {
    "left": {"type": "dirichlet", "value": 0.0},
    "right": {"type": "dirichlet", "value": 0.0},
    "top": {"type": "dirichlet", "value": 0.0},
    "bottom": {"type": "dirichlet", "value": 0.0}
}

# Reproduce the issue
try:
    solution = solver.solve(parameters, boundary_conditions)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### System Information

```python
import sys
import numpy as np
import scipy
import matplotlib

print("System Information:")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

try:
    import bayesian_pde_solver
    print(f"Bayesian PDE Solver: {bayesian_pde_solver.__version__}")
except:
    print("Bayesian PDE Solver: Not installed")

# Platform info
import platform
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
```

For additional help:
- Check GitHub issues: https://github.com/your-repo/issues
- Review examples in `examples/` directory
- Consult API documentation in `docs/api_reference.md`
- Join community discussions