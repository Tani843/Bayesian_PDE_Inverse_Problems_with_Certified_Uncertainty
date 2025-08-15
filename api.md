---
layout: page
title: API Reference
permalink: /api/
---

# API Reference

Complete reference for all classes and functions in the Bayesian PDE framework.

## Core Classes

### PDE Solvers

#### `HeatEquationSolver`
Finite difference solver for 2D heat conduction problems.

```python
class HeatEquationSolver:
    def __init__(self, domain_size=(1.0, 1.0), grid_size=(50, 50), 
                 boundary_conditions='dirichlet'):
        """
        Initialize 2D heat equation solver.
        
        Parameters:
        -----------
        domain_size : tuple
            Physical domain dimensions (Lx, Ly)
        grid_size : tuple  
            Number of grid points (nx, ny)
        boundary_conditions : str
            Boundary condition type ('dirichlet', 'neumann', 'mixed')
        """
    
    def solve(self, thermal_conductivity, heat_source=None):
        """Solve heat equation with given parameters."""
        
    def observe_at_points(self, solution, points):
        """Extract solution values at observation points."""
```

#### `WaveEquationSolver`  
Time-dependent wave equation solver.

```python
class WaveEquationSolver:
    def solve_time_dependent(self, wave_speed, initial_conditions, 
                           time_steps=100, dt=0.01):
        """Solve time-dependent wave equation."""
```

### Bayesian Inference

#### `BayesianInference`
Main class for parameter estimation with uncertainty quantification.

```python
class BayesianInference:
    def __init__(self, solver, observations, observation_points, 
                 prior_bounds=None, noise_level=0.1):
        """
        Set up Bayesian inference problem.
        
        Parameters:
        -----------
        solver : PDESolver
            PDE solver instance
        observations : array_like
            Observed data values  
        observation_points : array_like
            Spatial locations of observations
        prior_bounds : tuple
            Parameter bounds for uniform prior
        noise_level : float
            Observation noise standard deviation
        """
    
    def mcmc_sampling(self, n_samples=5000, burn_in=1000, 
                      algorithm='metropolis-hastings'):
        """
        Generate posterior samples using MCMC.
        
        Returns:
        --------
        samples : ndarray
            Posterior samples of parameters
        """
        
    def variational_inference(self, max_iterations=1000, learning_rate=0.01):
        """Approximate posterior using variational inference."""
        
    def certified_bounds(self, confidence=0.95, method='hoeffding'):
        """Compute certified confidence intervals."""
```

### Uncertainty Quantification

#### `UncertaintyQuantifier`
Tools for certified uncertainty bounds.

```python
class UncertaintyQuantifier:
    @staticmethod
    def hoeffding_bound(samples, confidence=0.95):
        """Compute Hoeffding concentration bound."""
        
    @staticmethod  
    def pac_bayes_bound(prior_samples, posterior_samples, 
                        observations, confidence=0.95):
        """Compute PAC-Bayes generalization bound."""
        
    @staticmethod
    def coverage_analysis(true_parameters, bounds_list):
        """Validate empirical coverage of bounds."""
```

### Visualization

#### `Visualizer`
Publication-quality plotting utilities.

```python
class Visualizer:
    @staticmethod
    def plot_solution(solution, domain_size, title="PDE Solution"):
        """Plot 2D PDE solution field."""
        
    @staticmethod
    def plot_posterior(samples, true_value=None, bins=50):
        """Plot posterior distribution histogram."""
        
    @staticmethod  
    def plot_convergence(samples, window_size=100):
        """Plot MCMC convergence diagnostics."""
        
    @staticmethod
    def plot_bounds_comparison(methods_dict, true_value):
        """Compare different uncertainty quantification methods."""
```

## Utility Functions

### Data Processing
```python
def generate_synthetic_data(solver, true_parameters, n_observations, 
                           noise_level=0.05):
    """Generate synthetic observations for testing."""

def load_experimental_data(filename, format='csv'):
    """Load experimental data from various formats."""
```

### Numerical Methods
```python  
def adaptive_mesh_refinement(solver, error_indicator, 
                            refinement_threshold=0.1):
    """Adaptively refine computational mesh."""
    
def convergence_diagnostics(samples, method='gelman_rubin'):
    """Compute MCMC convergence diagnostics."""
```

### Performance Utilities
```python
def enable_gpu_acceleration():
    """Enable GPU computations if available."""
    
def parallel_mcmc(n_chains=4, **mcmc_kwargs):
    """Run multiple MCMC chains in parallel."""
```

## Constants and Enums

### Solver Types
```python
class SolverType(Enum):
    FINITE_DIFFERENCE = "fd"
    FINITE_ELEMENT = "fe"
    SPECTRAL = "spectral"
```

### Boundary Conditions  
```python
class BoundaryCondition(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"  
    MIXED = "mixed"
    PERIODIC = "periodic"
```

### MCMC Algorithms
```python
class MCMCAlgorithm(Enum):
    METROPOLIS_HASTINGS = "mh"
    HAMILTONIAN_MONTE_CARLO = "hmc"
    NO_U_TURN_SAMPLER = "nuts"
```

## Error Classes

```python
class BayesianPDEError(Exception):
    """Base exception class."""

class ConvergenceError(BayesianPDEError):
    """Raised when MCMC fails to converge."""
    
class SolverError(BayesianPDEError):
    """Raised when PDE solver encounters problems."""
    
class BoundsError(BayesianPDEError):
    """Raised when uncertainty bounds cannot be computed."""
```

## Configuration

### Global Settings
```python
import bayesian_pde_solver as bps

# Set global precision
bps.set_precision('double')  # or 'single'

# Configure parallel backend  
bps.set_parallel_backend('multiprocessing')  # or 'mpi'

# Enable/disable progress bars
bps.set_progress_bars(True)
```

## Examples

### Basic Usage
```python
from bayesian_pde_solver import HeatEquationSolver, BayesianInference

# Solve inverse problem
solver = HeatEquationSolver()
inference = BayesianInference(solver, observations, obs_points)
samples = inference.mcmc_sampling(n_samples=2000)
bounds = inference.certified_bounds()
```

### Advanced Configuration
```python  
# Custom prior distribution
def log_prior(theta):
    return -0.5 * np.sum(theta**2)  # Gaussian prior

inference = BayesianInference(solver, observations, obs_points,
                             custom_prior=log_prior)

# Adaptive MCMC with custom proposal
samples = inference.mcmc_sampling(
    algorithm='hmc',
    step_size=0.01,
    n_leapfrog_steps=10,
    adapt_step_size=True
)
```