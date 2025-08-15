# API Reference

Complete API documentation for the Bayesian PDE Inverse Problems framework.

## Table of Contents

- [PDE Solvers](#pde-solvers)
- [Bayesian Inference](#bayesian-inference)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Utilities](#utilities)

## PDE Solvers

### `bayesian_pde_solver.pde_solvers`

#### `FiniteDifferenceSolver`

**Class**: `FiniteDifferenceSolver(domain_bounds, mesh_size, pde_type="elliptic")`

Finite difference solver for elliptic, parabolic, and hyperbolic PDEs.

**Parameters:**
- `domain_bounds` (Tuple[float, ...]): Domain boundaries
  - 1D: `(x_min, x_max)`
  - 2D: `(x_min, x_max, y_min, y_max)`
- `mesh_size` (Tuple[int, ...]): Mesh resolution
  - 1D: `(nx,)`
  - 2D: `(nx, ny)`
- `pde_type` (str): Type of PDE. Options: `"elliptic"`, `"parabolic"`, `"hyperbolic"`

**Attributes:**
- `dimension` (int): Spatial dimension
- `coordinates` (np.ndarray): Mesh coordinates
- `mesh` (Dict): Mesh information

**Methods:**

##### `solve(parameters, boundary_conditions)`

Solve the PDE with given parameters and boundary conditions.

**Parameters:**
- `parameters` (Dict[str, Any]): PDE parameters
  - `diffusion` (float or Callable): Diffusion coefficient D(x,y)
  - `reaction` (float or Callable): Reaction coefficient R(x,y)
  - `source` (float or Callable): Source term f(x,y)
- `boundary_conditions` (Dict[str, Any]): Boundary conditions
  ```python
  {
      "left": {"type": "dirichlet", "value": 0.0},
      "right": {"type": "neumann", "value": lambda x, y: x},
      # ... other boundaries
  }
  ```

**Returns:**
- `solution` (np.ndarray): PDE solution at mesh points

**Example:**
```python
solver = FiniteDifferenceSolver(
    domain_bounds=(0, 1, 0, 1),
    mesh_size=(51, 51),
    pde_type="elliptic"
)

parameters = {
    'diffusion': 1.0,
    'reaction': 0.0,
    'source': lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
}

boundary_conditions = {
    "left": {"type": "dirichlet", "value": 0.0},
    "right": {"type": "dirichlet", "value": 0.0},
    "top": {"type": "dirichlet", "value": 0.0},
    "bottom": {"type": "dirichlet", "value": 0.0}
}

solution = solver.solve(parameters, boundary_conditions)
```

##### `compute_gradient(solution)`

Compute spatial gradient of the solution.

**Parameters:**
- `solution` (np.ndarray): PDE solution

**Returns:**
- `gradient` (np.ndarray): Gradient field
  - 1D: shape `(n_points,)`
  - 2D: shape `(n_points, 2)`

##### `_interpolate_solution(solution, points)`

Interpolate solution at arbitrary points.

**Parameters:**
- `solution` (np.ndarray): PDE solution at mesh points
- `points` (np.ndarray): Query points, shape `(n_points, dimension)`

**Returns:**
- `interpolated` (np.ndarray): Interpolated values, shape `(n_points,)`

#### `FiniteElementSolver`

**Class**: `FiniteElementSolver(domain_bounds, mesh_size, pde_type="elliptic", element_type="linear_triangle")`

Finite element solver using linear triangular elements.

**Parameters:**
- `domain_bounds` (Tuple[float, ...]): Domain boundaries (2D only)
- `mesh_size` (Tuple[int, int]): Mesh resolution `(nx, ny)`
- `pde_type` (str): Type of PDE
- `element_type` (str): Element type (currently only `"linear_triangle"`)

**Attributes:**
- `dof_coordinates` (np.ndarray): Degrees of freedom coordinates
- `elements` (np.ndarray): Element connectivity, shape `(n_elements, 3)`
- `boundary_nodes` (np.ndarray): Boundary node indices
- `element_areas` (np.ndarray): Element areas

**Methods:**

##### `solve(parameters, boundary_conditions)`

Solve PDE using finite element method.

##### `solve_time_dependent(parameters, boundary_conditions, initial_condition, time_steps, final_time, method="backward_euler")`

Solve time-dependent PDE.

**Parameters:**
- `initial_condition` (np.ndarray): Initial solution
- `time_steps` (int): Number of time steps
- `final_time` (float): Final simulation time
- `method` (str): Time integration method. Options: `"backward_euler"`, `"crank_nicolson"`

**Returns:**
- `solutions` (np.ndarray): Solutions at all time steps, shape `(time_steps+1, n_dof)`
- `times` (np.ndarray): Time values

**Example:**
```python
fe_solver = FiniteElementSolver(
    domain_bounds=(0, 1, 0, 1),
    mesh_size=(31, 31),
    pde_type="parabolic"
)

# Heat equation
parameters = {
    'diffusion': 1.0,
    'reaction': 0.0,
    'source': lambda x, y: 0.0
}

# Initial condition: Gaussian
coords = fe_solver.dof_coordinates
initial_condition = np.exp(-((coords[:, 0] - 0.5)**2 + (coords[:, 1] - 0.5)**2) / 0.1)

solutions, times = fe_solver.solve_time_dependent(
    parameters=parameters,
    boundary_conditions=boundary_conditions,
    initial_condition=initial_condition,
    time_steps=100,
    final_time=1.0
)
```

## Bayesian Inference

### `bayesian_pde_solver.bayesian_inference`

#### `MCMCSampler`

**Class**: `MCMCSampler(log_posterior_fn, parameter_dim, sampler_type="metropolis_hastings")`

Markov Chain Monte Carlo sampler for Bayesian parameter estimation.

**Parameters:**
- `log_posterior_fn` (Callable): Function returning log posterior probability
- `parameter_dim` (int): Number of parameters
- `sampler_type` (str): MCMC algorithm. Options: `"metropolis_hastings"`, `"random_walk"`

**Methods:**

##### `sample(n_samples, initial_state, step_size=0.1, **kwargs)`

Run MCMC sampling.

**Parameters:**
- `n_samples` (int): Number of samples to generate
- `initial_state` (np.ndarray): Starting parameter values
- `step_size` (float): Proposal step size
- `adapt_step_size` (bool, optional): Enable adaptive step size tuning
- `target_acceptance_rate` (float, optional): Target acceptance rate for adaptation

**Returns:**
- `result` (Dict): Sampling results
  ```python
  {
      'samples': np.ndarray,          # Shape (n_samples, parameter_dim)
      'acceptance_rate': float,        # Overall acceptance rate
      'final_step_size': float,       # Final step size (if adapted)
      'log_probabilities': np.ndarray # Log posterior values
  }
  ```

**Example:**
```python
def log_posterior(params):
    # Define your posterior here
    return -0.5 * np.sum(params**2)  # Standard normal

sampler = MCMCSampler(
    log_posterior_fn=log_posterior,
    parameter_dim=2,
    sampler_type="metropolis_hastings"
)

result = sampler.sample(
    n_samples=5000,
    initial_state=np.zeros(2),
    step_size=0.5,
    adapt_step_size=True,
    target_acceptance_rate=0.4
)

samples = result['samples']
burn_in = 1000
posterior_samples = samples[burn_in:]
```

#### `VariationalInference`

**Class**: `VariationalInference(log_posterior_fn, parameter_dim, vi_type="mean_field")`

Variational inference for approximate Bayesian computation.

**Parameters:**
- `log_posterior_fn` (Callable): Log posterior function
- `parameter_dim` (int): Parameter dimension
- `vi_type` (str): VI type. Options: `"mean_field"`, `"full_rank"`

**Methods:**

##### `optimize(n_iterations=5000, learning_rate=0.01, n_samples=100, optimizer="adam", **kwargs)`

Optimize variational parameters.

**Parameters:**
- `n_iterations` (int): Number of optimization iterations
- `learning_rate` (float): Learning rate
- `n_samples` (int): Number of samples per iteration for ELBO estimation
- `optimizer` (str): Optimizer. Options: `"sgd"`, `"adam"`, `"adagrad"`
- `patience` (int, optional): Early stopping patience
- `min_delta` (float, optional): Minimum improvement for early stopping

**Returns:**
- `result` (Dict): Optimization results
  ```python
  {
      'final_elbo': float,
      'elbo_history': np.ndarray,
      'gradient_norms': np.ndarray,
      'variational_params': Dict,
      'n_iterations': int
  }
  ```

##### `sample(n_samples)`

Sample from optimized variational distribution.

**Parameters:**
- `n_samples` (int): Number of samples

**Returns:**
- `samples` (np.ndarray): Samples from variational posterior, shape `(n_samples, parameter_dim)`

**Example:**
```python
vi = VariationalInference(
    log_posterior_fn=log_posterior,
    parameter_dim=2,
    vi_type="mean_field"
)

# Optimize
result = vi.optimize(
    n_iterations=3000,
    learning_rate=0.02,
    n_samples=50,
    optimizer="adam"
)

# Sample from optimized distribution
samples = vi.sample(2000)

# Get summary statistics
stats = vi.get_summary_statistics()
print(f"Variational means: {stats['means']}")
print(f"Variational stds: {stats['stds']}")
```

## Uncertainty Quantification

### `bayesian_pde_solver.uncertainty_quantification`

#### `CertifiedBounds`

**Class**: `CertifiedBounds()`

Compute certified uncertainty bounds with mathematical guarantees.

**Methods:**

##### `compute_parameter_bounds(posterior_samples, parameter_names, parameter_bounds, confidence_level=0.95)`

Compute certified bounds for parameters.

**Parameters:**
- `posterior_samples` (np.ndarray): MCMC/VI samples, shape `(n_samples, n_params)`
- `parameter_names` (List[str]): Parameter names
- `parameter_bounds` (List[Tuple[float, float]]): Parameter bounds for each parameter
- `confidence_level` (float): Confidence level (0 < confidence_level < 1)

**Returns:**
- `bounds` (Dict): Bounds for each parameter
  ```python
  {
      'param_name': {
          'empirical_ci': {'lower': float, 'upper': float},
          'concentration': {'lower': float, 'upper': float},
          'pac_bayes': {'lower': float, 'upper': float}
      }
  }
  ```

##### `compute_prediction_bounds(predictions, confidence_level=0.95)`

Compute prediction uncertainty bounds.

**Parameters:**
- `predictions` (np.ndarray): Ensemble predictions, shape `(n_samples, n_predictions)`
- `confidence_level` (float): Confidence level

**Returns:**
- `bounds` (Dict): Prediction bounds
  ```python
  {
      'mean': np.ndarray,                    # Mean predictions
      'lower': np.ndarray,                   # Lower bounds
      'upper': np.ndarray,                   # Upper bounds
      'coverage_probability': float          # Theoretical coverage
  }
  ```

**Example:**
```python
cert_bounds = CertifiedBounds()

# Parameter bounds
param_bounds = cert_bounds.compute_parameter_bounds(
    posterior_samples=samples,
    parameter_names=['diffusion', 'source_strength'],
    parameter_bounds=[(0.1, 10.0), (0.1, 10.0)],
    confidence_level=0.95
)

# Check coverage
for param_name, bounds_dict in param_bounds.items():
    emp_bounds = bounds_dict['empirical_ci']
    conc_bounds = bounds_dict['concentration']
    
    print(f"{param_name}:")
    print(f"  Empirical 95% CI: [{emp_bounds['lower']:.3f}, {emp_bounds['upper']:.3f}]")
    print(f"  Concentration bound: [{conc_bounds['lower']:.3f}, {conc_bounds['upper']:.3f}]")
```

#### `ConcentrationInequalities`

**Class**: `ConcentrationInequalities()`

Implementation of concentration inequalities for uncertainty bounds.

**Methods:**

##### `compute_all_bounds(samples, confidence_level, bound_range, variance_estimate=None)`

Compute all available concentration bounds.

**Parameters:**
- `samples` (np.ndarray): Sample values
- `confidence_level` (float): Confidence level
- `bound_range` (Tuple[float, float]): Sample range bounds
- `variance_estimate` (float, optional): Variance estimate for Bernstein bound

**Returns:**
- `bounds` (Dict[str, float]): Dictionary of bound values by method name

#### `PACBayesBounds`

**Class**: `PACBayesBounds()`

PAC-Bayes bounds for generalization guarantees.

**Methods:**

##### `compute_all_bounds(posterior_samples, prior_mean, prior_cov, empirical_losses, confidence=0.95)`

Compute PAC-Bayes bounds.

**Parameters:**
- `posterior_samples` (np.ndarray): Posterior parameter samples
- `prior_mean` (np.ndarray): Prior mean
- `prior_cov` (np.ndarray): Prior covariance matrix
- `empirical_losses` (np.ndarray): Empirical loss values
- `confidence` (float): Confidence level

**Returns:**
- `bounds` (Dict[str, float]): PAC-Bayes bounds by method

## Visualization

### `bayesian_pde_solver.visualization`

#### `PDEPlotter`

**Class**: `PDEPlotter(style="academic", color_scheme="default", figure_size=(10, 6))`

Visualization for PDE solutions and related quantities.

**Methods:**

##### `plot_solution_field_2d(x, y, solution, title="Solution Field", **kwargs)`

Plot 2D solution field as color map.

**Parameters:**
- `x` (np.ndarray): x-coordinates
- `y` (np.ndarray): y-coordinates  
- `solution` (np.ndarray): 2D solution array
- `title` (str): Plot title
- `**kwargs`: Additional plotting arguments

**Returns:**
- `fig` (matplotlib.Figure): Figure object

##### `plot_solution_1d(x, solution, title="1D Solution", xlabel="x", ylabel="u(x)", **kwargs)`

Plot 1D solution.

##### `plot_contours(X, Y, Z, levels=10, title="Contour Plot", **kwargs)`

Plot contour lines.

##### `plot_mesh(coordinates, elements, title="Mesh", **kwargs)`

Visualize finite element mesh.

##### `plot_convergence(mesh_sizes, errors, expected_rate=None, title="Convergence Analysis", **kwargs)`

Plot convergence analysis.

**Example:**
```python
plotter = PDEPlotter(style="academic")

# 2D solution field
fig1 = plotter.plot_solution_field_2d(x, y, solution_2d, title="Temperature Field")

# Convergence plot
fig2 = plotter.plot_convergence(
    mesh_sizes=[10, 20, 40, 80],
    errors=[1e-1, 2.5e-2, 6e-3, 1.5e-3],
    expected_rate=2.0,
    title="Mesh Convergence"
)
```

#### `BayesianPlotter`

**Class**: `BayesianPlotter(style="academic", color_scheme="bayesian", figure_size=(10, 6))`

Visualization for Bayesian inference results.

**Methods:**

##### `plot_traces(samples, parameter_names, true_values=None, **kwargs)`

Plot MCMC traces.

**Parameters:**
- `samples` (np.ndarray): MCMC samples, shape `(n_samples, n_params)`
- `parameter_names` (List[str]): Parameter names
- `true_values` (List[float], optional): True parameter values for comparison

##### `plot_posterior_distributions(samples, parameter_names, true_values=None, **kwargs)`

Plot posterior distributions.

##### `plot_corner(samples, parameter_names, true_values=None, **kwargs)`

Create corner plot showing all pairwise parameter relationships.

##### `plot_elbo_convergence(elbo_history, title="ELBO Convergence", **kwargs)`

Plot ELBO convergence for variational inference.

**Example:**
```python
bayesian_plotter = BayesianPlotter(style="academic")

# Trace plots
fig1 = bayesian_plotter.plot_traces(
    samples=samples,
    parameter_names=['Diffusion', 'Source Strength'],
    true_values=[1.5, 2.0]
)

# Corner plot
fig2 = bayesian_plotter.plot_corner(
    samples=samples,
    parameter_names=['Diffusion', 'Source Strength'],
    true_values=[1.5, 2.0]
)
```

#### `UncertaintyPlotter`

**Class**: `UncertaintyPlotter(style="academic", color_scheme="uncertainty", figure_size=(10, 6))`

Specialized uncertainty visualization.

**Methods:**

##### `plot_confidence_bands(x, mean, std, observations=None, confidence_levels=[0.68, 0.95], **kwargs)`

Plot confidence bands with observations.

**Parameters:**
- `x` (np.ndarray): x-coordinates
- `mean` (np.ndarray): Mean prediction
- `std` (np.ndarray): Standard deviation
- `observations` (Dict, optional): Observation data with 'x' and 'y' keys
- `confidence_levels` (List[float]): Confidence levels to plot

##### `plot_certified_bounds(samples, parameter_names, certified_bounds, true_values=None, **kwargs)`

Plot certified bounds comparison.

**Parameters:**
- `certified_bounds` (Dict): Certified bounds for each parameter

##### `plot_coverage_analysis(coverage_results, methods, confidence_levels=[0.68, 0.95, 0.99], **kwargs)`

Plot coverage analysis for different UQ methods.

**Example:**
```python
uncertainty_plotter = UncertaintyPlotter(style="academic")

# Confidence bands
fig1 = uncertainty_plotter.plot_confidence_bands(
    x=x_pred,
    mean=mean_pred,
    std=std_pred,
    observations={'x': obs_x, 'y': obs_y},
    confidence_levels=[0.68, 0.95]
)

# Certified bounds
fig2 = uncertainty_plotter.plot_certified_bounds(
    samples=samples,
    parameter_names=['diffusion', 'source_strength'],
    certified_bounds={
        'diffusion': (0.8, 2.2),
        'source_strength': (1.3, 2.7)
    },
    true_values={'diffusion': 1.5, 'source_strength': 2.0}
)
```

## Configuration

### `bayesian_pde_solver.config`

#### `ConfigManager`

**Class**: `ConfigManager(config_path=None)`

Configuration management for the framework.

**Methods:**

##### `load_config(config_path)`

Load configuration from YAML file.

##### `get_default_config()`

Get default configuration dictionary.

**Example:**
```python
from bayesian_pde_solver.config import ConfigManager

config_manager = ConfigManager()
config = config_manager.get_default_config()

# Override defaults
config['mcmc']['n_samples'] = 20000
config['uncertainty']['confidence_level'] = 0.99
```

## Utilities

### Mathematical Utilities

#### `bayesian_pde_solver.utils.math_utils`

##### Functions

- `safe_log_sum_exp(log_values)`: Numerically stable log-sum-exp
- `compute_effective_sample_size(samples)`: Effective sample size for MCMC
- `autocorrelation_function(samples, max_lag=None)`: Autocorrelation analysis
- `geweke_diagnostic(samples)`: Geweke convergence diagnostic
- `split_rhat(samples)`: Split R-hat statistic

### Mesh Utilities

#### `bayesian_pde_solver.utils.mesh_utils`

##### Functions

- `create_structured_mesh(domain_bounds, mesh_size)`: Create structured mesh
- `refine_mesh(coordinates, elements, refinement_level)`: Adaptive mesh refinement
- `compute_mesh_quality(coordinates, elements)`: Mesh quality metrics

### I/O Utilities

#### `bayesian_pde_solver.utils.io_utils`

##### Functions

- `save_results(filename, results)`: Save results to HDF5
- `load_results(filename)`: Load results from HDF5
- `export_to_csv(data, filename)`: Export data to CSV
- `create_summary_report(results)`: Generate summary report

**Example:**
```python
from bayesian_pde_solver.utils import save_results, load_results

# Save MCMC results
results = {
    'samples': samples,
    'acceptance_rate': acceptance_rate,
    'metadata': {'n_samples': 5000, 'parameter_names': ['diffusion', 'source']}
}

save_results('mcmc_results.h5', results)

# Load results later
loaded_results = load_results('mcmc_results.h5')
```

## Error Handling

### Custom Exceptions

#### `bayesian_pde_solver.exceptions`

- `PDESolverError`: Raised when PDE solving fails
- `ConvergenceError`: Raised when iterative methods don't converge  
- `BoundaryConditionError`: Raised for invalid boundary conditions
- `MeshError`: Raised for mesh-related issues
- `InferenceError`: Raised during Bayesian inference failures

**Example:**
```python
from bayesian_pde_solver.exceptions import PDESolverError

try:
    solution = solver.solve(invalid_parameters, boundary_conditions)
except PDESolverError as e:
    print(f"PDE solving failed: {e}")
    # Handle error appropriately
```

## Performance Considerations

### Memory Management

- Use `dtype=np.float32` for large problems to reduce memory usage
- Enable chunked computation for very large sample sets
- Use sparse matrices for PDE systems when possible

### Parallel Computing

```python
# Set number of threads for numerical libraries
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

# Use parallel sampling (if available)
sampler = MCMCSampler(log_posterior, parameter_dim=2, n_parallel_chains=4)
```

### Optimization Tips

- Use adaptive step sizes in MCMC for better mixing
- Start with coarse meshes for initial parameter exploration
- Use variational inference for quick approximations before MCMC
- Enable early stopping in optimization routines

## Version Compatibility

This API documentation is for version 1.0.0. For compatibility with different versions:

```python
import bayesian_pde_solver
print(f"Version: {bayesian_pde_solver.__version__}")

# Check for feature availability
if hasattr(bayesian_pde_solver, 'some_new_feature'):
    # Use new feature
    pass
else:
    # Fallback implementation
    pass
```