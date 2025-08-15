# Contributing Guidelines

Thank you for your interest in contributing to the Bayesian PDE Inverse Problems framework! This document provides guidelines for contributing code, documentation, examples, and other improvements.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Code](#contributing-code)
- [Documentation](#documentation)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Types of Contributions

We welcome various types of contributions:

1. **Bug fixes**: Fix identified issues or edge cases
2. **New features**: Add PDE solvers, inference methods, or UQ techniques
3. **Documentation**: Improve guides, add examples, fix typos
4. **Examples**: Contribute real-world applications or test cases
5. **Performance improvements**: Optimize algorithms or add parallelization
6. **Testing**: Add test cases or improve test coverage

### Before You Start

1. **Check existing issues**: Look for open issues that match your interests
2. **Discuss major changes**: Create an issue to discuss significant additions
3. **Review the codebase**: Familiarize yourself with the project structure
4. **Read this guide**: Understand our development practices and standards

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Bayesian_PDE_Inverse_Problems_with_Certified_Uncertainty.git
cd Bayesian_PDE_Inverse_Problems_with_Certified_Uncertainty

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/Bayesian_PDE_Inverse_Problems_with_Certified_Uncertainty.git
```

### 2. Environment Setup

```bash
# Create development environment
conda create -n bayesian-pde-dev python=3.9
conda activate bayesian-pde-dev

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### 3. Development Dependencies

```bash
# Testing
pip install pytest pytest-cov pytest-xdist

# Code quality
pip install black flake8 mypy pre-commit

# Documentation
pip install sphinx sphinx-rtd-theme nbsphinx

# Optional: GPU support
pip install cupy  # For CUDA support
```

### 4. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contributing Code

### Project Structure

```
bayesian_pde_solver/
├── __init__.py                 # Main package interface
├── pde_solvers/               # PDE discretization methods
│   ├── __init__.py
│   ├── base.py               # Abstract base classes
│   ├── finite_difference.py  # FD implementations
│   └── finite_element.py     # FE implementations
├── bayesian_inference/       # Inference algorithms
│   ├── __init__.py
│   ├── mcmc_sampler.py       # MCMC methods
│   └── variational_inference.py # VI methods
├── uncertainty_quantification/ # UQ methods
│   ├── __init__.py
│   ├── certified_bounds.py   # Concentration inequalities
│   └── pac_bayes_bounds.py   # PAC-Bayes theory
├── visualization/            # Plotting and visualization
│   ├── __init__.py
│   ├── pde_plots.py         # PDE-specific plots
│   ├── bayesian_plots.py    # Bayesian inference plots
│   └── uncertainty_plots.py  # UQ visualizations
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── math_utils.py        # Mathematical utilities
│   └── io_utils.py          # I/O operations
└── config/                  # Configuration management
    ├── __init__.py
    └── config_manager.py    # Config handling
```

### Adding New Features

#### 1. Adding a New PDE Solver

```python
# bayesian_pde_solver/pde_solvers/your_solver.py

from .base import ForwardSolver
import numpy as np

class YourCustomSolver(ForwardSolver):
    """
    Your custom PDE solver.
    
    Solves the PDE: [describe your PDE here]
    
    Parameters
    ----------
    domain_bounds : Tuple[float, ...]
        Domain boundaries
    mesh_size : Tuple[int, ...]
        Mesh resolution
    custom_param : float
        Your custom parameter
        
    Examples
    --------
    >>> solver = YourCustomSolver(
    ...     domain_bounds=(0, 1, 0, 1),
    ...     mesh_size=(51, 51),
    ...     custom_param=1.0
    ... )
    >>> solution = solver.solve(parameters, boundary_conditions)
    """
    
    def __init__(self, domain_bounds, mesh_size, custom_param=1.0):
        super().__init__(domain_bounds, mesh_size, "custom")
        self.custom_param = custom_param
        
    def solve(self, parameters, boundary_conditions):
        """
        Solve the custom PDE.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            PDE parameters
        boundary_conditions : Dict[str, Any]
            Boundary conditions
            
        Returns
        -------
        solution : np.ndarray
            PDE solution at mesh points
        """
        # Your implementation here
        # Follow patterns from existing solvers
        
        # 1. Validate inputs
        self._validate_parameters(parameters)
        self._validate_boundary_conditions(boundary_conditions)
        
        # 2. Assemble system
        A, b = self._assemble_system(parameters)
        
        # 3. Apply boundary conditions
        A_bc, b_bc = self._apply_boundary_conditions(A, b, boundary_conditions)
        
        # 4. Solve system
        solution = self._solve_linear_system(A_bc, b_bc)
        
        return solution
    
    def _assemble_system(self, parameters):
        """Assemble the linear system for your PDE."""
        # Implementation specific to your PDE
        pass
    
    def _validate_parameters(self, parameters):
        """Validate PDE parameters."""
        required_params = ['your_required_params']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
```

#### 2. Adding a New Inference Method

```python
# bayesian_pde_solver/bayesian_inference/your_method.py

import numpy as np
from typing import Callable, Dict, Any

class YourInferenceMethod:
    """
    Your custom Bayesian inference method.
    
    Parameters
    ----------
    log_posterior_fn : Callable
        Log posterior function
    parameter_dim : int
        Number of parameters
    """
    
    def __init__(self, log_posterior_fn: Callable, parameter_dim: int):
        self.log_posterior_fn = log_posterior_fn
        self.parameter_dim = parameter_dim
        
    def fit(self, **kwargs) -> Dict[str, Any]:
        """
        Fit the model to find posterior approximation.
        
        Returns
        -------
        result : Dict[str, Any]
            Fitting results
        """
        # Your implementation
        pass
        
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from the fitted posterior approximation.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        samples : np.ndarray
            Posterior samples, shape (n_samples, parameter_dim)
        """
        # Your implementation
        pass
```

#### 3. Adding Uncertainty Quantification Methods

```python
# bayesian_pde_solver/uncertainty_quantification/your_uq_method.py

import numpy as np
from typing import Dict, Any

class YourUQMethod:
    """
    Your custom uncertainty quantification method.
    
    Implements [describe your method and theory].
    
    References
    ----------
    [Add relevant academic references]
    """
    
    def compute_bounds(self, samples: np.ndarray, 
                      confidence: float = 0.95) -> Dict[str, float]:
        """
        Compute uncertainty bounds.
        
        Parameters
        ----------
        samples : np.ndarray
            Posterior samples
        confidence : float
            Confidence level
            
        Returns
        -------
        bounds : Dict[str, float]
            Computed bounds
        """
        # Your implementation
        # Should return bounds that are valid with probability >= confidence
        pass
    
    def verify_coverage(self, samples: np.ndarray, 
                       true_values: np.ndarray,
                       confidence: float = 0.95) -> Dict[str, Any]:
        """
        Verify empirical coverage of the bounds.
        
        Parameters
        ----------
        samples : np.ndarray
            Posterior samples from multiple experiments
        true_values : np.ndarray
            True parameter values for each experiment
        confidence : float
            Nominal confidence level
            
        Returns
        -------
        coverage_stats : Dict[str, Any]
            Coverage statistics and diagnostics
        """
        # Implementation for empirical validation
        pass
```

### Code Quality Guidelines

#### 1. Documentation Standards

```python
def your_function(param1: np.ndarray, param2: float = 1.0) -> Dict[str, Any]:
    """
    Brief description of the function.
    
    Longer description providing context, mathematical background,
    and implementation details if necessary.
    
    Parameters
    ----------
    param1 : np.ndarray, shape (n, m)
        Description of param1, including shape if relevant
    param2 : float, default=1.0
        Description of param2 with default value
        
    Returns
    -------
    result : Dict[str, Any]
        Description of return value
        
    Raises
    ------
    ValueError
        When input validation fails
    RuntimeError
        When computation fails
        
    Examples
    --------
    >>> result = your_function(np.array([[1, 2], [3, 4]]), param2=2.0)
    >>> print(result['key'])
    expected_output
    
    Notes
    -----
    Additional mathematical or implementation notes.
    
    References
    ----------
    [1] Author, "Paper Title", Journal, Year.
    """
    # Implementation
    pass
```

#### 2. Type Hints

```python
from typing import Union, Optional, List, Dict, Tuple, Callable
import numpy as np

def typed_function(
    array_param: np.ndarray,
    optional_param: Optional[float] = None,
    callback: Callable[[np.ndarray], float] = None,
    config: Dict[str, Union[int, float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Function with comprehensive type hints."""
    pass
```

#### 3. Error Handling

```python
def robust_function(parameters: Dict[str, Any]) -> np.ndarray:
    """Function with proper error handling."""
    
    # Input validation
    if not isinstance(parameters, dict):
        raise TypeError("Parameters must be a dictionary")
    
    required_keys = ['diffusion', 'source']
    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Missing required parameter: {key}")
    
    # Value validation
    diffusion = parameters['diffusion']
    if isinstance(diffusion, (int, float)):
        if diffusion <= 0:
            raise ValueError("Diffusion coefficient must be positive")
    elif not callable(diffusion):
        raise TypeError("Diffusion must be a number or callable")
    
    try:
        # Main computation
        result = expensive_computation(parameters)
        
        # Output validation
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Computation produced non-finite values")
        
        return result
        
    except Exception as e:
        # Re-raise with context
        raise RuntimeError(f"Computation failed: {e}") from e
```

### Testing Your Code

#### 1. Unit Tests

```python
# tests/test_your_feature.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

from bayesian_pde_solver.your_module import YourClass

class TestYourClass:
    """Test suite for YourClass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.instance = YourClass(param1=1.0, param2=2.0)
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = self.instance.method(input_data)
        
        # Test output shape
        assert result.shape == expected_shape
        
        # Test output values
        assert_allclose(result, expected_result, rtol=1e-3)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty input
        with pytest.raises(ValueError):
            self.instance.method(np.array([]))
        
        # Invalid parameters
        with pytest.raises(ValueError):
            YourClass(param1=-1.0)  # Should fail for negative param1
    
    @pytest.mark.parametrize("param1,param2,expected", [
        (1.0, 2.0, 3.0),
        (0.5, 1.5, 2.0),
        (2.0, 3.0, 5.0),
    ])
    def test_parametrized(self, param1, param2, expected):
        """Test with multiple parameter combinations."""
        instance = YourClass(param1=param1, param2=param2)
        result = instance.simple_method()
        assert abs(result - expected) < 1e-10
    
    def test_numerical_accuracy(self):
        """Test numerical accuracy against analytical solution."""
        # Use a problem with known analytical solution
        analytical_result = analytical_function(test_input)
        numerical_result = self.instance.method(test_input)
        
        relative_error = np.abs(numerical_result - analytical_result) / np.abs(analytical_result)
        assert np.all(relative_error < 1e-6)
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance for large problems."""
        import time
        
        large_input = np.random.random((1000, 1000))
        
        start_time = time.time()
        result = self.instance.method(large_input)
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed_time < 60.0  # 1 minute
        assert result.shape == large_input.shape
```

#### 2. Integration Tests

```python
# tests/test_integration.py

def test_full_workflow():
    """Test complete workflow integration."""
    
    # Set up problem
    solver = PDESolver(domain=(0, 1, 0, 1), mesh_size=(21, 21))
    
    # Generate synthetic data
    true_params = {'diffusion': 1.5, 'source': 2.0}
    observations = generate_synthetic_data(solver, true_params)
    
    # Run inference
    posterior_fn = create_posterior(solver, observations)
    sampler = MCMCSampler(posterior_fn, parameter_dim=2)
    result = sampler.sample(n_samples=1000)
    
    # Check convergence
    assert result['acceptance_rate'] > 0.1
    assert result['acceptance_rate'] < 0.8
    
    # Check parameter recovery
    param_means = np.mean(result['samples'][200:], axis=0)
    assert abs(param_means[0] - 1.5) < 0.3
    assert abs(param_means[1] - 2.0) < 0.3
```

## Documentation

### 1. Docstring Guidelines

- Use NumPy-style docstrings for all public functions and classes
- Include parameter types, shapes, and descriptions
- Provide examples for complex functions
- Add mathematical background for algorithms
- Reference academic papers when appropriate

### 2. API Documentation

When adding new modules, update `docs/api_reference.md`:

```markdown
### `YourNewModule`

Brief description of the module's purpose.

#### `YourClass`

**Class**: `YourClass(param1, param2, **kwargs)`

Detailed description of the class.

**Parameters:**
- `param1` (type): Description
- `param2` (type): Description

**Methods:**

##### `method_name(arg1, arg2)`

Description of the method.

**Example:**
```python
instance = YourClass(param1=1.0)
result = instance.method_name(arg1, arg2)
```
```

### 3. Examples and Tutorials

When contributing examples:

1. **Add to examples/ directory**: Follow naming convention `your_example.py`
2. **Include comprehensive comments**: Explain each step
3. **Add to examples guide**: Update `docs/examples_guide.md`
4. **Test the example**: Ensure it runs without errors

```python
# examples/your_example.py

"""
Your Example: Description

This example demonstrates [specific capability].

Key features:
- Feature 1
- Feature 2
- Feature 3

Runtime: ~X minutes
"""

import numpy as np
from bayesian_pde_solver import *

def main():
    """Main function with complete workflow."""
    
    print("🚀 Starting Your Example")
    print("=" * 50)
    
    # Step 1: Problem setup
    print("Step 1: Setting up problem...")
    # Clear, commented code here
    
    # Step 2: Data generation
    print("Step 2: Generating data...")
    # Implementation
    
    # Step 3: Bayesian inference
    print("Step 3: Running inference...")
    # Implementation
    
    # Step 4: Results analysis
    print("Step 4: Analyzing results...")
    # Implementation
    
    print("✅ Example completed successfully!")

if __name__ == "__main__":
    main()
```

## Pull Request Process

### 1. Before Submitting

```bash
# Ensure your branch is up to date
git checkout main
git pull upstream main
git checkout your-feature-branch
git rebase main

# Run tests
pytest tests/

# Check code style
black bayesian_pde_solver/
flake8 bayesian_pde_solver/
mypy bayesian_pde_solver/

# Update documentation if needed
```

### 2. Pull Request Description

**Template:**
```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Changes Made

- Specific change 1
- Specific change 2
- Specific change 3

## Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested on multiple platforms/Python versions

## Documentation

- [ ] I have updated the documentation accordingly
- [ ] I have added docstrings to new functions/classes
- [ ] I have added examples if appropriate

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
```

### 3. Review Process

1. **Automated checks**: All CI tests must pass
2. **Code review**: At least one maintainer review required
3. **Documentation review**: Check that docs are updated
4. **Testing**: Verify test coverage and quality

## Community Guidelines

### Code of Conduct

- **Be respectful**: Treat all contributors with respect
- **Be inclusive**: Welcome newcomers and different perspectives  
- **Be collaborative**: Work together to improve the project
- **Be patient**: Remember that everyone is learning

### Communication

- **GitHub Issues**: For bug reports, feature requests, and discussions
- **Pull Requests**: For code contributions with detailed descriptions
- **Discussions**: For general questions and community interaction

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments

## Getting Help

### Resources

- **Documentation**: Check `docs/` directory first
- **Examples**: Look at `examples/` for usage patterns
- **Tests**: Review `tests/` for expected behavior
- **Issues**: Search existing issues for similar problems

### Contact

- **GitHub Issues**: For technical questions and bug reports
- **Discussions**: For general questions and community interaction
- **Email**: [maintainer email] for security issues or private concerns

Thank you for contributing to the Bayesian PDE Inverse Problems framework! Your contributions help advance scientific computing and uncertainty quantification for the entire community.