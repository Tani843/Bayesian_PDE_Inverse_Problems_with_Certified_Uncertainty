"""
Configuration Manager

Comprehensive configuration system for managing all project parameters
with validation, default values, and hierarchical configuration loading.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import warnings

from .default_configs import (
    DEFAULT_PDE_CONFIG,
    DEFAULT_MCMC_CONFIG,
    DEFAULT_UNCERTAINTY_CONFIG,
    DEFAULT_VISUALIZATION_CONFIG,
    DEFAULT_SOLVER_CONFIG
)


@dataclass
class PDEConfig:
    """Configuration for PDE solver parameters."""
    solver_type: str = "finite_difference"
    dimension: int = 2
    domain_bounds: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0, 1.0])
    mesh_size: List[int] = field(default_factory=lambda: [50, 50])
    pde_type: str = "elliptic"
    time_steps: Optional[int] = None
    final_time: Optional[float] = None
    scheme: str = "central"
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate PDE configuration parameters."""
        if self.solver_type not in ["finite_difference", "finite_element"]:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
        
        if self.dimension not in [1, 2, 3]:
            raise ValueError("Dimension must be 1, 2, or 3")
        
        expected_bounds = self.dimension * 2
        if len(self.domain_bounds) != expected_bounds:
            raise ValueError(f"Expected {expected_bounds} domain bounds for {self.dimension}D")
        
        if len(self.mesh_size) != self.dimension:
            raise ValueError(f"Expected {self.dimension} mesh size values")
        
        if any(n <= 0 for n in self.mesh_size):
            raise ValueError("All mesh sizes must be positive")


@dataclass  
class MCMCConfig:
    """Configuration for MCMC sampling parameters."""
    sampler_type: str = "metropolis_hastings"
    n_samples: int = 10000
    n_burn: int = 2000
    n_thin: int = 1
    n_chains: int = 1
    step_size: float = 0.1
    mass_matrix: Optional[str] = None
    adaptation: bool = True
    target_acceptance: float = 0.65
    max_tree_depth: int = 10
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate MCMC configuration parameters."""
        valid_samplers = ["metropolis_hastings", "hamiltonian_monte_carlo", "nuts"]
        if self.sampler_type not in valid_samplers:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")
        
        if self.n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        if self.n_burn < 0:
            raise ValueError("Burn-in samples cannot be negative")
        
        if self.n_thin <= 0:
            raise ValueError("Thinning interval must be positive")
        
        if not 0 < self.target_acceptance < 1:
            raise ValueError("Target acceptance rate must be between 0 and 1")


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification parameters."""
    confidence_level: float = 0.95
    certification_method: str = "pac_bayes"
    concentration_inequality: str = "hoeffding"
    pac_bayes_bound: str = "mcallester"
    bootstrap_samples: int = 1000
    cross_validation_folds: int = 5
    coverage_test: bool = True
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate uncertainty quantification parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        valid_methods = ["pac_bayes", "concentration", "bootstrap", "cross_validation"]
        if self.certification_method not in valid_methods:
            raise ValueError(f"Unknown certification method: {self.certification_method}")
        
        valid_inequalities = ["hoeffding", "bernstein", "mcdiarmid"]
        if self.concentration_inequality not in valid_inequalities:
            raise ValueError(f"Unknown concentration inequality: {self.concentration_inequality}")


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    style: str = "academic"
    figure_size: List[float] = field(default_factory=lambda: [10.0, 6.0])
    dpi: int = 300
    font_family: str = "serif"
    font_size: int = 12
    color_scheme: str = "academic"
    save_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    interactive: bool = False
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate visualization parameters."""
        valid_styles = ["academic", "presentation", "minimal"]
        if self.style not in valid_styles:
            raise ValueError(f"Unknown plot style: {self.style}")
        
        if len(self.figure_size) != 2:
            raise ValueError("Figure size must be [width, height]")
        
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")


@dataclass
class SolverConfig:
    """Configuration for numerical solver parameters."""
    linear_solver: str = "spsolve"
    tolerance: float = 1e-8
    max_iterations: int = 1000
    preconditioning: bool = True
    parallel: bool = False
    n_processes: Optional[int] = None
    memory_limit: Optional[float] = None
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate solver configuration parameters."""
        valid_solvers = ["spsolve", "cg", "gmres", "bicgstab"]
        if self.linear_solver not in valid_solvers:
            raise ValueError(f"Unknown linear solver: {self.linear_solver}")
        
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")


class ConfigManager:
    """
    Comprehensive configuration manager for the Bayesian PDE solver.
    
    Manages hierarchical configuration loading, validation, and provides
    convenient access to all configuration parameters.
    
    Examples
    --------
    >>> config = ConfigManager()
    >>> config.load_from_file("config.yaml")
    >>> print(config.mcmc.n_samples)
    10000
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config_dict : Optional[Dict[str, Any]]
            Initial configuration dictionary
        """
        # Initialize with default configurations
        self.pde = PDEConfig(**DEFAULT_PDE_CONFIG)
        self.mcmc = MCMCConfig(**DEFAULT_MCMC_CONFIG)
        self.uncertainty = UncertaintyConfig(**DEFAULT_UNCERTAINTY_CONFIG)
        self.visualization = VisualizationConfig(**DEFAULT_VISUALIZATION_CONFIG)
        self.solver = SolverConfig(**DEFAULT_SOLVER_CONFIG)
        
        # Load additional configuration if provided
        if config_dict is not None:
            self.update_from_dict(config_dict)
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path to configuration file
            
        Raises
        ------
        FileNotFoundError
            If configuration file doesn't exist
        yaml.YAMLError
            If YAML file is malformed
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                warnings.warn(f"Empty configuration file: {config_path}")
                return
            
            self.update_from_dict(config_dict)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path where to save configuration
        """
        config_dict = self.to_dict()
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary with nested structure
        """
        # Update PDE configuration
        if 'pde' in config_dict:
            pde_config = {**DEFAULT_PDE_CONFIG, **config_dict['pde']}
            self.pde = PDEConfig(**pde_config)
        
        # Update MCMC configuration
        if 'mcmc' in config_dict:
            mcmc_config = {**DEFAULT_MCMC_CONFIG, **config_dict['mcmc']}
            self.mcmc = MCMCConfig(**mcmc_config)
        
        # Update uncertainty configuration
        if 'uncertainty' in config_dict:
            uncertainty_config = {**DEFAULT_UNCERTAINTY_CONFIG, **config_dict['uncertainty']}
            self.uncertainty = UncertaintyConfig(**uncertainty_config)
        
        # Update visualization configuration
        if 'visualization' in config_dict:
            viz_config = {**DEFAULT_VISUALIZATION_CONFIG, **config_dict['visualization']}
            self.visualization = VisualizationConfig(**viz_config)
        
        # Update solver configuration
        if 'solver' in config_dict:
            solver_config = {**DEFAULT_SOLVER_CONFIG, **config_dict['solver']}
            self.solver = SolverConfig(**solver_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns
        -------
        config_dict : Dict[str, Any]
            Complete configuration as nested dictionary
        """
        return {
            'pde': {
                'solver_type': self.pde.solver_type,
                'dimension': self.pde.dimension,
                'domain_bounds': self.pde.domain_bounds,
                'mesh_size': self.pde.mesh_size,
                'pde_type': self.pde.pde_type,
                'time_steps': self.pde.time_steps,
                'final_time': self.pde.final_time,
                'scheme': self.pde.scheme
            },
            'mcmc': {
                'sampler_type': self.mcmc.sampler_type,
                'n_samples': self.mcmc.n_samples,
                'n_burn': self.mcmc.n_burn,
                'n_thin': self.mcmc.n_thin,
                'n_chains': self.mcmc.n_chains,
                'step_size': self.mcmc.step_size,
                'mass_matrix': self.mcmc.mass_matrix,
                'adaptation': self.mcmc.adaptation,
                'target_acceptance': self.mcmc.target_acceptance,
                'max_tree_depth': self.mcmc.max_tree_depth
            },
            'uncertainty': {
                'confidence_level': self.uncertainty.confidence_level,
                'certification_method': self.uncertainty.certification_method,
                'concentration_inequality': self.uncertainty.concentration_inequality,
                'pac_bayes_bound': self.uncertainty.pac_bayes_bound,
                'bootstrap_samples': self.uncertainty.bootstrap_samples,
                'cross_validation_folds': self.uncertainty.cross_validation_folds,
                'coverage_test': self.uncertainty.coverage_test
            },
            'visualization': {
                'style': self.visualization.style,
                'figure_size': self.visualization.figure_size,
                'dpi': self.visualization.dpi,
                'font_family': self.visualization.font_family,
                'font_size': self.visualization.font_size,
                'color_scheme': self.visualization.color_scheme,
                'save_formats': self.visualization.save_formats,
                'interactive': self.visualization.interactive
            },
            'solver': {
                'linear_solver': self.solver.linear_solver,
                'tolerance': self.solver.tolerance,
                'max_iterations': self.solver.max_iterations,
                'preconditioning': self.solver.preconditioning,
                'parallel': self.solver.parallel,
                'n_processes': self.solver.n_processes,
                'memory_limit': self.solver.memory_limit
            }
        }
    
    def get_pde_params(self) -> Dict[str, Any]:
        """Get PDE solver parameters."""
        return {
            'domain_bounds': tuple(self.pde.domain_bounds),
            'mesh_size': tuple(self.pde.mesh_size),
            'pde_type': self.pde.pde_type,
            'scheme': self.pde.scheme
        }
    
    def get_mcmc_params(self) -> Dict[str, Any]:
        """Get MCMC sampling parameters."""
        return {
            'n_samples': self.mcmc.n_samples,
            'n_burn': self.mcmc.n_burn,
            'n_thin': self.mcmc.n_thin,
            'sampler_type': self.mcmc.sampler_type,
            'step_size': self.mcmc.step_size,
            'target_acceptance': self.mcmc.target_acceptance
        }
    
    def get_solver_params(self) -> Dict[str, Any]:
        """Get numerical solver parameters."""
        return {
            'method': self.solver.linear_solver,
            'tol': self.solver.tolerance,
            'maxiter': self.solver.max_iterations
        }
    
    def validate_all(self) -> None:
        """Validate all configuration sections."""
        self.pde.validate()
        self.mcmc.validate()
        self.uncertainty.validate()
        self.visualization.validate() 
        self.solver.validate()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(\n  PDE: {self.pde.pde_type} {self.pde.dimension}D\n  MCMC: {self.mcmc.sampler_type} ({self.mcmc.n_samples} samples)\n  Uncertainty: {self.uncertainty.certification_method}\n)"


def load_config(config_path: Union[str, Path]) -> ConfigManager:
    """
    Load configuration from file.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration file
        
    Returns
    -------
    config : ConfigManager
        Loaded configuration manager
    """
    config = ConfigManager()
    config.load_from_file(config_path)
    return config


def save_config(config: ConfigManager, config_path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Parameters
    ----------
    config : ConfigManager
        Configuration manager to save
    config_path : Union[str, Path]
        Path where to save configuration
    """
    config.save_to_file(config_path)