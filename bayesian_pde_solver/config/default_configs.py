"""
Default Configuration Values

Provides default configuration dictionaries for all project components.
These serve as fallbacks and can be overridden by user configuration files.
"""

from typing import Dict, Any, List, Optional


# PDE Solver Default Configuration
DEFAULT_PDE_CONFIG: Dict[str, Any] = {
    "solver_type": "finite_difference",
    "dimension": 2,
    "domain_bounds": [0.0, 1.0, 0.0, 1.0],  # [x_min, x_max, y_min, y_max] for 2D
    "mesh_size": [50, 50],  # [nx, ny] for 2D
    "pde_type": "elliptic",
    "time_steps": None,  # For time-dependent problems
    "final_time": None,  # For time-dependent problems
    "scheme": "central"  # Finite difference scheme
}

# MCMC Sampling Default Configuration
DEFAULT_MCMC_CONFIG: Dict[str, Any] = {
    "sampler_type": "metropolis_hastings",
    "n_samples": 10000,
    "n_burn": 2000,
    "n_thin": 1,
    "n_chains": 1,
    "step_size": 0.1,
    "mass_matrix": None,  # For HMC/NUTS
    "adaptation": True,
    "target_acceptance": 0.65,
    "max_tree_depth": 10  # For NUTS
}

# Uncertainty Quantification Default Configuration
DEFAULT_UNCERTAINTY_CONFIG: Dict[str, Any] = {
    "confidence_level": 0.95,
    "certification_method": "pac_bayes",
    "concentration_inequality": "hoeffding",
    "pac_bayes_bound": "mcallester",
    "bootstrap_samples": 1000,
    "cross_validation_folds": 5,
    "coverage_test": True
}

# Visualization Default Configuration
DEFAULT_VISUALIZATION_CONFIG: Dict[str, Any] = {
    "style": "academic",
    "figure_size": [10.0, 6.0],
    "dpi": 300,
    "font_family": "serif",
    "font_size": 12,
    "color_scheme": "academic",
    "save_formats": ["png", "pdf"],
    "interactive": False
}

# Numerical Solver Default Configuration
DEFAULT_SOLVER_CONFIG: Dict[str, Any] = {
    "linear_solver": "spsolve",
    "tolerance": 1e-8,
    "max_iterations": 1000,
    "preconditioning": True,
    "parallel": False,
    "n_processes": None,
    "memory_limit": None  # In GB
}

# Complete Default Configuration
DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": DEFAULT_PDE_CONFIG,
    "mcmc": DEFAULT_MCMC_CONFIG,
    "uncertainty": DEFAULT_UNCERTAINTY_CONFIG,
    "visualization": DEFAULT_VISUALIZATION_CONFIG,
    "solver": DEFAULT_SOLVER_CONFIG
}

# Problem-Specific Default Configurations
ELLIPTIC_1D_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "dimension": 1,
        "domain_bounds": [0.0, 1.0],
        "mesh_size": [100],
        "pde_type": "elliptic"
    },
    "mcmc": DEFAULT_MCMC_CONFIG,
    "uncertainty": DEFAULT_UNCERTAINTY_CONFIG,
    "visualization": {
        **DEFAULT_VISUALIZATION_CONFIG,
        "figure_size": [8.0, 6.0]
    },
    "solver": DEFAULT_SOLVER_CONFIG
}

ELLIPTIC_2D_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "dimension": 2,
        "domain_bounds": [0.0, 1.0, 0.0, 1.0],
        "mesh_size": [50, 50],
        "pde_type": "elliptic"
    },
    "mcmc": DEFAULT_MCMC_CONFIG,
    "uncertainty": DEFAULT_UNCERTAINTY_CONFIG,
    "visualization": {
        **DEFAULT_VISUALIZATION_CONFIG,
        "figure_size": [10.0, 8.0]
    },
    "solver": DEFAULT_SOLVER_CONFIG
}

PARABOLIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "pde_type": "parabolic",
        "time_steps": 100,
        "final_time": 1.0,
        "scheme": "backward_euler"
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "n_samples": 20000,  # More samples for time-dependent problems
        "sampler_type": "hamiltonian_monte_carlo"
    },
    "uncertainty": DEFAULT_UNCERTAINTY_CONFIG,
    "visualization": DEFAULT_VISUALIZATION_CONFIG,
    "solver": {
        **DEFAULT_SOLVER_CONFIG,
        "linear_solver": "cg"  # Better for time-stepping
    }
}

HYPERBOLIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "pde_type": "hyperbolic",
        "time_steps": 200,
        "final_time": 2.0,
        "scheme": "leapfrog"
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "n_samples": 15000,
        "sampler_type": "nuts"
    },
    "uncertainty": {
        **DEFAULT_UNCERTAINTY_CONFIG,
        "certification_method": "concentration",
        "concentration_inequality": "bernstein"
    },
    "visualization": DEFAULT_VISUALIZATION_CONFIG,
    "solver": {
        **DEFAULT_SOLVER_CONFIG,
        "linear_solver": "bicgstab"
    }
}

# High-Performance Computing Configuration
HPC_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "mesh_size": [200, 200]  # High resolution
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "n_samples": 50000,
        "n_chains": 4,
        "sampler_type": "nuts"
    },
    "uncertainty": {
        **DEFAULT_UNCERTAINTY_CONFIG,
        "bootstrap_samples": 5000
    },
    "visualization": {
        **DEFAULT_VISUALIZATION_CONFIG,
        "dpi": 600  # High resolution plots
    },
    "solver": {
        **DEFAULT_SOLVER_CONFIG,
        "parallel": True,
        "n_processes": 8,
        "memory_limit": 16.0  # 16 GB
    }
}

# Development/Testing Configuration
DEV_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "mesh_size": [20, 20]  # Small for fast testing
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "n_samples": 1000,  # Few samples for testing
        "n_burn": 100
    },
    "uncertainty": {
        **DEFAULT_UNCERTAINTY_CONFIG,
        "bootstrap_samples": 100
    },
    "visualization": {
        **DEFAULT_VISUALIZATION_CONFIG,
        "dpi": 150,  # Lower resolution for speed
        "save_formats": ["png"]
    },
    "solver": DEFAULT_SOLVER_CONFIG
}

# Configuration for specific applications
GROUNDWATER_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        "solver_type": "finite_element",
        "dimension": 2,
        "domain_bounds": [0.0, 1000.0, 0.0, 800.0],  # Realistic spatial scale (meters)
        "mesh_size": [100, 80],
        "pde_type": "elliptic"
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "sampler_type": "hamiltonian_monte_carlo",
        "n_samples": 15000
    },
    "uncertainty": {
        **DEFAULT_UNCERTAINTY_CONFIG,
        "certification_method": "pac_bayes",
        "confidence_level": 0.90
    },
    "visualization": {
        **DEFAULT_VISUALIZATION_CONFIG,
        "color_scheme": "earth_tones"
    },
    "solver": {
        **DEFAULT_SOLVER_CONFIG,
        "linear_solver": "gmres",
        "preconditioning": True
    }
}

HEAT_TRANSFER_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "pde_type": "parabolic",
        "time_steps": 1000,
        "final_time": 10.0,
        "scheme": "crank_nicolson"
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "sampler_type": "nuts",
        "n_samples": 25000
    },
    "uncertainty": {
        **DEFAULT_UNCERTAINTY_CONFIG,
        "certification_method": "concentration",
        "concentration_inequality": "bernstein"
    },
    "visualization": {
        **DEFAULT_VISUALIZATION_CONFIG,
        "interactive": True
    },
    "solver": {
        **DEFAULT_SOLVER_CONFIG,
        "linear_solver": "cg"
    }
}

# Configuration templates by problem size
SMALL_PROBLEM_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "mesh_size": [25, 25]
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "n_samples": 5000
    }
}

MEDIUM_PROBLEM_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "mesh_size": [50, 50]
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "n_samples": 10000
    }
}

LARGE_PROBLEM_CONFIG: Dict[str, Dict[str, Any]] = {
    "pde": {
        **DEFAULT_PDE_CONFIG,
        "mesh_size": [100, 100]
    },
    "mcmc": {
        **DEFAULT_MCMC_CONFIG,
        "n_samples": 25000,
        "sampler_type": "nuts"
    },
    "solver": {
        **DEFAULT_SOLVER_CONFIG,
        "parallel": True,
        "linear_solver": "cg"
    }
}

# Configuration validation schemas
CONFIG_SCHEMA: Dict[str, Dict[str, type]] = {
    "pde": {
        "solver_type": str,
        "dimension": int,
        "domain_bounds": list,
        "mesh_size": list,
        "pde_type": str,
        "scheme": str
    },
    "mcmc": {
        "sampler_type": str,
        "n_samples": int,
        "n_burn": int,
        "n_thin": int,
        "step_size": float,
        "target_acceptance": float
    },
    "uncertainty": {
        "confidence_level": float,
        "certification_method": str,
        "concentration_inequality": str,
        "pac_bayes_bound": str
    },
    "visualization": {
        "style": str,
        "figure_size": list,
        "dpi": int,
        "font_family": str,
        "color_scheme": str
    },
    "solver": {
        "linear_solver": str,
        "tolerance": float,
        "max_iterations": int,
        "preconditioning": bool
    }
}

# Helper functions for configuration management
def get_config_by_name(config_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get predefined configuration by name.
    
    Parameters
    ----------
    config_name : str
        Name of configuration
        
    Returns
    -------
    config : Dict[str, Dict[str, Any]]
        Configuration dictionary
        
    Raises
    ------
    ValueError
        If configuration name is unknown
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "elliptic_1d": ELLIPTIC_1D_CONFIG,
        "elliptic_2d": ELLIPTIC_2D_CONFIG,
        "parabolic": PARABOLIC_CONFIG,
        "hyperbolic": HYPERBOLIC_CONFIG,
        "hpc": HPC_CONFIG,
        "dev": DEV_CONFIG,
        "groundwater": GROUNDWATER_CONFIG,
        "heat_transfer": HEAT_TRANSFER_CONFIG,
        "small": SMALL_PROBLEM_CONFIG,
        "medium": MEDIUM_PROBLEM_CONFIG,
        "large": LARGE_PROBLEM_CONFIG
    }
    
    if config_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown configuration '{config_name}'. Available: {available}")
    
    return configs[config_name]


def list_available_configs() -> List[str]:
    """
    List all available predefined configurations.
    
    Returns
    -------
    config_names : List[str]
        List of available configuration names
    """
    return [
        "default", "elliptic_1d", "elliptic_2d", "parabolic", "hyperbolic",
        "hpc", "dev", "groundwater", "heat_transfer", "small", "medium", "large"
    ]


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration
    override_config : Dict[str, Any]
        Override configuration
        
    Returns
    -------
    merged_config : Dict[str, Any]
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged