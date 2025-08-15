"""
Configuration Management Module

Provides comprehensive configuration management for all project components
including PDE solvers, MCMC parameters, visualization settings, and more.
"""

from .config_manager import ConfigManager, load_config, save_config
from .default_configs import (
    DEFAULT_PDE_CONFIG,
    DEFAULT_MCMC_CONFIG, 
    DEFAULT_UNCERTAINTY_CONFIG,
    DEFAULT_VISUALIZATION_CONFIG,
    DEFAULT_SOLVER_CONFIG
)

__all__ = [
    "ConfigManager",
    "load_config",
    "save_config",
    "DEFAULT_PDE_CONFIG",
    "DEFAULT_MCMC_CONFIG",
    "DEFAULT_UNCERTAINTY_CONFIG", 
    "DEFAULT_VISUALIZATION_CONFIG",
    "DEFAULT_SOLVER_CONFIG"
]