"""
Utility Functions Module

Helper functions and utilities for the Bayesian PDE solver including
data handling, mathematical utilities, and validation functions.
"""

from .data_utils import (
    generate_synthetic_data, load_data, save_data,
    split_data, normalize_data, add_noise
)
from .math_utils import (
    safe_log, safe_exp, log_sum_exp, safe_divide,
    compute_gradient_finite_diff, integrate_trapz
)
from .validation_utils import (
    validate_array, validate_bounds, validate_parameters,
    check_convergence, estimate_error
)
from .io_utils import (
    create_directory, load_config_yaml, save_results,
    export_to_csv, import_from_csv
)

__all__ = [
    # Data utilities
    "generate_synthetic_data", "load_data", "save_data",
    "split_data", "normalize_data", "add_noise",
    
    # Math utilities  
    "safe_log", "safe_exp", "log_sum_exp", "safe_divide",
    "compute_gradient_finite_diff", "integrate_trapz",
    
    # Validation utilities
    "validate_array", "validate_bounds", "validate_parameters", 
    "check_convergence", "estimate_error",
    
    # I/O utilities
    "create_directory", "load_config_yaml", "save_results",
    "export_to_csv", "import_from_csv"
]