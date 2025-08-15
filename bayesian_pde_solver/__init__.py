"""
Bayesian PDE Inverse Problems Solver with Certified Uncertainty Quantification

A comprehensive framework for solving inverse problems in partial differential equations
using Bayesian methods with rigorous uncertainty quantification and certification.
"""

__version__ = "1.0.0"
__author__ = "Tanisha Gupta"
__email__ = "tanisha@example.com"

from .pde_solvers import *
from .bayesian_inference import *
from .uncertainty_quantification import *
from .visualization import *
from .utils import *

__all__ = [
    "pde_solvers",
    "bayesian_inference", 
    "uncertainty_quantification",
    "visualization",
    "utils"
]