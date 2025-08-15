"""
Visualization Module

Comprehensive plotting and visualization tools for Bayesian PDE inverse problems.
Creates publication-quality figures for analysis and presentation.
"""

from .plotting_utils import PlottingConfig, setup_matplotlib_style
from .solution_plots import SolutionPlotter
from .uncertainty_plots import UncertaintyPlotter
from .posterior_plots import PosteriorPlotter
from .convergence_plots import ConvergencePlotter
from .comparison_plots import ComparisonPlotter
from .interactive_plots import InteractivePlotter

__all__ = [
    "PlottingConfig",
    "setup_matplotlib_style",
    "SolutionPlotter",
    "UncertaintyPlotter", 
    "PosteriorPlotter",
    "ConvergencePlotter",
    "ComparisonPlotter",
    "InteractivePlotter"
]