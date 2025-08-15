"""
Bayesian Inference Module

Implements Bayesian methods for inverse problems including MCMC sampling,
variational inference, and posterior analysis.
"""

from .mcmc_sampler import MCMCSampler, MetropolisHastings, HamiltonianMonteCarlo
from .variational_inference import VariationalInference, MeanFieldVI
from .posterior_analysis import PosteriorAnalysis
from .priors import Prior, GaussianPrior, UniformPrior, LogNormalPrior
from .likelihood import Likelihood, GaussianLikelihood
from .inverse_solver import InverseSolver

__all__ = [
    "MCMCSampler",
    "MetropolisHastings", 
    "HamiltonianMonteCarlo",
    "VariationalInference",
    "MeanFieldVI",
    "PosteriorAnalysis",
    "Prior",
    "GaussianPrior",
    "UniformPrior", 
    "LogNormalPrior",
    "Likelihood",
    "GaussianLikelihood",
    "InverseSolver"
]