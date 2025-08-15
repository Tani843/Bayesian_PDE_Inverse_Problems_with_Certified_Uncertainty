"""
Likelihood Functions

Implements various likelihood functions for Bayesian parameter estimation.
"""

import numpy as np
from scipy import stats
from typing import Union, Optional, Dict, Any
from abc import ABC, abstractmethod


class Likelihood(ABC):
    """
    Abstract base class for likelihood functions.
    """
    
    @abstractmethod
    def log_prob(self, observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Compute log likelihood.
        
        Args:
            observations: Observed data
            predictions: Model predictions
            
        Returns:
            log_likelihood: Log likelihood value
        """
        pass
    
    @abstractmethod
    def sample_predictions(self, predictions: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Sample from predictive distribution.
        
        Args:
            predictions: Model predictions (means)
            n_samples: Number of samples
            
        Returns:
            samples: Samples from predictive distribution
        """
        pass


class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood with known or unknown noise variance.
    """
    
    def __init__(self, noise_std: Optional[float] = None,
                 estimate_noise: bool = False,
                 noise_prior: Optional[Dict[str, float]] = None):
        """
        Initialize Gaussian likelihood.
        
        Args:
            noise_std: Known noise standard deviation (if None, will be estimated)
            estimate_noise: Whether to estimate noise parameter
            noise_prior: Prior for noise parameter (if estimating)
        """
        self.noise_std = noise_std
        self.estimate_noise = estimate_noise
        self.noise_prior = noise_prior
        
        if estimate_noise and noise_prior is None:
            # Default inverse gamma prior for noise variance
            self.noise_prior = {"alpha": 2.0, "beta": 1.0}
    
    def log_prob(self, observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Compute Gaussian log likelihood.
        
        Args:
            observations: Observed data
            predictions: Model predictions
            
        Returns:
            log_likelihood: Log likelihood value
        """
        if len(observations) != len(predictions):
            raise ValueError("Observations and predictions must have same length")
        
        if self.noise_std is None:
            raise ValueError("Noise standard deviation must be specified or estimated")
        
        # Compute residuals
        residuals = observations - predictions
        
        # Gaussian log likelihood
        n = len(observations)
        log_likelihood = -0.5 * n * np.log(2 * np.pi)
        log_likelihood -= n * np.log(self.noise_std)
        log_likelihood -= 0.5 * np.sum(residuals**2) / (self.noise_std**2)
        
        return log_likelihood
    
    def log_prob_with_noise_estimation(self, observations: np.ndarray, 
                                      predictions: np.ndarray,
                                      noise_variance: float) -> float:
        """
        Compute log likelihood when estimating noise variance.
        
        Args:
            observations: Observed data
            predictions: Model predictions
            noise_variance: Noise variance parameter
            
        Returns:
            log_likelihood: Log likelihood including noise prior
        """
        if noise_variance <= 0:
            return -np.inf
        
        # Data likelihood
        residuals = observations - predictions
        n = len(observations)
        
        log_likelihood = -0.5 * n * np.log(2 * np.pi)
        log_likelihood -= 0.5 * n * np.log(noise_variance)
        log_likelihood -= 0.5 * np.sum(residuals**2) / noise_variance
        
        # Add noise prior (inverse gamma)
        if self.noise_prior is not None:
            alpha = self.noise_prior["alpha"]
            beta = self.noise_prior["beta"]
            log_likelihood += stats.invgamma.logpdf(noise_variance, a=alpha, scale=beta)
        
        return log_likelihood
    
    def sample_predictions(self, predictions: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from Gaussian predictive distribution."""
        if self.noise_std is None:
            raise ValueError("Noise standard deviation must be specified")
        
        samples = np.zeros((n_samples, len(predictions)))
        for i in range(n_samples):
            samples[i] = predictions + np.random.normal(0, self.noise_std, len(predictions))
        
        return samples
    
    def compute_residual_stats(self, observations: np.ndarray, 
                              predictions: np.ndarray) -> Dict[str, float]:
        """Compute residual statistics."""
        residuals = observations - predictions
        
        return {
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "rmse": np.sqrt(np.mean(residuals**2)),
            "mae": np.mean(np.abs(residuals)),
            "max_abs_residual": np.max(np.abs(residuals))
        }


class StudentTLikelihood(Likelihood):
    """
    Student's t-likelihood for robust inference with outliers.
    """
    
    def __init__(self, degrees_of_freedom: float = 3.0,
                 scale: Optional[float] = None,
                 estimate_scale: bool = False):
        """
        Initialize Student's t likelihood.
        
        Args:
            degrees_of_freedom: Degrees of freedom parameter (nu)
            scale: Scale parameter (if None, will be estimated)
            estimate_scale: Whether to estimate scale parameter
        """
        self.nu = degrees_of_freedom
        self.scale = scale
        self.estimate_scale = estimate_scale
        
        if self.nu <= 0:
            raise ValueError("Degrees of freedom must be positive")
    
    def log_prob(self, observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Compute Student's t log likelihood.
        
        Args:
            observations: Observed data
            predictions: Model predictions
            
        Returns:
            log_likelihood: Log likelihood value
        """
        if len(observations) != len(predictions):
            raise ValueError("Observations and predictions must have same length")
        
        if self.scale is None:
            raise ValueError("Scale parameter must be specified or estimated")
        
        # Compute residuals
        residuals = observations - predictions
        
        # Student's t log likelihood
        n = len(observations)
        log_likelihood = n * (stats.loggamma((self.nu + 1) / 2) - 
                             stats.loggamma(self.nu / 2) - 
                             0.5 * np.log(np.pi * self.nu) - 
                             np.log(self.scale))
        
        log_likelihood -= 0.5 * (self.nu + 1) * np.sum(
            np.log(1 + (residuals / self.scale)**2 / self.nu)
        )
        
        return log_likelihood
    
    def sample_predictions(self, predictions: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from Student's t predictive distribution."""
        if self.scale is None:
            raise ValueError("Scale parameter must be specified")
        
        samples = np.zeros((n_samples, len(predictions)))
        for i in range(n_samples):
            t_samples = stats.t.rvs(df=self.nu, scale=self.scale, size=len(predictions))
            samples[i] = predictions + t_samples
        
        return samples


class LaplaceLikelihood(Likelihood):
    """
    Laplace (double exponential) likelihood for robust inference.
    """
    
    def __init__(self, scale: Optional[float] = None, estimate_scale: bool = False):
        """
        Initialize Laplace likelihood.
        
        Args:
            scale: Scale parameter (if None, will be estimated)
            estimate_scale: Whether to estimate scale parameter
        """
        self.scale = scale
        self.estimate_scale = estimate_scale
    
    def log_prob(self, observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Compute Laplace log likelihood.
        
        Args:
            observations: Observed data
            predictions: Model predictions
            
        Returns:
            log_likelihood: Log likelihood value
        """
        if len(observations) != len(predictions):
            raise ValueError("Observations and predictions must have same length")
        
        if self.scale is None:
            raise ValueError("Scale parameter must be specified or estimated")
        
        # Compute residuals
        residuals = observations - predictions
        
        # Laplace log likelihood
        n = len(observations)
        log_likelihood = -n * np.log(2 * self.scale)
        log_likelihood -= np.sum(np.abs(residuals)) / self.scale
        
        return log_likelihood
    
    def sample_predictions(self, predictions: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from Laplace predictive distribution."""
        if self.scale is None:
            raise ValueError("Scale parameter must be specified")
        
        samples = np.zeros((n_samples, len(predictions)))
        for i in range(n_samples):
            laplace_samples = stats.laplace.rvs(scale=self.scale, size=len(predictions))
            samples[i] = predictions + laplace_samples
        
        return samples


class HeteroscedasticGaussianLikelihood(Likelihood):
    """
    Gaussian likelihood with heteroscedastic (spatially varying) noise.
    """
    
    def __init__(self, noise_model: callable,
                 noise_params: Optional[np.ndarray] = None):
        """
        Initialize heteroscedastic Gaussian likelihood.
        
        Args:
            noise_model: Function that computes noise std at each location
            noise_params: Parameters for noise model
        """
        self.noise_model = noise_model
        self.noise_params = noise_params
    
    def log_prob(self, observations: np.ndarray, predictions: np.ndarray,
                observation_locations: Optional[np.ndarray] = None) -> float:
        """
        Compute heteroscedastic Gaussian log likelihood.
        
        Args:
            observations: Observed data
            predictions: Model predictions
            observation_locations: Locations of observations
            
        Returns:
            log_likelihood: Log likelihood value
        """
        if len(observations) != len(predictions):
            raise ValueError("Observations and predictions must have same length")
        
        if observation_locations is None:
            raise ValueError("Observation locations required for heteroscedastic likelihood")
        
        # Compute spatially varying noise
        noise_stds = self.noise_model(observation_locations, self.noise_params)
        
        if np.any(noise_stds <= 0):
            return -np.inf
        
        # Compute residuals
        residuals = observations - predictions
        
        # Heteroscedastic Gaussian log likelihood
        n = len(observations)
        log_likelihood = -0.5 * n * np.log(2 * np.pi)
        log_likelihood -= np.sum(np.log(noise_stds))
        log_likelihood -= 0.5 * np.sum((residuals / noise_stds)**2)
        
        return log_likelihood
    
    def sample_predictions(self, predictions: np.ndarray, 
                          observation_locations: np.ndarray,
                          n_samples: int = 1) -> np.ndarray:
        """Sample from heteroscedastic Gaussian predictive distribution."""
        noise_stds = self.noise_model(observation_locations, self.noise_params)
        
        samples = np.zeros((n_samples, len(predictions)))
        for i in range(n_samples):
            samples[i] = predictions + np.random.normal(0, noise_stds)
        
        return samples


class PoissonLikelihood(Likelihood):
    """
    Poisson likelihood for count data.
    """
    
    def log_prob(self, observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Compute Poisson log likelihood.
        
        Args:
            observations: Observed counts (non-negative integers)
            predictions: Model predictions (rates)
            
        Returns:
            log_likelihood: Log likelihood value
        """
        if len(observations) != len(predictions):
            raise ValueError("Observations and predictions must have same length")
        
        # Check for non-negative predictions
        if np.any(predictions <= 0):
            return -np.inf
        
        # Check for non-negative integer observations
        if np.any(observations < 0) or not np.allclose(observations, np.round(observations)):
            return -np.inf
        
        # Poisson log likelihood
        log_likelihood = np.sum(observations * np.log(predictions) - predictions - 
                               stats.loggamma(observations + 1))
        
        return log_likelihood
    
    def sample_predictions(self, predictions: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from Poisson predictive distribution."""
        samples = np.zeros((n_samples, len(predictions)))
        for i in range(n_samples):
            samples[i] = stats.poisson.rvs(mu=predictions)
        
        return samples


class CompositeLikelihood(Likelihood):
    """
    Composite likelihood for multiple types of observations.
    """
    
    def __init__(self, likelihoods: Dict[str, Likelihood],
                 observation_types: Dict[str, np.ndarray]):
        """
        Initialize composite likelihood.
        
        Args:
            likelihoods: Dictionary of likelihood functions
            observation_types: Dictionary mapping types to observation indices
        """
        self.likelihoods = likelihoods
        self.observation_types = observation_types
        
        # Validate observation types
        all_indices = []
        for indices in observation_types.values():
            all_indices.extend(indices)
        
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Observation indices must be unique across types")
    
    def log_prob(self, observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Compute composite log likelihood.
        
        Args:
            observations: Observed data
            predictions: Model predictions
            
        Returns:
            log_likelihood: Log likelihood value
        """
        if len(observations) != len(predictions):
            raise ValueError("Observations and predictions must have same length")
        
        total_log_likelihood = 0.0
        
        for obs_type, likelihood in self.likelihoods.items():
            indices = self.observation_types[obs_type]
            obs_subset = observations[indices]
            pred_subset = predictions[indices]
            
            log_like = likelihood.log_prob(obs_subset, pred_subset)
            total_log_likelihood += log_like
        
        return total_log_likelihood
    
    def sample_predictions(self, predictions: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from composite predictive distribution."""
        samples = np.zeros((n_samples, len(predictions)))
        
        for obs_type, likelihood in self.likelihoods.items():
            indices = self.observation_types[obs_type]
            pred_subset = predictions[indices]
            
            subset_samples = likelihood.sample_predictions(pred_subset, n_samples)
            samples[:, indices] = subset_samples
        
        return samples