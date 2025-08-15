"""
Prior Distributions

Implements various prior distributions for Bayesian parameter estimation.
"""

import numpy as np
from scipy import stats
from typing import List, Union, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class Prior(ABC):
    """
    Abstract base class for prior distributions.
    """
    
    def __init__(self, parameter_names: List[str]):
        """
        Initialize prior distribution.
        
        Args:
            parameter_names: Names of parameters
        """
        self.parameter_names = parameter_names
        self.n_params = len(parameter_names)
    
    @abstractmethod
    def log_prob(self, parameters: np.ndarray) -> float:
        """
        Compute log probability density.
        
        Args:
            parameters: Parameter values
            
        Returns:
            log_prob: Log probability density
        """
        pass
    
    @abstractmethod
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from prior distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            samples: Prior samples
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get parameter bounds for optimization.
        
        Returns:
            bounds: List of (min, max) bounds for each parameter
        """
        pass
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about parameters."""
        return {
            "parameter_names": self.parameter_names,
            "n_params": self.n_params
        }


class GaussianPrior(Prior):
    """
    Multivariate Gaussian prior distribution.
    """
    
    def __init__(self, parameter_names: List[str],
                 means: Union[float, List[float], np.ndarray],
                 covariances: Union[float, List[List[float]], np.ndarray]):
        """
        Initialize Gaussian prior.
        
        Args:
            parameter_names: Parameter names
            means: Prior means
            covariances: Prior covariance matrix
        """
        super().__init__(parameter_names)
        
        # Convert to numpy arrays
        if isinstance(means, (int, float)):
            self.means = np.full(self.n_params, means)
        else:
            self.means = np.array(means)
        
        if isinstance(covariances, (int, float)):
            self.covariances = np.eye(self.n_params) * covariances
        else:
            self.covariances = np.array(covariances)
        
        # Validate dimensions
        if len(self.means) != self.n_params:
            raise ValueError("Dimension mismatch between means and parameter names")
        
        if self.covariances.shape != (self.n_params, self.n_params):
            raise ValueError("Covariance matrix dimension mismatch")
        
        # Create scipy distribution
        self.distribution = stats.multivariate_normal(
            mean=self.means,
            cov=self.covariances
        )
        
        # Compute bounds (3 sigma)
        self.stds = np.sqrt(np.diag(self.covariances))
        self._bounds = [(mu - 3*sigma, mu + 3*sigma) 
                       for mu, sigma in zip(self.means, self.stds)]
    
    def log_prob(self, parameters: np.ndarray) -> float:
        """Compute log probability density."""
        if len(parameters) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(parameters)}")
        
        return self.distribution.logpdf(parameters)
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from Gaussian prior."""
        samples = self.distribution.rvs(size=n_samples)
        if n_samples == 1:
            samples = samples.reshape(1, -1)
        return samples
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds."""
        return self._bounds
    
    def get_marginal_stats(self) -> Dict[str, Dict[str, float]]:
        """Get marginal statistics for each parameter."""
        stats_dict = {}
        for i, name in enumerate(self.parameter_names):
            stats_dict[name] = {
                "mean": self.means[i],
                "std": self.stds[i],
                "variance": self.covariances[i, i]
            }
        return stats_dict


class UniformPrior(Prior):
    """
    Independent uniform prior distributions.
    """
    
    def __init__(self, parameter_names: List[str],
                 bounds: List[Tuple[float, float]]):
        """
        Initialize uniform prior.
        
        Args:
            parameter_names: Parameter names
            bounds: List of (min, max) bounds for each parameter
        """
        super().__init__(parameter_names)
        
        if len(bounds) != self.n_params:
            raise ValueError("Number of bounds must match number of parameters")
        
        self.bounds = bounds
        self.mins = np.array([b[0] for b in bounds])
        self.maxs = np.array([b[1] for b in bounds])
        self.widths = self.maxs - self.mins
        
        # Check bounds validity
        if np.any(self.widths <= 0):
            raise ValueError("All bounds must have positive width")
        
        # Log probability density (constant)
        self.log_density = -np.sum(np.log(self.widths))
    
    def log_prob(self, parameters: np.ndarray) -> float:
        """Compute log probability density."""
        if len(parameters) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(parameters)}")
        
        # Check if parameters are within bounds
        if np.any(parameters < self.mins) or np.any(parameters > self.maxs):
            return -np.inf
        
        return self.log_density
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from uniform prior."""
        samples = np.random.uniform(
            low=self.mins,
            high=self.maxs,
            size=(n_samples, self.n_params)
        )
        return samples
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds."""
        return self.bounds


class LogNormalPrior(Prior):
    """
    Independent log-normal prior distributions.
    """
    
    def __init__(self, parameter_names: List[str],
                 log_means: Union[float, List[float], np.ndarray],
                 log_stds: Union[float, List[float], np.ndarray]):
        """
        Initialize log-normal prior.
        
        Args:
            parameter_names: Parameter names
            log_means: Means of log-transformed parameters
            log_stds: Standard deviations of log-transformed parameters
        """
        super().__init__(parameter_names)
        
        # Convert to numpy arrays
        if isinstance(log_means, (int, float)):
            self.log_means = np.full(self.n_params, log_means)
        else:
            self.log_means = np.array(log_means)
        
        if isinstance(log_stds, (int, float)):
            self.log_stds = np.full(self.n_params, log_stds)
        else:
            self.log_stds = np.array(log_stds)
        
        # Validate dimensions
        if len(self.log_means) != self.n_params:
            raise ValueError("Dimension mismatch between log_means and parameter names")
        
        if len(self.log_stds) != self.n_params:
            raise ValueError("Dimension mismatch between log_stds and parameter names")
        
        # Create scipy distributions
        self.distributions = [
            stats.lognorm(s=log_std, scale=np.exp(log_mean))
            for log_mean, log_std in zip(self.log_means, self.log_stds)
        ]
        
        # Compute bounds (based on quantiles)
        self._bounds = [(dist.ppf(0.001), dist.ppf(0.999)) 
                       for dist in self.distributions]
    
    def log_prob(self, parameters: np.ndarray) -> float:
        """Compute log probability density."""
        if len(parameters) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(parameters)}")
        
        # Check for positive parameters
        if np.any(parameters <= 0):
            return -np.inf
        
        log_prob = 0.0
        for i, (param, dist) in enumerate(zip(parameters, self.distributions)):
            log_prob += dist.logpdf(param)
        
        return log_prob
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from log-normal prior."""
        samples = np.zeros((n_samples, self.n_params))
        for i, dist in enumerate(self.distributions):
            samples[:, i] = dist.rvs(size=n_samples)
        return samples
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds."""
        return self._bounds


class MixturePrior(Prior):
    """
    Mixture of prior distributions.
    """
    
    def __init__(self, parameter_names: List[str],
                 components: List[Prior],
                 weights: Optional[List[float]] = None):
        """
        Initialize mixture prior.
        
        Args:
            parameter_names: Parameter names
            components: List of component prior distributions
            weights: Mixture weights (uniform if None)
        """
        super().__init__(parameter_names)
        
        self.components = components
        self.n_components = len(components)
        
        # Validate component dimensions
        for comp in components:
            if comp.n_params != self.n_params:
                raise ValueError("All components must have same parameter dimension")
        
        # Set weights
        if weights is None:
            self.weights = np.ones(self.n_components) / self.n_components
        else:
            self.weights = np.array(weights)
            if len(self.weights) != self.n_components:
                raise ValueError("Number of weights must match number of components")
            if not np.isclose(np.sum(self.weights), 1.0):
                raise ValueError("Weights must sum to 1")
        
        self.log_weights = np.log(self.weights)
    
    def log_prob(self, parameters: np.ndarray) -> float:
        """Compute log probability density."""
        if len(parameters) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(parameters)}")
        
        # Compute log probabilities for each component
        log_probs = []
        for comp in self.components:
            log_probs.append(comp.log_prob(parameters))
        
        log_probs = np.array(log_probs)
        
        # Use log-sum-exp trick for numerical stability
        max_log_prob = np.max(log_probs)
        log_sum_exp = max_log_prob + np.log(
            np.sum(self.weights * np.exp(log_probs - max_log_prob))
        )
        
        return log_sum_exp
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from mixture prior."""
        samples = np.zeros((n_samples, self.n_params))
        
        # Sample component indices
        component_indices = np.random.choice(
            self.n_components,
            size=n_samples,
            p=self.weights
        )
        
        # Sample from selected components
        for i in range(n_samples):
            comp_idx = component_indices[i]
            samples[i] = self.components[comp_idx].sample(1)[0]
        
        return samples
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds (union of component bounds)."""
        bounds = []
        for i in range(self.n_params):
            min_vals = []
            max_vals = []
            for comp in self.components:
                comp_bounds = comp.get_bounds()
                min_vals.append(comp_bounds[i][0])
                max_vals.append(comp_bounds[i][1])
            bounds.append((min(min_vals), max(max_vals)))
        return bounds


class ImproperPrior(Prior):
    """
    Improper (non-normalizable) prior distribution.
    """
    
    def __init__(self, parameter_names: List[str],
                 log_prob_fn: callable,
                 bounds: List[Tuple[float, float]]):
        """
        Initialize improper prior.
        
        Args:
            parameter_names: Parameter names
            log_prob_fn: Function that computes log probability density
            bounds: Parameter bounds for optimization
        """
        super().__init__(parameter_names)
        self.log_prob_fn = log_prob_fn
        self.bounds = bounds
        
        if len(bounds) != self.n_params:
            raise ValueError("Number of bounds must match number of parameters")
    
    def log_prob(self, parameters: np.ndarray) -> float:
        """Compute log probability density."""
        if len(parameters) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(parameters)}")
        
        return self.log_prob_fn(parameters)
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from improper prior (not implementable in general)."""
        raise NotImplementedError("Cannot sample from improper prior")
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds."""
        return self.bounds