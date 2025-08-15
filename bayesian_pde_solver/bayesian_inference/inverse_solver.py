"""
Bayesian Inverse Solver

Main class for solving inverse problems using Bayesian methods with
various sampling and optimization techniques.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable, List
from scipy.optimize import minimize
import time
from tqdm import tqdm

from .priors import Prior
from .likelihood import Likelihood
from .mcmc_sampler import MCMCSampler
from .variational_inference import VariationalInference
from .posterior_analysis import PosteriorAnalysis


class InverseSolver:
    """
    Bayesian inverse solver for PDE parameter estimation.
    """
    
    def __init__(self, forward_solver, prior: Prior, likelihood: Likelihood,
                 observation_points: np.ndarray, observations: np.ndarray,
                 noise_std: float = 0.01):
        """
        Initialize Bayesian inverse solver.
        
        Args:
            forward_solver: Forward PDE solver
            prior: Prior distribution for parameters
            likelihood: Likelihood function
            observation_points: Points where observations are made
            observations: Observed data
            noise_std: Standard deviation of observation noise
        """
        self.forward_solver = forward_solver
        self.prior = prior
        self.likelihood = likelihood
        self.observation_points = observation_points
        self.observations = observations
        self.noise_std = noise_std
        
        self.parameter_names = prior.parameter_names
        self.parameter_dim = len(self.parameter_names)
        
        # Storage for results
        self.map_estimate = None
        self.samples = None
        self.posterior_analysis = None
        
    def log_posterior(self, parameters: np.ndarray, 
                     boundary_conditions: Dict[str, Any]) -> float:
        """
        Compute log posterior probability.
        
        Args:
            parameters: Parameter values
            boundary_conditions: Boundary conditions for forward solve
            
        Returns:
            log_posterior: Log posterior probability
        """
        # Convert parameters to dictionary
        param_dict = dict(zip(self.parameter_names, parameters))
        
        # Compute log prior
        log_prior = self.prior.log_prob(parameters)
        if not np.isfinite(log_prior):
            return -np.inf
        
        try:
            # Solve forward problem
            solution = self.forward_solver.solve(param_dict, boundary_conditions)
            
            # Compute observables
            predicted = self.forward_solver.compute_observables(solution, self.observation_points)
            
            # Compute log likelihood
            log_like = self.likelihood.log_prob(self.observations, predicted)
            
            return log_prior + log_like
            
        except Exception as e:
            print(f"Forward solve failed: {e}")
            return -np.inf
    
    def negative_log_posterior(self, parameters: np.ndarray,
                              boundary_conditions: Dict[str, Any]) -> float:
        """Negative log posterior for optimization."""
        return -self.log_posterior(parameters, boundary_conditions)
    
    def find_map_estimate(self, boundary_conditions: Dict[str, Any],
                         initial_guess: Optional[np.ndarray] = None,
                         method: str = "L-BFGS-B",
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find Maximum A Posteriori (MAP) estimate.
        
        Args:
            boundary_conditions: Boundary conditions
            initial_guess: Initial parameter guess
            method: Optimization method
            options: Optimization options
            
        Returns:
            result: Optimization result with MAP estimate
        """
        if initial_guess is None:
            initial_guess = self.prior.sample(1)[0]
        
        if options is None:
            options = {"maxiter": 1000, "disp": True}
        
        print("Finding MAP estimate...")
        start_time = time.time()
        
        result = minimize(
            fun=self.negative_log_posterior,
            x0=initial_guess,
            args=(boundary_conditions,),
            method=method,
            bounds=self.prior.get_bounds(),
            options=options
        )
        
        elapsed_time = time.time() - start_time
        
        if result.success:
            self.map_estimate = result.x
            print(f"MAP estimation completed in {elapsed_time:.2f} seconds")
            print(f"MAP estimate: {dict(zip(self.parameter_names, result.x))}")
        else:
            print(f"MAP estimation failed: {result.message}")
        
        return {
            "success": result.success,
            "map_estimate": result.x,
            "log_posterior": -result.fun,
            "message": result.message,
            "elapsed_time": elapsed_time
        }
    
    def sample_posterior_mcmc(self, boundary_conditions: Dict[str, Any],
                             sampler_type: str = "metropolis_hastings",
                             n_samples: int = 10000,
                             n_burn: int = 2000,
                             n_thin: int = 10,
                             initial_state: Optional[np.ndarray] = None,
                             sampler_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sample from posterior using MCMC.
        
        Args:
            boundary_conditions: Boundary conditions
            sampler_type: Type of MCMC sampler
            n_samples: Number of samples
            n_burn: Number of burn-in samples
            n_thin: Thinning interval
            initial_state: Initial state for chain
            sampler_options: Sampler-specific options
            
        Returns:
            results: Sampling results
        """
        if initial_state is None:
            if self.map_estimate is not None:
                initial_state = self.map_estimate
            else:
                initial_state = self.prior.sample(1)[0]
        
        if sampler_options is None:
            sampler_options = {}
        
        # Create sampler
        sampler = MCMCSampler(
            log_posterior_fn=lambda x: self.log_posterior(x, boundary_conditions),
            parameter_dim=self.parameter_dim,
            sampler_type=sampler_type,
            **sampler_options
        )
        
        print(f"Starting MCMC sampling with {sampler_type}...")
        start_time = time.time()
        
        # Run sampling
        samples, log_probs, acceptance_rate = sampler.sample(
            n_samples=n_samples + n_burn,
            initial_state=initial_state,
            n_thin=n_thin
        )
        
        # Remove burn-in
        samples = samples[n_burn:]
        log_probs = log_probs[n_burn:]
        
        elapsed_time = time.time() - start_time
        
        self.samples = samples
        
        print(f"MCMC sampling completed in {elapsed_time:.2f} seconds")
        print(f"Acceptance rate: {acceptance_rate:.3f}")
        print(f"Effective samples: {len(samples)}")
        
        return {
            "samples": samples,
            "log_posteriors": log_probs,
            "acceptance_rate": acceptance_rate,
            "elapsed_time": elapsed_time,
            "n_effective": len(samples)
        }
    
    def sample_posterior_vi(self, boundary_conditions: Dict[str, Any],
                           vi_type: str = "mean_field",
                           n_iterations: int = 5000,
                           learning_rate: float = 0.01,
                           n_samples: int = 1000,
                           vi_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Approximate posterior using variational inference.
        
        Args:
            boundary_conditions: Boundary conditions
            vi_type: Type of variational inference
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            n_samples: Number of samples from approximate posterior
            vi_options: VI-specific options
            
        Returns:
            results: VI results
        """
        if vi_options is None:
            vi_options = {}
        
        # Create VI solver
        vi_solver = VariationalInference(
            log_posterior_fn=lambda x: self.log_posterior(x, boundary_conditions),
            parameter_dim=self.parameter_dim,
            vi_type=vi_type,
            **vi_options
        )
        
        print(f"Starting variational inference with {vi_type}...")
        start_time = time.time()
        
        # Optimize variational parameters
        vi_result = vi_solver.optimize(
            n_iterations=n_iterations,
            learning_rate=learning_rate
        )
        
        # Sample from approximate posterior
        samples = vi_solver.sample(n_samples)
        
        elapsed_time = time.time() - start_time
        
        self.samples = samples
        
        print(f"Variational inference completed in {elapsed_time:.2f} seconds")
        print(f"Final ELBO: {vi_result['final_elbo']:.4f}")
        
        return {
            "samples": samples,
            "variational_params": vi_result["variational_params"],
            "elbo_history": vi_result["elbo_history"],
            "final_elbo": vi_result["final_elbo"],
            "elapsed_time": elapsed_time
        }
    
    def analyze_posterior(self, confidence_levels: List[float] = [0.68, 0.95],
                         compute_correlations: bool = True) -> PosteriorAnalysis:
        """
        Analyze posterior samples.
        
        Args:
            confidence_levels: Confidence levels for credible intervals
            compute_correlations: Whether to compute parameter correlations
            
        Returns:
            analysis: Posterior analysis results
        """
        if self.samples is None:
            raise ValueError("No samples available. Run MCMC or VI first.")
        
        self.posterior_analysis = PosteriorAnalysis(
            samples=self.samples,
            parameter_names=self.parameter_names,
            confidence_levels=confidence_levels,
            compute_correlations=compute_correlations
        )
        
        return self.posterior_analysis
    
    def predict_forward(self, boundary_conditions: Dict[str, Any],
                       prediction_points: Optional[np.ndarray] = None,
                       n_samples: int = 1000,
                       return_uncertainty: bool = True) -> Dict[str, Any]:
        """
        Forward prediction with uncertainty quantification.
        
        Args:
            boundary_conditions: Boundary conditions
            prediction_points: Points for prediction
            n_samples: Number of posterior samples to use
            return_uncertainty: Whether to compute prediction uncertainty
            
        Returns:
            predictions: Prediction results with uncertainty
        """
        if self.samples is None:
            raise ValueError("No samples available. Run MCMC or VI first.")
        
        if prediction_points is None:
            prediction_points = self.observation_points
        
        # Select random subset of samples
        if len(self.samples) > n_samples:
            indices = np.random.choice(len(self.samples), n_samples, replace=False)
            sample_subset = self.samples[indices]
        else:
            sample_subset = self.samples
        
        predictions = []
        print(f"Computing forward predictions for {len(sample_subset)} samples...")
        
        for i, parameters in enumerate(tqdm(sample_subset)):
            param_dict = dict(zip(self.parameter_names, parameters))
            
            try:
                solution = self.forward_solver.solve(param_dict, boundary_conditions)
                pred = self.forward_solver.compute_observables(solution, prediction_points)
                predictions.append(pred)
            except Exception as e:
                print(f"Forward solve failed for sample {i}: {e}")
                continue
        
        predictions = np.array(predictions)
        
        result = {
            "prediction_points": prediction_points,
            "predictions": predictions,
            "mean_prediction": np.mean(predictions, axis=0)
        }
        
        if return_uncertainty:
            result.update({
                "std_prediction": np.std(predictions, axis=0),
                "quantiles": {
                    "q05": np.percentile(predictions, 5, axis=0),
                    "q25": np.percentile(predictions, 25, axis=0),
                    "q75": np.percentile(predictions, 75, axis=0),
                    "q95": np.percentile(predictions, 95, axis=0)
                }
            })
        
        return result
    
    def compute_model_evidence(self, boundary_conditions: Dict[str, Any],
                              method: str = "harmonic_mean") -> float:
        """
        Compute marginal likelihood (model evidence).
        
        Args:
            boundary_conditions: Boundary conditions
            method: Method for evidence computation
            
        Returns:
            log_evidence: Log marginal likelihood
        """
        if self.samples is None:
            raise ValueError("No samples available. Run MCMC first.")
        
        if method == "harmonic_mean":
            # Harmonic mean estimator (biased but simple)
            log_likes = []
            for parameters in self.samples:
                log_post = self.log_posterior(parameters, boundary_conditions)
                log_prior = self.prior.log_prob(parameters)
                log_like = log_post - log_prior
                log_likes.append(log_like)
            
            log_likes = np.array(log_likes)
            # Harmonic mean of likelihoods
            max_log_like = np.max(log_likes)
            log_evidence = max_log_like - np.log(np.mean(np.exp(max_log_like - log_likes)))
            
        else:
            raise NotImplementedError(f"Evidence computation method {method} not implemented")
        
        return log_evidence
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of inverse problem results."""
        summary = {
            "parameter_names": self.parameter_names,
            "n_observations": len(self.observations),
            "observation_noise_std": self.noise_std
        }
        
        if self.map_estimate is not None:
            summary["map_estimate"] = dict(zip(self.parameter_names, self.map_estimate))
        
        if self.samples is not None:
            summary["n_samples"] = len(self.samples)
            summary["posterior_mean"] = dict(zip(self.parameter_names, np.mean(self.samples, axis=0)))
            summary["posterior_std"] = dict(zip(self.parameter_names, np.std(self.samples, axis=0)))
        
        if self.posterior_analysis is not None:
            summary["credible_intervals"] = self.posterior_analysis.credible_intervals
        
        return summary