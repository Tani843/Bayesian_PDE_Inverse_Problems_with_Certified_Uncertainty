"""
Comprehensive Benchmark Comparison System for Bayesian PDE Inverse Problems

This module provides a complete benchmarking framework for comparing different
inverse problem solution methods across multiple PDE types with rigorous
statistical analysis and publication-quality visualization.

Includes implementations of:
- Classical methods (Tikhonov regularization, adjoint-based optimization)
- Modern methods (Ensemble Kalman Filter, standard MCMC)
- Performance metrics with statistical significance testing
- Publication-ready plots and tables
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import norm, chi2, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


@dataclass
class BenchmarkResult:
    """Data structure for storing benchmark results."""
    method_name: str
    parameter_estimate: np.ndarray
    parameter_uncertainty: Optional[np.ndarray]
    computational_time: float
    convergence_iterations: int
    log_likelihood: float
    mse_error: float
    coverage_probability: float
    uncertainty_quality_score: float
    additional_metrics: Dict[str, Any]


class TikhonovRegularization:
    """
    Classical Tikhonov regularization for PDE inverse problems.
    
    Solves regularized least squares problem:
    min_θ ||F(θ) - y||² + λ||L(θ - θ_prior)||²
    
    where F is the forward operator, L is regularization operator,
    and λ is regularization parameter.
    """
    
    def __init__(self, regularization_parameter: float = 1e-3,
                 regularization_operator: Optional[np.ndarray] = None):
        """
        Initialize Tikhonov regularization solver.
        
        Parameters:
        -----------
        regularization_parameter : float
            Regularization strength λ
        regularization_operator : np.ndarray, optional
            Regularization matrix L (identity if None)
        """
        self.regularization_parameter = regularization_parameter
        self.regularization_operator = regularization_operator
        self.solution_path = []
        
    def solve(self, forward_operator: Callable, jacobian_operator: Callable,
              observations: np.ndarray, initial_guess: np.ndarray,
              prior_mean: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Solve Tikhonov regularized inverse problem.
        
        Parameters:
        -----------
        forward_operator : Callable
            Forward map F: θ → y
        jacobian_operator : Callable  
            Jacobian ∂F/∂θ
        observations : np.ndarray
            Observed data y
        initial_guess : np.ndarray
            Initial parameter estimate
        prior_mean : np.ndarray, optional
            Prior mean θ_prior
            
        Returns:
        --------
        Dict[str, Any]
            Solution dictionary with estimate, uncertainty, and diagnostics
        """
        start_time = time.time()
        
        if prior_mean is None:
            prior_mean = np.zeros_like(initial_guess)
            
        # Define regularized objective function
        def objective(theta):
            residual = forward_operator(theta) - observations
            data_fit = 0.5 * np.sum(residual**2)
            
            # Regularization term
            if self.regularization_operator is not None:
                reg_term = self.regularization_operator @ (theta - prior_mean)
            else:
                reg_term = theta - prior_mean
                
            regularization = 0.5 * self.regularization_parameter * np.sum(reg_term**2)
            
            return data_fit + regularization
        
        def gradient(theta):
            residual = forward_operator(theta) - observations
            jacobian = jacobian_operator(theta)
            data_grad = jacobian.T @ residual
            
            # Regularization gradient
            if self.regularization_operator is not None:
                reg_grad = (self.regularization_parameter * 
                           self.regularization_operator.T @ 
                           self.regularization_operator @ (theta - prior_mean))
            else:
                reg_grad = self.regularization_parameter * (theta - prior_mean)
                
            return data_grad + reg_grad
        
        # Optimize using L-BFGS-B
        result = opt.minimize(objective, initial_guess, jac=gradient, 
                            method='L-BFGS-B', 
                            options={'disp': False, 'maxiter': 1000})
        
        # Compute uncertainty estimate using Hessian approximation
        jacobian_final = jacobian_operator(result.x)
        hessian_approx = jacobian_final.T @ jacobian_final
        
        if self.regularization_operator is not None:
            reg_hessian = (self.regularization_parameter * 
                          self.regularization_operator.T @ self.regularization_operator)
        else:
            reg_hessian = self.regularization_parameter * np.eye(len(result.x))
            
        total_hessian = hessian_approx + reg_hessian
        
        try:
            # Uncertainty from inverse Hessian
            uncertainty_covariance = la.inv(total_hessian)
            parameter_std = np.sqrt(np.diag(uncertainty_covariance))
        except la.LinAlgError:
            parameter_std = np.full_like(result.x, np.inf)
            
        computational_time = time.time() - start_time
        
        return {
            'parameter_estimate': result.x,
            'parameter_uncertainty': parameter_std,
            'computational_time': computational_time,
            'convergence_iterations': result.nit,
            'objective_value': result.fun,
            'convergence_success': result.success,
            'optimization_message': result.message
        }


class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter for PDE inverse problems.
    
    Iterative ensemble-based method that approximates posterior distribution
    using ensemble of particles evolved through data assimilation steps.
    """
    
    def __init__(self, ensemble_size: int = 100, inflation_factor: float = 1.01):
        """
        Initialize Ensemble Kalman Filter.
        
        Parameters:
        -----------
        ensemble_size : int
            Number of ensemble members
        inflation_factor : float
            Covariance inflation to prevent collapse
        """
        self.ensemble_size = ensemble_size
        self.inflation_factor = inflation_factor
        self.ensemble_history = []
        
    def solve(self, forward_operator: Callable, observations: np.ndarray,
              observation_noise_cov: np.ndarray, prior_mean: np.ndarray,
              prior_covariance: np.ndarray, n_iterations: int = 10) -> Dict[str, Any]:
        """
        Solve inverse problem using Ensemble Kalman Filter.
        
        Parameters:
        -----------
        forward_operator : Callable
            Forward operator F(θ)
        observations : np.ndarray
            Observation vector y
        observation_noise_cov : np.ndarray
            Observation noise covariance R
        prior_mean : np.ndarray
            Prior mean
        prior_covariance : np.ndarray
            Prior covariance
        n_iterations : int
            Number of EnKF iterations
            
        Returns:
        --------
        Dict[str, Any]
            Results with ensemble-based estimates
        """
        start_time = time.time()
        
        # Initialize ensemble from prior
        ensemble = np.random.multivariate_normal(
            prior_mean, prior_covariance, self.ensemble_size
        )
        
        for iteration in range(n_iterations):
            # Forecast step: apply forward operator
            forecasted_observations = np.array([
                forward_operator(member) for member in ensemble
            ])
            
            # Compute ensemble statistics
            ensemble_mean = np.mean(ensemble, axis=0)
            obs_ensemble_mean = np.mean(forecasted_observations, axis=0)
            
            # Ensemble anomalies
            param_anomalies = ensemble - ensemble_mean[None, :]
            obs_anomalies = forecasted_observations - obs_ensemble_mean[None, :]
            
            # Covariances
            param_cov = (param_anomalies.T @ param_anomalies) / (self.ensemble_size - 1)
            cross_cov = (param_anomalies.T @ obs_anomalies) / (self.ensemble_size - 1)
            obs_cov = (obs_anomalies.T @ obs_anomalies) / (self.ensemble_size - 1)
            
            # Innovation covariance
            innovation_cov = obs_cov + observation_noise_cov
            
            # Kalman gain
            try:
                kalman_gain = cross_cov @ la.inv(innovation_cov)
            except la.LinAlgError:
                kalman_gain = cross_cov @ la.pinv(innovation_cov)
            
            # Update ensemble members
            for i in range(self.ensemble_size):
                # Perturbed observations
                obs_perturbed = observations + np.random.multivariate_normal(
                    np.zeros(len(observations)), observation_noise_cov
                )
                
                # Innovation
                innovation = obs_perturbed - forecasted_observations[i]
                
                # Update
                ensemble[i] += kalman_gain @ innovation
            
            # Covariance inflation
            ensemble_mean_updated = np.mean(ensemble, axis=0)
            ensemble = (ensemble - ensemble_mean_updated[None, :]) * self.inflation_factor + ensemble_mean_updated[None, :]
            
            self.ensemble_history.append(ensemble.copy())
        
        # Final statistics
        final_mean = np.mean(ensemble, axis=0)
        final_cov = np.cov(ensemble.T)
        final_std = np.sqrt(np.diag(final_cov))
        
        computational_time = time.time() - start_time
        
        return {
            'parameter_estimate': final_mean,
            'parameter_uncertainty': final_std,
            'parameter_covariance': final_cov,
            'final_ensemble': ensemble,
            'computational_time': computational_time,
            'convergence_iterations': n_iterations,
            'ensemble_history': self.ensemble_history
        }


class StandardMCMC:
    """
    Standard MCMC (Metropolis-Hastings) for Bayesian PDE inverse problems.
    
    Implements adaptive Metropolis-Hastings algorithm with covariance adaptation.
    """
    
    def __init__(self, step_size: float = 0.1, adaptation_rate: float = 0.01):
        """
        Initialize MCMC sampler.
        
        Parameters:
        -----------
        step_size : float
            Initial proposal step size
        adaptation_rate : float
            Rate of step size adaptation
        """
        self.step_size = step_size
        self.adaptation_rate = adaptation_rate
        self.chain_history = []
        self.acceptance_history = []
        
    def log_posterior(self, theta: np.ndarray, forward_operator: Callable,
                     observations: np.ndarray, noise_precision: np.ndarray,
                     log_prior: Callable) -> float:
        """
        Compute log posterior probability.
        
        Parameters:
        -----------
        theta : np.ndarray
            Parameter vector
        forward_operator : Callable
            Forward operator F(θ)
        observations : np.ndarray
            Observations y
        noise_precision : np.ndarray
            Precision matrix Σ^(-1)
        log_prior : Callable
            Log prior density
            
        Returns:
        --------
        float
            Log posterior probability
        """
        try:
            # Forward model evaluation
            predicted = forward_operator(theta)
            residual = observations - predicted
            
            # Log likelihood (Gaussian)
            log_likelihood = -0.5 * residual.T @ noise_precision @ residual
            
            # Log prior
            log_prior_val = log_prior(theta)
            
            return log_likelihood + log_prior_val
            
        except Exception:
            return -np.inf
    
    def solve(self, forward_operator: Callable, observations: np.ndarray,
              noise_precision: np.ndarray, log_prior: Callable,
              initial_theta: np.ndarray, n_samples: int = 10000,
              burn_in: int = 2000) -> Dict[str, Any]:
        """
        Run MCMC sampling for inverse problem.
        
        Parameters:
        -----------
        forward_operator : Callable
            Forward operator
        observations : np.ndarray
            Observation data
        noise_precision : np.ndarray
            Noise precision matrix
        log_prior : Callable
            Log prior function
        initial_theta : np.ndarray
            Initial parameter guess
        n_samples : int
            Number of MCMC samples
        burn_in : int
            Burn-in period
            
        Returns:
        --------
        Dict[str, Any]
            MCMC results and diagnostics
        """
        start_time = time.time()
        
        dimension = len(initial_theta)
        chain = np.zeros((n_samples, dimension))
        log_post_values = np.zeros(n_samples)
        n_accepted = 0
        
        # Initialize
        current_theta = initial_theta.copy()
        current_log_post = self.log_posterior(
            current_theta, forward_operator, observations, 
            noise_precision, log_prior
        )
        
        # Proposal covariance (adaptive)
        proposal_cov = self.step_size**2 * np.eye(dimension)
        
        for i in range(n_samples):
            # Propose new state
            proposed_theta = np.random.multivariate_normal(current_theta, proposal_cov)
            proposed_log_post = self.log_posterior(
                proposed_theta, forward_operator, observations,
                noise_precision, log_prior
            )
            
            # Metropolis-Hastings acceptance
            log_alpha = proposed_log_post - current_log_post
            alpha = min(1.0, np.exp(log_alpha))
            
            if np.random.rand() < alpha:
                current_theta = proposed_theta
                current_log_post = proposed_log_post
                n_accepted += 1
            
            # Store sample
            chain[i] = current_theta
            log_post_values[i] = current_log_post
            
            # Adapt proposal covariance
            if i > 100 and i % 100 == 0:
                recent_samples = chain[max(0, i-500):i+1]
                sample_cov = np.cov(recent_samples.T)
                
                # Robbins-Monro adaptation
                target_acceptance = 0.234  # Optimal for Gaussian targets
                current_acceptance = n_accepted / (i + 1)
                
                adaptation_factor = 1 + self.adaptation_rate * (current_acceptance - target_acceptance)
                proposal_cov = adaptation_factor * (2.38**2 / dimension) * sample_cov + 1e-8 * np.eye(dimension)
        
        # Remove burn-in
        samples = chain[burn_in:]
        log_posts = log_post_values[burn_in:]
        
        # Compute statistics
        parameter_mean = np.mean(samples, axis=0)
        parameter_std = np.std(samples, axis=0)
        acceptance_rate = n_accepted / n_samples
        
        computational_time = time.time() - start_time
        
        return {
            'parameter_estimate': parameter_mean,
            'parameter_uncertainty': parameter_std,
            'parameter_samples': samples,
            'log_posterior_values': log_posts,
            'acceptance_rate': acceptance_rate,
            'computational_time': computational_time,
            'convergence_iterations': n_samples,
            'effective_sample_size': self._effective_sample_size(samples)
        }
    
    def _effective_sample_size(self, samples: np.ndarray) -> np.ndarray:
        """Compute effective sample size for each parameter."""
        n_samples, n_params = samples.shape
        ess = np.zeros(n_params)
        
        for i in range(n_params):
            # Autocorrelation computation
            x = samples[:, i] - np.mean(samples[:, i])
            autocorr = np.correlate(x, x, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first negative value
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                cutoff = first_negative[0]
            else:
                cutoff = len(autocorr)
            
            # Integrated autocorrelation time
            tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
            ess[i] = n_samples / (2 * tau_int) if tau_int > 0 else n_samples
        
        return ess


class AdjointBasedOptimization:
    """
    Adjoint-based optimization for PDE inverse problems.
    
    Uses adjoint sensitivity analysis for efficient gradient computation
    in PDE-constrained optimization problems.
    """
    
    def __init__(self, optimization_method: str = 'L-BFGS-B'):
        """
        Initialize adjoint-based optimizer.
        
        Parameters:
        -----------
        optimization_method : str
            Optimization algorithm ('L-BFGS-B', 'CG', 'Newton-CG')
        """
        self.optimization_method = optimization_method
        self.iteration_history = []
        
    def solve(self, forward_operator: Callable, adjoint_operator: Callable,
              observations: np.ndarray, initial_guess: np.ndarray,
              bounds: Optional[List[Tuple]] = None) -> Dict[str, Any]:
        """
        Solve inverse problem using adjoint-based optimization.
        
        Parameters:
        -----------
        forward_operator : Callable
            Forward operator F(θ)
        adjoint_operator : Callable
            Adjoint operator for gradient computation
        observations : np.ndarray
            Observation data
        initial_guess : np.ndarray
            Initial parameter estimate
        bounds : List[Tuple], optional
            Parameter bounds for constrained optimization
            
        Returns:
        --------
        Dict[str, Any]
            Optimization results and diagnostics
        """
        start_time = time.time()
        
        def objective(theta):
            predicted = forward_operator(theta)
            residual = predicted - observations
            return 0.5 * np.sum(residual**2)
        
        def gradient(theta):
            return adjoint_operator(theta, observations)
        
        # Store iteration history
        def callback(theta):
            self.iteration_history.append({
                'parameters': theta.copy(),
                'objective': objective(theta)
            })
        
        # Optimize
        result = opt.minimize(
            objective, initial_guess, 
            jac=gradient, method=self.optimization_method,
            bounds=bounds, callback=callback,
            options={'disp': False, 'maxiter': 1000}
        )
        
        # Estimate uncertainty using finite differences (Hessian approximation)
        try:
            hessian = opt.approx_fprime(result.x, gradient, 1e-8)
            uncertainty_cov = la.inv(hessian)
            parameter_std = np.sqrt(np.diag(uncertainty_cov))
        except:
            parameter_std = np.full_like(result.x, np.inf)
        
        computational_time = time.time() - start_time
        
        return {
            'parameter_estimate': result.x,
            'parameter_uncertainty': parameter_std,
            'computational_time': computational_time,
            'convergence_iterations': result.nit,
            'objective_value': result.fun,
            'convergence_success': result.success,
            'optimization_message': result.message,
            'iteration_history': self.iteration_history
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for comparing inverse problem methods.
    
    Tests multiple methods across different PDE types with rigorous
    statistical analysis and publication-quality visualization.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize benchmark suite.
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.results = {}
        
    def run_heat_equation_benchmark(self, parameter_ranges: Dict[str, Tuple],
                                  noise_levels: List[float],
                                  n_observations_list: List[int]) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark methods on heat equation inverse problems.
        
        Parameters:
        -----------
        parameter_ranges : Dict[str, Tuple]
            Ranges for thermal conductivity and source strength
        noise_levels : List[float]
            Different noise levels to test
        n_observations_list : List[int]
            Different numbers of observations
            
        Returns:
        --------
        Dict[str, List[BenchmarkResult]]
            Results for each method
        """
        print("Running Heat Equation Benchmark...")
        
        methods = {
            'Tikhonov': TikhonovRegularization(),
            'EnKF': EnsembleKalmanFilter(),
            'MCMC': StandardMCMC(),
            'Adjoint': AdjointBasedOptimization()
        }
        
        benchmark_results = {method: [] for method in methods}
        
        # True parameters
        true_conductivity = 1.5
        true_source_strength = 2.0
        true_params = np.array([true_conductivity, true_source_strength])
        
        for noise_level in noise_levels:
            for n_obs in n_observations_list:
                print(f"  Testing: noise={noise_level:.3f}, n_obs={n_obs}")
                
                # Generate synthetic data
                obs_points = np.random.uniform(0, 1, (n_obs, 2))
                true_observations = self._heat_equation_forward(true_params, obs_points)
                noisy_observations = true_observations + noise_level * np.random.randn(n_obs)
                
                # Test each method
                for method_name, method in methods.items():
                    try:
                        result = self._run_single_heat_benchmark(
                            method, method_name, true_params, obs_points,
                            noisy_observations, noise_level
                        )
                        benchmark_results[method_name].append(result)
                    except Exception as e:
                        print(f"    {method_name} failed: {str(e)}")
                        # Add failed result
                        failed_result = BenchmarkResult(
                            method_name=method_name,
                            parameter_estimate=np.full_like(true_params, np.nan),
                            parameter_uncertainty=np.full_like(true_params, np.inf),
                            computational_time=np.inf,
                            convergence_iterations=0,
                            log_likelihood=-np.inf,
                            mse_error=np.inf,
                            coverage_probability=0.0,
                            uncertainty_quality_score=0.0,
                            additional_metrics={'error': str(e)}
                        )
                        benchmark_results[method_name].append(failed_result)
        
        return benchmark_results
    
    def _heat_equation_forward(self, params: np.ndarray, obs_points: np.ndarray) -> np.ndarray:
        """Simplified heat equation forward model."""
        conductivity, source_strength = params
        
        # Simplified analytical solution for demonstration
        # u(x,y) = source_strength * exp(-((x-0.5)² + (y-0.5)²) / conductivity)
        x, y = obs_points[:, 0], obs_points[:, 1]
        distances_sq = (x - 0.5)**2 + (y - 0.5)**2
        solution = source_strength * np.exp(-distances_sq / max(conductivity, 0.01))
        
        return solution
    
    def _heat_equation_jacobian(self, params: np.ndarray, obs_points: np.ndarray) -> np.ndarray:
        """Jacobian of heat equation forward model."""
        conductivity, source_strength = params
        x, y = obs_points[:, 0], obs_points[:, 1]
        distances_sq = (x - 0.5)**2 + (y - 0.5)**2
        
        # Partial derivatives
        exp_term = np.exp(-distances_sq / max(conductivity, 0.01))
        
        # ∂u/∂conductivity
        du_dk = source_strength * exp_term * distances_sq / (conductivity**2)
        
        # ∂u/∂source_strength  
        du_ds = exp_term
        
        return np.column_stack([du_dk, du_ds])
    
    def _run_single_heat_benchmark(self, method, method_name: str, true_params: np.ndarray,
                                 obs_points: np.ndarray, observations: np.ndarray,
                                 noise_level: float) -> BenchmarkResult:
        """Run single benchmark test for heat equation."""
        
        # Initial guess
        initial_guess = true_params + 0.5 * np.random.randn(len(true_params))
        
        if method_name == 'Tikhonov':
            result = method.solve(
                forward_operator=lambda p: self._heat_equation_forward(p, obs_points),
                jacobian_operator=lambda p: self._heat_equation_jacobian(p, obs_points),
                observations=observations,
                initial_guess=initial_guess
            )
            
        elif method_name == 'EnKF':
            result = method.solve(
                forward_operator=lambda p: self._heat_equation_forward(p, obs_points),
                observations=observations,
                observation_noise_cov=noise_level**2 * np.eye(len(observations)),
                prior_mean=np.array([1.0, 1.0]),
                prior_covariance=np.eye(2),
                n_iterations=20
            )
            
        elif method_name == 'MCMC':
            def log_prior(theta):
                # Gaussian prior
                return -0.5 * np.sum((theta - np.array([1.0, 1.0]))**2)
            
            noise_precision = (1.0 / noise_level**2) * np.eye(len(observations))
            result = method.solve(
                forward_operator=lambda p: self._heat_equation_forward(p, obs_points),
                observations=observations,
                noise_precision=noise_precision,
                log_prior=log_prior,
                initial_theta=initial_guess,
                n_samples=5000,
                burn_in=1000
            )
            
        elif method_name == 'Adjoint':
            def adjoint_grad(params, obs):
                predicted = self._heat_equation_forward(params, obs_points)
                residual = predicted - obs
                jacobian = self._heat_equation_jacobian(params, obs_points)
                return jacobian.T @ residual
                
            result = method.solve(
                forward_operator=lambda p: self._heat_equation_forward(p, obs_points),
                adjoint_operator=adjoint_grad,
                observations=observations,
                initial_guess=initial_guess
            )
        
        # Compute performance metrics
        param_estimate = result['parameter_estimate']
        param_uncertainty = result.get('parameter_uncertainty', np.full_like(param_estimate, np.inf))
        
        mse_error = np.mean((param_estimate - true_params)**2)
        
        # Coverage probability (if uncertainty available)
        if np.all(np.isfinite(param_uncertainty)):
            # 95% confidence intervals
            lower_bounds = param_estimate - 1.96 * param_uncertainty
            upper_bounds = param_estimate + 1.96 * param_uncertainty
            coverage = np.mean((lower_bounds <= true_params) & (true_params <= upper_bounds))
        else:
            coverage = 0.0
        
        # Uncertainty quality score (calibration)
        uncertainty_score = self._compute_uncertainty_quality(
            param_estimate, param_uncertainty, true_params
        )
        
        # Log likelihood
        predicted_obs = self._heat_equation_forward(param_estimate, obs_points)
        residual = observations - predicted_obs
        log_likelihood = -0.5 * np.sum(residual**2) / noise_level**2
        
        return BenchmarkResult(
            method_name=method_name,
            parameter_estimate=param_estimate,
            parameter_uncertainty=param_uncertainty,
            computational_time=result['computational_time'],
            convergence_iterations=result['convergence_iterations'],
            log_likelihood=log_likelihood,
            mse_error=mse_error,
            coverage_probability=coverage,
            uncertainty_quality_score=uncertainty_score,
            additional_metrics={'noise_level': noise_level, 'n_observations': len(observations)}
        )
    
    def _compute_uncertainty_quality(self, estimate: np.ndarray, uncertainty: np.ndarray,
                                   true_value: np.ndarray) -> float:
        """Compute uncertainty quality score based on calibration."""
        if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty <= 0):
            return 0.0
        
        # Z-scores (should be standard normal if well-calibrated)
        z_scores = (estimate - true_value) / uncertainty
        
        # Kolmogorov-Smirnov test against standard normal
        from scipy.stats import kstest
        ks_statistic, p_value = kstest(z_scores, 'norm')
        
        # Quality score: 1 - KS statistic (higher is better)
        return max(0.0, 1.0 - ks_statistic)
    
    def run_wave_equation_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark methods on wave equation inverse problems."""
        print("Wave equation benchmark not yet implemented")
        return {}
    
    def run_reaction_diffusion_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark methods on reaction-diffusion inverse problems."""
        print("Reaction-diffusion benchmark not yet implemented")
        return {}
    
    def generate_comparison_plots(self, results: Dict[str, List[BenchmarkResult]],
                                save_path: Optional[str] = None) -> plt.Figure:
        """Generate comprehensive comparison plots."""
        
        # Set up publication-quality style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (15, 10)
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Collect data for plotting
        methods = list(results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        # 1. MSE comparison
        ax = axes[0, 0]
        mse_data = []
        for method in methods:
            method_results = results[method]
            mse_values = [r.mse_error for r in method_results if np.isfinite(r.mse_error)]
            mse_data.append(mse_values)
        
        bp = ax.boxplot(mse_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('MSE')
        ax.set_title('Parameter Estimation Error')
        ax.set_yscale('log')
        
        # 2. Computational time comparison
        ax = axes[0, 1]
        time_data = []
        for method in methods:
            method_results = results[method]
            time_values = [r.computational_time for r in method_results if np.isfinite(r.computational_time)]
            time_data.append(time_values)
        
        bp = ax.boxplot(time_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computational Efficiency')
        ax.set_yscale('log')
        
        # 3. Coverage probability
        ax = axes[0, 2]
        coverage_means = []
        coverage_stds = []
        for method in methods:
            method_results = results[method]
            coverage_values = [r.coverage_probability for r in method_results]
            coverage_means.append(np.mean(coverage_values))
            coverage_stds.append(np.std(coverage_values))
        
        bars = ax.bar(methods, coverage_means, yerr=coverage_stds, 
                     color=colors, alpha=0.7, capsize=5)
        ax.axhline(y=0.95, color='red', linestyle='--', label='Target (95%)')
        ax.set_ylabel('Coverage Probability')
        ax.set_title('Uncertainty Calibration')
        ax.legend()
        
        # 4. Uncertainty quality scores
        ax = axes[1, 0]
        quality_data = []
        for method in methods:
            method_results = results[method]
            quality_values = [r.uncertainty_quality_score for r in method_results]
            quality_data.append(quality_values)
        
        bp = ax.boxplot(quality_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Uncertainty Quality Score')
        ax.set_title('Uncertainty Quality')
        
        # 5. Performance vs noise level
        ax = axes[1, 1]
        for i, method in enumerate(methods):
            method_results = results[method]
            noise_levels = [r.additional_metrics.get('noise_level', 0) for r in method_results]
            mse_values = [r.mse_error for r in method_results]
            
            # Group by noise level
            unique_noise = sorted(set(noise_levels))
            mse_by_noise = []
            for noise in unique_noise:
                noise_mse = [mse for mse, n in zip(mse_values, noise_levels) if n == noise]
                mse_by_noise.append(np.mean(noise_mse))
            
            ax.plot(unique_noise, mse_by_noise, 'o-', color=colors[i], 
                   label=method, linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('MSE')
        ax.set_title('Robustness to Noise')
        ax.legend()
        ax.set_yscale('log')
        
        # 6. Efficiency frontier (MSE vs Time)
        ax = axes[1, 2]
        for i, method in enumerate(methods):
            method_results = results[method]
            times = [r.computational_time for r in method_results if np.isfinite(r.computational_time)]
            mses = [r.mse_error for r in method_results if np.isfinite(r.mse_error)]
            
            if times and mses:
                avg_time = np.mean(times)
                avg_mse = np.mean(mses)
                ax.scatter(avg_time, avg_mse, color=colors[i], s=100, 
                          label=method, alpha=0.7)
        
        ax.set_xlabel('Computational Time (seconds)')
        ax.set_ylabel('MSE')
        ax.set_title('Efficiency Frontier')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_tables(self, results: Dict[str, List[BenchmarkResult]]) -> pd.DataFrame:
        """Create performance comparison tables."""
        
        # Aggregate results by method
        summary_data = []
        
        for method, method_results in results.items():
            if not method_results:
                continue
                
            # Compute statistics
            mse_values = [r.mse_error for r in method_results if np.isfinite(r.mse_error)]
            time_values = [r.computational_time for r in method_results if np.isfinite(r.computational_time)]
            coverage_values = [r.coverage_probability for r in method_results]
            quality_values = [r.uncertainty_quality_score for r in method_results]
            
            summary_data.append({
                'Method': method,
                'MSE Mean': np.mean(mse_values) if mse_values else np.nan,
                'MSE Std': np.std(mse_values) if mse_values else np.nan,
                'Time Mean (s)': np.mean(time_values) if time_values else np.nan,
                'Time Std (s)': np.std(time_values) if time_values else np.nan,
                'Coverage Mean': np.mean(coverage_values),
                'Coverage Std': np.std(coverage_values),
                'Quality Mean': np.mean(quality_values),
                'Quality Std': np.std(quality_values),
                'Success Rate': len(mse_values) / len(method_results) if method_results else 0
            })
        
        return pd.DataFrame(summary_data)
    
    def statistical_significance_tests(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Perform statistical significance tests between methods."""
        
        methods = list(results.keys())
        n_methods = len(methods)
        
        # Prepare data
        mse_data = {}
        time_data = {}
        
        for method in methods:
            method_results = results[method]
            mse_data[method] = [r.mse_error for r in method_results if np.isfinite(r.mse_error)]
            time_data[method] = [r.computational_time for r in method_results if np.isfinite(r.computational_time)]
        
        # Pairwise comparisons
        mse_comparisons = {}
        time_comparisons = {}
        
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                method1, method2 = methods[i], methods[j]
                
                # MSE comparison (Mann-Whitney U test for non-parametric comparison)
                if mse_data[method1] and mse_data[method2]:
                    mse_stat, mse_pval = mannwhitneyu(mse_data[method1], mse_data[method2], alternative='two-sided')
                    mse_comparisons[f"{method1}_vs_{method2}"] = {
                        'statistic': mse_stat,
                        'p_value': mse_pval,
                        'significant': mse_pval < 0.05
                    }
                
                # Time comparison
                if time_data[method1] and time_data[method2]:
                    time_stat, time_pval = mannwhitneyu(time_data[method1], time_data[method2], alternative='two-sided')
                    time_comparisons[f"{method1}_vs_{method2}"] = {
                        'statistic': time_stat,
                        'p_value': time_pval,
                        'significant': time_pval < 0.05
                    }
        
        return {
            'mse_comparisons': mse_comparisons,
            'time_comparisons': time_comparisons,
            'sample_sizes': {method: len(mse_data[method]) for method in methods}
        }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparison."""
    print("Starting Comprehensive Benchmark Suite")
    print("=" * 50)
    
    # Initialize benchmark suite
    benchmark = BenchmarkSuite(random_seed=42)
    
    # Heat equation benchmark
    parameter_ranges = {
        'conductivity': (0.5, 3.0),
        'source_strength': (1.0, 4.0)
    }
    noise_levels = [0.01, 0.05, 0.1]
    n_observations_list = [20, 50, 100]
    
    results = benchmark.run_heat_equation_benchmark(
        parameter_ranges, noise_levels, n_observations_list
    )
    
    # Generate performance analysis
    print("\nGenerating Performance Analysis...")
    
    # Create plots
    fig = benchmark.generate_comparison_plots(results, save_path='benchmark_comparison.pdf')
    plt.show()
    
    # Create performance table
    performance_table = benchmark.create_performance_tables(results)
    print("\nPerformance Summary:")
    print(performance_table.round(4))
    
    # Statistical significance tests
    significance_results = benchmark.statistical_significance_tests(results)
    print("\nStatistical Significance Tests:")
    for comparison, result in significance_results['mse_comparisons'].items():
        print(f"{comparison}: p-value = {result['p_value']:.4f}, significant = {result['significant']}")
    
    return results, performance_table, significance_results


if __name__ == "__main__":
    results, table, significance = run_comprehensive_benchmark()