"""
Variational Inference

Implements variational Bayesian methods for approximate posterior inference
including mean-field variational Bayes and automatic differentiation variational inference.
"""

import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, List
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import warnings
from tqdm import tqdm


class VariationalInference:
    """
    Variational inference for Bayesian parameter estimation.
    
    Approximates the posterior distribution p(θ|y) with a tractable
    variational distribution q(θ; λ) by minimizing KL divergence.
    
    Parameters
    ----------
    log_posterior_fn : Callable[[np.ndarray], float]
        Function that computes log posterior probability
    parameter_dim : int
        Dimension of parameter space
    vi_type : str, default="mean_field"
        Type of variational inference ("mean_field", "full_rank")
    
    Examples
    --------
    >>> vi = VariationalInference(log_posterior_fn, parameter_dim=2)
    >>> result = vi.optimize(n_iterations=5000)
    >>> samples = vi.sample(n_samples=1000)
    """
    
    def __init__(self, log_posterior_fn: Callable[[np.ndarray], float],
                 parameter_dim: int,
                 vi_type: str = "mean_field"):
        """Initialize variational inference."""
        self.log_posterior_fn = log_posterior_fn
        self.parameter_dim = parameter_dim
        self.vi_type = vi_type
        
        # Initialize variational parameters
        self._initialize_variational_parameters()
        
        # Track optimization history
        self.elbo_history = []
        self.gradient_norms = []
        
    def _initialize_variational_parameters(self) -> None:
        """Initialize variational distribution parameters."""
        if self.vi_type == "mean_field":
            # Mean-field: independent Gaussians q(θ) = ∏ N(θ_i; μ_i, σ_i²)
            self.variational_params = {
                'means': np.zeros(self.parameter_dim),
                'log_stds': np.zeros(self.parameter_dim)  # log(σ) for numerical stability
            }
            
        elif self.vi_type == "full_rank":
            # Full-rank Gaussian: q(θ) = N(θ; μ, Σ)
            self.variational_params = {
                'mean': np.zeros(self.parameter_dim),
                'chol_cov': np.eye(self.parameter_dim)  # Cholesky factor of covariance
            }
        else:
            raise ValueError(f"Unknown VI type: {self.vi_type}")
    
    def _pack_parameters(self) -> np.ndarray:
        """Pack variational parameters into a single vector."""
        if self.vi_type == "mean_field":
            return np.concatenate([
                self.variational_params['means'],
                self.variational_params['log_stds']
            ])
        elif self.vi_type == "full_rank":
            # Pack mean and lower triangular part of Cholesky factor
            chol = self.variational_params['chol_cov']
            chol_vec = chol[np.tril_indices(self.parameter_dim)]
            return np.concatenate([self.variational_params['mean'], chol_vec])
    
    def _unpack_parameters(self, params: np.ndarray) -> None:
        """Unpack parameter vector into variational parameters."""
        if self.vi_type == "mean_field":
            self.variational_params['means'] = params[:self.parameter_dim]
            self.variational_params['log_stds'] = params[self.parameter_dim:]
        elif self.vi_type == "full_rank":
            self.variational_params['mean'] = params[:self.parameter_dim]
            
            # Reconstruct Cholesky factor
            chol_vec = params[self.parameter_dim:]
            chol = np.zeros((self.parameter_dim, self.parameter_dim))
            chol[np.tril_indices(self.parameter_dim)] = chol_vec
            self.variational_params['chol_cov'] = chol
    
    def sample_variational(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from current variational distribution.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw
            
        Returns
        -------
        samples : np.ndarray, shape (n_samples, parameter_dim)
            Samples from variational distribution
        """
        if self.vi_type == "mean_field":
            means = self.variational_params['means']
            stds = np.exp(self.variational_params['log_stds'])
            
            samples = np.random.normal(
                loc=means[None, :],
                scale=stds[None, :],
                size=(n_samples, self.parameter_dim)
            )
            
        elif self.vi_type == "full_rank":
            mean = self.variational_params['mean']
            chol = self.variational_params['chol_cov']
            
            # Sample from N(0,I) and transform
            z = np.random.normal(size=(n_samples, self.parameter_dim))
            samples = mean[None, :] + z @ chol.T
        
        return samples
    
    def log_variational_prob(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute log probability of samples under variational distribution.
        
        Parameters
        ----------
        samples : np.ndarray, shape (n_samples, parameter_dim)
            Sample points
            
        Returns
        -------
        log_probs : np.ndarray, shape (n_samples,)
            Log probabilities
        """
        n_samples = samples.shape[0]
        log_probs = np.zeros(n_samples)
        
        if self.vi_type == "mean_field":
            means = self.variational_params['means']
            stds = np.exp(self.variational_params['log_stds'])
            
            for i in range(n_samples):
                log_probs[i] = np.sum(
                    -0.5 * np.log(2 * np.pi * stds**2) -
                    0.5 * (samples[i] - means)**2 / stds**2
                )
                
        elif self.vi_type == "full_rank":
            mean = self.variational_params['mean']
            chol = self.variational_params['chol_cov']
            
            # Compute covariance matrix
            cov = chol @ chol.T
            
            try:
                for i in range(n_samples):
                    log_probs[i] = multivariate_normal.logpdf(samples[i], mean, cov)
            except np.linalg.LinAlgError:
                # Fallback if covariance is singular
                log_probs.fill(-np.inf)
        
        return log_probs
    
    def compute_elbo(self, samples: np.ndarray) -> float:
        """
        Compute Evidence Lower BOund (ELBO).
        
        ELBO = E_q[log p(θ,y)] - E_q[log q(θ)]
             = E_q[log p(θ|y)] + log p(y) - E_q[log q(θ)]
             ≈ (1/S) Σ [log p(θ^s|y) - log q(θ^s)]
        
        Parameters
        ----------
        samples : np.ndarray, shape (n_samples, parameter_dim)
            Samples from variational distribution
            
        Returns
        -------
        elbo : float
            Evidence Lower BOund estimate
        """
        n_samples = samples.shape[0]
        
        # Compute log posterior for each sample
        log_posteriors = np.array([
            self.log_posterior_fn(sample) for sample in samples
        ])
        
        # Compute log variational probabilities
        log_var_probs = self.log_variational_prob(samples)
        
        # ELBO estimate
        elbo = np.mean(log_posteriors - log_var_probs)
        
        return elbo
    
    def compute_elbo_gradient(self, samples: np.ndarray, 
                             h: float = 1e-6) -> np.ndarray:
        """
        Compute gradient of ELBO with respect to variational parameters.
        
        Uses finite differences for gradient estimation.
        
        Parameters
        ----------
        samples : np.ndarray
            Samples from variational distribution  
        h : float, default=1e-6
            Finite difference step size
            
        Returns
        -------
        gradient : np.ndarray
            ELBO gradient
        """
        current_params = self._pack_parameters()
        n_params = len(current_params)
        gradient = np.zeros(n_params)
        
        # Current ELBO
        elbo_current = self.compute_elbo(samples)
        
        # Compute finite difference gradient
        for i in range(n_params):
            # Forward step
            params_plus = current_params.copy()
            params_plus[i] += h
            self._unpack_parameters(params_plus)
            
            # Resample with updated parameters (for consistency)
            samples_plus = self.sample_variational(len(samples))
            elbo_plus = self.compute_elbo(samples_plus)
            
            # Backward step  
            params_minus = current_params.copy()
            params_minus[i] -= h
            self._unpack_parameters(params_minus)
            
            samples_minus = self.sample_variational(len(samples))
            elbo_minus = self.compute_elbo(samples_minus)
            
            # Central difference
            gradient[i] = (elbo_plus - elbo_minus) / (2 * h)
        
        # Restore original parameters
        self._unpack_parameters(current_params)
        
        return gradient
    
    def compute_elbo_gradient_analytic(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute analytical gradient of ELBO (for mean-field case).
        
        Uses reparameterization trick and score function estimator.
        
        Parameters
        ----------
        samples : np.ndarray
            Samples from variational distribution
            
        Returns
        -------
        gradient : np.ndarray
            Analytical ELBO gradient
        """
        if self.vi_type != "mean_field":
            warnings.warn("Analytical gradient only implemented for mean-field VI")
            return self.compute_elbo_gradient(samples)
        
        means = self.variational_params['means']
        log_stds = self.variational_params['log_stds']
        stds = np.exp(log_stds)
        
        n_samples = samples.shape[0]
        
        # Initialize gradients
        grad_means = np.zeros(self.parameter_dim)
        grad_log_stds = np.zeros(self.parameter_dim)
        
        for sample in samples:
            # Compute log posterior
            log_post = self.log_posterior_fn(sample)
            
            if np.isfinite(log_post):
                # Gradient with respect to means (reparameterization trick)
                # ∇_μ ELBO = ∇_θ log p(θ|y) where θ = μ + σε
                try:
                    grad_log_post = self._compute_gradient_finite_diff(sample)
                    grad_means += grad_log_post
                except:
                    # Fallback to score function
                    grad_means += log_post * (sample - means) / stds**2
                
                # Gradient with respect to log(σ) (reparameterization trick)  
                # ∇_{log σ} ELBO = σ * ∇_θ log p(θ|y) * ε + 1
                # where ε = (θ - μ)/σ
                epsilon = (sample - means) / stds
                try:
                    grad_log_stds += stds * grad_log_post * epsilon + 1
                except:
                    # Fallback
                    grad_log_stds += log_post * epsilon + 1
        
        # Average over samples
        grad_means /= n_samples
        grad_log_stds /= n_samples
        
        return np.concatenate([grad_means, grad_log_stds])
    
    def _compute_gradient_finite_diff(self, x: np.ndarray, 
                                     h: float = 1e-8) -> np.ndarray:
        """Compute gradient of log posterior using finite differences."""
        gradient = np.zeros(len(x))
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            gradient[i] = (self.log_posterior_fn(x_plus) - 
                          self.log_posterior_fn(x_minus)) / (2 * h)
        
        return gradient
    
    def optimize(self, n_iterations: int = 5000,
                learning_rate: float = 0.01,
                n_samples: int = 100,
                optimizer: str = "adam",
                patience: int = 100,
                min_delta: float = 1e-6) -> Dict[str, Any]:
        """
        Optimize variational parameters to maximize ELBO.
        
        Parameters
        ----------
        n_iterations : int, default=5000
            Number of optimization iterations
        learning_rate : float, default=0.01
            Learning rate for gradient ascent
        n_samples : int, default=100
            Number of samples per iteration for ELBO estimation
        optimizer : str, default="adam"
            Optimization algorithm ("sgd", "adam", "adagrad")
        patience : int, default=100
            Early stopping patience
        min_delta : float, default=1e-6
            Minimum change for early stopping
            
        Returns
        -------
        result : Dict[str, Any]
            Optimization results
        """
        print(f"Starting variational inference optimization...")
        print(f"Method: {self.vi_type}, Optimizer: {optimizer}")
        
        # Initialize optimizer state
        if optimizer == "adam":
            m = np.zeros_like(self._pack_parameters())  # First moment
            v = np.zeros_like(self._pack_parameters())  # Second moment
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
        elif optimizer == "adagrad":
            G = np.zeros_like(self._pack_parameters())  # Accumulated squared gradients
            eps = 1e-8
        
        best_elbo = -np.inf
        patience_counter = 0
        
        # Progress bar
        pbar = tqdm(range(n_iterations), desc="VI Optimization")
        
        for iteration in pbar:
            # Sample from current variational distribution
            samples = self.sample_variational(n_samples)
            
            # Compute ELBO
            elbo = self.compute_elbo(samples)
            self.elbo_history.append(elbo)
            
            # Compute gradient
            if self.vi_type == "mean_field":
                gradient = self.compute_elbo_gradient_analytic(samples)
            else:
                gradient = self.compute_elbo_gradient(samples)
            
            gradient_norm = np.linalg.norm(gradient)
            self.gradient_norms.append(gradient_norm)
            
            # Update parameters
            current_params = self._pack_parameters()
            
            if optimizer == "sgd":
                # Stochastic gradient descent
                updated_params = current_params + learning_rate * gradient
                
            elif optimizer == "adam":
                # Adam optimizer
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * gradient**2
                
                # Bias correction
                m_corrected = m / (1 - beta1**(iteration + 1))
                v_corrected = v / (1 - beta2**(iteration + 1))
                
                updated_params = current_params + learning_rate * m_corrected / (np.sqrt(v_corrected) + eps)
                
            elif optimizer == "adagrad":
                # Adagrad optimizer
                G += gradient**2
                updated_params = current_params + learning_rate * gradient / (np.sqrt(G) + eps)
            
            self._unpack_parameters(updated_params)
            
            # Update progress bar
            pbar.set_postfix({
                'ELBO': f'{elbo:.4f}',
                'Grad': f'{gradient_norm:.2e}'
            })
            
            # Early stopping
            if elbo > best_elbo + min_delta:
                best_elbo = elbo
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at iteration {iteration}")
                break
            
            # Check for numerical issues
            if not np.isfinite(elbo) or gradient_norm > 1e6:
                warnings.warn("Numerical instability detected, stopping optimization")
                break
        
        pbar.close()
        
        final_elbo = self.elbo_history[-1] if self.elbo_history else -np.inf
        
        print(f"Optimization completed:")
        print(f"  Final ELBO: {final_elbo:.4f}")
        print(f"  Iterations: {len(self.elbo_history)}")
        print(f"  Final gradient norm: {gradient_norm:.2e}")
        
        return {
            'final_elbo': final_elbo,
            'elbo_history': np.array(self.elbo_history),
            'gradient_norms': np.array(self.gradient_norms),
            'variational_params': self.variational_params.copy(),
            'n_iterations': len(self.elbo_history)
        }
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from optimized variational distribution.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        samples : np.ndarray, shape (n_samples, parameter_dim)
            Samples from variational posterior approximation
        """
        return self.sample_variational(n_samples)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of variational distribution."""
        if self.vi_type == "mean_field":
            means = self.variational_params['means']
            stds = np.exp(self.variational_params['log_stds'])
            
            return {
                'means': means,
                'stds': stds,
                'variances': stds**2,
                'covariance_matrix': np.diag(stds**2)
            }
            
        elif self.vi_type == "full_rank":
            mean = self.variational_params['mean']
            chol = self.variational_params['chol_cov']
            cov = chol @ chol.T
            
            return {
                'mean': mean,
                'covariance_matrix': cov,
                'stds': np.sqrt(np.diag(cov)),
                'variances': np.diag(cov)
            }
    
    def kl_divergence_to_prior(self, prior_mean: np.ndarray, 
                              prior_cov: np.ndarray) -> float:
        """
        Compute KL divergence from variational distribution to prior.
        
        Parameters
        ----------
        prior_mean : np.ndarray
            Prior mean
        prior_cov : np.ndarray  
            Prior covariance matrix
            
        Returns
        -------
        kl_div : float
            KL divergence KL(q||p)
        """
        if self.vi_type == "mean_field":
            # KL for diagonal covariance
            q_mean = self.variational_params['means']
            q_var = np.exp(2 * self.variational_params['log_stds'])
            
            # Assume diagonal prior covariance
            prior_var = np.diag(prior_cov)
            
            kl = 0.5 * np.sum(
                np.log(prior_var / q_var) +
                q_var / prior_var +
                (q_mean - prior_mean)**2 / prior_var - 1
            )
            
        elif self.vi_type == "full_rank":
            # KL for full covariance matrices
            q_mean = self.variational_params['mean']
            q_chol = self.variational_params['chol_cov']
            q_cov = q_chol @ q_chol.T
            
            try:
                k = len(q_mean)
                
                # Compute terms
                inv_prior_cov = np.linalg.inv(prior_cov)
                trace_term = np.trace(inv_prior_cov @ q_cov)
                
                quad_term = (q_mean - prior_mean).T @ inv_prior_cov @ (q_mean - prior_mean)
                
                _, logdet_prior = np.linalg.slogdet(prior_cov)
                _, logdet_q = np.linalg.slogdet(q_cov)
                
                kl = 0.5 * (trace_term + quad_term - k + logdet_prior - logdet_q)
                
            except np.linalg.LinAlgError:
                kl = np.inf
        
        return kl


class MeanFieldVI(VariationalInference):
    """
    Mean-field variational inference with Gaussian factors.
    
    Approximates posterior as q(θ) = ∏_i N(θ_i; μ_i, σ_i²)
    """
    
    def __init__(self, log_posterior_fn: Callable[[np.ndarray], float],
                 parameter_dim: int):
        """Initialize mean-field VI."""
        super().__init__(log_posterior_fn, parameter_dim, vi_type="mean_field")
    
    def get_marginal_distributions(self) -> List[Dict[str, float]]:
        """Get parameters of marginal distributions."""
        means = self.variational_params['means']
        stds = np.exp(self.variational_params['log_stds'])
        
        marginals = []
        for i in range(self.parameter_dim):
            marginals.append({
                'mean': means[i],
                'std': stds[i],
                'variance': stds[i]**2
            })
        
        return marginals