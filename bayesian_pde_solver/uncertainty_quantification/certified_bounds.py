"""
Certified Uncertainty Bounds

Implements rigorous uncertainty quantification methods including
concentration inequalities and PAC-Bayes bounds.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Dict, Any, Tuple, List, Optional, Callable
import warnings


class CertifiedBounds:
    """
    Base class for certified uncertainty bounds.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize certified bounds.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compute_bounds(self, data: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Compute certified bounds.
        
        Args:
            data: Data samples
            **kwargs: Additional parameters
            
        Returns:
            lower_bound, upper_bound: Certified bounds
        """
        raise NotImplementedError("Subclasses must implement compute_bounds")
    
    def validate_coverage(self, true_values: np.ndarray, 
                         bounds: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Validate empirical coverage of bounds.
        
        Args:
            true_values: True parameter values
            bounds: List of (lower, upper) bound pairs
            
        Returns:
            coverage_stats: Coverage statistics
        """
        n_total = len(true_values)
        n_covered = 0
        
        for i, (lower, upper) in enumerate(bounds):
            if lower <= true_values[i] <= upper:
                n_covered += 1
        
        empirical_coverage = n_covered / n_total
        
        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": self.confidence_level,
            "coverage_error": empirical_coverage - self.confidence_level,
            "n_covered": n_covered,
            "n_total": n_total
        }


class ConcentrationBounds(CertifiedBounds):
    """
    Concentration inequality based bounds (Hoeffding, Bernstein, etc.).
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 inequality_type: str = "hoeffding"):
        """
        Initialize concentration bounds.
        
        Args:
            confidence_level: Confidence level
            inequality_type: Type of concentration inequality
        """
        super().__init__(confidence_level)
        self.inequality_type = inequality_type
    
    def hoeffding_bounds(self, data: np.ndarray, 
                        data_range: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute Hoeffding concentration bounds.
        
        Args:
            data: Sample data
            data_range: Range of data values (a, b)
            
        Returns:
            lower_bound, upper_bound: Hoeffding bounds
        """
        n = len(data)
        mean = np.mean(data)
        a, b = data_range
        
        # Hoeffding bound: P(|X_bar - E[X]| >= t) <= 2 exp(-2n t^2 / (b-a)^2)
        # For confidence level 1-alpha: t = (b-a) * sqrt(-log(alpha/2) / (2n))
        t = (b - a) * np.sqrt(-np.log(self.alpha / 2) / (2 * n))
        
        return mean - t, mean + t
    
    def bernstein_bounds(self, data: np.ndarray, 
                        variance_bound: float) -> Tuple[float, float]:
        """
        Compute Bernstein concentration bounds.
        
        Args:
            data: Sample data
            variance_bound: Upper bound on variance
            
        Returns:
            lower_bound, upper_bound: Bernstein bounds
        """
        n = len(data)
        mean = np.mean(data)
        
        # Bernstein bound is more complex, using empirical variance
        emp_var = np.var(data, ddof=1)
        sigma2 = min(emp_var, variance_bound)
        
        # Simplified Bernstein bound
        t = np.sqrt(2 * sigma2 * np.log(2 / self.alpha) / n) + \
            (2 * np.log(2 / self.alpha)) / (3 * n)
        
        return mean - t, mean + t
    
    def mcdiarmid_bounds(self, data: np.ndarray, 
                        lipschitz_constant: float) -> Tuple[float, float]:
        """
        Compute McDiarmid concentration bounds for functions with bounded differences.
        
        Args:
            data: Sample data
            lipschitz_constant: Lipschitz constant of the function
            
        Returns:
            lower_bound, upper_bound: McDiarmid bounds
        """
        n = len(data)
        mean = np.mean(data)
        
        # McDiarmid bound: P(|f(X) - E[f(X)]| >= t) <= 2 exp(-2t^2 / sum(c_i^2))
        # where c_i is the bounded difference constant
        c = lipschitz_constant
        t = c * np.sqrt(-np.log(self.alpha / 2) / (2 * n))
        
        return mean - t, mean + t
    
    def compute_bounds(self, data: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Compute concentration bounds based on specified inequality type.
        
        Args:
            data: Sample data
            **kwargs: Additional parameters for specific bounds
            
        Returns:
            lower_bound, upper_bound: Concentration bounds
        """
        if self.inequality_type == "hoeffding":
            data_range = kwargs.get("data_range", (np.min(data), np.max(data)))
            return self.hoeffding_bounds(data, data_range)
        
        elif self.inequality_type == "bernstein":
            variance_bound = kwargs.get("variance_bound", np.var(data) * 2)
            return self.bernstein_bounds(data, variance_bound)
        
        elif self.inequality_type == "mcdiarmid":
            lipschitz_constant = kwargs.get("lipschitz_constant", 1.0)
            return self.mcdiarmid_bounds(data, lipschitz_constant)
        
        else:
            raise ValueError(f"Unknown inequality type: {self.inequality_type}")


class PACBayesBounds(CertifiedBounds):
    """
    PAC-Bayes bounds for Bayesian learning with finite sample guarantees.
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 bound_type: str = "mcallester"):
        """
        Initialize PAC-Bayes bounds.
        
        Args:
            confidence_level: Confidence level
            bound_type: Type of PAC-Bayes bound
        """
        super().__init__(confidence_level)
        self.bound_type = bound_type
    
    def mcallester_bound(self, empirical_risk: float, 
                        kl_divergence: float, 
                        n_samples: int) -> float:
        """
        Compute McAllester PAC-Bayes bound.
        
        Args:
            empirical_risk: Empirical risk on training data
            kl_divergence: KL divergence between posterior and prior
            n_samples: Number of training samples
            
        Returns:
            risk_bound: Upper bound on true risk
        """
        # McAllester bound: R(ρ) <= R_emp(ρ) + sqrt((KL(ρ||π) + ln(2√n/δ)) / (2n))
        term = (kl_divergence + np.log(2 * np.sqrt(n_samples) / self.alpha)) / (2 * n_samples)
        return empirical_risk + np.sqrt(term)
    
    def seeger_bound(self, empirical_risk: float, 
                    kl_divergence: float, 
                    n_samples: int) -> float:
        """
        Compute Seeger PAC-Bayes bound (tighter than McAllester).
        
        Args:
            empirical_risk: Empirical risk on training data
            kl_divergence: KL divergence between posterior and prior
            n_samples: Number of training samples
            
        Returns:
            risk_bound: Upper bound on true risk
        """
        # Seeger bound: solve for ρ such that empirical_risk + φ(ρ) = ρ
        # where φ(ρ) = (KL + ln(2√n/δ)) / (2n(1-ρ))
        
        def seeger_equation(rho):
            if rho >= 1:
                return np.inf
            phi = (kl_divergence + np.log(2 * np.sqrt(n_samples) / self.alpha)) / (2 * n_samples * (1 - rho))
            return empirical_risk + phi - rho
        
        # Solve for rho
        try:
            result = minimize_scalar(lambda x: abs(seeger_equation(x)), 
                                   bounds=(empirical_risk, 0.99), 
                                   method='bounded')
            return result.x if result.success else self.mcallester_bound(empirical_risk, kl_divergence, n_samples)
        except:
            return self.mcallester_bound(empirical_risk, kl_divergence, n_samples)
    
    def catoni_bound(self, losses: np.ndarray, 
                    kl_divergence: float, 
                    lambda_param: float = 1.0) -> float:
        """
        Compute Catoni PAC-Bayes bound for unbounded losses.
        
        Args:
            losses: Array of loss values
            kl_divergence: KL divergence between posterior and prior
            lambda_param: Temperature parameter
            
        Returns:
            risk_bound: Upper bound on true risk
        """
        n = len(losses)
        
        # Catoni bound uses exponential moment bounds
        # E[exp(λ(L - E[L]))] <= exp(λ²σ²/2) for sub-Gaussian losses
        empirical_risk = np.mean(losses)
        
        # Use robust estimator for variance
        median_loss = np.median(losses)
        mad = np.median(np.abs(losses - median_loss))
        robust_std = 1.4826 * mad  # Convert MAD to std estimate
        
        # Catoni bound
        term = (kl_divergence + np.log(1 / self.alpha)) / (lambda_param * n)
        variance_term = lambda_param * robust_std**2 / 2
        
        return empirical_risk + term + variance_term
    
    def compute_pac_bayes_bound(self, posterior_samples: np.ndarray,
                               prior_samples: np.ndarray,
                               loss_function: Callable,
                               training_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute PAC-Bayes bounds for given posterior and prior samples.
        
        Args:
            posterior_samples: Samples from posterior distribution
            prior_samples: Samples from prior distribution
            loss_function: Loss function L(θ, data)
            training_data: Training data
            
        Returns:
            bounds: Dictionary of different PAC-Bayes bounds
        """
        n_samples = len(training_data.get("observations", []))
        
        # Compute empirical risk for posterior samples
        posterior_losses = []
        for sample in posterior_samples:
            loss = loss_function(sample, training_data)
            posterior_losses.append(loss)
        
        empirical_risk = np.mean(posterior_losses)
        
        # Estimate KL divergence between posterior and prior
        kl_div = self._estimate_kl_divergence(posterior_samples, prior_samples)
        
        bounds = {}
        
        if self.bound_type == "mcallester" or self.bound_type == "all":
            bounds["mcallester"] = self.mcallester_bound(empirical_risk, kl_div, n_samples)
        
        if self.bound_type == "seeger" or self.bound_type == "all":
            bounds["seeger"] = self.seeger_bound(empirical_risk, kl_div, n_samples)
        
        if self.bound_type == "catoni" or self.bound_type == "all":
            bounds["catoni"] = self.catoni_bound(np.array(posterior_losses), kl_div)
        
        bounds.update({
            "empirical_risk": empirical_risk,
            "kl_divergence": kl_div,
            "n_samples": n_samples
        })
        
        return bounds
    
    def _estimate_kl_divergence(self, posterior_samples: np.ndarray, 
                               prior_samples: np.ndarray) -> float:
        """
        Estimate KL divergence between posterior and prior using samples.
        
        Args:
            posterior_samples: Samples from posterior
            prior_samples: Samples from prior
            
        Returns:
            kl_estimate: Estimated KL divergence
        """
        # Use k-nearest neighbor estimation or kernel density estimation
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Simple k-NN based KL estimation
            k = min(10, len(posterior_samples) // 10)
            
            # Fit k-NN on prior samples
            nbrs_prior = NearestNeighbors(n_neighbors=k).fit(prior_samples)
            nbrs_posterior = NearestNeighbors(n_neighbors=k).fit(posterior_samples)
            
            # Compute distances
            dist_prior, _ = nbrs_prior.kneighbors(posterior_samples)
            dist_posterior, _ = nbrs_posterior.kneighbors(posterior_samples)
            
            # KL estimate (simplified)
            log_ratio = np.log(dist_prior[:, -1] + 1e-10) - np.log(dist_posterior[:, -1] + 1e-10)
            kl_estimate = np.mean(log_ratio) + np.log(len(prior_samples) / len(posterior_samples))
            
            return max(0, kl_estimate)  # KL divergence is non-negative
            
        except ImportError:
            warnings.warn("sklearn not available, using simple KL estimate")
            # Fallback: assume Gaussian distributions
            post_mean, post_cov = np.mean(posterior_samples, axis=0), np.cov(posterior_samples.T)
            prior_mean, prior_cov = np.mean(prior_samples, axis=0), np.cov(prior_samples.T)
            
            # KL divergence between multivariate Gaussians
            d = len(post_mean)
            try:
                inv_prior_cov = np.linalg.inv(prior_cov + 1e-6 * np.eye(d))
                trace_term = np.trace(inv_prior_cov @ post_cov)
                quad_term = (prior_mean - post_mean).T @ inv_prior_cov @ (prior_mean - post_mean)
                log_det_term = np.log(np.linalg.det(prior_cov) / np.linalg.det(post_cov))
                
                kl = 0.5 * (trace_term + quad_term - d + log_det_term)
                return max(0, kl)
            except:
                return 1.0  # Conservative estimate
    
    def compute_bounds(self, data: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Compute PAC-Bayes bounds for parameter estimates.
        
        Args:
            data: Posterior samples
            **kwargs: Additional parameters
            
        Returns:
            lower_bound, upper_bound: PAC-Bayes bounds
        """
        # This is a simplified interface - in practice, PAC-Bayes bounds
        # are more complex and require additional information
        posterior_samples = data
        prior_samples = kwargs.get("prior_samples", np.random.randn(*data.shape))
        
        # Estimate bounds based on sample statistics
        post_mean = np.mean(posterior_samples, axis=0)
        kl_div = self._estimate_kl_divergence(posterior_samples, prior_samples)
        n_samples = kwargs.get("n_training_samples", 1000)
        
        # Use McAllester bound as default
        empirical_risk = 0.1  # Placeholder
        bound_width = np.sqrt((kl_div + np.log(2 * np.sqrt(n_samples) / self.alpha)) / (2 * n_samples))
        
        if post_mean.ndim == 0:
            return post_mean - bound_width, post_mean + bound_width
        else:
            # For multivariate case, return bounds for first component
            return post_mean[0] - bound_width, post_mean[0] + bound_width