"""
Mathematical Utilities

Helper functions for numerical computations, special functions,
and mathematical operations commonly used in PDE and Bayesian methods.
"""

import numpy as np
from typing import Union, Callable, Tuple, Optional
import warnings
from scipy import special


def safe_log(x: Union[float, np.ndarray], 
            min_val: float = 1e-12) -> Union[float, np.ndarray]:
    """
    Compute logarithm with numerical safety.
    
    Parameters
    ----------
    x : Union[float, np.ndarray]
        Input values
    min_val : float, default=1e-12
        Minimum value to prevent log(0)
        
    Returns
    -------
    result : Union[float, np.ndarray]
        Safe logarithm values
    """
    x_safe = np.maximum(x, min_val)
    return np.log(x_safe)


def safe_exp(x: Union[float, np.ndarray],
            max_val: float = 700.0) -> Union[float, np.ndarray]:
    """
    Compute exponential with overflow protection.
    
    Parameters
    ----------
    x : Union[float, np.ndarray]
        Input values
    max_val : float, default=700.0
        Maximum value to prevent overflow
        
    Returns
    -------
    result : Union[float, np.ndarray]
        Safe exponential values
    """
    x_safe = np.minimum(x, max_val)
    return np.exp(x_safe)


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Numerically stable computation of log(sum(exp(x))).
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : Optional[int], default=None
        Axis along which to compute
        
    Returns
    -------
    result : Union[float, np.ndarray]
        Log-sum-exp value
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    if np.any(~np.isfinite(x_max)):
        x_max = np.where(np.isfinite(x_max), x_max, 0)
    
    exp_sum = np.sum(np.exp(x - x_max), axis=axis, keepdims=True)
    result = x_max + np.log(exp_sum)
    
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    
    return result


def safe_divide(a: Union[float, np.ndarray], 
               b: Union[float, np.ndarray],
               default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safe division with handling of division by zero.
    
    Parameters
    ----------
    a, b : Union[float, np.ndarray]
        Numerator and denominator
    default : float, default=0.0
        Value to return when b=0
        
    Returns
    -------
    result : Union[float, np.ndarray]
        Division result
    """
    return np.where(np.abs(b) > 1e-15, a / b, default)


def compute_gradient_finite_diff(f: Callable[[np.ndarray], float],
                                x: np.ndarray,
                                h: float = 1e-8,
                                method: str = "central") -> np.ndarray:
    """
    Compute gradient using finite differences.
    
    Parameters
    ----------
    f : Callable[[np.ndarray], float]
        Function to differentiate
    x : np.ndarray
        Point at which to compute gradient
    h : float, default=1e-8
        Step size
    method : str, default="central"
        Finite difference method ("forward", "backward", "central")
        
    Returns
    -------
    grad : np.ndarray
        Gradient vector
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        if method == "forward":
            x_plus = x.copy()
            x_plus[i] += h
            grad[i] = (f(x_plus) - f(x)) / h
            
        elif method == "backward":
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x) - f(x_minus)) / h
            
        elif method == "central":
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return grad


def integrate_trapz(y: np.ndarray, 
                   x: Optional[np.ndarray] = None,
                   dx: float = 1.0) -> float:
    """
    Trapezoidal rule integration.
    
    Parameters
    ----------
    y : np.ndarray
        Function values
    x : Optional[np.ndarray], default=None
        x coordinates (uniform spacing if None)
    dx : float, default=1.0
        Spacing if x is None
        
    Returns
    -------
    integral : float
        Integral value
    """
    if x is None:
        return np.trapz(y, dx=dx)
    else:
        return np.trapz(y, x=x)


def compute_autocorrelation(x: np.ndarray, 
                          max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation function.
    
    Parameters
    ----------
    x : np.ndarray
        Time series data
    max_lag : Optional[int], default=None
        Maximum lag (default: len(x)//4)
        
    Returns
    -------
    autocorr : np.ndarray
        Autocorrelation function
    """
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    
    x_centered = x - np.mean(x)
    
    # Use numpy correlate for efficiency
    autocorr_full = np.correlate(x_centered, x_centered, mode='full')
    
    # Extract positive lags
    mid = len(autocorr_full) // 2
    autocorr = autocorr_full[mid:mid + max_lag + 1]
    
    # Normalize
    autocorr = autocorr / autocorr[0]
    
    return autocorr


def effective_sample_size(x: np.ndarray) -> float:
    """
    Estimate effective sample size from autocorrelation.
    
    Parameters
    ----------
    x : np.ndarray
        MCMC samples
        
    Returns
    -------
    ess : float
        Effective sample size
    """
    autocorr = compute_autocorrelation(x)
    
    # Find first negative value
    first_negative = None
    for i, ac in enumerate(autocorr[1:], 1):
        if ac <= 0:
            first_negative = i
            break
    
    if first_negative is None:
        # Use all lags if no negative found
        integrated_time = 1 + 2 * np.sum(autocorr[1:])
    else:
        # Sum until first negative
        integrated_time = 1 + 2 * np.sum(autocorr[1:first_negative])
    
    # Effective sample size
    ess = len(x) / (2 * integrated_time)
    return max(1, ess)


def gelman_rubin_statistic(chains: list) -> float:
    """
    Compute Gelman-Rubin R-hat convergence diagnostic.
    
    Parameters
    ----------
    chains : list
        List of MCMC chains (each a 1D array)
        
    Returns
    -------
    r_hat : float
        R-hat statistic
    """
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains")
    
    # Convert to array
    chains_array = np.array(chains)
    m, n = chains_array.shape  # m chains, n samples per chain
    
    # Chain means
    chain_means = np.mean(chains_array, axis=1)
    overall_mean = np.mean(chain_means)
    
    # Within-chain variance
    W = np.mean([np.var(chain, ddof=1) for chain in chains_array])
    
    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)
    
    # Pooled variance estimate
    var_plus = ((n - 1) * W + B) / n
    
    # R-hat statistic
    if W > 0:
        r_hat = np.sqrt(var_plus / W)
    else:
        r_hat = np.nan
    
    return r_hat


def numerical_hessian(f: Callable[[np.ndarray], float],
                     x: np.ndarray,
                     h: float = 1e-5) -> np.ndarray:
    """
    Compute Hessian matrix using finite differences.
    
    Parameters
    ----------
    f : Callable[[np.ndarray], float]
        Scalar function
    x : np.ndarray
        Point at which to compute Hessian
    h : float, default=1e-5
        Step size
        
    Returns
    -------
    hess : np.ndarray
        Hessian matrix
    """
    n = len(x)
    hess = np.zeros((n, n))
    
    # Compute diagonal elements
    for i in range(n):
        x_pp = x.copy()
        x_mm = x.copy()
        x_pp[i] += h
        x_mm[i] -= h
        
        hess[i, i] = (f(x_pp) - 2*f(x) + f(x_mm)) / h**2
    
    # Compute off-diagonal elements
    for i in range(n):
        for j in range(i+1, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += h
            x_pp[j] += h
            
            x_pm[i] += h
            x_pm[j] -= h
            
            x_mp[i] -= h
            x_mp[j] += h
            
            x_mm[i] -= h
            x_mm[j] -= h
            
            hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
            hess[j, i] = hess[i, j]  # Symmetry
    
    return hess


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax function with numerical stability.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int, default=-1
        Axis along which to compute softmax
        
    Returns
    -------
    result : np.ndarray
        Softmax values
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log-softmax function with numerical stability.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int, default=-1
        Axis along which to compute log-softmax
        
    Returns
    -------
    result : np.ndarray
        Log-softmax values
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp_val = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return x - log_sum_exp_val


def multivariate_normal_logpdf(x: np.ndarray,
                              mean: np.ndarray,
                              cov: np.ndarray) -> float:
    """
    Compute log PDF of multivariate normal distribution.
    
    Parameters
    ----------
    x : np.ndarray
        Input vector
    mean : np.ndarray
        Mean vector
    cov : np.ndarray
        Covariance matrix
        
    Returns
    -------
    logpdf : float
        Log probability density
    """
    k = len(mean)
    
    try:
        # Compute Cholesky decomposition
        L = np.linalg.cholesky(cov)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # Solve linear system
        diff = x - mean
        y = np.linalg.solve(L, diff)
        quad_form = np.dot(y, y)
        
    except np.linalg.LinAlgError:
        # Fallback to SVD if Cholesky fails
        try:
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                return -np.inf
            
            diff = x - mean
            quad_form = np.dot(diff, np.linalg.solve(cov, diff))
        except np.linalg.LinAlgError:
            return -np.inf
    
    # Log PDF
    logpdf = -0.5 * (k * np.log(2 * np.pi) + log_det + quad_form)
    
    return logpdf


def chi2_quantile(p: float, df: int) -> float:
    """
    Compute quantile of chi-squared distribution.
    
    Parameters
    ----------
    p : float
        Probability level
    df : int
        Degrees of freedom
        
    Returns
    -------
    quantile : float
        Chi-squared quantile
    """
    return special.chi2.ppf(p, df)


def normal_quantile(p: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """
    Compute quantile of normal distribution.
    
    Parameters
    ----------
    p : float
        Probability level
    loc : float, default=0.0
        Location parameter (mean)
    scale : float, default=1.0
        Scale parameter (std)
        
    Returns
    -------
    quantile : float
        Normal quantile
    """
    return special.norm.ppf(p, loc=loc, scale=scale)


def kl_divergence_gaussian(mu1: np.ndarray, sigma1: np.ndarray,
                          mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Compute KL divergence between two multivariate Gaussians.
    
    KL(p||q) where p ~ N(mu1, sigma1), q ~ N(mu2, sigma2)
    
    Parameters
    ----------
    mu1, mu2 : np.ndarray
        Mean vectors
    sigma1, sigma2 : np.ndarray
        Covariance matrices
        
    Returns
    -------
    kl_div : float
        KL divergence
    """
    k = len(mu1)
    
    try:
        # Compute determinants
        sign1, logdet1 = np.linalg.slogdet(sigma1)
        sign2, logdet2 = np.linalg.slogdet(sigma2)
        
        if sign1 <= 0 or sign2 <= 0:
            return np.inf
        
        # Compute trace and quadratic terms
        sigma2_inv = np.linalg.inv(sigma2)
        trace_term = np.trace(sigma2_inv @ sigma1)
        
        diff = mu2 - mu1
        quad_term = diff.T @ sigma2_inv @ diff
        
        # KL divergence formula
        kl_div = 0.5 * (trace_term + quad_term - k + logdet2 - logdet1)
        
        return kl_div
        
    except np.linalg.LinAlgError:
        return np.inf