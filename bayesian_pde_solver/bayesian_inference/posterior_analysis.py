"""
Posterior Analysis

Tools for analyzing MCMC samples including convergence diagnostics,
parameter statistics, and posterior summaries.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
from scipy import stats
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


class PosteriorAnalysis:
    """
    Comprehensive posterior analysis for MCMC samples.
    
    Provides parameter statistics, convergence diagnostics, and
    credible intervals for Bayesian parameter estimation.
    
    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_params)
        MCMC samples from posterior distribution
    parameter_names : List[str]
        Names of parameters
    confidence_levels : List[float], default=[0.68, 0.95]
        Confidence levels for credible intervals
    compute_correlations : bool, default=True
        Whether to compute parameter correlations
        
    Examples
    --------
    >>> analysis = PosteriorAnalysis(samples, ['param1', 'param2'])
    >>> print(analysis.parameter_means)
    {'param1': 1.23, 'param2': 0.87}
    >>> print(analysis.credible_intervals[0.95])
    {'param1': (0.98, 1.48), 'param2': (0.62, 1.12)}
    """
    
    def __init__(self, samples: np.ndarray,
                 parameter_names: List[str],
                 confidence_levels: List[float] = [0.68, 0.95],
                 compute_correlations: bool = True):
        """Initialize posterior analysis."""
        if samples.ndim != 2:
            raise ValueError("Samples must be 2D array (n_samples, n_params)")
        
        if len(parameter_names) != samples.shape[1]:
            raise ValueError("Number of parameter names must match sample dimension")
        
        self.samples = samples
        self.parameter_names = parameter_names
        self.confidence_levels = confidence_levels
        self.n_samples, self.n_params = samples.shape
        
        # Compute basic statistics
        self._compute_basic_statistics()
        
        # Compute credible intervals
        self._compute_credible_intervals()
        
        # Compute correlations if requested
        if compute_correlations:
            self._compute_correlations()
        
        # Compute convergence diagnostics
        self._compute_convergence_diagnostics()
    
    def _compute_basic_statistics(self) -> None:
        """Compute basic parameter statistics."""
        self.parameter_means = {}
        self.parameter_medians = {}
        self.parameter_stds = {}
        self.parameter_vars = {}
        self.parameter_modes = {}
        
        for i, name in enumerate(self.parameter_names):
            param_samples = self.samples[:, i]
            
            self.parameter_means[name] = np.mean(param_samples)
            self.parameter_medians[name] = np.median(param_samples)
            self.parameter_stds[name] = np.std(param_samples, ddof=1)
            self.parameter_vars[name] = np.var(param_samples, ddof=1)
            
            # Estimate mode using kernel density estimation
            try:
                kde = stats.gaussian_kde(param_samples)
                x_grid = np.linspace(np.min(param_samples), np.max(param_samples), 1000)
                density = kde(x_grid)
                mode_idx = np.argmax(density)
                self.parameter_modes[name] = x_grid[mode_idx]
            except:
                # Fallback to median if KDE fails
                self.parameter_modes[name] = self.parameter_medians[name]
    
    def _compute_credible_intervals(self) -> None:
        """Compute credible intervals for all confidence levels."""
        self.credible_intervals = {}
        self.highest_density_intervals = {}
        
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            self.credible_intervals[conf_level] = {}
            self.highest_density_intervals[conf_level] = {}
            
            for i, name in enumerate(self.parameter_names):
                param_samples = self.samples[:, i]
                
                # Equal-tailed credible interval
                lower = np.quantile(param_samples, lower_quantile)
                upper = np.quantile(param_samples, upper_quantile)
                self.credible_intervals[conf_level][name] = (lower, upper)
                
                # Highest density interval (HDI)
                hdi = self._compute_hdi(param_samples, conf_level)
                self.highest_density_intervals[conf_level][name] = hdi
    
    def _compute_hdi(self, samples: np.ndarray, conf_level: float) -> Tuple[float, float]:
        """
        Compute highest density interval.
        
        Parameters
        ----------
        samples : np.ndarray
            Parameter samples
        conf_level : float
            Confidence level
            
        Returns
        -------
        hdi : Tuple[float, float]
            Highest density interval (lower, upper)
        """
        # Sort samples
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        
        # Number of samples in interval
        interval_size = int(np.ceil(conf_level * n))
        
        # Find interval with minimum width
        min_width = np.inf
        best_lower = best_upper = 0
        
        for i in range(n - interval_size + 1):
            lower = sorted_samples[i]
            upper = sorted_samples[i + interval_size - 1]
            width = upper - lower
            
            if width < min_width:
                min_width = width
                best_lower = lower
                best_upper = upper
        
        return best_lower, best_upper
    
    def _compute_correlations(self) -> None:
        """Compute parameter correlations and covariance matrix."""
        self.correlation_matrix = np.corrcoef(self.samples.T)
        self.covariance_matrix = np.cov(self.samples.T)
        
        # Parameter-wise correlations
        self.parameter_correlations = {}
        for i, name1 in enumerate(self.parameter_names):
            self.parameter_correlations[name1] = {}
            for j, name2 in enumerate(self.parameter_names):
                self.parameter_correlations[name1][name2] = self.correlation_matrix[i, j]
    
    def _compute_convergence_diagnostics(self) -> None:
        """Compute MCMC convergence diagnostics."""
        # Effective sample size
        self.effective_sample_sizes = {}
        self.autocorrelation_functions = {}
        
        for i, name in enumerate(self.parameter_names):
            param_samples = self.samples[:, i]
            
            # Compute autocorrelation function
            autocorr = self._compute_autocorrelation(param_samples)
            self.autocorrelation_functions[name] = autocorr
            
            # Estimate effective sample size
            eff_size = self._estimate_effective_sample_size(autocorr)
            self.effective_sample_sizes[name] = eff_size
        
        # Monte Carlo standard errors
        self.mc_standard_errors = {}
        for name in self.parameter_names:
            eff_size = self.effective_sample_sizes[name]
            std = self.parameter_stds[name]
            self.mc_standard_errors[name] = std / np.sqrt(eff_size)
    
    def _compute_autocorrelation(self, samples: np.ndarray, max_lag: int = None) -> np.ndarray:
        """
        Compute autocorrelation function using FFT.
        
        Parameters
        ----------
        samples : np.ndarray
            Parameter samples
        max_lag : int, optional
            Maximum lag to compute (default: n_samples // 4)
            
        Returns
        -------
        autocorr : np.ndarray
            Autocorrelation function
        """
        n = len(samples)
        if max_lag is None:
            max_lag = n // 4
        
        # Center the data
        centered = samples - np.mean(samples)
        
        # Pad with zeros for FFT
        padded = np.zeros(2 * n)
        padded[:n] = centered
        
        # Compute autocorrelation via FFT
        f_samples = fft(padded)
        autocorr_fft = ifft(f_samples * np.conj(f_samples)).real
        
        # Normalize
        autocorr = autocorr_fft[:max_lag + 1]
        autocorr = autocorr / autocorr[0]
        
        return autocorr
    
    def _estimate_effective_sample_size(self, autocorr: np.ndarray) -> float:
        """
        Estimate effective sample size from autocorrelation function.
        
        Parameters
        ----------
        autocorr : np.ndarray
            Autocorrelation function
            
        Returns
        -------
        eff_size : float
            Effective sample size
        """
        # Find first negative autocorrelation
        first_negative = None
        for i, ac in enumerate(autocorr[1:], 1):
            if ac <= 0:
                first_negative = i
                break
        
        if first_negative is None:
            # If no negative found, use all lags
            integrated_time = 1 + 2 * np.sum(autocorr[1:])
        else:
            # Sum until first negative
            integrated_time = 1 + 2 * np.sum(autocorr[1:first_negative])
        
        # Effective sample size
        eff_size = self.n_samples / (2 * integrated_time)
        
        return max(1, eff_size)
    
    def compute_gelman_rubin_statistic(self, chains: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute Gelman-Rubin R-hat statistic for multiple chains.
        
        Parameters
        ----------
        chains : List[np.ndarray]
            List of MCMC chains, each with shape (n_samples, n_params)
            
        Returns
        -------
        r_hat : Dict[str, float]
            R-hat statistics for each parameter
        """
        if len(chains) < 2:
            warnings.warn("Need at least 2 chains for R-hat computation")
            return {name: np.nan for name in self.parameter_names}
        
        n_chains = len(chains)
        n_samples = chains[0].shape[0]
        
        # Check all chains have same shape
        for chain in chains[1:]:
            if chain.shape != chains[0].shape:
                raise ValueError("All chains must have same shape")
        
        r_hat = {}
        
        for i, name in enumerate(self.parameter_names):
            # Extract parameter samples from all chains
            chain_samples = [chain[:, i] for chain in chains]
            
            # Chain means
            chain_means = [np.mean(chain) for chain in chain_samples]
            overall_mean = np.mean(chain_means)
            
            # Within-chain variance
            within_var = np.mean([np.var(chain, ddof=1) for chain in chain_samples])
            
            # Between-chain variance
            between_var = n_samples * np.var(chain_means, ddof=1)
            
            # Pooled variance estimate
            pooled_var = ((n_samples - 1) * within_var + between_var) / n_samples
            
            # R-hat statistic
            if within_var > 0:
                r_hat[name] = np.sqrt(pooled_var / within_var)
            else:
                r_hat[name] = np.nan
        
        return r_hat
    
    def summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive summary statistics.
        
        Returns
        -------
        summary : Dict[str, Dict[str, float]]
            Summary statistics for each parameter
        """
        summary = {}
        
        for name in self.parameter_names:
            summary[name] = {
                'mean': self.parameter_means[name],
                'median': self.parameter_medians[name],
                'mode': self.parameter_modes[name],
                'std': self.parameter_stds[name],
                'var': self.parameter_vars[name],
                'mc_se': self.mc_standard_errors[name],
                'eff_size': self.effective_sample_sizes[name],
                'eff_size_ratio': self.effective_sample_sizes[name] / self.n_samples
            }
            
            # Add credible intervals
            for conf_level in self.confidence_levels:
                ci = self.credible_intervals[conf_level][name]
                hdi = self.highest_density_intervals[conf_level][name]
                summary[name][f'ci_{int(conf_level*100)}'] = ci
                summary[name][f'hdi_{int(conf_level*100)}'] = hdi
        
        return summary
    
    def parameter_comparison(self, param1: str, param2: str) -> Dict[str, float]:
        """
        Compare two parameters.
        
        Parameters
        ----------
        param1, param2 : str
            Parameter names to compare
            
        Returns
        -------
        comparison : Dict[str, float]
            Comparison statistics
        """
        if param1 not in self.parameter_names or param2 not in self.parameter_names:
            raise ValueError("Parameter names must be in parameter_names list")
        
        idx1 = self.parameter_names.index(param1)
        idx2 = self.parameter_names.index(param2)
        
        samples1 = self.samples[:, idx1]
        samples2 = self.samples[:, idx2]
        
        # Compute comparison statistics
        diff_samples = samples1 - samples2
        ratio_samples = samples1 / (samples2 + 1e-10)  # Avoid division by zero
        
        return {
            'correlation': self.correlation_matrix[idx1, idx2],
            'mean_diff': np.mean(diff_samples),
            'std_diff': np.std(diff_samples, ddof=1),
            'prob_param1_greater': np.mean(samples1 > samples2),
            'mean_ratio': np.mean(ratio_samples),
            'median_ratio': np.median(ratio_samples)
        }
    
    def convergence_summary(self) -> Dict[str, Any]:
        """
        Get convergence diagnostic summary.
        
        Returns
        -------
        convergence : Dict[str, Any]
            Convergence diagnostic summary
        """
        min_eff_size = min(self.effective_sample_sizes.values())
        mean_eff_size = np.mean(list(self.effective_sample_sizes.values()))
        
        return {
            'n_samples': self.n_samples,
            'n_parameters': self.n_params,
            'min_effective_size': min_eff_size,
            'mean_effective_size': mean_eff_size,
            'min_efficiency': min_eff_size / self.n_samples,
            'mean_efficiency': mean_eff_size / self.n_samples,
            'effective_sample_sizes': self.effective_sample_sizes,
            'mc_standard_errors': self.mc_standard_errors
        }
    
    def print_summary(self, precision: int = 4) -> None:
        """
        Print formatted summary of posterior analysis.
        
        Parameters
        ----------
        precision : int, default=4
            Number of decimal places for output
        """
        print("="*60)
        print("POSTERIOR ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of parameters: {self.n_params}")
        print()
        
        # Parameter statistics
        print("PARAMETER STATISTICS")
        print("-"*60)
        
        header = f"{'Parameter':<12} {'Mean':<8} {'Median':<8} {'Std':<8} {'MC SE':<8} {'ESS':<6}"
        print(header)
        print("-"*60)
        
        for name in self.parameter_names:
            mean_val = self.parameter_means[name]
            median_val = self.parameter_medians[name]
            std_val = self.parameter_stds[name]
            mc_se = self.mc_standard_errors[name]
            ess = int(self.effective_sample_sizes[name])
            
            print(f"{name:<12} {mean_val:<8.{precision}f} {median_val:<8.{precision}f} "
                  f"{std_val:<8.{precision}f} {mc_se:<8.{precision}f} {ess:<6}")
        
        print()
        
        # Credible intervals
        for conf_level in self.confidence_levels:
            print(f"{int(conf_level*100)}% CREDIBLE INTERVALS")
            print("-"*40)
            
            for name in self.parameter_names:
                ci = self.credible_intervals[conf_level][name]
                hdi = self.highest_density_intervals[conf_level][name]
                print(f"{name}: CI = ({ci[0]:.{precision}f}, {ci[1]:.{precision}f}), "
                      f"HDI = ({hdi[0]:.{precision}f}, {hdi[1]:.{precision}f})")
            print()
        
        # Convergence summary
        conv_summary = self.convergence_summary()
        print("CONVERGENCE DIAGNOSTICS")
        print("-"*40)
        print(f"Minimum effective sample size: {conv_summary['min_effective_size']:.0f}")
        print(f"Mean sampling efficiency: {conv_summary['mean_efficiency']:.3f}")
        
        if conv_summary['min_efficiency'] < 0.1:
            print("WARNING: Low sampling efficiency detected!")
        if conv_summary['min_effective_size'] < 100:
            print("WARNING: Low effective sample size detected!")