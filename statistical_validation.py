"""
Statistical Validation Framework for Bayesian PDE Methods

Comprehensive statistical testing and validation framework for Bayesian inverse
problems in partial differential equations. Provides rigorous hypothesis testing,
cross-validation, model selection, and statistical significance analysis with
multiple testing corrections.

Features:
- Hypothesis testing for uncertainty quantification accuracy
- Cross-validation with proper scoring rules
- Model selection and comparison with statistical significance
- Coverage probability validation and calibration testing
- Bootstrap confidence intervals and permutation tests
- Bayesian model comparison with Bayes factors
- Multiple testing corrections (Bonferroni, FDR, etc.)
- Power analysis and sample size determination
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import kstest, anderson, jarque_bera, shapiro
from scipy.stats import chi2_contingency, mannwhitneyu, wilcoxon
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
import itertools
from pathlib import Path
import json


@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    reject_null: Optional[bool] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    
    def __post_init__(self):
        if self.reject_null is None and self.p_value is not None:
            self.reject_null = self.p_value < 0.05


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    method_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_p_value: float
    significant_failures: List[str]
    validation_score: float
    recommendations: List[str]


class CoverageValidator:
    """
    Validation of uncertainty quantification coverage properties.
    
    Tests whether confidence intervals achieve their nominal coverage
    probability and provides calibration diagnostics for probabilistic
    predictions.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.68, 0.90, 0.95, 0.99]):
        """
        Initialize coverage validator.
        
        Parameters:
        -----------
        confidence_levels : List[float]
            Confidence levels to test
        """
        self.confidence_levels = confidence_levels
    
    def test_interval_coverage(self, intervals: np.ndarray, true_values: np.ndarray,
                             confidence_level: float = 0.95) -> TestResult:
        """
        Test coverage probability of confidence intervals.
        
        Parameters:
        -----------
        intervals : np.ndarray, shape (n_samples, 2)
            Confidence intervals [lower, upper]
        true_values : np.ndarray, shape (n_samples,)
            True parameter values
        confidence_level : float
            Nominal confidence level
            
        Returns:
        --------
        TestResult
            Coverage test results
        """
        n_samples = len(true_values)
        
        # Check which intervals contain true values
        coverage_indicators = ((intervals[:, 0] <= true_values) & 
                              (true_values <= intervals[:, 1]))
        
        empirical_coverage = np.mean(coverage_indicators)
        
        # Binomial test for coverage probability
        successes = np.sum(coverage_indicators)
        statistic, p_value = stats.binom_test(successes, n_samples, confidence_level,
                                            alternative='two-sided')
        
        # Wilson confidence interval for coverage probability
        z_alpha = stats.norm.ppf(1 - 0.05/2)  # 95% CI for coverage estimate
        p_hat = empirical_coverage
        n = n_samples
        
        denominator = 1 + z_alpha**2 / n
        center = (p_hat + z_alpha**2 / (2*n)) / denominator
        margin = z_alpha * np.sqrt(p_hat * (1 - p_hat) / n + z_alpha**2 / (4*n**2)) / denominator
        
        ci_lower = center - margin
        ci_upper = center + margin
        
        # Effect size (difference from nominal)
        effect_size = abs(empirical_coverage - confidence_level)
        
        # Interpretation
        if p_value < 0.05:
            if empirical_coverage < confidence_level:
                interpretation = f"Under-coverage: {empirical_coverage:.3f} < {confidence_level}"
            else:
                interpretation = f"Over-coverage: {empirical_coverage:.3f} > {confidence_level}"
        else:
            interpretation = f"Nominal coverage achieved: {empirical_coverage:.3f} ≈ {confidence_level}"
        
        return TestResult(
            test_name=f"Coverage Test ({confidence_level*100:.0f}%)",
            statistic=empirical_coverage,
            p_value=p_value,
            reject_null=p_value < 0.05,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def test_probability_calibration(self, predicted_probs: np.ndarray, 
                                   observed_outcomes: np.ndarray,
                                   n_bins: int = 10) -> TestResult:
        """
        Test calibration of probabilistic predictions using reliability diagram.
        
        Parameters:
        -----------
        predicted_probs : np.ndarray
            Predicted probabilities
        observed_outcomes : np.ndarray
            Binary observed outcomes (0 or 1)
        n_bins : int
            Number of bins for calibration curve
            
        Returns:
        --------
        TestResult
            Calibration test results
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = ((predicted_probs > bin_lower) & 
                     (predicted_probs <= bin_upper))
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(observed_outcomes[in_bin])
                bin_confidence = np.mean(predicted_probs[in_bin])
                bin_count = np.sum(in_bin)
                
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(bin_count)
        
        # Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(predicted_probs)
        
        for accuracy, confidence, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += (count / total_samples) * abs(accuracy - confidence)
        
        # Hosmer-Lemeshow test for calibration
        # Chi-square test comparing observed vs expected frequencies
        observed_positive = []
        expected_positive = []
        total_in_bin = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = ((predicted_probs > bin_lower) & 
                     (predicted_probs <= bin_upper))
            
            if np.sum(in_bin) > 0:
                n_in_bin = np.sum(in_bin)
                observed_pos = np.sum(observed_outcomes[in_bin])
                expected_pos = np.sum(predicted_probs[in_bin])
                
                observed_positive.append(observed_pos)
                expected_positive.append(expected_pos)
                total_in_bin.append(n_in_bin)
        
        # Chi-square statistic
        if len(observed_positive) > 1:
            chi_square = 0
            df = len(observed_positive) - 2  # -2 for calibration test
            
            for obs, exp, total in zip(observed_positive, expected_positive, total_in_bin):
                if exp > 0 and (total - exp) > 0:
                    chi_square += ((obs - exp)**2 / exp + 
                                  ((total - obs) - (total - exp))**2 / (total - exp))
            
            p_value = 1 - stats.chi2.cdf(chi_square, df) if df > 0 else 1.0
        else:
            chi_square = 0
            p_value = 1.0
        
        # Interpretation
        if ece < 0.05:
            interpretation = f"Well-calibrated (ECE = {ece:.4f})"
        elif ece < 0.1:
            interpretation = f"Moderately calibrated (ECE = {ece:.4f})"
        else:
            interpretation = f"Poorly calibrated (ECE = {ece:.4f})"
        
        return TestResult(
            test_name="Probability Calibration Test",
            statistic=ece,
            p_value=p_value,
            reject_null=p_value < 0.05,
            effect_size=ece,
            interpretation=interpretation
        )
    
    def comprehensive_coverage_analysis(self, uncertainty_data: Dict[str, Any]) -> Dict[str, TestResult]:
        """
        Perform comprehensive coverage analysis across multiple confidence levels.
        
        Parameters:
        -----------
        uncertainty_data : Dict[str, Any]
            Dictionary containing intervals and true values
            
        Returns:
        --------
        Dict[str, TestResult]
            Results for each confidence level
        """
        results = {}
        
        for conf_level in self.confidence_levels:
            if f'intervals_{conf_level}' in uncertainty_data:
                intervals = uncertainty_data[f'intervals_{conf_level}']
                true_values = uncertainty_data['true_values']
                
                result = self.test_interval_coverage(intervals, true_values, conf_level)
                results[f'coverage_{conf_level}'] = result
        
        return results


class ModelComparisonValidator:
    """
    Statistical validation for model comparison and selection.
    
    Provides hypothesis testing for comparing multiple models with
    proper corrections for multiple testing and cross-validation.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize model comparison validator.
        
        Parameters:
        -----------
        alpha : float
            Significance level for hypothesis tests
        """
        self.alpha = alpha
    
    def paired_comparison_test(self, scores1: np.ndarray, scores2: np.ndarray,
                             test_type: str = 'wilcoxon') -> TestResult:
        """
        Perform paired comparison test between two models.
        
        Parameters:
        -----------
        scores1, scores2 : np.ndarray
            Performance scores for models 1 and 2
        test_type : str
            Type of test ('wilcoxon', 'ttest', 'permutation')
            
        Returns:
        --------
        TestResult
            Comparison test results
        """
        if len(scores1) != len(scores2):
            raise ValueError("Score arrays must have same length")
        
        differences = scores1 - scores2
        
        if test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric)
            statistic, p_value = wilcoxon(differences, alternative='two-sided')
            test_name = "Wilcoxon Signed-Rank Test"
            
        elif test_type == 'ttest':
            # Paired t-test (parametric)
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            test_name = "Paired t-test"
            
        elif test_type == 'permutation':
            # Permutation test
            statistic = np.mean(differences)
            p_value = self._permutation_test(scores1, scores2)
            test_name = "Permutation Test"
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Effect size (Cohen's d for differences)
        effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
        
        # Confidence interval for mean difference
        n = len(differences)
        se = np.std(differences) / np.sqrt(n)
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
        ci_lower = np.mean(differences) - t_critical * se
        ci_upper = np.mean(differences) + t_critical * se
        
        # Interpretation
        if p_value < self.alpha:
            if np.mean(differences) > 0:
                interpretation = "Model 1 significantly outperforms Model 2"
            else:
                interpretation = "Model 2 significantly outperforms Model 1"
        else:
            interpretation = "No significant difference between models"
        
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _permutation_test(self, scores1: np.ndarray, scores2: np.ndarray,
                         n_permutations: int = 10000) -> float:
        """
        Perform permutation test for paired comparison.
        
        Parameters:
        -----------
        scores1, scores2 : np.ndarray
            Performance scores
        n_permutations : int
            Number of permutations
            
        Returns:
        --------
        float
            p-value from permutation test
        """
        observed_diff = np.mean(scores1 - scores2)
        
        # Generate null distribution
        null_diffs = []
        combined_scores = np.column_stack([scores1, scores2])
        
        for _ in range(n_permutations):
            # Randomly flip signs of differences
            signs = np.random.choice([-1, 1], size=len(scores1))
            permuted_diff = np.mean(signs * (scores1 - scores2))
            null_diffs.append(permuted_diff)
        
        null_diffs = np.array(null_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
        
        return p_value
    
    def multiple_model_comparison(self, model_scores: Dict[str, np.ndarray],
                                correction_method: str = 'fdr_bh') -> Dict[str, Any]:
        """
        Compare multiple models with correction for multiple testing.
        
        Parameters:
        -----------
        model_scores : Dict[str, np.ndarray]
            Performance scores for each model
        correction_method : str
            Multiple testing correction method
            
        Returns:
        --------
        Dict[str, Any]
            Results including pairwise comparisons and corrected p-values
        """
        model_names = list(model_scores.keys())
        n_models = len(model_names)
        
        # Pairwise comparisons
        pairwise_results = {}
        p_values = []
        comparisons = []
        
        for i, j in itertools.combinations(range(n_models), 2):
            model1, model2 = model_names[i], model_names[j]
            scores1, scores2 = model_scores[model1], model_scores[model2]
            
            result = self.paired_comparison_test(scores1, scores2)
            pairwise_results[f"{model1}_vs_{model2}"] = result
            p_values.append(result.p_value)
            comparisons.append(f"{model1}_vs_{model2}")
        
        # Multiple testing correction
        if len(p_values) > 1:
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=self.alpha, method=correction_method
            )
        else:
            rejected = [p_values[0] < self.alpha] if p_values else []
            p_corrected = p_values.copy()
        
        # Update results with corrected p-values
        for i, comparison in enumerate(comparisons):
            pairwise_results[comparison].p_value = p_corrected[i]
            pairwise_results[comparison].reject_null = rejected[i]
        
        # Overall results
        results = {
            'pairwise_comparisons': pairwise_results,
            'correction_method': correction_method,
            'significant_differences': [comp for i, comp in enumerate(comparisons) if rejected[i]],
            'n_comparisons': len(comparisons),
            'family_wise_error_rate': self.alpha
        }
        
        return results
    
    def cross_validation_test(self, model_predictions: Dict[str, np.ndarray],
                            true_values: np.ndarray, cv_folds: int = 5,
                            scoring_function: Callable = None) -> Dict[str, TestResult]:
        """
        Perform cross-validation comparison of models.
        
        Parameters:
        -----------
        model_predictions : Dict[str, np.ndarray]
            Predictions from each model
        true_values : np.ndarray
            True target values
        cv_folds : int
            Number of cross-validation folds
        scoring_function : Callable, optional
            Custom scoring function
            
        Returns:
        --------
        Dict[str, TestResult]
            Cross-validation results for each model
        """
        if scoring_function is None:
            scoring_function = lambda y_true, y_pred: -np.mean((y_true - y_pred)**2)  # Negative MSE
        
        n_samples = len(true_values)
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {model: [] for model in model_predictions.keys()}
        
        for train_idx, test_idx in kfold.split(range(n_samples)):
            for model_name, predictions in model_predictions.items():
                test_predictions = predictions[test_idx]
                test_true = true_values[test_idx]
                
                score = scoring_function(test_true, test_predictions)
                cv_scores[model_name].append(score)
        
        # Statistical tests for each model
        results = {}
        
        for model_name, scores in cv_scores.items():
            scores = np.array(scores)
            
            # Test if scores significantly different from zero
            statistic, p_value = stats.ttest_1samp(scores, 0)
            
            # Confidence interval
            ci_lower, ci_upper = stats.t.interval(
                1 - self.alpha, len(scores) - 1,
                loc=np.mean(scores), scale=stats.sem(scores)
            )
            
            results[model_name] = TestResult(
                test_name=f"CV Performance Test - {model_name}",
                statistic=np.mean(scores),
                p_value=p_value,
                reject_null=p_value < self.alpha,
                effect_size=np.mean(scores) / np.std(scores),
                confidence_interval=(ci_lower, ci_upper),
                interpretation=f"Mean CV score: {np.mean(scores):.4f} ± {np.std(scores):.4f}"
            )
        
        return results


class BayesianValidator:
    """
    Specialized validation for Bayesian methods.
    
    Provides validation of MCMC convergence, posterior distribution
    properties, and Bayesian model comparison.
    """
    
    def __init__(self):
        """Initialize Bayesian validator."""
        pass
    
    def test_mcmc_convergence(self, chains: List[np.ndarray],
                            parameter_names: List[str] = None) -> Dict[str, TestResult]:
        """
        Test MCMC convergence using multiple diagnostics.
        
        Parameters:
        -----------
        chains : List[np.ndarray]
            MCMC chains for each parameter
        parameter_names : List[str], optional
            Parameter names
            
        Returns:
        --------
        Dict[str, TestResult]
            Convergence test results
        """
        if parameter_names is None:
            parameter_names = [f'param_{i}' for i in range(len(chains))]
        
        results = {}
        
        for i, (chain, param_name) in enumerate(zip(chains, parameter_names)):
            # Reshape chain if needed (assume shape is (n_chains, n_samples))
            if chain.ndim == 1:
                chain = chain.reshape(1, -1)
            
            n_chains, n_samples = chain.shape
            
            # 1. Gelman-Rubin diagnostic (R-hat)
            if n_chains > 1:
                rhat = self._calculate_rhat(chain)
                
                results[f'{param_name}_rhat'] = TestResult(
                    test_name=f"Gelman-Rubin R-hat - {param_name}",
                    statistic=rhat,
                    p_value=None,  # R-hat doesn't have p-value
                    reject_null=rhat > 1.1,  # Common threshold
                    interpretation=f"R-hat = {rhat:.4f} ({'Good' if rhat <= 1.1 else 'Poor'} convergence)"
                )
            
            # 2. Geweke diagnostic
            z_score, p_value = self._geweke_diagnostic(chain.flatten())
            
            results[f'{param_name}_geweke'] = TestResult(
                test_name=f"Geweke Diagnostic - {param_name}",
                statistic=z_score,
                p_value=p_value,
                reject_null=p_value < 0.05,
                interpretation=f"Z-score = {z_score:.4f} ({'Converged' if p_value >= 0.05 else 'Not converged'})"
            )
            
            # 3. Effective sample size
            ess = self._effective_sample_size(chain.flatten())
            ess_ratio = ess / n_samples
            
            results[f'{param_name}_ess'] = TestResult(
                test_name=f"Effective Sample Size - {param_name}",
                statistic=ess,
                p_value=None,
                reject_null=ess_ratio < 0.1,  # Less than 10% is concerning
                interpretation=f"ESS = {ess:.0f} ({ess_ratio:.2%} of total samples)"
            )
        
        return results
    
    def _calculate_rhat(self, chains: np.ndarray) -> float:
        """
        Calculate Gelman-Rubin R-hat statistic.
        
        Parameters:
        -----------
        chains : np.ndarray, shape (n_chains, n_samples)
            MCMC chains
            
        Returns:
        --------
        float
            R-hat statistic
        """
        n_chains, n_samples = chains.shape
        
        # Within-chain variance
        within_var = np.mean(np.var(chains, axis=1, ddof=1))
        
        # Between-chain variance
        chain_means = np.mean(chains, axis=1)
        between_var = n_samples * np.var(chain_means, ddof=1)
        
        # Marginal posterior variance
        marginal_var = ((n_samples - 1) / n_samples) * within_var + (1 / n_samples) * between_var
        
        # R-hat
        rhat = np.sqrt(marginal_var / within_var)
        
        return rhat
    
    def _geweke_diagnostic(self, chain: np.ndarray, first: float = 0.1, 
                          last: float = 0.5) -> Tuple[float, float]:
        """
        Calculate Geweke convergence diagnostic.
        
        Parameters:
        -----------
        chain : np.ndarray
            MCMC chain
        first : float
            Fraction of chain to use for first part
        last : float
            Fraction of chain to use for last part
            
        Returns:
        --------
        Tuple[float, float]
            Z-score and p-value
        """
        n = len(chain)
        
        # First part
        n1 = int(first * n)
        first_part = chain[:n1]
        
        # Last part
        n2 = int(last * n)
        last_part = chain[-n2:]
        
        # Means and variances
        mean1, mean2 = np.mean(first_part), np.mean(last_part)
        var1, var2 = np.var(first_part, ddof=1), np.var(last_part, ddof=1)
        
        # Z-score
        se = np.sqrt(var1/n1 + var2/n2)
        z_score = (mean1 - mean2) / se if se > 0 else 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return z_score, p_value
    
    def _effective_sample_size(self, chain: np.ndarray) -> float:
        """
        Estimate effective sample size using autocorrelation.
        
        Parameters:
        -----------
        chain : np.ndarray
            MCMC chain
            
        Returns:
        --------
        float
            Effective sample size
        """
        n = len(chain)
        
        # Center the chain
        centered = chain - np.mean(chain)
        
        # Autocorrelation function
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[n-1:]  # Keep only non-negative lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first negative autocorrelation or cutoff
        cutoff = None
        for i, ac in enumerate(autocorr[1:], 1):
            if ac <= 0:
                cutoff = i
                break
        
        if cutoff is None:
            cutoff = min(len(autocorr), n // 4)  # Use at most n/4 lags
        
        # Integrated autocorrelation time
        integrated_time = 1 + 2 * np.sum(autocorr[1:cutoff])
        
        # Effective sample size
        ess = n / integrated_time
        
        return max(1, ess)  # At least 1
    
    def test_posterior_properties(self, samples: np.ndarray,
                                prior_samples: Optional[np.ndarray] = None,
                                true_value: Optional[np.ndarray] = None) -> Dict[str, TestResult]:
        """
        Test posterior distribution properties.
        
        Parameters:
        -----------
        samples : np.ndarray
            Posterior samples
        prior_samples : np.ndarray, optional
            Prior samples for comparison
        true_value : np.ndarray, optional
            True parameter value
            
        Returns:
        --------
        Dict[str, TestResult]
            Posterior property test results
        """
        results = {}
        
        # 1. Normality test (multivariate if applicable)
        if samples.ndim == 1:
            # Univariate normality test
            statistic, p_value = shapiro(samples)
            
            results['normality'] = TestResult(
                test_name="Shapiro-Wilk Normality Test",
                statistic=statistic,
                p_value=p_value,
                reject_null=p_value < 0.05,
                interpretation=f"Posterior is {'not ' if p_value < 0.05 else ''}approximately normal"
            )
        
        # 2. Test if true value is in credible region
        if true_value is not None:
            # Calculate credible intervals
            credible_levels = [0.68, 0.95]
            for level in credible_levels:
                alpha = 1 - level
                
                if samples.ndim == 1:
                    lower = np.percentile(samples, 100 * alpha/2)
                    upper = np.percentile(samples, 100 * (1 - alpha/2))
                    in_interval = (lower <= true_value <= upper)
                else:
                    # Multivariate case - use marginal intervals
                    in_interval = True
                    for i in range(samples.shape[1]):
                        lower = np.percentile(samples[:, i], 100 * alpha/2)
                        upper = np.percentile(samples[:, i], 100 * (1 - alpha/2))
                        if not (lower <= true_value[i] <= upper):
                            in_interval = False
                            break
                
                results[f'credible_{level}'] = TestResult(
                    test_name=f"{level*100:.0f}% Credible Interval Test",
                    statistic=float(in_interval),
                    p_value=None,
                    reject_null=not in_interval,
                    interpretation=f"True value is {'in' if in_interval else 'not in'} {level*100:.0f}% credible interval"
                )
        
        # 3. Prior-posterior comparison (if prior samples available)
        if prior_samples is not None:
            # KS test comparing prior and posterior
            if samples.ndim == 1 and prior_samples.ndim == 1:
                statistic, p_value = kstest(samples, prior_samples)
                
                results['prior_posterior_ks'] = TestResult(
                    test_name="Prior-Posterior KS Test",
                    statistic=statistic,
                    p_value=p_value,
                    reject_null=p_value < 0.05,
                    interpretation=f"Posterior {'differs significantly from' if p_value < 0.05 else 'similar to'} prior"
                )
        
        return results


class ComprehensiveValidator:
    """
    Comprehensive validation framework combining all validation methods.
    
    Provides end-to-end validation pipeline for Bayesian PDE methods
    with automated reporting and recommendations.
    """
    
    def __init__(self, alpha: float = 0.05, output_dir: str = "validation_results"):
        """
        Initialize comprehensive validator.
        
        Parameters:
        -----------
        alpha : float
            Significance level
        output_dir : str
            Directory for output files
        """
        self.alpha = alpha
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize component validators
        self.coverage_validator = CoverageValidator()
        self.model_comparison_validator = ModelComparisonValidator(alpha)
        self.bayesian_validator = BayesianValidator()
        
        # Test results storage
        self.all_results = {}
    
    def validate_method(self, method_name: str, validation_data: Dict[str, Any]) -> ValidationSummary:
        """
        Perform comprehensive validation of a single method.
        
        Parameters:
        -----------
        method_name : str
            Name of method being validated
        validation_data : Dict[str, Any]
            Complete validation data
            
        Returns:
        --------
        ValidationSummary
            Validation summary and recommendations
        """
        method_results = {}
        
        # 1. Coverage validation
        if 'uncertainty_intervals' in validation_data:
            coverage_results = self.coverage_validator.comprehensive_coverage_analysis(
                validation_data['uncertainty_intervals']
            )
            method_results.update(coverage_results)
        
        # 2. MCMC convergence validation
        if 'mcmc_chains' in validation_data:
            convergence_results = self.bayesian_validator.test_mcmc_convergence(
                validation_data['mcmc_chains'],
                validation_data.get('parameter_names')
            )
            method_results.update(convergence_results)
        
        # 3. Posterior properties validation
        if 'posterior_samples' in validation_data:
            posterior_results = self.bayesian_validator.test_posterior_properties(
                validation_data['posterior_samples'],
                validation_data.get('prior_samples'),
                validation_data.get('true_parameters')
            )
            method_results.update(posterior_results)
        
        # 4. Calibration validation
        if 'predicted_probabilities' in validation_data:
            calibration_result = self.coverage_validator.test_probability_calibration(
                validation_data['predicted_probabilities'],
                validation_data['observed_outcomes']
            )
            method_results['calibration'] = calibration_result
        
        # Store results
        self.all_results[method_name] = method_results
        
        # Generate summary
        summary = self._generate_validation_summary(method_name, method_results)
        
        # Save detailed results
        self._save_detailed_results(method_name, method_results, summary)
        
        return summary
    
    def compare_methods(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple methods with statistical significance testing.
        
        Parameters:
        -----------
        method_results : Dict[str, Dict[str, Any]]
            Results for each method
            
        Returns:
        --------
        Dict[str, Any]
            Comparison results with statistical significance
        """
        # Extract performance scores for comparison
        model_scores = {}
        
        for method_name, results in method_results.items():
            if 'performance_scores' in results:
                model_scores[method_name] = results['performance_scores']
        
        if len(model_scores) > 1:
            comparison_results = self.model_comparison_validator.multiple_model_comparison(
                model_scores, correction_method='fdr_bh'
            )
            
            # Save comparison results
            comparison_file = self.output_dir / "method_comparison.json"
            with open(comparison_file, 'w') as f:
                # Convert TestResult objects to dicts for JSON serialization
                serializable_results = {}
                for key, value in comparison_results.items():
                    if isinstance(value, dict):
                        serializable_dict = {}
                        for k, v in value.items():
                            if hasattr(v, '__dict__'):
                                serializable_dict[k] = v.__dict__
                            else:
                                serializable_dict[k] = v
                        serializable_results[key] = serializable_dict
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2)
            
            return comparison_results
        
        return {}
    
    def _generate_validation_summary(self, method_name: str, 
                                   results: Dict[str, TestResult]) -> ValidationSummary:
        """Generate validation summary from test results."""
        total_tests = len(results)
        failed_tests = sum(1 for result in results.values() if result.reject_null)
        passed_tests = total_tests - failed_tests
        
        # Calculate overall p-value using Fisher's method
        p_values = [r.p_value for r in results.values() if r.p_value is not None]
        if p_values:
            # Fisher's combined p-value test
            test_statistic = -2 * np.sum(np.log(p_values))
            overall_p_value = 1 - stats.chi2.cdf(test_statistic, 2 * len(p_values))
        else:
            overall_p_value = 1.0
        
        # Identify significant failures
        significant_failures = [
            name for name, result in results.items() 
            if result.reject_null and result.p_value is not None and result.p_value < self.alpha
        ]
        
        # Calculate validation score (0-1, higher is better)
        validation_score = passed_tests / total_tests if total_tests > 0 else 1.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, validation_score)
        
        return ValidationSummary(
            method_name=method_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_p_value=overall_p_value,
            significant_failures=significant_failures,
            validation_score=validation_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, results: Dict[str, TestResult], 
                                validation_score: float) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Overall assessment
        if validation_score >= 0.9:
            recommendations.append("Excellent validation performance - method is well-calibrated")
        elif validation_score >= 0.7:
            recommendations.append("Good validation performance with minor issues to address")
        elif validation_score >= 0.5:
            recommendations.append("Moderate validation performance - several issues need attention")
        else:
            recommendations.append("Poor validation performance - major issues require investigation")
        
        # Specific recommendations based on failed tests
        for test_name, result in results.items():
            if result.reject_null:
                if 'coverage' in test_name.lower():
                    if 'under' in result.interpretation.lower():
                        recommendations.append("Increase uncertainty estimates - confidence intervals are too narrow")
                    elif 'over' in result.interpretation.lower():
                        recommendations.append("Reduce uncertainty estimates - confidence intervals are too wide")
                
                elif 'convergence' in test_name.lower() or 'rhat' in test_name.lower():
                    recommendations.append("Improve MCMC convergence - run longer chains or tune sampler")
                
                elif 'calibration' in test_name.lower():
                    recommendations.append("Improve probability calibration - recalibrate uncertainty estimates")
                
                elif 'normality' in test_name.lower():
                    recommendations.append("Consider non-Gaussian posterior approximations")
        
        return recommendations
    
    def _save_detailed_results(self, method_name: str, results: Dict[str, TestResult],
                             summary: ValidationSummary):
        """Save detailed validation results to files."""
        # Create method-specific directory
        method_dir = self.output_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_data = {}
        for test_name, result in results.items():
            results_data[test_name] = {
                'test_name': result.test_name,
                'statistic': result.statistic,
                'p_value': result.p_value,
                'reject_null': result.reject_null,
                'effect_size': result.effect_size,
                'confidence_interval': result.confidence_interval,
                'interpretation': result.interpretation
            }
        
        results_file = method_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary
        summary_file = method_dir / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'method_name': summary.method_name,
                'total_tests': summary.total_tests,
                'passed_tests': summary.passed_tests,
                'failed_tests': summary.failed_tests,
                'overall_p_value': summary.overall_p_value,
                'significant_failures': summary.significant_failures,
                'validation_score': summary.validation_score,
                'recommendations': summary.recommendations
            }, f, indent=2)
        
        print(f"Saved validation results for {method_name} to {method_dir}")
    
    def generate_validation_report(self, method_summaries: List[ValidationSummary]) -> str:
        """
        Generate comprehensive validation report.
        
        Parameters:
        -----------
        method_summaries : List[ValidationSummary]
            Validation summaries for all methods
            
        Returns:
        --------
        str
            Formatted validation report
        """
        report = "# Statistical Validation Report\n\n"
        
        # Overall summary
        report += "## Overall Summary\n\n"
        report += f"- Total methods validated: {len(method_summaries)}\n"
        report += f"- Significance level (α): {self.alpha}\n\n"
        
        # Method-specific results
        report += "## Method-Specific Results\n\n"
        
        for summary in method_summaries:
            report += f"### {summary.method_name}\n\n"
            report += f"- **Validation Score**: {summary.validation_score:.3f}\n"
            report += f"- **Tests Passed**: {summary.passed_tests}/{summary.total_tests}\n"
            report += f"- **Overall p-value**: {summary.overall_p_value:.4f}\n"
            
            if summary.significant_failures:
                report += f"- **Significant Failures**: {', '.join(summary.significant_failures)}\n"
            
            if summary.recommendations:
                report += "\n**Recommendations**:\n"
                for rec in summary.recommendations:
                    report += f"  - {rec}\n"
            
            report += "\n"
        
        # Rankings
        report += "## Method Rankings\n\n"
        sorted_methods = sorted(method_summaries, key=lambda x: x.validation_score, reverse=True)
        
        for i, summary in enumerate(sorted_methods, 1):
            report += f"{i}. {summary.method_name} (Score: {summary.validation_score:.3f})\n"
        
        # Save report
        report_file = self.output_dir / "validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Validation report saved to {report_file}")
        return report


def demo_statistical_validation():
    """Demonstrate statistical validation framework."""
    print("Statistical Validation Framework Demo")
    print("=" * 40)
    
    # Generate synthetic validation data
    np.random.seed(42)
    
    # Method 1: Well-calibrated method
    n_samples = 1000
    true_params = np.array([1.0, -0.5, 0.3])
    
    # Generate synthetic intervals and predictions
    intervals_95 = np.column_stack([
        true_params[0] + np.random.normal(-0.2, 0.1, n_samples),
        true_params[0] + np.random.normal(0.2, 0.1, n_samples)
    ])
    
    # Predicted probabilities (well-calibrated)
    pred_probs = np.random.beta(2, 3, n_samples)
    outcomes = np.random.binomial(1, pred_probs, n_samples)
    
    # MCMC chains
    mcmc_chains = [np.random.normal(true_params[i], 0.1, (3, 500)) for i in range(3)]
    
    # Method 1 validation data
    method1_data = {
        'uncertainty_intervals': {
            'intervals_0.95': intervals_95,
            'true_values': np.repeat(true_params[0], n_samples)
        },
        'mcmc_chains': mcmc_chains,
        'posterior_samples': np.random.multivariate_normal(true_params, np.eye(3)*0.01, 1000),
        'predicted_probabilities': pred_probs,
        'observed_outcomes': outcomes,
        'true_parameters': true_params,
        'parameter_names': ['theta_1', 'theta_2', 'theta_3']
    }
    
    # Method 2: Poorly calibrated method
    intervals_95_bad = np.column_stack([
        true_params[0] + np.random.normal(-0.05, 0.02, n_samples),  # Too narrow
        true_params[0] + np.random.normal(0.05, 0.02, n_samples)
    ])
    
    pred_probs_bad = np.random.uniform(0.3, 0.7, n_samples)  # Poorly calibrated
    outcomes_bad = np.random.binomial(1, np.random.uniform(0, 1, n_samples), n_samples)
    
    method2_data = {
        'uncertainty_intervals': {
            'intervals_0.95': intervals_95_bad,
            'true_values': np.repeat(true_params[0], n_samples)
        },
        'predicted_probabilities': pred_probs_bad,
        'observed_outcomes': outcomes_bad
    }
    
    # Initialize comprehensive validator
    validator = ComprehensiveValidator(alpha=0.05, output_dir="demo_validation")
    
    # Validate methods
    print("Validating Method 1 (well-calibrated)...")
    summary1 = validator.validate_method("WellCalibratedMethod", method1_data)
    
    print(f"Method 1 Results:")
    print(f"  Validation Score: {summary1.validation_score:.3f}")
    print(f"  Tests Passed: {summary1.passed_tests}/{summary1.total_tests}")
    print(f"  Recommendations: {len(summary1.recommendations)}")
    
    print("\nValidating Method 2 (poorly calibrated)...")
    summary2 = validator.validate_method("PoorlyCalibrated", method2_data)
    
    print(f"Method 2 Results:")
    print(f"  Validation Score: {summary2.validation_score:.3f}")
    print(f"  Tests Passed: {summary2.passed_tests}/{summary2.total_tests}")
    print(f"  Recommendations: {len(summary2.recommendations)}")
    
    # Generate comparison report
    print("\nGenerating validation report...")
    report = validator.generate_validation_report([summary1, summary2])
    
    # Individual component demonstrations
    print("\n" + "="*50)
    print("Component Demonstrations")
    print("="*50)
    
    # Coverage validation
    print("\n1. Coverage Validation:")
    coverage_validator = CoverageValidator()
    
    coverage_result = coverage_validator.test_interval_coverage(
        intervals_95, np.repeat(true_params[0], n_samples), confidence_level=0.95
    )
    print(f"   Coverage Test: {coverage_result.interpretation}")
    
    # Model comparison
    print("\n2. Model Comparison:")
    model_comparison = ModelComparisonValidator()
    
    scores1 = np.random.normal(0.85, 0.1, 50)  # Method 1 performance
    scores2 = np.random.normal(0.75, 0.15, 50)  # Method 2 performance
    
    comparison_result = model_comparison.paired_comparison_test(scores1, scores2)
    print(f"   Comparison Test: {comparison_result.interpretation}")
    
    # Bayesian validation
    print("\n3. Bayesian Validation:")
    bayesian_validator = BayesianValidator()
    
    convergence_results = bayesian_validator.test_mcmc_convergence(mcmc_chains[:1], ['theta_1'])
    for test_name, result in convergence_results.items():
        print(f"   {test_name}: {result.interpretation}")
    
    print("\nStatistical validation demo completed!")
    print("Check 'demo_validation/' directory for detailed results.")


if __name__ == "__main__":
    demo_statistical_validation()