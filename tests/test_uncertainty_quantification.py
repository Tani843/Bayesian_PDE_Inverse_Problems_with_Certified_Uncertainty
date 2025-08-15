"""
Test suite for uncertainty quantification methods.

Tests certified bounds, concentration inequalities, PAC-Bayes bounds,
coverage analysis, and uncertainty calibration.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
import matplotlib.pyplot as plt

from bayesian_pde_solver.uncertainty_quantification.certified_bounds import CertifiedBounds
from bayesian_pde_solver.uncertainty_quantification.concentration_inequalities import (
    HoeffdingBound, BernsteinBound, McDiarmidBound, ConcentrationInequalities
)
from bayesian_pde_solver.uncertainty_quantification.pac_bayes_bounds import (
    PACBayesBounds, McAllesterBound, SeegerBound, CatoniBound
)


class TestConcentrationInequalities:
    """Test concentration inequality implementations."""
    
    def test_hoeffding_bound_basic(self):
        """Test basic Hoeffding bound computation."""
        # Generate bounded random variables
        np.random.seed(42)
        n_samples = 1000
        samples = np.random.uniform(-1, 1, n_samples)  # Bounded in [-1, 1]
        
        bound = HoeffdingBound(bound_range=(-1, 1))
        
        # Compute bound for different confidence levels
        confidence_levels = [0.9, 0.95, 0.99]
        
        for confidence in confidence_levels:
            delta = 1 - confidence
            bound_value = bound.compute_bound(samples, delta)
            
            # Bound should be positive and decrease with more samples
            assert bound_value > 0
            
            # Check that empirical mean is within bounds with high probability
            empirical_mean = np.mean(samples)
            theoretical_bound = bound_value
            
            # For this test, we know the theoretical bound
            expected_bound = np.sqrt(-np.log(delta/2) / (2 * n_samples))
            assert abs(bound_value - expected_bound) < 1e-10
    
    def test_hoeffding_convergence(self):
        """Test Hoeffding bound convergence with sample size."""
        np.random.seed(42)
        bound = HoeffdingBound(bound_range=(0, 1))
        
        sample_sizes = [100, 400, 1600, 6400]
        bounds = []
        
        for n in sample_sizes:
            samples = np.random.beta(2, 2, n)  # Bounded in [0, 1]
            bound_value = bound.compute_bound(samples, delta=0.05)
            bounds.append(bound_value)
        
        # Bounds should decrease with sqrt(n)
        for i in range(len(bounds) - 1):
            ratio = bounds[i] / bounds[i + 1]
            expected_ratio = np.sqrt(sample_sizes[i + 1] / sample_sizes[i])
            # Should be approximately sqrt(4) = 2
            assert 1.8 < ratio < 2.2
    
    def test_bernstein_bound_basic(self):
        """Test Bernstein bound for sub-exponential random variables."""
        np.random.seed(42)
        n_samples = 2000
        
        # Generate sub-exponential samples (exponential distribution shifted)
        samples = np.random.exponential(1.0, n_samples) - 1.0
        
        # Estimate sub-exponential parameters
        variance_estimate = np.var(samples)
        range_estimate = np.max(samples) - np.min(samples)
        
        bound = BernsteinBound(
            variance_proxy=variance_estimate,
            range_bound=range_estimate
        )
        
        bound_value = bound.compute_bound(samples, delta=0.05)
        
        assert bound_value > 0
        assert np.isfinite(bound_value)
        
        # Bernstein bound should be tighter than Hoeffding for small variance
        hoeffding = HoeffdingBound(bound_range=(np.min(samples), np.max(samples)))
        hoeffding_bound = hoeffding.compute_bound(samples, delta=0.05)
        
        # Bernstein should give smaller bound when variance is small
        if variance_estimate < (range_estimate**2 / 4):
            assert bound_value <= hoeffding_bound
    
    def test_mcdiarmid_bound_basic(self):
        """Test McDiarmid's inequality for functions with bounded differences."""
        np.random.seed(42)
        n_samples = 1000
        
        def bounded_difference_function(x_vec):
            """Function with bounded difference property."""
            return np.mean(x_vec**2)  # Each coordinate affects result by at most 1/n
        
        samples = np.random.uniform(-1, 1, n_samples)
        function_values = [bounded_difference_function(samples[:i+1]) 
                          for i in range(10, n_samples, 50)]
        
        # Bounded difference parameter (each variable affects result by at most c_i)
        c_values = [2.0 / i for i in range(10, n_samples, 50)]  # |x_i|^2 ≤ 1, affects mean by 1/n
        
        bound = McDiarmidBound(bounded_differences=c_values)
        
        # Test bound computation
        bound_value = bound.compute_bound(function_values, delta=0.05)
        
        assert bound_value > 0
        assert np.isfinite(bound_value)
    
    def test_concentration_inequalities_wrapper(self):
        """Test the concentration inequalities wrapper class."""
        np.random.seed(42)
        n_samples = 1500
        samples = np.random.uniform(0, 1, n_samples)
        
        concentrator = ConcentrationInequalities()
        
        # Test multiple inequality types
        bounds = concentrator.compute_all_bounds(
            samples=samples,
            confidence_level=0.95,
            bound_range=(0, 1),
            variance_estimate=np.var(samples)
        )
        
        # All bounds should be positive and finite
        for bound_type, bound_value in bounds.items():
            assert bound_value > 0, f"{bound_type} bound should be positive"
            assert np.isfinite(bound_value), f"{bound_type} bound should be finite"
        
        # Hoeffding bound should exist
        assert 'hoeffding' in bounds
        
        # Test empirical coverage
        empirical_mean = np.mean(samples)
        true_mean = 0.5  # For uniform[0,1]
        
        # At least one bound should contain the true deviation
        deviation = abs(empirical_mean - true_mean)
        assert any(bound_value >= deviation for bound_value in bounds.values())


class TestPACBayesBounds:
    """Test PAC-Bayes bound implementations."""
    
    def test_mcallester_bound_basic(self):
        """Test McAllester PAC-Bayes bound."""
        np.random.seed(42)
        
        # Generate synthetic posterior samples
        n_samples = 1000
        true_parameter = 0.3
        posterior_samples = np.random.normal(true_parameter, 0.1, n_samples)
        
        # Define prior (standard normal)
        prior_mean = 0.0
        prior_std = 1.0
        
        bound = McAllesterBound()
        
        # Compute bound
        kl_divergence = bound.compute_kl_divergence(
            posterior_samples=posterior_samples,
            prior_mean=prior_mean,
            prior_std=prior_std
        )
        
        pac_bound = bound.compute_bound(
            kl_divergence=kl_divergence,
            n_samples=n_samples,
            confidence=0.95
        )
        
        assert kl_divergence >= 0  # KL divergence is non-negative
        assert pac_bound > 0
        assert np.isfinite(pac_bound)
        
        # KL should be reasonable for this setup
        expected_kl = 0.5 * ((true_parameter - prior_mean)**2 / prior_std**2 + 
                            0.1**2 / prior_std**2 - 1 - 2*np.log(0.1/prior_std))
        assert abs(kl_divergence - expected_kl) < 0.1
    
    def test_seeger_bound_basic(self):
        """Test Seeger PAC-Bayes bound."""
        bound = SeegerBound()
        
        # Test with different KL values and sample sizes
        test_cases = [
            (0.5, 100, 0.95),
            (1.0, 500, 0.99),
            (2.0, 1000, 0.9)
        ]
        
        for kl_div, n_samples, confidence in test_cases:
            bound_value = bound.compute_bound(kl_div, n_samples, confidence)
            
            assert bound_value > 0
            assert np.isfinite(bound_value)
            
            # Bound should decrease with more samples
            bound_more_samples = bound.compute_bound(kl_div, n_samples * 2, confidence)
            assert bound_more_samples < bound_value
    
    def test_catoni_bound_basic(self):
        """Test Catoni PAC-Bayes bound."""
        bound = CatoniBound()
        
        # Generate test data with bounded loss
        np.random.seed(42)
        n_samples = 800
        losses = np.random.beta(2, 5, n_samples)  # Losses in [0, 1]
        kl_divergence = 1.5
        
        bound_value = bound.compute_bound(
            empirical_losses=losses,
            kl_divergence=kl_divergence,
            confidence=0.95,
            loss_bound=1.0
        )
        
        assert bound_value > 0
        assert np.isfinite(bound_value)
        
        # Bound should be close to empirical risk plus complexity term
        empirical_risk = np.mean(losses)
        assert bound_value > empirical_risk  # Should upper bound empirical risk
    
    def test_pac_bayes_bounds_wrapper(self):
        """Test PAC-Bayes bounds wrapper class."""
        pac_bounds = PACBayesBounds()
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1200
        posterior_samples = np.random.normal(1.0, 0.5, n_samples)
        prior_mean = 0.0
        prior_cov = np.eye(1) * 2.0
        losses = np.random.exponential(0.5, n_samples)
        
        # Compute all bounds
        bounds = pac_bounds.compute_all_bounds(
            posterior_samples=posterior_samples[:, np.newaxis],  # Make 2D
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            empirical_losses=losses,
            confidence=0.95
        )
        
        # Check that all bounds are computed
        expected_bounds = ['mcallester', 'seeger', 'catoni']
        for bound_name in expected_bounds:
            assert bound_name in bounds
            assert bounds[bound_name] > 0
            assert np.isfinite(bounds[bound_name])
    
    def test_pac_bayes_tightness_comparison(self):
        """Compare tightness of different PAC-Bayes bounds."""
        # Generate data where we expect certain orderings
        np.random.seed(42)
        n_samples = 2000
        
        # Small KL divergence case (posterior close to prior)
        kl_small = 0.1
        
        mcallester = McAllesterBound()
        seeger = SeegerBound()
        
        bound_mcallester = mcallester.compute_bound(kl_small, n_samples, 0.95)
        bound_seeger = seeger.compute_bound(kl_small, n_samples, 0.95)
        
        # For small KL, both bounds should be similar
        ratio = max(bound_mcallester, bound_seeger) / min(bound_mcallester, bound_seeger)
        assert ratio < 2.0, "Bounds should be similar for small KL divergence"


class TestCertifiedBounds:
    """Test the main certified bounds framework."""
    
    def test_certified_bounds_initialization(self):
        """Test certified bounds class initialization."""
        cb = CertifiedBounds()
        
        assert hasattr(cb, 'concentration_ineq')
        assert hasattr(cb, 'pac_bayes')
        assert cb.concentration_ineq is not None
        assert cb.pac_bayes is not None
    
    def test_parameter_bounds_computation(self):
        """Test parameter bounds computation."""
        cb = CertifiedBounds()
        
        # Generate synthetic posterior samples for 2 parameters
        np.random.seed(42)
        n_samples = 1500
        
        # Parameter 1: centered around 1.0
        param1_samples = np.random.normal(1.0, 0.2, n_samples)
        # Parameter 2: centered around -0.5  
        param2_samples = np.random.normal(-0.5, 0.3, n_samples)
        
        posterior_samples = np.column_stack([param1_samples, param2_samples])
        
        # Define parameter constraints
        parameter_bounds = [(-2, 3), (-2, 1)]  # [min, max] for each parameter
        
        bounds = cb.compute_parameter_bounds(
            posterior_samples=posterior_samples,
            parameter_names=['param1', 'param2'],
            parameter_bounds=parameter_bounds,
            confidence_level=0.95
        )
        
        # Check output structure
        assert 'param1' in bounds
        assert 'param2' in bounds
        
        for param_name in ['param1', 'param2']:
            param_bounds = bounds[param_name]
            assert 'concentration' in param_bounds
            assert 'pac_bayes' in param_bounds
            assert 'empirical_ci' in param_bounds
            
            # Each method should provide lower and upper bounds
            for method in ['concentration', 'pac_bayes', 'empirical_ci']:
                method_bounds = param_bounds[method]
                assert 'lower' in method_bounds
                assert 'upper' in method_bounds
                assert method_bounds['lower'] < method_bounds['upper']
    
    def test_prediction_bounds_computation(self):
        """Test prediction uncertainty bounds."""
        cb = CertifiedBounds()
        
        np.random.seed(42)
        n_samples = 1000
        n_predictions = 50
        
        # Generate synthetic predictions (e.g., PDE solutions at different points)
        true_values = np.sin(np.linspace(0, np.pi, n_predictions))
        
        # Add noise to create ensemble predictions
        predictions = np.zeros((n_samples, n_predictions))
        for i in range(n_samples):
            noise = np.random.normal(0, 0.1, n_predictions)
            predictions[i] = true_values + noise
        
        bounds = cb.compute_prediction_bounds(
            predictions=predictions,
            confidence_level=0.95
        )
        
        assert 'mean' in bounds
        assert 'lower' in bounds
        assert 'upper' in bounds
        assert 'coverage_probability' in bounds
        
        # Check dimensions
        assert bounds['mean'].shape == (n_predictions,)
        assert bounds['lower'].shape == (n_predictions,)
        assert bounds['upper'].shape == (n_predictions,)
        
        # Bounds should contain mean
        assert np.all(bounds['lower'] <= bounds['mean'])
        assert np.all(bounds['mean'] <= bounds['upper'])
    
    def test_coverage_analysis(self):
        """Test coverage analysis for uncertainty bounds."""
        cb = CertifiedBounds()
        
        np.random.seed(42)
        n_experiments = 100
        n_samples = 500
        
        # Generate multiple experiments
        true_parameter = 2.0
        confidence_levels = [0.8, 0.9, 0.95, 0.99]
        coverage_results = {cl: 0 for cl in confidence_levels}
        
        for _ in range(n_experiments):
            samples = np.random.normal(true_parameter, 0.5, n_samples)
            
            for confidence_level in confidence_levels:
                # Compute concentration bound
                concentrator = ConcentrationInequalities()
                bounds = concentrator.compute_all_bounds(
                    samples=samples,
                    confidence_level=confidence_level,
                    bound_range=(true_parameter - 3, true_parameter + 3)
                )
                
                empirical_mean = np.mean(samples)
                bound_width = bounds['hoeffding']
                
                # Check if true parameter is within bounds
                lower_bound = empirical_mean - bound_width
                upper_bound = empirical_mean + bound_width
                
                if lower_bound <= true_parameter <= upper_bound:
                    coverage_results[confidence_level] += 1
        
        # Check empirical coverage rates
        for confidence_level in confidence_levels:
            empirical_coverage = coverage_results[confidence_level] / n_experiments
            
            # Should be at least the theoretical coverage (possibly conservative)
            assert empirical_coverage >= confidence_level - 0.1, \
                f"Coverage {empirical_coverage} below theoretical {confidence_level}"
            
            print(f"Confidence {confidence_level}: Coverage {empirical_coverage}")
    
    def test_bound_tightness_comparison(self):
        """Test relative tightness of different bound methods."""
        cb = CertifiedBounds()
        
        np.random.seed(42)
        n_samples = 2000
        
        # Generate samples from known distribution
        true_mean = 1.5
        true_std = 0.8
        samples = np.random.normal(true_mean, true_std, n_samples)
        
        # Compute bounds using different methods
        concentrator = ConcentrationInequalities()
        concentration_bounds = concentrator.compute_all_bounds(
            samples=samples,
            confidence_level=0.95,
            bound_range=(true_mean - 4*true_std, true_mean + 4*true_std),
            variance_estimate=np.var(samples)
        )
        
        # Empirical confidence interval (for comparison)
        empirical_mean = np.mean(samples)
        empirical_std = np.std(samples)
        t_critical = stats.t.ppf(0.975, n_samples - 1)
        empirical_bound = t_critical * empirical_std / np.sqrt(n_samples)
        
        print("Bound comparison:")
        print(f"Empirical t-interval: ±{empirical_bound:.4f}")
        for bound_type, bound_value in concentration_bounds.items():
            print(f"{bound_type}: ±{bound_value:.4f}")
        
        # Concentration bounds should be wider (more conservative)
        for bound_value in concentration_bounds.values():
            assert bound_value >= empirical_bound, \
                "Concentration bounds should be at least as wide as empirical bounds"
    
    @pytest.mark.integration
    def test_end_to_end_uncertainty_quantification(self):
        """End-to-end test of uncertainty quantification pipeline."""
        cb = CertifiedBounds()
        
        # Simulate a complete UQ workflow
        np.random.seed(42)
        
        # 1. Generate synthetic posterior samples (as from MCMC)
        n_samples = 1000
        n_params = 3
        
        true_params = np.array([1.0, -0.5, 2.0])
        posterior_samples = np.random.multivariate_normal(
            true_params, 
            0.1 * np.eye(n_params),
            n_samples
        )
        
        # 2. Generate predictions using these parameters
        n_prediction_points = 30
        predictions = np.zeros((n_samples, n_prediction_points))
        
        for i in range(n_samples):
            params = posterior_samples[i]
            # Synthetic prediction model
            x = np.linspace(0, 1, n_prediction_points)
            predictions[i] = params[0] * np.sin(params[1] * x) + params[2] * x**2
        
        # 3. Compute parameter bounds
        param_bounds = cb.compute_parameter_bounds(
            posterior_samples=posterior_samples,
            parameter_names=[f'param_{i}' for i in range(n_params)],
            parameter_bounds=[(-5, 5)] * n_params,
            confidence_level=0.95
        )
        
        # 4. Compute prediction bounds
        pred_bounds = cb.compute_prediction_bounds(
            predictions=predictions,
            confidence_level=0.95
        )
        
        # 5. Validate results
        for i, param_name in enumerate([f'param_{i}' for i in range(n_params)]):
            bounds_dict = param_bounds[param_name]
            
            # Check that true parameter is within most bounds
            for method in ['concentration', 'empirical_ci']:
                lower = bounds_dict[method]['lower']
                upper = bounds_dict[method]['upper']
                
                # True parameter should be within bounds (most of the time)
                if not (lower <= true_params[i] <= upper):
                    print(f"Warning: {param_name} {method} bounds don't contain true value")
                    print(f"True: {true_params[i]}, Bounds: [{lower:.3f}, {upper:.3f}]")
        
        # Check prediction bounds contain true predictions
        true_predictions = true_params[0] * np.sin(true_params[1] * np.linspace(0, 1, n_prediction_points)) + \
                          true_params[2] * np.linspace(0, 1, n_prediction_points)**2
        
        within_bounds = np.logical_and(
            pred_bounds['lower'] <= true_predictions,
            true_predictions <= pred_bounds['upper']
        )
        
        coverage_rate = np.mean(within_bounds)
        print(f"Prediction coverage rate: {coverage_rate:.3f}")
        
        # Should have reasonable coverage
        assert coverage_rate >= 0.85, f"Coverage rate {coverage_rate} too low"
    
    def test_bound_calibration(self):
        """Test calibration of uncertainty bounds."""
        # Generate multiple datasets and check if bounds have correct coverage
        np.random.seed(42)
        n_experiments = 50
        n_samples = 800
        true_parameter = 0.0
        
        confidence_levels = [0.8, 0.9, 0.95]
        
        for confidence_level in confidence_levels:
            covered_count = 0
            
            for exp in range(n_experiments):
                # Generate samples
                samples = np.random.normal(true_parameter, 1.0, n_samples)
                
                # Compute concentration bound
                concentrator = ConcentrationInequalities()
                bounds = concentrator.compute_all_bounds(
                    samples=samples,
                    confidence_level=confidence_level,
                    bound_range=(-5, 5)
                )
                
                empirical_mean = np.mean(samples)
                bound_width = bounds['hoeffding']
                
                # Check coverage
                if abs(empirical_mean - true_parameter) <= bound_width:
                    covered_count += 1
            
            empirical_coverage = covered_count / n_experiments
            
            # Check calibration (allowing for some variability)
            lower_threshold = confidence_level - 0.1
            upper_threshold = min(1.0, confidence_level + 0.1)
            
            assert lower_threshold <= empirical_coverage <= upper_threshold, \
                f"Coverage {empirical_coverage} not well-calibrated for confidence {confidence_level}"
            
            print(f"Confidence {confidence_level}: Empirical coverage {empirical_coverage:.3f}")


class TestUncertaintyVisualization:
    """Test uncertainty visualization components."""
    
    def test_uncertainty_plot_data_preparation(self):
        """Test preparation of data for uncertainty plots."""
        cb = CertifiedBounds()
        
        np.random.seed(42)
        n_samples = 500
        n_points = 20
        
        # Generate ensemble predictions
        x_values = np.linspace(0, 1, n_points)
        predictions = np.zeros((n_samples, n_points))
        
        for i in range(n_samples):
            noise_level = 0.1 * (1 + 0.5 * x_values)  # Heteroscedastic noise
            predictions[i] = np.sin(2 * np.pi * x_values) + np.random.normal(0, noise_level)
        
        # Compute bounds
        bounds = cb.compute_prediction_bounds(predictions, confidence_level=0.95)
        
        # Check data is properly formatted for plotting
        plot_data = {
            'x': x_values,
            'mean': bounds['mean'],
            'lower': bounds['lower'],
            'upper': bounds['upper'],
            'std': np.std(predictions, axis=0)
        }
        
        # Validate plot data
        assert len(plot_data['x']) == n_points
        assert len(plot_data['mean']) == n_points
        assert len(plot_data['lower']) == n_points
        assert len(plot_data['upper']) == n_points
        assert np.all(plot_data['lower'] <= plot_data['upper'])
        assert np.all(plot_data['std'] >= 0)
    
    @pytest.mark.slow
    def test_coverage_visualization_data(self):
        """Test data preparation for coverage analysis visualization."""
        # Generate coverage analysis data
        np.random.seed(42)
        
        confidence_levels = np.linspace(0.5, 0.99, 20)
        n_experiments = 30
        true_param = 1.0
        
        coverage_data = []
        
        for conf_level in confidence_levels:
            covered = 0
            
            for _ in range(n_experiments):
                samples = np.random.normal(true_param, 0.5, 200)
                empirical_mean = np.mean(samples)
                
                # Simple concentration bound
                bound_width = np.sqrt(-np.log((1 - conf_level)/2) / (2 * len(samples)))
                
                if abs(empirical_mean - true_param) <= bound_width:
                    covered += 1
            
            empirical_coverage = covered / n_experiments
            coverage_data.append({
                'confidence_level': conf_level,
                'empirical_coverage': empirical_coverage,
                'theoretical_coverage': conf_level
            })
        
        # Check that coverage data is reasonable
        coverage_array = np.array([item['empirical_coverage'] for item in coverage_data])
        theoretical_array = np.array([item['theoretical_coverage'] for item in coverage_data])
        
        # Empirical coverage should generally increase with confidence level
        assert np.corrcoef(coverage_array, theoretical_array)[0, 1] > 0.8
        
        # Most coverage rates should be above theoretical (conservative bounds)
        above_theoretical = np.mean(coverage_array >= theoretical_array - 0.05)
        assert above_theoretical >= 0.7