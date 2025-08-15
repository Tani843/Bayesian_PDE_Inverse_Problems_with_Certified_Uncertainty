"""
Test suite for Bayesian inference methods.

Tests MCMC sampling, variational inference, convergence diagnostics,
and posterior accuracy against known analytical distributions.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

from bayesian_pde_solver.bayesian_inference.mcmc_sampler import MCMCSampler
from bayesian_pde_solver.bayesian_inference.variational_inference import VariationalInference, MeanFieldVI


class TestMCMCSampler:
    """Test suite for MCMC sampling methods."""
    
    def test_initialization(self, simple_posterior_function):
        """Test MCMC sampler initialization."""
        sampler = MCMCSampler(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        assert sampler.parameter_dim == 2
        assert sampler.sampler_type == "metropolis_hastings"
        assert callable(sampler.log_posterior_fn)
    
    def test_gaussian_posterior_sampling(self, test_tolerances):
        """Test MCMC sampling of a known Gaussian posterior."""
        # Define 2D Gaussian posterior N(μ=[1,2], Σ=[[2,0.5],[0.5,1]])
        true_mean = np.array([1.0, 2.0])
        true_cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        inv_cov = np.linalg.inv(true_cov)
        
        def gaussian_log_posterior(params):
            diff = params - true_mean
            return -0.5 * diff.T @ inv_cov @ diff
        
        sampler = MCMCSampler(
            log_posterior_fn=gaussian_log_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        # Run MCMC
        result = sampler.sample(
            n_samples=5000,
            initial_state=np.array([0.0, 0.0]),
            step_size=0.5
        )
        
        samples = result['samples']
        
        # Remove burn-in
        burn_in = 1000
        samples_clean = samples[burn_in:]
        
        # Check convergence statistics
        empirical_mean = np.mean(samples_clean, axis=0)
        empirical_cov = np.cov(samples_clean.T)
        
        # Compare to true distribution
        assert_allclose(empirical_mean, true_mean, rtol=test_tolerances['medium'])
        assert_allclose(empirical_cov, true_cov, rtol=test_tolerances['loose'])
        
        # Check acceptance rate is reasonable
        acceptance_rate = result['acceptance_rate']
        assert 0.2 < acceptance_rate < 0.7, f"Acceptance rate {acceptance_rate} not optimal"
    
    def test_different_samplers(self, simple_posterior_function):
        """Test different MCMC sampling algorithms."""
        samplers_to_test = ["metropolis_hastings", "random_walk"]
        
        for sampler_type in samplers_to_test:
            sampler = MCMCSampler(
                log_posterior_fn=simple_posterior_function,
                parameter_dim=2,
                sampler_type=sampler_type
            )
            
            result = sampler.sample(
                n_samples=1000,
                initial_state=np.zeros(2),
                step_size=0.1
            )
            
            assert result['samples'].shape == (1000, 2)
            assert 0.0 < result['acceptance_rate'] < 1.0
            assert np.all(np.isfinite(result['samples']))
    
    def test_adaptive_step_size(self, simple_posterior_function):
        """Test adaptive step size adjustment."""
        sampler = MCMCSampler(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        # Enable step size adaptation
        result = sampler.sample(
            n_samples=2000,
            initial_state=np.zeros(2),
            step_size=0.1,
            adapt_step_size=True,
            target_acceptance_rate=0.4
        )
        
        # Check that step size was adjusted
        final_step_size = result.get('final_step_size', 0.1)
        initial_step_size = 0.1
        
        # Step size should have changed during adaptation
        assert final_step_size != initial_step_size or len(result['samples']) < 500
        
        # Acceptance rate should be closer to target
        acceptance_rate = result['acceptance_rate']
        assert 0.2 < acceptance_rate < 0.6
    
    @pytest.mark.slow
    def test_convergence_diagnostics(self, simple_posterior_function):
        """Test MCMC convergence diagnostics."""
        sampler = MCMCSampler(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        # Run multiple chains
        n_chains = 4
        n_samples_per_chain = 2500
        chains = []
        
        for chain_id in range(n_chains):
            np.random.seed(42 + chain_id)
            initial_state = np.random.normal(0, 1, 2)
            
            result = sampler.sample(
                n_samples=n_samples_per_chain,
                initial_state=initial_state,
                step_size=0.2
            )
            chains.append(result['samples'])
        
        # Compute R-hat diagnostic
        chains_array = np.array(chains)  # Shape: (n_chains, n_samples, n_params)
        
        # Simple R-hat computation for each parameter
        burn_in = 500
        chains_clean = chains_array[:, burn_in:, :]
        
        for param_idx in range(2):
            param_chains = chains_clean[:, :, param_idx]
            
            # Between-chain variance
            chain_means = np.mean(param_chains, axis=1)
            overall_mean = np.mean(chain_means)
            B = len(param_chains[0]) * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in param_chains])
            
            # R-hat statistic
            var_est = ((len(param_chains[0]) - 1) * W + B) / len(param_chains[0])
            r_hat = np.sqrt(var_est / W)
            
            # R-hat should be close to 1 for converged chains
            assert r_hat < 1.2, f"R-hat {r_hat} indicates poor convergence for parameter {param_idx}"
    
    def test_multimodal_distribution(self):
        """Test MCMC on multimodal distribution."""
        # Define bimodal distribution (mixture of Gaussians)
        def bimodal_log_posterior(params):
            x, y = params
            
            # Two modes at (-2, 0) and (2, 0)
            mode1 = -0.5 * ((x + 2)**2 + y**2) / 0.5
            mode2 = -0.5 * ((x - 2)**2 + y**2) / 0.5
            
            # Log-sum-exp for numerical stability
            max_val = max(mode1, mode2)
            return max_val + np.log(0.5 * (np.exp(mode1 - max_val) + np.exp(mode2 - max_val)))
        
        sampler = MCMCSampler(
            log_posterior_fn=bimodal_log_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        result = sampler.sample(
            n_samples=10000,
            initial_state=np.array([0.0, 0.0]),
            step_size=1.0  # Larger step size to help mode switching
        )
        
        samples = result['samples'][2000:]  # Remove burn-in
        
        # Check that both modes are visited
        x_samples = samples[:, 0]
        left_mode_visits = np.sum(x_samples < 0)
        right_mode_visits = np.sum(x_samples > 0)
        
        # Both modes should be visited (though not necessarily equally)
        assert left_mode_visits > len(samples) * 0.1
        assert right_mode_visits > len(samples) * 0.1
    
    def test_parameter_bounds(self, test_tolerances):
        """Test MCMC with parameter bounds/constraints."""
        # Posterior constrained to positive quadrant
        def constrained_log_posterior(params):
            x, y = params
            if x <= 0 or y <= 0:
                return -np.inf
            
            # Log-normal-like distribution
            return -0.5 * (np.log(x)**2 + np.log(y)**2)
        
        sampler = MCMCSampler(
            log_posterior_fn=constrained_log_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        result = sampler.sample(
            n_samples=3000,
            initial_state=np.array([1.0, 1.0]),
            step_size=0.3
        )
        
        samples = result['samples'][500:]  # Remove burn-in
        
        # All samples should be positive
        assert np.all(samples > 0), "Samples should respect positivity constraint"
        
        # Check that samples follow approximately log-normal distribution
        log_samples = np.log(samples)
        log_mean = np.mean(log_samples, axis=0)
        
        # Should be centered around (0, 0) in log space
        assert_allclose(log_mean, [0, 0], atol=test_tolerances['loose'])
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Invalid parameter dimension
        with pytest.raises(ValueError):
            MCMCSampler(
                log_posterior_fn=lambda x: -x**2,
                parameter_dim=0,
                sampler_type="metropolis_hastings"
            )
        
        # Invalid sampler type
        with pytest.raises(ValueError):
            MCMCSampler(
                log_posterior_fn=lambda x: -x**2,
                parameter_dim=1,
                sampler_type="invalid_sampler"
            )
        
        # Test with NaN/infinite posterior
        def problematic_posterior(params):
            if np.any(np.abs(params) > 10):
                return -np.inf
            return np.nan  # Problematic return
        
        sampler = MCMCSampler(
            log_posterior_fn=problematic_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        # Should handle NaN gracefully
        result = sampler.sample(
            n_samples=100,
            initial_state=np.zeros(2),
            step_size=0.1
        )
        
        # Should still produce samples (with low acceptance rate)
        assert result['samples'].shape == (100, 2)
        assert result['acceptance_rate'] >= 0  # Could be 0 due to NaN issues


class TestVariationalInference:
    """Test suite for variational inference methods."""
    
    def test_initialization_mean_field(self, simple_posterior_function):
        """Test mean-field VI initialization."""
        vi = VariationalInference(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        assert vi.parameter_dim == 2
        assert vi.vi_type == "mean_field"
        assert 'means' in vi.variational_params
        assert 'log_stds' in vi.variational_params
    
    def test_initialization_full_rank(self, simple_posterior_function):
        """Test full-rank VI initialization."""
        vi = VariationalInference(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            vi_type="full_rank"
        )
        
        assert vi.vi_type == "full_rank"
        assert 'mean' in vi.variational_params
        assert 'chol_cov' in vi.variational_params
    
    def test_parameter_packing_unpacking(self, simple_posterior_function):
        """Test parameter packing and unpacking."""
        vi = VariationalInference(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        # Test packing
        packed = vi._pack_parameters()
        assert len(packed) == 4  # 2 means + 2 log_stds
        
        # Modify parameters
        vi.variational_params['means'][0] = 1.5
        vi.variational_params['log_stds'][1] = -0.5
        
        # Pack and unpack
        packed_modified = vi._pack_parameters()
        vi._unpack_parameters(packed)  # Restore original
        
        # Check restoration
        assert vi.variational_params['means'][0] == 0.0
        assert vi.variational_params['log_stds'][1] == 0.0
        
        # Unpack modified
        vi._unpack_parameters(packed_modified)
        assert vi.variational_params['means'][0] == 1.5
        assert vi.variational_params['log_stds'][1] == -0.5
    
    def test_variational_sampling(self, simple_posterior_function):
        """Test sampling from variational distribution."""
        vi = VariationalInference(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        # Set known parameters
        vi.variational_params['means'] = np.array([1.0, -0.5])
        vi.variational_params['log_stds'] = np.array([0.0, np.log(2.0)])  # stds = [1.0, 2.0]
        
        # Sample
        samples = vi.sample_variational(1000)
        
        assert samples.shape == (1000, 2)
        assert np.all(np.isfinite(samples))
        
        # Check sample statistics
        sample_means = np.mean(samples, axis=0)
        sample_stds = np.std(samples, axis=0)
        
        assert_allclose(sample_means, [1.0, -0.5], atol=0.2)
        assert_allclose(sample_stds, [1.0, 2.0], atol=0.3)
    
    def test_elbo_computation(self, test_tolerances):
        """Test ELBO computation accuracy."""
        # Use known Gaussian posterior
        true_mean = np.array([0.0, 0.0])
        true_cov = np.eye(2)
        
        def gaussian_log_posterior(params):
            return stats.multivariate_normal.logpdf(params, true_mean, true_cov)
        
        vi = VariationalInference(
            log_posterior_fn=gaussian_log_posterior,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        # Set variational parameters close to true posterior
        vi.variational_params['means'] = np.array([0.1, -0.1])
        vi.variational_params['log_stds'] = np.array([0.05, -0.05])  # stds ≈ [1.05, 0.95]
        
        # Compute ELBO
        samples = vi.sample_variational(2000)
        elbo = vi.compute_elbo(samples)
        
        # ELBO should be finite and negative (since it's a lower bound on log evidence)
        assert np.isfinite(elbo)
        
        # For this simple case, we can compare to analytical KL divergence
        # KL(q||p) for diagonal Gaussian vs standard Gaussian
        q_means = vi.variational_params['means']
        q_stds = np.exp(vi.variational_params['log_stds'])
        
        analytical_kl = 0.5 * np.sum(
            q_stds**2 + q_means**2 - 1 - 2*vi.variational_params['log_stds']
        )
        
        # ELBO ≈ -KL (ignoring constant terms)
        expected_elbo_approx = -analytical_kl
        
        # Should be reasonably close (allowing for Monte Carlo error)
        assert abs(elbo - expected_elbo_approx) < 1.0
    
    @pytest.mark.slow
    def test_vi_optimization_gaussian(self, test_tolerances):
        """Test VI optimization on Gaussian posterior."""
        # Define 2D Gaussian posterior
        true_mean = np.array([2.0, -1.0])
        true_cov = np.array([[1.5, 0.3], [0.3, 0.8]])
        inv_cov = np.linalg.inv(true_cov)
        
        def gaussian_log_posterior(params):
            diff = params - true_mean
            return -0.5 * diff.T @ inv_cov @ diff
        
        vi = VariationalInference(
            log_posterior_fn=gaussian_log_posterior,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        # Optimize
        result = vi.optimize(
            n_iterations=2000,
            learning_rate=0.01,
            n_samples=50,
            optimizer="adam"
        )
        
        # Check convergence
        assert result['final_elbo'] > -10  # Should converge to reasonable ELBO
        assert len(result['elbo_history']) > 100  # Should run for reasonable iterations
        
        # Check parameter estimates
        final_means = vi.variational_params['means']
        final_stds = np.exp(vi.variational_params['log_stds'])
        
        # Means should be close to true means
        assert_allclose(final_means, true_mean, rtol=test_tolerances['loose'])
        
        # Stds should be close to true marginal standard deviations
        true_marginal_stds = np.sqrt(np.diag(true_cov))
        assert_allclose(final_stds, true_marginal_stds, rtol=test_tolerances['loose'])
    
    def test_mean_field_vi_class(self):
        """Test MeanFieldVI convenience class."""
        def simple_posterior(params):
            return -0.5 * np.sum(params**2)
        
        vi = MeanFieldVI(
            log_posterior_fn=simple_posterior,
            parameter_dim=3
        )
        
        assert vi.vi_type == "mean_field"
        assert vi.parameter_dim == 3
        
        # Test marginal distributions
        marginals = vi.get_marginal_distributions()
        assert len(marginals) == 3
        
        for marginal in marginals:
            assert 'mean' in marginal
            assert 'std' in marginal
            assert 'variance' in marginal
    
    def test_kl_divergence_computation(self, simple_posterior_function):
        """Test KL divergence computation."""
        vi = VariationalInference(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        # Set variational parameters
        vi.variational_params['means'] = np.array([0.5, -0.5])
        vi.variational_params['log_stds'] = np.array([0.1, -0.1])
        
        # Compute KL to standard Gaussian prior
        prior_mean = np.zeros(2)
        prior_cov = np.eye(2)
        
        kl_div = vi.kl_divergence_to_prior(prior_mean, prior_cov)
        
        assert np.isfinite(kl_div)
        assert kl_div >= 0  # KL divergence is non-negative
    
    def test_summary_statistics(self, simple_posterior_function):
        """Test summary statistics extraction."""
        vi = VariationalInference(
            log_posterior_fn=simple_posterior_function,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        # Set parameters
        vi.variational_params['means'] = np.array([1.0, 2.0])
        vi.variational_params['log_stds'] = np.array([0.0, np.log(0.5)])
        
        stats = vi.get_summary_statistics()
        
        assert_allclose(stats['means'], [1.0, 2.0])
        assert_allclose(stats['stds'], [1.0, 0.5])
        assert_allclose(stats['variances'], [1.0, 0.25])
        
        expected_cov = np.diag([1.0, 0.25])
        assert_allclose(stats['covariance_matrix'], expected_cov)


class TestBayesianInferenceIntegration:
    """Integration tests combining different inference methods."""
    
    @pytest.mark.integration
    def test_mcmc_vs_vi_comparison(self, test_tolerances):
        """Compare MCMC and VI on same posterior."""
        # Define test posterior
        true_mean = np.array([1.0, -0.5])
        true_cov = np.diag([2.0, 0.5])
        inv_cov = np.linalg.inv(true_cov)
        
        def test_posterior(params):
            diff = params - true_mean
            return -0.5 * diff.T @ inv_cov @ diff
        
        # MCMC sampling
        mcmc = MCMCSampler(
            log_posterior_fn=test_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        mcmc_result = mcmc.sample(
            n_samples=3000,
            initial_state=np.zeros(2),
            step_size=0.5
        )
        mcmc_samples = mcmc_result['samples'][500:]  # Remove burn-in
        mcmc_mean = np.mean(mcmc_samples, axis=0)
        mcmc_cov = np.cov(mcmc_samples.T)
        
        # Variational inference
        vi = VariationalInference(
            log_posterior_fn=test_posterior,
            parameter_dim=2,
            vi_type="mean_field"
        )
        
        vi_result = vi.optimize(
            n_iterations=1500,
            learning_rate=0.01,
            n_samples=100
        )
        
        vi_samples = vi.sample(2000)
        vi_mean = np.mean(vi_samples, axis=0)
        vi_cov = np.cov(vi_samples.T)
        
        # Both methods should give similar results
        assert_allclose(mcmc_mean, true_mean, rtol=test_tolerances['medium'])
        assert_allclose(vi_mean, true_mean, rtol=test_tolerances['medium'])
        
        # MCMC should be more accurate for covariance (VI assumes independence)
        mcmc_cov_error = np.linalg.norm(mcmc_cov - true_cov)
        vi_cov_error = np.linalg.norm(np.diag(np.diag(vi_cov)) - true_cov)  # Compare only diagonal
        
        print(f"MCMC covariance error: {mcmc_cov_error:.4f}")
        print(f"VI covariance error: {vi_cov_error:.4f}")
    
    @pytest.mark.integration 
    def test_pde_parameter_inference(self, synthetic_observations, fd_solver_2d, 
                                   dirichlet_zero_bc):
        """Test Bayesian inference for PDE parameters."""
        obs_data = synthetic_observations
        
        def log_posterior(params):
            """Log posterior for PDE parameters."""
            diffusion, source_strength = params
            
            if diffusion <= 0 or source_strength <= 0:
                return -np.inf
            
            # Log-normal priors
            log_prior = (
                -0.5 * (np.log(diffusion) - 0)**2 / 0.5**2 +
                -0.5 * (np.log(source_strength) - 0)**2 / 0.5**2
            )
            
            try:
                # Define source function
                def source_func(x, y):
                    return source_strength * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
                
                # Solve PDE
                pde_params = {
                    "diffusion": diffusion,
                    "reaction": 0.0,
                    "source": source_func
                }
                
                solution = fd_solver_2d.solve(pde_params, dirichlet_zero_bc)
                
                # Interpolate at observation points  
                predictions = fd_solver_2d._interpolate_solution(solution, obs_data["points"])
                
                # Gaussian likelihood
                residuals = obs_data["values"] - predictions
                log_likelihood = -0.5 * np.sum(residuals**2) / obs_data["noise_std"]**2
                
                return log_prior + log_likelihood
                
            except Exception:
                return -np.inf
        
        # Test both MCMC and VI
        mcmc = MCMCSampler(
            log_posterior_fn=log_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        mcmc_result = mcmc.sample(
            n_samples=1000,  # Reduced for testing
            initial_state=np.array([1.0, 1.0]),
            step_size=0.05
        )
        
        # Check results are reasonable
        samples = mcmc_result['samples'][200:]  # Remove burn-in
        param_means = np.mean(samples, axis=0)
        
        true_params = [obs_data["true_params"]["diffusion"], 
                      obs_data["true_params"]["source_strength"]]
        
        # Should be within reasonable range of true values
        rel_errors = np.abs(param_means - true_params) / true_params
        assert np.all(rel_errors < 0.5), f"Parameter estimates too far from truth: {param_means} vs {true_params}"
        
        print(f"True parameters: {true_params}")
        print(f"Estimated parameters: {param_means}")
        print(f"Relative errors: {rel_errors}")


class TestInferenceRobustness:
    """Test robustness of inference methods."""
    
    def test_ill_conditioned_posterior(self):
        """Test inference on ill-conditioned posterior."""
        # Very elongated Gaussian (high condition number)
        def ill_conditioned_posterior(params):
            x, y = params
            return -0.5 * (x**2 / 0.01 + y**2 / 100)  # Condition number = 10000
        
        # Test MCMC
        mcmc = MCMCSampler(
            log_posterior_fn=ill_conditioned_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        result = mcmc.sample(
            n_samples=2000,
            initial_state=np.zeros(2),
            step_size=0.1
        )
        
        samples = result['samples'][500:]
        
        # Should still produce reasonable samples
        assert np.all(np.isfinite(samples))
        
        # Check that samples explore both dimensions appropriately
        std_x = np.std(samples[:, 0])
        std_y = np.std(samples[:, 1])
        
        assert 0.05 < std_x < 0.2  # Should reflect narrow dimension
        assert 5 < std_y < 15      # Should reflect wide dimension
    
    def test_inference_with_noise(self):
        """Test inference robustness to computational noise."""
        def noisy_posterior(params):
            base_val = -0.5 * np.sum(params**2)
            # Add small amount of computational noise
            noise = 1e-10 * np.random.normal()
            return base_val + noise
        
        mcmc = MCMCSampler(
            log_posterior_fn=noisy_posterior,
            parameter_dim=2,
            sampler_type="metropolis_hastings"
        )
        
        result = mcmc.sample(
            n_samples=1000,
            initial_state=np.zeros(2),
            step_size=0.5
        )
        
        # Should handle noise gracefully
        assert np.all(np.isfinite(result['samples']))
        assert result['acceptance_rate'] > 0.1
    
    @pytest.mark.parametrize("dimension", [1, 3, 5])
    def test_scalability_with_dimension(self, dimension):
        """Test inference scalability with parameter dimension."""
        def high_dim_posterior(params):
            return -0.5 * np.sum(params**2)
        
        vi = VariationalInference(
            log_posterior_fn=high_dim_posterior,
            parameter_dim=dimension,
            vi_type="mean_field"
        )
        
        # VI should be more scalable than MCMC
        result = vi.optimize(
            n_iterations=500,
            learning_rate=0.05,
            n_samples=50
        )
        
        assert result['final_elbo'] > -dimension * 2  # Reasonable bound
        
        samples = vi.sample(100)
        assert samples.shape == (100, dimension)
        assert np.all(np.isfinite(samples))