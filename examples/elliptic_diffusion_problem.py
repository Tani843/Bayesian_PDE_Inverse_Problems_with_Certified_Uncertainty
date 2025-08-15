"""
Example: Elliptic Diffusion Parameter Estimation

This example demonstrates Bayesian parameter estimation for an elliptic PDE
with unknown diffusion coefficient and source term strength.

Problem: -∇·(D(x,y) ∇u) + R u = S(x,y) f(x,y) in Ω = [0,1]²
         u = 0 on ∂Ω

Unknown parameters: D (diffusion coefficient), S (source strength)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
import os

# Add the package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bayesian_pde_solver.pde_solvers import FiniteDifferenceSolver
from bayesian_pde_solver.bayesian_inference import (
    InverseSolver, GaussianPrior, GaussianLikelihood
)
from bayesian_pde_solver.uncertainty_quantification import (
    CertifiedBounds, ConcentrationBounds, PACBayesBounds
)
from bayesian_pde_solver.visualization import (
    setup_matplotlib_style, SolutionPlotter, PosteriorPlotter, UncertaintyPlotter
)


def true_diffusion_field(x, y):
    """True spatially varying diffusion coefficient."""
    return 1.0 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def source_function(x, y, source_strength=1.0):
    """Source term with unknown strength parameter."""
    return source_strength * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)


def generate_synthetic_data(nx=50, ny=50, noise_std=0.01, random_seed=42):
    """
    Generate synthetic observation data.
    
    Args:
        nx, ny: Grid resolution
        noise_std: Observation noise standard deviation
        random_seed: Random seed for reproducibility
        
    Returns:
        observation_points: Points where observations are made
        observations: Noisy observation data
        true_solution: True solution field
        true_parameters: True parameter values
    """
    np.random.seed(random_seed)
    
    # True parameter values
    true_diffusion_param = 1.0  # Average diffusion coefficient
    true_source_strength = 1.5
    true_parameters = [true_diffusion_param, true_source_strength]
    
    # Set up forward solver
    solver = FiniteDifferenceSolver(
        domain_bounds=(0, 1, 0, 1),
        mesh_size=(nx, ny),
        pde_type="elliptic"
    )
    
    # Define spatially varying parameters
    X, Y = solver.mesh["X"], solver.mesh["Y"]
    diffusion_field = true_diffusion_field(X, Y) * true_diffusion_param
    source_field = source_function(X, Y, true_source_strength)
    
    # Set up PDE parameters
    pde_params = {
        "diffusion": diffusion_field,
        "reaction": 0.1,  # Small reaction term
        "source": source_field
    }
    
    # Boundary conditions (homogeneous Dirichlet)
    boundary_conditions = {
        "left": {"type": "dirichlet", "value": 0.0},
        "right": {"type": "dirichlet", "value": 0.0},
        "bottom": {"type": "dirichlet", "value": 0.0},
        "top": {"type": "dirichlet", "value": 0.0}
    }
    
    # Solve forward problem
    true_solution = solver.solve(pde_params, boundary_conditions)
    
    # Generate observation points (random subset)
    n_obs = 100
    mesh_indices = np.random.choice(len(solver.dof_coordinates), n_obs, replace=False)
    observation_points = solver.dof_coordinates[mesh_indices]
    
    # Extract true values at observation points
    true_values = true_solution[mesh_indices]
    
    # Add noise
    observations = true_values + np.random.normal(0, noise_std, n_obs)
    
    return observation_points, observations, true_solution, true_parameters, solver


def run_bayesian_inference(observation_points, observations, noise_std=0.01):
    """
    Run Bayesian parameter estimation.
    
    Args:
        observation_points: Observation locations
        observations: Observation data
        noise_std: Noise standard deviation
        
    Returns:
        inverse_solver: Configured inverse solver
        results: Dictionary of results
    """
    print("Setting up Bayesian inverse problem...")
    
    # Set up forward solver (same as for data generation)
    solver = FiniteDifferenceSolver(
        domain_bounds=(0, 1, 0, 1),
        mesh_size=(50, 50),
        pde_type="elliptic"
    )
    
    # Define prior distributions
    prior = GaussianPrior(
        parameter_names=['diffusion_param', 'source_strength'],
        means=[1.2, 1.0],  # Slightly biased initial guess
        covariances=[[0.25, 0.0], [0.0, 0.25]]  # Moderate uncertainty
    )
    
    # Define likelihood
    likelihood = GaussianLikelihood(noise_std=noise_std)
    
    # Create inverse solver with custom forward model
    def parameter_to_fields(params):
        """Convert parameters to spatially varying fields."""
        diffusion_param, source_strength = params
        X, Y = solver.mesh["X"], solver.mesh["Y"]
        
        diffusion_field = true_diffusion_field(X, Y) * diffusion_param
        source_field = source_function(X, Y, source_strength)
        
        return {
            "diffusion": diffusion_field,
            "reaction": 0.1,
            "source": source_field
        }
    
    # Create custom inverse solver
    class CustomInverseSolver(InverseSolver):
        def log_posterior(self, parameters, boundary_conditions):
            # Convert parameters to fields
            param_dict = parameter_to_fields(parameters)
            
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
                return -np.inf
    
    inverse_solver = CustomInverseSolver(
        forward_solver=solver,
        prior=prior,
        likelihood=likelihood,
        observation_points=observation_points,
        observations=observations,
        noise_std=noise_std
    )
    
    # Boundary conditions
    boundary_conditions = {
        "left": {"type": "dirichlet", "value": 0.0},
        "right": {"type": "dirichlet", "value": 0.0},
        "bottom": {"type": "dirichlet", "value": 0.0},
        "top": {"type": "dirichlet", "value": 0.0}
    }
    
    results = {}
    
    # Find MAP estimate
    print("Finding MAP estimate...")
    map_result = inverse_solver.find_map_estimate(
        boundary_conditions,
        initial_guess=np.array([1.0, 1.0])
    )
    results['map'] = map_result
    
    # MCMC sampling
    print("Running MCMC sampling...")
    mcmc_result = inverse_solver.sample_posterior_mcmc(
        boundary_conditions,
        sampler_type="metropolis_hastings",
        n_samples=5000,
        n_burn=1000,
        n_thin=5
    )
    results['mcmc'] = mcmc_result
    
    # Posterior analysis
    print("Analyzing posterior...")
    posterior_analysis = inverse_solver.analyze_posterior()
    results['posterior_analysis'] = posterior_analysis
    
    return inverse_solver, results


def compute_certified_bounds(samples, confidence_level=0.95):
    """
    Compute certified uncertainty bounds.
    
    Args:
        samples: MCMC samples
        confidence_level: Confidence level for bounds
        
    Returns:
        bounds_results: Dictionary of certified bounds
    """
    print("Computing certified uncertainty bounds...")
    
    bounds_results = {}
    
    # Concentration bounds
    conc_bounds = ConcentrationBounds(confidence_level=confidence_level)
    
    for i, param_name in enumerate(['diffusion_param', 'source_strength']):
        param_samples = samples[:, i]
        
        # Hoeffding bounds
        hoeff_bounds = conc_bounds.compute_bounds(
            param_samples,
            data_range=(0.1, 3.0)  # Reasonable parameter range
        )
        
        # Bernstein bounds
        conc_bounds.inequality_type = "bernstein"
        bern_bounds = conc_bounds.compute_bounds(
            param_samples,
            variance_bound=1.0
        )
        
        bounds_results[param_name] = {
            'hoeffding': hoeff_bounds,
            'bernstein': bern_bounds,
            'empirical_mean': np.mean(param_samples),
            'empirical_std': np.std(param_samples),
            'empirical_quantiles': {
                'q025': np.quantile(param_samples, 0.025),
                'q975': np.quantile(param_samples, 0.975)
            }
        }
    
    return bounds_results


def create_visualizations(inverse_solver, results, true_parameters, save_path="./figures/"):
    """
    Create comprehensive visualizations.
    
    Args:
        inverse_solver: Trained inverse solver
        results: Results dictionary
        true_parameters: True parameter values
        save_path: Path to save figures
    """
    print("Creating visualizations...")
    
    # Set up plotting style
    setup_matplotlib_style('academic')
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Posterior distributions
    samples = results['mcmc']['samples']
    param_names = inverse_solver.parameter_names
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (ax, param_name) in enumerate(zip(axes, param_names)):
        # Posterior histogram
        ax.hist(samples[:, i], bins=50, density=True, alpha=0.7, 
               color='skyblue', label='Posterior samples')
        
        # True value
        ax.axvline(true_parameters[i], color='red', linestyle='--', linewidth=2,
                  label=f'True value: {true_parameters[i]:.2f}')
        
        # MAP estimate
        if results['map']['success']:
            ax.axvline(results['map']['map_estimate'][i], color='green', 
                      linestyle=':', linewidth=2,
                      label=f'MAP: {results["map"]["map_estimate"][i]:.2f}')
        
        # Posterior mean
        post_mean = np.mean(samples[:, i])
        ax.axvline(post_mean, color='orange', linestyle='-', linewidth=2,
                  label=f'Posterior mean: {post_mean:.2f}')
        
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior Distribution: {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'posterior_distributions.png'), dpi=300)
    plt.show()
    
    # 2. Corner plot (joint distributions)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Marginal distributions
    for i in range(2):
        axes[i, i].hist(samples[:, i], bins=30, density=True, alpha=0.7)
        axes[i, i].axvline(true_parameters[i], color='red', linestyle='--')
        axes[i, i].set_xlabel(param_names[i])
        axes[i, i].set_ylabel('Density')
    
    # Joint distribution
    axes[1, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
    axes[1, 0].scatter(true_parameters[0], true_parameters[1], 
                      color='red', s=100, marker='*', label='True values')
    axes[1, 0].set_xlabel(param_names[0])
    axes[1, 0].set_ylabel(param_names[1])
    axes[1, 0].legend()
    
    # Hide upper triangle
    axes[0, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'corner_plot.png'), dpi=300)
    plt.show()
    
    # 3. MCMC trace plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    for i, (ax, param_name) in enumerate(zip(axes, param_names)):
        ax.plot(samples[:, i], alpha=0.8, linewidth=0.5)
        ax.axhline(true_parameters[i], color='red', linestyle='--', 
                  label=f'True value: {true_parameters[i]:.2f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(param_name.replace('_', ' ').title())
        ax.set_title(f'MCMC Trace: {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mcmc_traces.png'), dpi=300)
    plt.show()
    
    # 4. Convergence diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Running average
    for i, param_name in enumerate(param_names):
        running_mean = np.cumsum(samples[:, i]) / np.arange(1, len(samples) + 1)
        axes[0].plot(running_mean, label=f'{param_name}')
        axes[0].axhline(true_parameters[i], color=f'C{i}', linestyle='--', alpha=0.7)
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Running Average')
    axes[0].set_title('Convergence of Running Average')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Autocorrelation (simplified)
    for i, param_name in enumerate(param_names):
        lags = np.arange(0, min(200, len(samples) // 4))
        autocorr = [np.corrcoef(samples[:-lag if lag > 0 else None, i], 
                               samples[lag:, i])[0, 1] if lag > 0 else 1.0 
                   for lag in lags]
        axes[1].plot(lags, autocorr, label=f'{param_name}')
    
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title('Autocorrelation Function')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'convergence_diagnostics.png'), dpi=300)
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Bayesian PDE Inverse Problems: Elliptic Diffusion Example")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic observation data...")
    observation_points, observations, true_solution, true_parameters, solver = generate_synthetic_data(
        nx=50, ny=50, noise_std=0.01
    )
    
    print(f"   - Generated {len(observations)} observations")
    print(f"   - True parameters: {dict(zip(['diffusion_param', 'source_strength'], true_parameters))}")
    print(f"   - Observation noise std: 0.01")
    
    # Run Bayesian inference
    print("\n2. Running Bayesian parameter estimation...")
    inverse_solver, results = run_bayesian_inference(observation_points, observations)
    
    # Compute certified bounds
    print("\n3. Computing certified uncertainty bounds...")
    bounds_results = compute_certified_bounds(results['mcmc']['samples'])
    
    # Print results summary
    print("\n4. Results Summary:")
    print("-" * 40)
    
    samples = results['mcmc']['samples']
    param_names = inverse_solver.parameter_names
    
    for i, param_name in enumerate(param_names):
        print(f"\n{param_name.upper()}:")
        print(f"   True value:        {true_parameters[i]:.4f}")
        if results['map']['success']:
            print(f"   MAP estimate:      {results['map']['map_estimate'][i]:.4f}")
        print(f"   Posterior mean:    {np.mean(samples[:, i]):.4f}")
        print(f"   Posterior std:     {np.std(samples[:, i]):.4f}")
        print(f"   95% Credible:      [{np.quantile(samples[:, i], 0.025):.4f}, {np.quantile(samples[:, i], 0.975):.4f}]")
        
        bounds = bounds_results[param_name]
        print(f"   Hoeffding bounds:  [{bounds['hoeffding'][0]:.4f}, {bounds['hoeffding'][1]:.4f}]")
        print(f"   Bernstein bounds:  [{bounds['bernstein'][0]:.4f}, {bounds['bernstein'][1]:.4f}]")
    
    print(f"\nMCMC Summary:")
    print(f"   Acceptance rate:   {results['mcmc']['acceptance_rate']:.3f}")
    print(f"   Effective samples: {results['mcmc']['n_effective']}")
    print(f"   Runtime:           {results['mcmc']['elapsed_time']:.2f} seconds")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(inverse_solver, results, true_parameters)
    
    print("\nExample completed successfully!")
    print("Check the './figures/' directory for generated plots.")


if __name__ == "__main__":
    main()