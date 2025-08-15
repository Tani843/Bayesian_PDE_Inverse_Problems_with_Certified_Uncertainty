"""
Complete Workflow Demonstration

This example demonstrates the full Bayesian PDE inverse problem workflow
including data generation, parameter estimation, uncertainty quantification,
and comprehensive visualization.

Problem: 2D elliptic diffusion equation
-∇·(D(x,y) ∇u) + R u = S(x,y) in Ω = [0,1]²
u = 0 on ∂Ω

Parameters to estimate: D (diffusion coefficient), R (reaction coefficient)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import sys
import os
from pathlib import Path

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from bayesian_pde_solver.pde_solvers import FiniteDifferenceSolver
from bayesian_pde_solver.bayesian_inference import (
    InverseSolver, GaussianPrior, GaussianLikelihood, 
    MetropolisHastings, PosteriorAnalysis
)
from bayesian_pde_solver.uncertainty_quantification import (
    ConcentrationBounds, PACBayesBounds
)
from bayesian_pde_solver.visualization import (
    setup_matplotlib_style, SolutionPlotter
)
from bayesian_pde_solver.config import ConfigManager, get_config_by_name
from bayesian_pde_solver.utils import generate_synthetic_data, normalize_data


class CompleteDemoSolver:
    """
    Complete demonstration of Bayesian PDE inverse problem solving.
    """
    
    def __init__(self, config_name: str = "elliptic_2d"):
        """Initialize with configuration."""
        print("🔧 Initializing Bayesian PDE solver...")
        
        # Load configuration
        config_dict = get_config_by_name(config_name)
        self.config = ConfigManager(config_dict)
        
        # Set up forward solver
        self.forward_solver = FiniteDifferenceSolver(
            domain_bounds=tuple(self.config.pde.domain_bounds),
            mesh_size=tuple(self.config.pde.mesh_size),
            pde_type=self.config.pde.pde_type,
            scheme=self.config.pde.scheme
        )
        
        # Set up visualization
        setup_matplotlib_style(self.config.visualization.style)
        self.plotter = SolutionPlotter(
            style=self.config.visualization.style,
            figure_size=self.config.visualization.figure_size
        )
        
        # Storage for results
        self.results = {}
        
        print(f"✅ Solver initialized with {self.config.pde.dimension}D {self.config.pde.pde_type} PDE")
        print(f"   Domain: {self.config.pde.domain_bounds}")
        print(f"   Mesh size: {self.config.pde.mesh_size}")
    
    def define_problem(self):
        """Define the specific PDE problem and true parameters."""
        print("\n📊 Defining PDE problem...")
        
        # True parameter values (what we want to recover)
        self.true_parameters = {
            'diffusion': 1.5,      # Diffusion coefficient
            'reaction': 0.2,       # Reaction coefficient  
            'source': self.source_function  # Source function
        }
        
        # Boundary conditions (homogeneous Dirichlet)
        self.boundary_conditions = {
            "left": {"type": "dirichlet", "value": 0.0},
            "right": {"type": "dirichlet", "value": 0.0}, 
            "bottom": {"type": "dirichlet", "value": 0.0},
            "top": {"type": "dirichlet", "value": 0.0}
        }
        
        print(f"✅ Problem defined:")
        print(f"   True diffusion coefficient: {self.true_parameters['diffusion']}")
        print(f"   True reaction coefficient: {self.true_parameters['reaction']}")
        print(f"   Source: Gaussian peak at domain center")
        
    def source_function(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Define source term as Gaussian peak."""
        return 2.0 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.05)
    
    def generate_data(self):
        """Generate synthetic observation data."""
        print("\n🎲 Generating synthetic observation data...")
        
        # Solve forward problem with true parameters to create synthetic data
        X, Y = self.forward_solver.mesh["X"], self.forward_solver.mesh["Y"]
        source_field = self.source_function(X, Y)
        
        true_pde_params = {
            'diffusion': np.full_like(X, self.true_parameters['diffusion']),
            'reaction': np.full_like(X, self.true_parameters['reaction']),
            'source': source_field
        }
        
        # Get true solution
        self.true_solution = self.forward_solver.solve(true_pde_params, self.boundary_conditions)
        
        # Generate observation points (random locations)
        n_obs = 150
        domain_bounds = self.forward_solver.domain_bounds
        observation_points = np.random.uniform(
            low=[domain_bounds[0], domain_bounds[2]],
            high=[domain_bounds[1], domain_bounds[3]],
            size=(n_obs, 2)
        )
        
        # Extract true values at observation points
        true_observations = self.forward_solver.compute_observables(
            self.true_solution, observation_points
        )
        
        # Add noise
        noise_std = 0.02
        noise = np.random.normal(0, noise_std, n_obs)
        observations = true_observations + noise
        
        # Store data
        self.observation_data = {
            'points': observation_points,
            'values': observations,
            'true_values': true_observations,
            'noise': noise,
            'noise_std': noise_std
        }
        
        print(f"✅ Generated {n_obs} observations")
        print(f"   Noise level: {noise_std:.3f} (SNR ≈ {np.std(true_observations)/noise_std:.1f})")
        print(f"   Observation range: [{np.min(observations):.3f}, {np.max(observations):.3f}]")
    
    def setup_bayesian_inverse_problem(self):
        """Set up the Bayesian inverse problem."""
        print("\n🔬 Setting up Bayesian inverse problem...")
        
        # Define prior distributions  
        self.prior = GaussianPrior(
            parameter_names=['diffusion', 'reaction'],
            means=[1.0, 0.1],  # Initial guess (biased from true values)
            covariances=[[0.25, 0.0], [0.0, 0.01]]  # Prior uncertainty
        )
        
        # Define likelihood
        self.likelihood = GaussianLikelihood(
            noise_std=self.observation_data['noise_std']
        )
        
        # Create custom inverse solver 
        class CustomInverseSolver(InverseSolver):
            def log_posterior(self, parameters, boundary_conditions):
                diffusion, reaction = parameters
                
                # Prior check
                log_prior = self.prior.log_prob(parameters)
                if not np.isfinite(log_prior):
                    return -np.inf
                
                # Parameter bounds check
                if diffusion <= 0 or reaction < 0:
                    return -np.inf
                
                try:
                    # Set up PDE parameters
                    X, Y = self.forward_solver.mesh["X"], self.forward_solver.mesh["Y"]
                    pde_params = {
                        'diffusion': np.full_like(X, diffusion),
                        'reaction': np.full_like(X, reaction),
                        'source': source_function(X, Y)
                    }
                    
                    # Solve forward problem
                    solution = self.forward_solver.solve(pde_params, boundary_conditions)
                    
                    # Compute observables
                    predicted = self.forward_solver.compute_observables(
                        solution, self.observation_points
                    )
                    
                    # Compute likelihood
                    log_like = self.likelihood.log_prob(self.observations, predicted)
                    
                    return log_prior + log_like
                    
                except Exception as e:
                    return -np.inf
        
        # Create source function accessible to solver
        def source_function(x, y):
            return self.source_function(x, y)
        
        # Initialize inverse solver
        self.inverse_solver = CustomInverseSolver(
            forward_solver=self.forward_solver,
            prior=self.prior,
            likelihood=self.likelihood,
            observation_points=self.observation_data['points'],
            observations=self.observation_data['values'],
            noise_std=self.observation_data['noise_std']
        )
        
        print(f"✅ Bayesian inverse problem configured")
        print(f"   Parameters: {self.prior.parameter_names}")
        print(f"   Prior means: {dict(zip(self.prior.parameter_names, self.prior.means))}")
        print(f"   Prior stds: {dict(zip(self.prior.parameter_names, np.sqrt(np.diag(self.prior.covariances))))}")
    
    def find_map_estimate(self):
        """Find Maximum A Posteriori (MAP) estimate."""
        print("\n🎯 Finding MAP estimate...")
        
        start_time = time.time()
        
        # Find MAP estimate
        map_result = self.inverse_solver.find_map_estimate(
            self.boundary_conditions,
            initial_guess=np.array([1.0, 0.1]),
            method="L-BFGS-B"
        )
        
        elapsed = time.time() - start_time
        
        if map_result['success']:
            self.results['map'] = map_result
            est_diffusion, est_reaction = map_result['map_estimate']
            
            print(f"✅ MAP estimation completed in {elapsed:.2f} seconds")
            print(f"   Estimated diffusion: {est_diffusion:.4f} (true: {self.true_parameters['diffusion']})")
            print(f"   Estimated reaction: {est_reaction:.4f} (true: {self.true_parameters['reaction']})")
            print(f"   Log posterior: {map_result['log_posterior']:.2f}")
            
            # Compute estimation errors
            diff_error = abs(est_diffusion - self.true_parameters['diffusion'])
            react_error = abs(est_reaction - self.true_parameters['reaction'])
            print(f"   Absolute errors: diffusion={diff_error:.4f}, reaction={react_error:.4f}")
        else:
            print(f"❌ MAP estimation failed: {map_result['message']}")
    
    def run_mcmc_sampling(self):
        """Run MCMC sampling for posterior exploration."""
        print("\n🔗 Running MCMC sampling...")
        
        start_time = time.time()
        
        # MCMC parameters from config
        mcmc_params = self.config.get_mcmc_params()
        
        # Run MCMC
        mcmc_result = self.inverse_solver.sample_posterior_mcmc(
            self.boundary_conditions,
            sampler_type=mcmc_params['sampler_type'],
            n_samples=mcmc_params['n_samples'],
            n_burn=mcmc_params['n_burn'],
            n_thin=mcmc_params['n_thin'],
            initial_state=self.results['map']['map_estimate'] if 'map' in self.results else None
        )
        
        elapsed = time.time() - start_time
        
        self.results['mcmc'] = mcmc_result
        self.samples = mcmc_result['samples']
        
        print(f"✅ MCMC sampling completed in {elapsed:.2f} seconds")
        print(f"   Effective samples: {mcmc_result['n_effective']}")
        print(f"   Acceptance rate: {mcmc_result['acceptance_rate']:.3f}")
        
        # Compute posterior statistics
        post_means = np.mean(self.samples, axis=0)
        post_stds = np.std(self.samples, axis=0)
        
        print(f"   Posterior means: diffusion={post_means[0]:.4f}, reaction={post_means[1]:.4f}")
        print(f"   Posterior stds: diffusion={post_stds[0]:.4f}, reaction={post_stds[1]:.4f}")
    
    def analyze_posterior(self):
        """Perform comprehensive posterior analysis."""
        print("\n📈 Analyzing posterior distribution...")
        
        # Create posterior analysis object
        self.posterior_analysis = PosteriorAnalysis(
            samples=self.samples,
            parameter_names=self.prior.parameter_names,
            confidence_levels=[0.68, 0.95],
            compute_correlations=True
        )
        
        # Print summary
        self.posterior_analysis.print_summary()
        
        # Store results
        self.results['posterior_analysis'] = self.posterior_analysis
        
        print("✅ Posterior analysis completed")
    
    def compute_certified_bounds(self):
        """Compute certified uncertainty bounds."""
        print("\n🔒 Computing certified uncertainty bounds...")
        
        # Concentration bounds
        conc_bounds = ConcentrationBounds(
            confidence_level=self.config.uncertainty.confidence_level,
            inequality_type=self.config.uncertainty.concentration_inequality
        )
        
        # PAC-Bayes bounds  
        pac_bounds = PACBayesBounds(
            confidence_level=self.config.uncertainty.confidence_level,
            bound_type=self.config.uncertainty.pac_bayes_bound
        )
        
        bounds_results = {}
        
        for i, param_name in enumerate(self.prior.parameter_names):
            param_samples = self.samples[:, i]
            
            # Concentration bounds
            if self.config.uncertainty.concentration_inequality == "hoeffding":
                # Reasonable parameter bounds for Hoeffding
                param_bounds = (0.1, 3.0) if param_name == 'diffusion' else (0.0, 1.0)
                bounds = conc_bounds.compute_bounds(param_samples, data_range=param_bounds)
            else:
                bounds = conc_bounds.compute_bounds(param_samples, variance_bound=1.0)
            
            bounds_results[param_name] = {
                'concentration_bounds': bounds,
                'empirical_mean': np.mean(param_samples),
                'empirical_std': np.std(param_samples),
                'true_value': self.true_parameters[param_name],
                'coverage': (bounds[0] <= self.true_parameters[param_name] <= bounds[1])
            }
        
        self.results['certified_bounds'] = bounds_results
        
        # Print results
        print("✅ Certified bounds computed:")
        for param_name, results in bounds_results.items():
            bounds = results['concentration_bounds']
            true_val = results['true_value']
            covered = "✓" if results['coverage'] else "✗"
            print(f"   {param_name}: [{bounds[0]:.4f}, {bounds[1]:.4f}] {covered} (true: {true_val:.4f})")
    
    def create_visualizations(self, save_path: str = "./results/"):
        """Create comprehensive visualizations."""
        print("\n🎨 Creating visualizations...")
        
        # Create output directory
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. True solution plot
        print("   📊 Plotting true solution...")
        fig1 = self.plotter.plot_2d_solution(
            self.true_solution,
            self.forward_solver.mesh,
            title="True PDE Solution",
            colormap="viridis"
        )
        self.plotter.save_figure(fig1, save_dir / "true_solution")
        plt.close(fig1)
        
        # 2. Observation data plot
        print("   📍 Plotting observation data...")
        fig2, ax = plt.subplots(figsize=self.config.visualization.figure_size)
        
        # Background: true solution
        X, Y = self.forward_solver.mesh["X"], self.forward_solver.mesh["Y"]
        solution_2d = self.true_solution.reshape(X.shape)
        im = ax.contourf(X, Y, solution_2d, levels=20, cmap='viridis', alpha=0.6)
        
        # Scatter plot of observations
        scatter = ax.scatter(
            self.observation_data['points'][:, 0],
            self.observation_data['points'][:, 1], 
            c=self.observation_data['values'],
            s=30, cmap='plasma', edgecolors='black', linewidth=0.5
        )
        
        ax.set_xlabel("x")
        ax.set_ylabel("y") 
        ax.set_title("Observation Data on True Solution")
        ax.set_aspect('equal')
        
        # Colorbars
        cbar1 = plt.colorbar(im, ax=ax, alpha=0.6)
        cbar1.set_label("True Solution")
        cbar2 = plt.colorbar(scatter, ax=ax)
        cbar2.set_label("Observations")
        
        plt.tight_layout()
        self.plotter.save_figure(fig2, save_dir / "observations")
        plt.close(fig2)
        
        # 3. Posterior distributions
        print("   📊 Plotting posterior distributions...")
        fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, param_name in enumerate(self.prior.parameter_names):
            ax = axes[i]
            param_samples = self.samples[:, i]
            true_val = self.true_parameters[param_name]
            
            # Posterior histogram
            ax.hist(param_samples, bins=50, density=True, alpha=0.7, 
                   color='skyblue', label='Posterior samples')
            
            # True value
            ax.axvline(true_val, color='red', linestyle='--', linewidth=2,
                      label=f'True value: {true_val:.3f}')
            
            # Posterior mean
            post_mean = np.mean(param_samples)
            ax.axvline(post_mean, color='orange', linestyle='-', linewidth=2,
                      label=f'Posterior mean: {post_mean:.3f}')
            
            # MAP estimate
            if 'map' in self.results:
                map_est = self.results['map']['map_estimate'][i]
                ax.axvline(map_est, color='green', linestyle=':', linewidth=2,
                          label=f'MAP: {map_est:.3f}')
            
            # Certified bounds
            if 'certified_bounds' in self.results:
                bounds = self.results['certified_bounds'][param_name]['concentration_bounds']
                ax.axvspan(bounds[0], bounds[1], alpha=0.2, color='gray',
                          label=f'95% Certified bounds')
            
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior Distribution: {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.plotter.save_figure(fig3, save_dir / "posterior_distributions")
        plt.close(fig3)
        
        # 4. MCMC trace plots
        print("   🔗 Plotting MCMC traces...")
        fig4, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for i, param_name in enumerate(self.prior.parameter_names):
            ax = axes[i]
            param_samples = self.samples[:, i]
            true_val = self.true_parameters[param_name]
            
            ax.plot(param_samples, alpha=0.8, linewidth=0.5, color='blue')
            ax.axhline(true_val, color='red', linestyle='--', alpha=0.7,
                      label=f'True value: {true_val:.3f}')
            ax.set_ylabel(param_name.replace('_', ' ').title())
            ax.set_title(f'MCMC Trace: {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        self.plotter.save_figure(fig4, save_dir / "mcmc_traces")
        plt.close(fig4)
        
        # 5. Corner plot (parameter correlations)
        print("   📊 Creating corner plot...")
        fig5, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        param_names = self.prior.parameter_names
        
        # Diagonal: marginal distributions
        for i in range(2):
            ax = axes[i, i]
            param_samples = self.samples[:, i]
            ax.hist(param_samples, bins=30, density=True, alpha=0.7, color='skyblue')
            ax.axvline(self.true_parameters[param_names[i]], color='red', linestyle='--')
            ax.set_xlabel(param_names[i])
            ax.set_ylabel('Density')
        
        # Off-diagonal: joint distributions
        ax = axes[1, 0]
        ax.scatter(self.samples[:, 0], self.samples[:, 1], alpha=0.6, s=1, color='blue')
        ax.scatter(self.true_parameters[param_names[0]], 
                  self.true_parameters[param_names[1]], 
                  color='red', s=100, marker='*', label='True values')
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.legend()
        
        # Hide upper triangle
        axes[0, 1].set_visible(False)
        
        plt.tight_layout()
        self.plotter.save_figure(fig5, save_dir / "corner_plot")
        plt.close(fig5)
        
        print(f"✅ All visualizations saved to {save_dir}")
    
    def generate_summary_report(self, save_path: str = "./results/"):
        """Generate comprehensive summary report."""
        print("\n📋 Generating summary report...")
        
        save_dir = Path(save_path)
        report_file = save_dir / "summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BAYESIAN PDE INVERSE PROBLEM - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Problem specification
            f.write("PROBLEM SPECIFICATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"PDE type: {self.config.pde.pde_type}\n")
            f.write(f"Dimension: {self.config.pde.dimension}D\n")
            f.write(f"Domain: {self.config.pde.domain_bounds}\n")
            f.write(f"Mesh size: {self.config.pde.mesh_size}\n")
            f.write(f"Unknown parameters: {', '.join(self.prior.parameter_names)}\n\n")
            
            # True parameters
            f.write("TRUE PARAMETERS\n")
            f.write("-" * 40 + "\n")
            for name, value in self.true_parameters.items():
                if name != 'source':
                    f.write(f"{name}: {value}\n")
            f.write("\n")
            
            # Observation data
            f.write("OBSERVATION DATA\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of observations: {len(self.observation_data['values'])}\n")
            f.write(f"Noise standard deviation: {self.observation_data['noise_std']}\n")
            f.write(f"SNR: {np.std(self.observation_data['true_values'])/self.observation_data['noise_std']:.2f}\n\n")
            
            # MAP results
            if 'map' in self.results:
                f.write("MAP ESTIMATION RESULTS\n")
                f.write("-" * 40 + "\n")
                map_est = self.results['map']['map_estimate']
                for i, name in enumerate(self.prior.parameter_names):
                    true_val = self.true_parameters[name]
                    est_val = map_est[i]
                    error = abs(est_val - true_val)
                    rel_error = error / abs(true_val) * 100
                    f.write(f"{name}: {est_val:.6f} (true: {true_val:.6f}, error: {error:.6f}, rel_error: {rel_error:.2f}%)\n")
                f.write(f"Log posterior: {self.results['map']['log_posterior']:.2f}\n\n")
            
            # MCMC results
            if 'mcmc' in self.results:
                f.write("MCMC SAMPLING RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Sampler: {self.config.mcmc.sampler_type}\n")
                f.write(f"Total samples: {self.results['mcmc']['n_effective']}\n")
                f.write(f"Acceptance rate: {self.results['mcmc']['acceptance_rate']:.3f}\n")
                f.write(f"Runtime: {self.results['mcmc']['elapsed_time']:.2f} seconds\n\n")
            
            # Posterior statistics
            if 'posterior_analysis' in self.results:
                f.write("POSTERIOR STATISTICS\n")
                f.write("-" * 40 + "\n")
                analysis = self.results['posterior_analysis']
                for name in self.prior.parameter_names:
                    true_val = self.true_parameters[name]
                    mean_val = analysis.parameter_means[name]
                    std_val = analysis.parameter_stds[name]
                    ci_95 = analysis.credible_intervals[0.95][name]
                    ess = analysis.effective_sample_sizes[name]
                    
                    f.write(f"{name}:\n")
                    f.write(f"  True value: {true_val:.6f}\n")
                    f.write(f"  Posterior mean: {mean_val:.6f} ± {std_val:.6f}\n") 
                    f.write(f"  95% credible interval: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]\n")
                    f.write(f"  Effective sample size: {ess:.0f}\n")
                    
                    # Check if true value is in credible interval
                    in_ci = ci_95[0] <= true_val <= ci_95[1]
                    f.write(f"  True value in 95% CI: {'Yes' if in_ci else 'No'}\n\n")
            
            # Certified bounds
            if 'certified_bounds' in self.results:
                f.write("CERTIFIED UNCERTAINTY BOUNDS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Method: {self.config.uncertainty.concentration_inequality} inequality\n")
                f.write(f"Confidence level: {self.config.uncertainty.confidence_level}\n\n")
                
                for name, results in self.results['certified_bounds'].items():
                    bounds = results['concentration_bounds']
                    true_val = results['true_value']
                    covered = results['coverage']
                    
                    f.write(f"{name}:\n")
                    f.write(f"  Certified bounds: [{bounds[0]:.6f}, {bounds[1]:.6f}]\n")
                    f.write(f"  True value: {true_val:.6f}\n")
                    f.write(f"  Coverage: {'Yes' if covered else 'No'}\n\n")
        
        print(f"✅ Summary report saved to {report_file}")
    
    def run_complete_analysis(self, save_path: str = "./results/"):
        """Run the complete Bayesian analysis workflow."""
        print("🚀 Starting complete Bayesian PDE inverse problem analysis")
        print("="*80)
        
        overall_start = time.time()
        
        try:
            # Step 1: Define problem
            self.define_problem()
            
            # Step 2: Generate synthetic data
            self.generate_data()
            
            # Step 3: Set up Bayesian inverse problem
            self.setup_bayesian_inverse_problem()
            
            # Step 4: Find MAP estimate
            self.find_map_estimate()
            
            # Step 5: Run MCMC sampling
            self.run_mcmc_sampling()
            
            # Step 6: Analyze posterior
            self.analyze_posterior()
            
            # Step 7: Compute certified bounds
            self.compute_certified_bounds()
            
            # Step 8: Create visualizations
            self.create_visualizations(save_path)
            
            # Step 9: Generate report
            self.generate_summary_report(save_path)
            
            overall_time = time.time() - overall_start
            
            print("\n" + "="*80)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total runtime: {overall_time:.2f} seconds")
            print(f"📁 Results saved to: {Path(save_path).absolute()}")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution function."""
    # Create and run complete demo
    demo = CompleteDemoSolver(config_name="elliptic_2d")
    success = demo.run_complete_analysis(save_path="./demo_results/")
    
    if success:
        print("\n🔍 Key findings:")
        if 'map' in demo.results:
            map_est = demo.results['map']['map_estimate']
            true_vals = [demo.true_parameters['diffusion'], demo.true_parameters['reaction']]
            print(f"   Parameter recovery accuracy:")
            for i, name in enumerate(demo.prior.parameter_names):
                error = abs(map_est[i] - true_vals[i]) / true_vals[i] * 100
                print(f"     {name}: {error:.1f}% relative error")
        
        if 'certified_bounds' in demo.results:
            coverage_count = sum(1 for r in demo.results['certified_bounds'].values() if r['coverage'])
            total_params = len(demo.results['certified_bounds'])
            print(f"   Certified bound coverage: {coverage_count}/{total_params} parameters")
        
        print("\n📊 Generated visualizations:")
        print("   - True PDE solution")
        print("   - Observation data overlay") 
        print("   - Posterior distributions")
        print("   - MCMC trace plots")
        print("   - Parameter correlation (corner plot)")
        
        print("\n📋 Summary report available in demo_results/summary_report.txt")


if __name__ == "__main__":
    main()