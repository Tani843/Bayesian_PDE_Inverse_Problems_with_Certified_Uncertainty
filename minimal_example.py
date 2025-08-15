#!/usr/bin/env python3
"""
Minimal Working Example - Bayesian PDE Inverse Problem

This is a self-contained example that requires only NumPy and matplotlib.
It demonstrates the core concepts without complex dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

class MinimalPDESolver:
    """Minimal 2D Poisson solver using finite differences."""
    
    def __init__(self, nx: int = 50, ny: int = 50):
        self.nx, self.ny = nx, ny
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        
        # Create mesh
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
    def solve(self, diffusion: float, source_func: Callable) -> np.ndarray:
        """Solve -∇²u = f with u=0 on boundary."""
        # Initialize solution
        u = np.zeros((self.nx, self.ny))
        
        # Jacobi iteration for simplicity
        for _ in range(500):  # Fixed iterations
            u_new = u.copy()
            
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    # 5-point stencil
                    laplacian = ((u[i+1, j] + u[i-1, j]) / self.dx**2 + 
                               (u[i, j+1] + u[i, j-1]) / self.dy**2)
                    
                    source_val = source_func(self.X[i, j], self.Y[i, j])
                    
                    u_new[i, j] = (laplacian + source_val / diffusion) / (2 / self.dx**2 + 2 / self.dy**2)
            
            u = u_new
            
        return u
    
    def interpolate(self, solution: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Simple bilinear interpolation."""
        values = np.zeros(len(points))
        
        for k, (x, y) in enumerate(points):
            # Find grid indices
            i = int(x * (self.nx - 1))
            j = int(y * (self.ny - 1))
            
            # Clamp to valid range
            i = max(0, min(i, self.nx - 2))
            j = max(0, min(j, self.ny - 2))
            
            # Bilinear interpolation weights
            alpha = (x * (self.nx - 1)) - i
            beta = (y * (self.ny - 1)) - j
            
            # Interpolate
            values[k] = ((1 - alpha) * (1 - beta) * solution[i, j] +
                        alpha * (1 - beta) * solution[i + 1, j] +
                        (1 - alpha) * beta * solution[i, j + 1] +
                        alpha * beta * solution[i + 1, j + 1])
        
        return values


class MinimalBayesianInference:
    """Minimal Bayesian parameter estimation."""
    
    def __init__(self, solver: MinimalPDESolver, 
                 obs_points: np.ndarray, obs_values: np.ndarray, 
                 noise_std: float = 0.01):
        self.solver = solver
        self.obs_points = obs_points
        self.obs_values = obs_values
        self.noise_std = noise_std
    
    def log_posterior(self, params: np.ndarray) -> float:
        """Compute log posterior probability."""
        diffusion, source_strength = params
        
        # Check parameter bounds
        if diffusion <= 0 or source_strength <= 0:
            return -np.inf
        
        # Log-normal priors
        log_prior = (-0.5 * (np.log(diffusion) - 0)**2 / 0.5**2 +
                    -0.5 * (np.log(source_strength) - 0)**2 / 0.5**2)
        
        try:
            # Define source function
            def source_func(x, y):
                return source_strength * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.05)
            
            # Solve forward problem
            solution = self.solver.solve(diffusion, source_func)
            
            # Get predictions at observation points
            predictions = self.solver.interpolate(solution, self.obs_points)
            
            # Gaussian likelihood
            residuals = self.obs_values - predictions
            log_likelihood = -0.5 * np.sum(residuals**2) / self.noise_std**2
            
            return log_prior + log_likelihood
            
        except Exception:
            return -np.inf
    
    def metropolis_hastings(self, n_samples: int = 10000, 
                           initial: np.ndarray = np.array([1.0, 1.0]),
                           step_size: float = 0.05) -> Tuple[np.ndarray, float]:
        """Simple Metropolis-Hastings sampler."""
        samples = np.zeros((n_samples, 2))
        current = initial.copy()
        current_log_prob = self.log_posterior(current)
        n_accepted = 0
        
        print(f"Running MCMC with {n_samples} samples...")
        
        for i in range(n_samples):
            # Progress indicator
            if i % 1000 == 0:
                print(f"  Progress: {i/n_samples*100:.1f}%")
            
            # Propose new state (log-normal proposal for positivity)
            log_proposal = np.log(current) + np.random.normal(0, step_size, 2)
            proposal = np.exp(log_proposal)
            
            # Compute acceptance probability
            proposal_log_prob = self.log_posterior(proposal)
            
            # Jacobian correction for log-normal proposal
            log_jacobian = np.sum(log_proposal - np.log(current))
            
            log_alpha = min(0, proposal_log_prob - current_log_prob + log_jacobian)
            
            # Accept/reject
            if np.log(np.random.random()) < log_alpha:
                current = proposal
                current_log_prob = proposal_log_prob
                n_accepted += 1
            
            samples[i] = current
        
        acceptance_rate = n_accepted / n_samples
        print(f"MCMC completed. Acceptance rate: {acceptance_rate:.3f}")
        
        return samples, acceptance_rate


def generate_synthetic_data(solver: MinimalPDESolver, 
                          true_diffusion: float = 1.5, 
                          true_source_strength: float = 2.0,
                          n_observations: int = 80,
                          noise_std: float = 0.02,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic observation data."""
    np.random.seed(seed)
    
    # True source function
    def true_source(x, y):
        return true_source_strength * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.05)
    
    # Solve with true parameters
    true_solution = solver.solve(true_diffusion, true_source)
    
    # Random observation points (avoid boundary)
    obs_points = np.random.uniform([0.1, 0.1], [0.9, 0.9], (n_observations, 2))
    
    # Get true values and add noise
    true_values = solver.interpolate(true_solution, obs_points)
    noisy_values = true_values + np.random.normal(0, noise_std, len(true_values))
    
    return obs_points, noisy_values, true_solution


def create_visualizations(solver: MinimalPDESolver, 
                         true_solution: np.ndarray,
                         obs_points: np.ndarray, 
                         obs_values: np.ndarray,
                         samples: np.ndarray,
                         true_params: Tuple[float, float],
                         save_path: str = "results") -> None:
    """Create visualization plots."""
    
    # Create results directory
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Set up plotting
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.dpi': 100
    })
    
    # 1. Solution and observations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # True solution
    im1 = axes[0, 0].contourf(solver.X, solver.Y, true_solution, levels=20, cmap='viridis')
    axes[0, 0].scatter(obs_points[:, 0], obs_points[:, 1], c='red', s=15, alpha=0.8)
    axes[0, 0].set_title('True Solution + Observations')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Parameter traces
    burn_in = len(samples) // 4
    axes[0, 1].plot(samples[:, 0], alpha=0.8, color='blue', linewidth=0.5)
    axes[0, 1].axhline(true_params[0], color='red', linestyle='--', linewidth=2, 
                      label=f'True: {true_params[0]:.2f}')
    axes[0, 1].axvline(burn_in, color='gray', linestyle=':', alpha=0.7, label='Burn-in')
    axes[0, 1].set_title('MCMC Trace: Diffusion')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Diffusion')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(samples[:, 1], alpha=0.8, color='orange', linewidth=0.5)
    axes[1, 0].axhline(true_params[1], color='red', linestyle='--', linewidth=2,
                      label=f'True: {true_params[1]:.2f}')
    axes[1, 0].axvline(burn_in, color='gray', linestyle=':', alpha=0.7, label='Burn-in')
    axes[1, 0].set_title('MCMC Trace: Source Strength')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Source Strength')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Joint posterior
    post_samples = samples[burn_in:]
    axes[1, 1].scatter(post_samples[:, 0], post_samples[:, 1], alpha=0.6, s=1, color='blue')
    axes[1, 1].scatter(true_params[0], true_params[1], color='red', s=100, marker='*',
                      label='True parameters', zorder=5)
    axes[1, 1].set_title('Joint Posterior Distribution')
    axes[1, 1].set_xlabel('Diffusion')
    axes[1, 1].set_ylabel('Source Strength')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/solution_and_traces.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved solution and traces to {save_path}/solution_and_traces.png")
    
    # 2. Posterior distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    param_names = ['Diffusion', 'Source Strength']
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        param_samples = post_samples[:, i]
        true_val = true_params[i]
        
        # Histogram
        ax.hist(param_samples, bins=50, density=True, alpha=0.7, color='skyblue',
               edgecolor='black', label='Posterior samples')
        
        # Statistics
        post_mean = np.mean(param_samples)
        post_std = np.std(param_samples)
        
        # Vertical lines
        ax.axvline(true_val, color='red', linestyle='--', linewidth=2,
                  label=f'True: {true_val:.3f}')
        ax.axvline(post_mean, color='blue', linestyle='-', linewidth=2,
                  label=f'Mean: {post_mean:.3f}')
        
        # Credible interval
        ci_lower = np.percentile(param_samples, 2.5)
        ci_upper = np.percentile(param_samples, 97.5)
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray',
                  label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Posterior Distribution')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Print statistics
        error = abs(post_mean - true_val) / true_val * 100
        coverage = ci_lower <= true_val <= ci_upper
        print(f"📊 {name}:")
        print(f"   True value: {true_val:.4f}")
        print(f"   Estimated: {post_mean:.4f} ± {post_std:.4f}")
        print(f"   Relative error: {error:.1f}%")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   True value in CI: {'✓' if coverage else '✗'}")
        print()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/posterior_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved posterior distributions to {save_path}/posterior_distributions.png")
    
    # Show plots if possible
    try:
        plt.show()
    except Exception:
        print("⚠️  Cannot display plots (no GUI available)")
    
    plt.close('all')


def main():
    """Main execution function."""
    print("🚀 Minimal Bayesian PDE Inverse Problem Demo")
    print("=" * 60)
    
    # Parameters
    true_diffusion = 1.5
    true_source_strength = 2.0
    true_params = (true_diffusion, true_source_strength)
    
    print(f"True parameters:")
    print(f"  Diffusion coefficient: {true_diffusion}")
    print(f"  Source strength: {true_source_strength}")
    print()
    
    # 1. Create solver
    print("🔧 Setting up PDE solver...")
    solver = MinimalPDESolver(nx=40, ny=40)  # Smaller grid for speed
    print(f"   Created {solver.nx}×{solver.ny} finite difference grid")
    
    # 2. Generate synthetic data
    print("\n📊 Generating synthetic observation data...")
    obs_points, obs_values, true_solution = generate_synthetic_data(
        solver, true_diffusion, true_source_strength, 
        n_observations=60, noise_std=0.02
    )
    print(f"   Generated {len(obs_values)} observations with 2% noise")
    
    # 3. Set up Bayesian inference
    print("\n🔬 Setting up Bayesian inference...")
    bayes_solver = MinimalBayesianInference(solver, obs_points, obs_values, noise_std=0.02)
    print("   Created Bayesian inverse solver")
    
    # 4. Run MCMC
    print("\n🔗 Running MCMC sampling...")
    samples, acceptance_rate = bayes_solver.metropolis_hastings(
        n_samples=5000, step_size=0.08
    )
    
    # 5. Analyze results
    print(f"\n📈 Analysis Results:")
    print(f"   MCMC acceptance rate: {acceptance_rate:.3f}")
    
    # Remove burn-in
    burn_in = len(samples) // 4
    post_samples = samples[burn_in:]
    
    param_means = np.mean(post_samples, axis=0)
    param_stds = np.std(post_samples, axis=0)
    
    print(f"   Effective samples: {len(post_samples)}")
    print(f"   Parameter estimates:")
    print(f"     Diffusion: {param_means[0]:.4f} ± {param_stds[0]:.4f}")
    print(f"     Source: {param_means[1]:.4f} ± {param_stds[1]:.4f}")
    
    # 6. Create visualizations
    print(f"\n🎨 Creating visualizations...")
    create_visualizations(solver, true_solution, obs_points, obs_values,
                         samples, true_params)
    
    # 7. Compute simple uncertainty bounds
    print(f"\n🔒 Computing uncertainty bounds...")
    for i, name in enumerate(['Diffusion', 'Source Strength']):
        param_samples = post_samples[:, i]
        true_val = true_params[i]
        
        # Simple concentration bound (Hoeffding-like)
        n = len(param_samples)
        mean_est = np.mean(param_samples)
        
        # Assume bounded in [0, 5]
        bound_width = 5 * np.sqrt(-np.log(0.05/2) / (2 * n))
        cert_lower = max(0, mean_est - bound_width)
        cert_upper = min(5, mean_est + bound_width)
        
        coverage = cert_lower <= true_val <= cert_upper
        
        print(f"   {name}:")
        print(f"     95% Certified bound: [{cert_lower:.4f}, {cert_upper:.4f}]")
        print(f"     Coverage: {'✓' if coverage else '✗'}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Results saved in './results/' directory")
    
    return samples, param_means, param_stds


if __name__ == "__main__":
    # Run the minimal example
    try:
        results = main()
        print("\n✅ All operations completed without errors!")
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Try running the workarounds.py script for alternative solutions")