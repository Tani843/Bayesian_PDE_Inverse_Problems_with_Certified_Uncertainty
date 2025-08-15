"""
Common Workarounds for Bayesian PDE Project Issues

This file provides solutions to common problems and alternative implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# =============================================================================
# WORKAROUND 1: Import Issues
# =============================================================================

def fix_import_paths():
    """Add project paths to sys.path for imports."""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Add subpackages
    subpackages = [
        'bayesian_pde_solver',
        'bayesian_pde_solver/pde_solvers',
        'bayesian_pde_solver/bayesian_inference',
        'bayesian_pde_solver/uncertainty_quantification',
        'bayesian_pde_solver/visualization',
        'bayesian_pde_solver/utils'
    ]
    
    for subpkg in subpackages:
        path = project_root / subpkg
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

# =============================================================================
# WORKAROUND 2: Missing Dependencies
# =============================================================================

def install_missing_packages():
    """Install missing packages with pip."""
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas',
        'scikit-learn', 'tqdm', 'pyyaml', 'h5py'
    ]
    
    import subprocess
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} not found, installing...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {package}")

# =============================================================================
# WORKAROUND 3: Simplified Implementations
# =============================================================================

class SimplifiedFiniteDifferenceSolver:
    """Simplified PDE solver for basic functionality."""
    
    def __init__(self, nx=50, ny=50, domain=(0, 1, 0, 1)):
        self.nx, self.ny = nx, ny
        self.x_min, self.x_max, self.y_min, self.y_max = domain
        
        # Create mesh
        self.dx = (self.x_max - self.x_min) / (nx - 1)
        self.dy = (self.y_max - self.y_min) / (ny - 1)
        
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        self.coordinates = np.column_stack([self.X.ravel(), self.Y.ravel()])
    
    def solve_poisson_2d(self, diffusion=1.0, source_func=None):
        """Solve 2D Poisson equation: -∇²u = f with u=0 on boundary."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve
        
        n_total = self.nx * self.ny
        
        # Index mapping
        def idx(i, j):
            return i * self.ny + j
        
        # Build system matrix
        A = sp.lil_matrix((n_total, n_total))
        b = np.zeros(n_total)
        
        # Interior points
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                k = idx(i, j)
                
                # 5-point stencil
                A[k, idx(i-1, j)] = -diffusion / self.dx**2
                A[k, idx(i+1, j)] = -diffusion / self.dx**2
                A[k, idx(i, j-1)] = -diffusion / self.dy**2
                A[k, idx(i, j+1)] = -diffusion / self.dy**2
                A[k, k] = 2 * diffusion * (1/self.dx**2 + 1/self.dy**2)
                
                # Source term
                if source_func:
                    b[k] = source_func(self.X[i, j], self.Y[i, j])
        
        # Boundary conditions (Dirichlet u = 0)
        for i in [0, self.nx-1]:
            for j in range(self.ny):
                k = idx(i, j)
                A[k, :] = 0
                A[k, k] = 1
                b[k] = 0
        
        for j in [0, self.ny-1]:
            for i in range(1, self.nx-1):
                k = idx(i, j)
                A[k, :] = 0
                A[k, k] = 1
                b[k] = 0
        
        # Solve
        solution = spsolve(A.tocsc(), b)
        return solution.reshape(self.nx, self.ny)


class SimplifiedBayesianSolver:
    """Simplified Bayesian parameter estimation."""
    
    def __init__(self, forward_solver, obs_points, obs_values, noise_std=0.01):
        self.solver = forward_solver
        self.obs_points = obs_points
        self.obs_values = obs_values
        self.noise_std = noise_std
    
    def log_posterior(self, params):
        """Compute log posterior probability."""
        diffusion, source_strength = params
        
        # Prior (log-normal for positivity)
        if diffusion <= 0 or source_strength <= 0:
            return -np.inf
        
        log_prior = (-0.5 * (np.log(diffusion) - np.log(1.0))**2 / 0.25 +
                    -0.5 * (np.log(source_strength) - np.log(1.0))**2 / 0.25)
        
        # Likelihood
        try:
            # Define source function
            def source_func(x, y):
                return source_strength * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
            
            # Solve forward problem
            solution = self.solver.solve_poisson_2d(diffusion, source_func)
            
            # Interpolate at observation points
            from scipy.interpolate import RegularGridInterpolator
            x = np.linspace(0, 1, self.solver.nx)
            y = np.linspace(0, 1, self.solver.ny)
            interp = RegularGridInterpolator((x, y), solution, method='linear', 
                                           bounds_error=False, fill_value=0)
            predictions = interp(self.obs_points)
            
            # Gaussian likelihood
            residuals = self.obs_values - predictions
            log_likelihood = -0.5 * np.sum(residuals**2) / self.noise_std**2
            
            return log_prior + log_likelihood
            
        except Exception:
            return -np.inf
    
    def metropolis_hastings(self, n_samples=5000, step_size=0.1, initial=[1.0, 1.0]):
        """Simple Metropolis-Hastings sampler."""
        samples = np.zeros((n_samples, 2))
        current = np.array(initial)
        current_log_prob = self.log_posterior(current)
        n_accepted = 0
        
        for i in range(n_samples):
            # Propose
            proposal = current + np.random.normal(0, step_size, 2)
            proposal_log_prob = self.log_posterior(proposal)
            
            # Accept/reject
            if (proposal_log_prob > current_log_prob or 
                np.random.random() < np.exp(proposal_log_prob - current_log_prob)):
                current = proposal
                current_log_prob = proposal_log_prob
                n_accepted += 1
            
            samples[i] = current
        
        return samples, n_accepted / n_samples


# =============================================================================
# WORKAROUND 4: Matplotlib Issues
# =============================================================================

def fix_matplotlib_backend():
    """Fix matplotlib backend issues."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        print("✅ Using Agg backend for matplotlib")
        return True
    except Exception as e:
        print(f"❌ Matplotlib backend issue: {e}")
        return False

def simple_plot(x, y, title="Simple Plot", save_path=None):
    """Create a simple plot with minimal dependencies."""
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    try:
        plt.show()
    except:
        print("Cannot display plot (no GUI available)")
    
    plt.close()

# =============================================================================
# WORKAROUND 5: Memory Issues
# =============================================================================

def reduce_memory_usage(array_list, dtype=np.float32):
    """Reduce memory usage by converting to lower precision."""
    converted = []
    for arr in array_list:
        if isinstance(arr, np.ndarray):
            converted.append(arr.astype(dtype))
        else:
            converted.append(arr)
    return converted

def chunked_computation(data, func, chunk_size=1000):
    """Process large arrays in chunks to avoid memory issues."""
    n_data = len(data)
    results = []
    
    for i in range(0, n_data, chunk_size):
        chunk = data[i:i+chunk_size]
        chunk_result = func(chunk)
        results.append(chunk_result)
    
    return np.concatenate(results) if results else np.array([])

# =============================================================================
# WORKAROUND 6: Quick Demo Function
# =============================================================================

def run_quick_demo():
    """Run a quick demonstration with simplified components."""
    print("🚀 Running Quick Demo with Workarounds")
    print("="*50)
    
    # Fix paths and check dependencies
    fix_import_paths()
    
    # Create simplified solver
    solver = SimplifiedFiniteDifferenceSolver(nx=30, ny=30)
    print(f"✅ Created solver with {solver.nx}x{solver.ny} grid")
    
    # Generate synthetic data
    np.random.seed(42)
    n_obs = 50
    obs_points = np.random.uniform([0.1, 0.1], [0.9, 0.9], (n_obs, 2))
    
    # True parameters
    true_diffusion = 1.5
    true_source_strength = 2.0
    
    def true_source(x, y):
        return true_source_strength * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
    
    # Solve with true parameters
    true_solution = solver.solve_poisson_2d(true_diffusion, true_source)
    
    # Generate observations
    from scipy.interpolate import RegularGridInterpolator
    x = np.linspace(0, 1, solver.nx)
    y = np.linspace(0, 1, solver.ny)
    interp = RegularGridInterpolator((x, y), true_solution, method='linear')
    
    true_obs = interp(obs_points)
    noisy_obs = true_obs + np.random.normal(0, 0.02, len(true_obs))
    
    print(f"✅ Generated {n_obs} synthetic observations")
    
    # Set up Bayesian solver
    bayes_solver = SimplifiedBayesianSolver(solver, obs_points, noisy_obs)
    
    # Run MCMC
    print("🔗 Running MCMC sampling...")
    samples, acceptance_rate = bayes_solver.metropolis_hastings(n_samples=2000)
    
    print(f"✅ MCMC completed with {acceptance_rate:.3f} acceptance rate")
    
    # Analyze results
    param_means = np.mean(samples[500:], axis=0)  # Remove burn-in
    param_stds = np.std(samples[500:], axis=0)
    
    print("\n📊 Results:")
    print(f"True diffusion: {true_diffusion:.3f}")
    print(f"Estimated diffusion: {param_means[0]:.3f} ± {param_stds[0]:.3f}")
    print(f"True source strength: {true_source_strength:.3f}")
    print(f"Estimated source strength: {param_means[1]:.3f} ± {param_stds[1]:.3f}")
    
    # Simple visualization
    try:
        if fix_matplotlib_backend():
            # Plot true solution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.contourf(solver.X, solver.Y, true_solution, levels=20, cmap='viridis')
            plt.colorbar(label='Solution')
            plt.title('True Solution')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Plot observation locations
            plt.scatter(obs_points[:, 0], obs_points[:, 1], c='red', s=20, alpha=0.8)
            
            # Plot parameter traces
            plt.subplot(1, 3, 2)
            plt.plot(samples[:, 0], alpha=0.8, label='Diffusion')
            plt.axhline(true_diffusion, color='red', linestyle='--', label='True')
            plt.xlabel('Iteration')
            plt.ylabel('Diffusion')
            plt.legend()
            plt.title('MCMC Trace: Diffusion')
            
            plt.subplot(1, 3, 3)
            plt.plot(samples[:, 1], alpha=0.8, label='Source Strength')
            plt.axhline(true_source_strength, color='red', linestyle='--', label='True')
            plt.xlabel('Iteration')
            plt.ylabel('Source Strength')
            plt.legend()
            plt.title('MCMC Trace: Source')
            
            plt.tight_layout()
            plt.savefig('quick_demo_results.png', dpi=150, bbox_inches='tight')
            print("✅ Results saved to 'quick_demo_results.png'")
            
            try:
                plt.show()
            except:
                print("⚠️  Cannot display plots (no GUI)")
            
            plt.close()
    
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
    
    print("\n🎉 Quick demo completed successfully!")
    return samples, param_means, param_stds

# =============================================================================
# WORKAROUND 7: Alternative Package Imports
# =============================================================================

def try_alternative_imports():
    """Try alternative package imports if main ones fail."""
    alternatives = {
        'scipy': ['numpy'],
        'matplotlib': ['numpy'],
        'seaborn': ['matplotlib'],
        'pandas': ['numpy'],
        'sklearn': ['numpy', 'scipy']
    }
    
    available = {}
    
    for package, fallbacks in alternatives.items():
        try:
            available[package] = __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available, trying fallbacks...")
            for fallback in fallbacks:
                try:
                    available[fallback] = __import__(fallback)
                    print(f"✅ Using {fallback} as fallback")
                    break
                except ImportError:
                    continue
            else:
                print(f"⚠️  No fallback found for {package}")
    
    return available

if __name__ == "__main__":
    print("Bayesian PDE Project - Workarounds and Quick Demo")
    print("="*60)
    
    # Check what's available
    print("Checking package availability...")
    available_packages = try_alternative_imports()
    
    # Run quick demo if basic packages are available
    if 'numpy' in available_packages:
        print("\n" + "="*60)
        run_quick_demo()
    else:
        print("❌ Cannot run demo - numpy not available")
        print("Please install numpy: pip install numpy")