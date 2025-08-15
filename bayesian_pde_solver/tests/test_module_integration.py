"""
Module integration tests within the bayesian_pde_solver package.

Tests inter-module compatibility and integration within the package structure.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add package to path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))

# Import package modules
from bayesian_pde_solver.pde_solvers.finite_difference_solver import FiniteDifferenceSolver
from bayesian_pde_solver.bayesian_inference.mcmc_sampler import MCMCSampler
from bayesian_pde_solver.uncertainty_quantification.certified_bounds import CertifiedBounds
from bayesian_pde_solver.visualization.pde_plots import PDEPlotter


class TestModuleIntegration:
    """Test integration between package modules."""
    
    def test_solver_to_inference_integration(self):
        """Test PDE solver to Bayesian inference integration."""
        # Create solver
        solver = FiniteDifferenceSolver(
            domain_bounds=(0, 1, 0, 1),
            mesh_size=(11, 11),
            pde_type="elliptic"
        )
        
        # Define posterior that uses solver
        def log_posterior(params):
            diffusion = params[0]
            if diffusion <= 0:
                return -np.inf
            
            try:
                pde_params = {
                    'diffusion': diffusion,
                    'reaction': 0.0,
                    'source': lambda x, y: 1.0
                }
                bc = {
                    "left": {"type": "dirichlet", "value": 0.0},
                    "right": {"type": "dirichlet", "value": 0.0},
                    "top": {"type": "dirichlet", "value": 0.0},
                    "bottom": {"type": "dirichlet", "value": 0.0}
                }
                
                solution = solver.solve(pde_params, bc)
                
                # Simple likelihood based on solution norm
                return -0.5 * (np.linalg.norm(solution) - 1.0)**2
                
            except Exception:
                return -np.inf
        
        # Test MCMC with solver
        mcmc = MCMCSampler(log_posterior, parameter_dim=1)
        result = mcmc.sample(n_samples=100, initial_state=np.array([1.0]), step_size=0.1)
        
        assert result['samples'].shape == (100, 1)
        assert result['acceptance_rate'] > 0
    
    def test_inference_to_uq_integration(self):
        """Test Bayesian inference to uncertainty quantification integration."""
        # Generate synthetic posterior samples
        np.random.seed(42)
        samples = np.random.normal([1.0, -0.5], [0.2, 0.3], (500, 2))
        
        # Test certified bounds computation
        cert_bounds = CertifiedBounds()
        
        bounds = cert_bounds.compute_parameter_bounds(
            posterior_samples=samples,
            parameter_names=['param1', 'param2'],
            parameter_bounds=[(-2, 3), (-2, 1)],
            confidence_level=0.95
        )
        
        assert 'param1' in bounds
        assert 'param2' in bounds
        
        for param in ['param1', 'param2']:
            assert 'concentration' in bounds[param]
            assert 'empirical_ci' in bounds[param]
    
    def test_solver_to_visualization_integration(self):
        """Test PDE solver to visualization integration."""
        # Create solver and solution
        solver = FiniteDifferenceSolver(
            domain_bounds=(0, 1, 0, 1),
            mesh_size=(11, 11),
            pde_type="elliptic"
        )
        
        pde_params = {
            'diffusion': 1.0,
            'reaction': 0.0,
            'source': lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        }
        
        bc = {
            "left": {"type": "dirichlet", "value": 0.0},
            "right": {"type": "dirichlet", "value": 0.0},
            "top": {"type": "dirichlet", "value": 0.0},
            "bottom": {"type": "dirichlet", "value": 0.0}
        }
        
        solution = solver.solve(pde_params, bc)
        
        # Test visualization
        plotter = PDEPlotter()
        
        # Reshape solution for 2D plotting
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        solution_2d = solution.reshape((11, 11))
        
        fig = plotter.plot_solution_field_2d(x, y, solution_2d)
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_package_imports(self):
        """Test that all package imports work correctly."""
        # Test main module imports
        try:
            from bayesian_pde_solver.pde_solvers import FiniteDifferenceSolver, FiniteElementSolver
            from bayesian_pde_solver.bayesian_inference import MCMCSampler, VariationalInference
            from bayesian_pde_solver.uncertainty_quantification import CertifiedBounds
            from bayesian_pde_solver.visualization import PDEPlotter, BayesianPlotter, UncertaintyPlotter
            
            # All imports should succeed
            assert True
            
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_configuration_integration(self):
        """Test configuration management integration."""
        try:
            from bayesian_pde_solver.config.config_manager import ConfigManager
            
            # Test default configuration loading
            config_manager = ConfigManager()
            default_config = config_manager.get_default_config()
            
            assert 'pde' in default_config
            assert 'mcmc' in default_config
            assert 'uncertainty' in default_config
            
        except ImportError:
            # Config manager might not be fully implemented
            pytest.skip("Configuration manager not available")


class TestModuleCompatibility:
    """Test compatibility between different module versions and configurations."""
    
    def test_solver_output_compatibility(self):
        """Test that different solvers produce compatible outputs."""
        domain = (0, 1, 0, 1)
        mesh_size = (11, 11)
        
        # Create both solver types
        fd_solver = FiniteDifferenceSolver(domain, mesh_size, "elliptic")
        
        try:
            from bayesian_pde_solver.pde_solvers.finite_element_solver import FiniteElementSolver
            fe_solver = FiniteElementSolver(domain, mesh_size, "elliptic")
            
            # Both should have compatible interfaces
            assert hasattr(fd_solver, 'solve')
            assert hasattr(fe_solver, 'solve')
            assert hasattr(fd_solver, 'coordinates')
            assert hasattr(fe_solver, 'dof_coordinates')
            
        except ImportError:
            pytest.skip("Finite element solver not available")
    
    def test_inference_method_compatibility(self):
        """Test that different inference methods have compatible interfaces."""
        def simple_posterior(params):
            return -0.5 * np.sum(params**2)
        
        # MCMC
        mcmc = MCMCSampler(simple_posterior, parameter_dim=2)
        mcmc_result = mcmc.sample(n_samples=50, initial_state=np.zeros(2), step_size=0.1)
        
        # VI
        try:
            from bayesian_pde_solver.bayesian_inference.variational_inference import VariationalInference
            vi = VariationalInference(simple_posterior, parameter_dim=2)
            vi_result = vi.optimize(n_iterations=50, n_samples=20)
            vi_samples = vi.sample(50)
            
            # Both should produce samples with same shape
            assert mcmc_result['samples'].shape == vi_samples.shape
            
        except ImportError:
            pytest.skip("Variational inference not available")
    
    def test_data_format_consistency(self):
        """Test that data formats are consistent across modules."""
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        n_params = 2
        
        samples = np.random.normal(0, 1, (n_samples, n_params))
        
        # Test that different modules accept the same data format
        try:
            cert_bounds = CertifiedBounds()
            
            # Should accept samples in standard format
            bounds = cert_bounds.compute_parameter_bounds(
                posterior_samples=samples,
                parameter_names=['p1', 'p2'],
                parameter_bounds=[(-3, 3), (-3, 3)],
                confidence_level=0.95
            )
            
            assert isinstance(bounds, dict)
            
        except Exception as e:
            pytest.fail(f"Data format compatibility issue: {e}")


class TestModuleErrorHandling:
    """Test error handling across module boundaries."""
    
    def test_solver_error_propagation(self):
        """Test that solver errors are properly propagated."""
        solver = FiniteDifferenceSolver(
            domain_bounds=(0, 1, 0, 1),
            mesh_size=(5, 5),
            pde_type="elliptic"
        )
        
        # Invalid parameters should raise appropriate errors
        invalid_params = {
            'diffusion': -1.0,  # Negative diffusion
            'reaction': 0.0,
            'source': lambda x, y: 1.0
        }
        
        bc = {
            "left": {"type": "dirichlet", "value": 0.0},
            "right": {"type": "dirichlet", "value": 0.0},
            "top": {"type": "dirichlet", "value": 0.0},
            "bottom": {"type": "dirichlet", "value": 0.0}
        }
        
        with pytest.raises((ValueError, RuntimeError)):
            solver.solve(invalid_params, bc)
    
    def test_inference_error_handling(self):
        """Test inference error handling with problematic posteriors."""
        def problematic_posterior(params):
            # Returns NaN for some inputs
            if np.any(np.abs(params) > 5):
                return np.nan
            return -0.5 * np.sum(params**2)
        
        mcmc = MCMCSampler(problematic_posterior, parameter_dim=2)
        
        # Should handle NaN gracefully
        result = mcmc.sample(n_samples=50, initial_state=np.zeros(2), step_size=0.1)
        
        # Should complete without crashing
        assert result['samples'].shape == (50, 2)
        assert 0 <= result['acceptance_rate'] <= 1
    
    def test_visualization_error_handling(self):
        """Test visualization error handling with invalid data."""
        plotter = PDEPlotter()
        
        # Test with mismatched dimensions
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 11)  # Different size
        solution = np.random.random((10, 10))
        
        # Should handle gracefully or raise informative error
        try:
            fig = plotter.plot_solution_field_2d(x, y, solution)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except (ValueError, IndexError) as e:
            # Expected error for mismatched dimensions
            assert "shape" in str(e) or "dimension" in str(e)


class TestModulePerformance:
    """Test performance characteristics across modules."""
    
    @pytest.mark.slow
    def test_solver_scaling(self):
        """Test solver performance scaling."""
        mesh_sizes = [(11, 11), (21, 21)]
        solve_times = []
        
        for mesh_size in mesh_sizes:
            solver = FiniteDifferenceSolver(
                domain_bounds=(0, 1, 0, 1),
                mesh_size=mesh_size,
                pde_type="elliptic"
            )
            
            pde_params = {
                'diffusion': 1.0,
                'reaction': 0.0,
                'source': lambda x, y: 1.0
            }
            
            bc = {
                "left": {"type": "dirichlet", "value": 0.0},
                "right": {"type": "dirichlet", "value": 0.0},
                "top": {"type": "dirichlet", "value": 0.0},
                "bottom": {"type": "dirichlet", "value": 0.0}
            }
            
            import time
            start_time = time.time()
            solution = solver.solve(pde_params, bc)
            solve_time = time.time() - start_time
            
            solve_times.append(solve_time)
        
        # Should complete in reasonable time
        for solve_time in solve_times:
            assert solve_time < 10.0, f"Solve time {solve_time}s too long"
    
    def test_inference_efficiency(self):
        """Test inference computational efficiency."""
        def simple_posterior(params):
            return -0.5 * np.sum(params**2)
        
        mcmc = MCMCSampler(simple_posterior, parameter_dim=2)
        
        import time
        start_time = time.time()
        result = mcmc.sample(n_samples=200, initial_state=np.zeros(2), step_size=0.1)
        mcmc_time = time.time() - start_time
        
        # Should be reasonably fast
        assert mcmc_time < 30.0, f"MCMC time {mcmc_time}s too long"
        
        # Should produce reasonable acceptance rate
        assert 0.1 < result['acceptance_rate'] < 0.8