"""
Test suite for visualization modules.

Tests plotting functionality, figure generation, styling consistency,
and output correctness for all visualization components.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from bayesian_pde_solver.visualization.plotting_utils import (
    PlottingConfig, setup_matplotlib_style, create_figure_grid, save_figure
)
from bayesian_pde_solver.visualization.pde_plots import PDEPlotter
from bayesian_pde_solver.visualization.bayesian_plots import BayesianPlotter
from bayesian_pde_solver.visualization.uncertainty_plots import UncertaintyPlotter


class TestPlottingUtils:
    """Test plotting utility functions."""
    
    def test_plotting_config_constants(self):
        """Test PlottingConfig class constants."""
        # Check that required constants exist
        assert hasattr(PlottingConfig, 'FONT_SIZES')
        assert hasattr(PlottingConfig, 'FIGURE_SIZES')
        assert hasattr(PlottingConfig, 'COLORS')
        assert hasattr(PlottingConfig, 'ACADEMIC_COLORS')
        assert hasattr(PlottingConfig, 'UNCERTAINTY_COLORS')
        
        # Check font sizes are reasonable
        font_sizes = PlottingConfig.FONT_SIZES
        assert 8 <= font_sizes['small'] <= 12
        assert 10 <= font_sizes['medium'] <= 16
        assert 12 <= font_sizes['large'] <= 20
        
        # Check figure sizes are tuples
        for size_name, size_value in PlottingConfig.FIGURE_SIZES.items():
            assert isinstance(size_value, tuple)
            assert len(size_value) == 2
            assert all(isinstance(s, (int, float)) for s in size_value)
    
    def test_matplotlib_style_setup(self):
        """Test matplotlib style setup."""
        # Test different styles
        styles = ['academic', 'presentation', 'default']
        
        for style in styles:
            setup_matplotlib_style(style)
            
            # Check that rcParams are set
            assert plt.rcParams['font.size'] > 0
            assert plt.rcParams['figure.dpi'] > 0
            
            # Reset to default after each test
            plt.style.use('default')
    
    def test_figure_grid_creation(self):
        """Test figure grid creation utility."""
        # Test different grid configurations
        test_configs = [
            (1, 1),
            (2, 2), 
            (1, 3),
            (3, 1)
        ]
        
        for nrows, ncols in test_configs:
            fig, axes = create_figure_grid(nrows, ncols)
            
            assert isinstance(fig, plt.Figure)
            
            # Check axes structure
            if nrows == 1 and ncols == 1:
                assert hasattr(axes, 'plot')  # Single axes object
            else:
                if nrows == 1 or ncols == 1:
                    assert len(axes) == max(nrows, ncols)
                else:
                    assert axes.shape == (nrows, ncols)
            
            plt.close(fig)
    
    def test_figure_saving(self, temp_dir):
        """Test figure saving utility."""
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Figure")
        
        # Test saving in different formats
        filename = temp_dir / "test_figure"
        formats = ['png', 'pdf', 'svg']
        
        save_figure(fig, str(filename), formats=formats, dpi=150)
        
        # Check that files were created
        for fmt in formats:
            filepath = temp_dir / f"test_figure.{fmt}"
            assert filepath.exists(), f"File {filepath} was not created"
            assert filepath.stat().st_size > 0, f"File {filepath} is empty"
        
        plt.close(fig)


class TestPDEPlotter:
    """Test PDE visualization functionality."""
    
    @pytest.fixture
    def pde_plotter(self):
        """Create PDE plotter instance."""
        return PDEPlotter()
    
    @pytest.fixture
    def sample_2d_solution(self):
        """Generate sample 2D PDE solution data."""
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y)
        
        # Simple analytical solution
        solution = np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        return {
            'x': x,
            'y': y,
            'X': X,
            'Y': Y,
            'solution': solution,
            'coordinates': np.column_stack([X.ravel(), Y.ravel()])
        }
    
    def test_pde_plotter_initialization(self, pde_plotter):
        """Test PDE plotter initialization."""
        assert hasattr(pde_plotter, 'style')
        assert hasattr(pde_plotter, 'color_scheme')
        assert hasattr(pde_plotter, 'figure_size')
    
    def test_solution_field_plot_2d(self, pde_plotter, sample_2d_solution, temp_dir):
        """Test 2D solution field plotting."""
        data = sample_2d_solution
        
        fig = pde_plotter.plot_solution_field_2d(
            x=data['x'],
            y=data['y'], 
            solution=data['solution'],
            title="Test Solution Field"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check that colorbar was added
        axes = fig.get_axes()
        assert len(axes) >= 2  # Main plot + colorbar
        
        # Save and verify
        save_path = temp_dir / "solution_field_2d.png"
        fig.savefig(save_path)
        assert save_path.exists()
        
        plt.close(fig)
    
    def test_contour_plot(self, pde_plotter, sample_2d_solution, temp_dir):
        """Test contour plotting."""
        data = sample_2d_solution
        
        fig = pde_plotter.plot_contours(
            X=data['X'],
            Y=data['Y'],
            Z=data['solution'],
            levels=10,
            title="Test Contours"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check contour plot elements
        axes = fig.get_axes()
        contour_collections = [child for child in axes[0].get_children() 
                             if hasattr(child, 'get_paths')]
        assert len(contour_collections) > 0, "No contour lines found"
        
        plt.close(fig)
    
    def test_mesh_visualization(self, pde_plotter, temp_dir):
        """Test mesh visualization."""
        # Create simple triangular mesh
        np.random.seed(42)
        n_points = 20
        points = np.random.uniform(0, 1, (n_points, 2))
        
        # Create simple triangulation
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        
        fig = pde_plotter.plot_mesh(
            coordinates=points,
            elements=tri.simplices,
            title="Test Mesh"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should show mesh structure
        axes = fig.get_axes()
        assert len(axes[0].get_lines()) > 0 or len(axes[0].collections) > 0
        
        plt.close(fig)
    
    def test_1d_solution_plot(self, pde_plotter, temp_dir):
        """Test 1D solution plotting."""
        x = np.linspace(0, 1, 51)
        solution = np.sin(2 * np.pi * x)
        
        fig = pde_plotter.plot_solution_1d(
            x=x,
            solution=solution,
            title="Test 1D Solution",
            xlabel="x",
            ylabel="u(x)"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check that line plot was created
        axes = fig.get_axes()
        assert len(axes[0].get_lines()) >= 1
        
        plt.close(fig)
    
    def test_error_visualization(self, pde_plotter, sample_2d_solution, temp_dir):
        """Test error visualization."""
        data = sample_2d_solution
        
        # Create synthetic error field
        error = 0.1 * np.random.random(data['solution'].shape)
        
        fig = pde_plotter.plot_error_field(
            X=data['X'],
            Y=data['Y'],
            error=error,
            title="Test Error Field"
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_convergence_plot(self, pde_plotter, temp_dir):
        """Test convergence analysis plotting."""
        mesh_sizes = [10, 20, 40, 80]
        errors = [1e-1, 2.5e-2, 6e-3, 1.5e-3]  # O(h^2) convergence
        
        fig = pde_plotter.plot_convergence(
            mesh_sizes=mesh_sizes,
            errors=errors,
            expected_rate=2.0,
            title="Test Convergence"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should be log-log plot
        axes = fig.get_axes()
        assert axes[0].get_xscale() == 'log'
        assert axes[0].get_yscale() == 'log'
        
        plt.close(fig)


class TestBayesianPlotter:
    """Test Bayesian inference visualization."""
    
    @pytest.fixture
    def bayesian_plotter(self):
        """Create Bayesian plotter instance."""
        return BayesianPlotter()
    
    @pytest.fixture
    def sample_mcmc_data(self):
        """Generate sample MCMC data."""
        np.random.seed(42)
        n_samples = 1000
        n_params = 2
        
        # Generate correlated samples
        mean = [1.0, -0.5]
        cov = [[1.0, 0.3], [0.3, 0.8]]
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        
        return {
            'samples': samples,
            'parameter_names': ['param_1', 'param_2'],
            'true_values': mean
        }
    
    def test_bayesian_plotter_initialization(self, bayesian_plotter):
        """Test Bayesian plotter initialization."""
        assert hasattr(bayesian_plotter, 'style')
        assert hasattr(bayesian_plotter, 'color_scheme')
    
    def test_trace_plots(self, bayesian_plotter, sample_mcmc_data, temp_dir):
        """Test MCMC trace plots."""
        data = sample_mcmc_data
        
        fig = bayesian_plotter.plot_traces(
            samples=data['samples'],
            parameter_names=data['parameter_names'],
            true_values=data['true_values']
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should have subplots for each parameter
        axes = fig.get_axes()
        assert len(axes) >= len(data['parameter_names'])
        
        plt.close(fig)
    
    def test_posterior_distributions(self, bayesian_plotter, sample_mcmc_data, temp_dir):
        """Test posterior distribution plots."""
        data = sample_mcmc_data
        
        fig = bayesian_plotter.plot_posterior_distributions(
            samples=data['samples'],
            parameter_names=data['parameter_names'],
            true_values=data['true_values']
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should have histogram/density plots
        axes = fig.get_axes()
        for ax in axes:
            # Check for histogram or density plot elements
            has_hist = len(ax.patches) > 0
            has_lines = len(ax.get_lines()) > 0
            assert has_hist or has_lines, "No histogram or density plot found"
        
        plt.close(fig)
    
    def test_corner_plot(self, bayesian_plotter, sample_mcmc_data, temp_dir):
        """Test corner plot (pairwise parameter plots)."""
        data = sample_mcmc_data
        
        fig = bayesian_plotter.plot_corner(
            samples=data['samples'],
            parameter_names=data['parameter_names'],
            true_values=data['true_values']
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Corner plot should have n_params^2 subplots
        n_params = len(data['parameter_names'])
        axes = fig.get_axes()
        assert len(axes) == n_params * n_params
        
        plt.close(fig)
    
    def test_convergence_diagnostics(self, bayesian_plotter, temp_dir):
        """Test convergence diagnostic plots."""
        # Generate multiple chains
        np.random.seed(42)
        n_chains = 4
        n_samples = 500
        chains = []
        
        for i in range(n_chains):
            chain = np.random.normal([1.0, -0.5], [0.2, 0.3], (n_samples, 2))
            chains.append(chain)
        
        fig = bayesian_plotter.plot_convergence_diagnostics(
            chains=chains,
            parameter_names=['param_1', 'param_2']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_prior_posterior_comparison(self, bayesian_plotter, sample_mcmc_data, temp_dir):
        """Test prior vs posterior comparison plots."""
        data = sample_mcmc_data
        
        # Define simple prior distributions
        prior_dists = [
            {'type': 'normal', 'params': {'loc': 0, 'scale': 2}},
            {'type': 'normal', 'params': {'loc': 0, 'scale': 2}}
        ]
        
        fig = bayesian_plotter.plot_prior_posterior_comparison(
            samples=data['samples'],
            parameter_names=data['parameter_names'],
            prior_distributions=prior_dists
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_elbo_convergence(self, bayesian_plotter, temp_dir):
        """Test ELBO convergence plot for variational inference."""
        # Generate synthetic ELBO history
        n_iterations = 1000
        iterations = np.arange(n_iterations)
        
        # Realistic ELBO convergence (asymptotic approach)
        elbo_history = -10 + 8 * (1 - np.exp(-iterations / 200)) + \
                      0.1 * np.random.normal(size=n_iterations)
        
        fig = bayesian_plotter.plot_elbo_convergence(
            elbo_history=elbo_history,
            title="Test ELBO Convergence"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should show increasing trend
        axes = fig.get_axes()
        lines = axes[0].get_lines()
        assert len(lines) >= 1
        
        plt.close(fig)


class TestUncertaintyPlotter:
    """Test uncertainty visualization."""
    
    @pytest.fixture
    def uncertainty_plotter(self):
        """Create uncertainty plotter instance."""
        return UncertaintyPlotter()
    
    @pytest.fixture
    def sample_uncertainty_data(self):
        """Generate sample uncertainty data."""
        np.random.seed(42)
        n_points = 50
        x = np.linspace(0, 1, n_points)
        
        # True function
        true_function = np.sin(2 * np.pi * x)
        
        # Mean prediction (slightly biased)
        mean_pred = true_function + 0.05 * np.sin(8 * np.pi * x)
        
        # Standard deviation (heteroscedastic)
        std_pred = 0.1 + 0.05 * x
        
        # Observation data
        n_obs = 15
        obs_x = np.random.uniform(0, 1, n_obs)
        obs_y = np.sin(2 * np.pi * obs_x) + 0.1 * np.random.normal(size=n_obs)
        
        return {
            'x': x,
            'mean': mean_pred,
            'std': std_pred,
            'true_function': true_function,
            'observations': {'x': obs_x, 'y': obs_y}
        }
    
    def test_uncertainty_plotter_initialization(self, uncertainty_plotter):
        """Test uncertainty plotter initialization."""
        assert hasattr(uncertainty_plotter, 'style')
        assert hasattr(uncertainty_plotter, 'color_scheme')
        assert hasattr(uncertainty_plotter, 'colors')
    
    def test_confidence_bands_plot(self, uncertainty_plotter, sample_uncertainty_data, temp_dir):
        """Test confidence bands plotting."""
        data = sample_uncertainty_data
        
        fig = uncertainty_plotter.plot_confidence_bands(
            x=data['x'],
            mean=data['mean'],
            std=data['std'],
            observations=data['observations'],
            confidence_levels=[0.68, 0.95],
            title="Test Confidence Bands"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check plot elements
        axes = fig.get_axes()
        
        # Should have mean line
        lines = axes[0].get_lines()
        assert len(lines) >= 1
        
        # Should have confidence bands (fill_between creates PolyCollection)
        collections = axes[0].collections
        assert len(collections) >= 2  # At least 2 confidence levels
        
        # Should have observation points
        scatter_collections = [c for c in collections if hasattr(c, 'get_offsets')]
        assert len(scatter_collections) >= 1
        
        plt.close(fig)
    
    def test_prediction_intervals_plot(self, uncertainty_plotter, temp_dir):
        """Test prediction intervals plotting."""
        np.random.seed(42)
        n_samples = 100
        n_points = 30
        
        # Generate ensemble predictions
        x = np.linspace(0, 1, n_points)
        predictions = np.zeros((n_samples, n_points))
        
        for i in range(n_samples):
            predictions[i] = np.sin(2 * np.pi * x) + 0.1 * np.random.normal(size=n_points)
        
        fig = uncertainty_plotter.plot_prediction_intervals(
            x=x,
            predictions=predictions,
            quantiles=[0.05, 0.25, 0.75, 0.95],
            title="Test Prediction Intervals"
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_certified_bounds_plot(self, uncertainty_plotter, temp_dir):
        """Test certified bounds visualization."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate posterior samples
        samples = np.random.multivariate_normal([1.0, -0.5], [[1.0, 0.2], [0.2, 0.8]], n_samples)
        parameter_names = ['diffusion', 'source_strength']
        
        # Define certified bounds
        certified_bounds = {
            'diffusion': (0.5, 1.8),
            'source_strength': (-1.2, 0.2)
        }
        
        # True values
        true_values = {'diffusion': 1.0, 'source_strength': -0.5}
        
        fig = uncertainty_plotter.plot_certified_bounds(
            samples=samples,
            parameter_names=parameter_names,
            certified_bounds=certified_bounds,
            true_values=true_values
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should have subplots for each parameter
        axes = fig.get_axes()
        assert len(axes) >= len(parameter_names)
        
        plt.close(fig)
    
    def test_coverage_analysis_plot(self, uncertainty_plotter, temp_dir):
        """Test coverage analysis visualization."""
        # Generate synthetic coverage results
        methods = ['empirical', 'concentration', 'pac_bayes']
        confidence_levels = [0.68, 0.95, 0.99]
        
        coverage_results = {}
        for method in methods:
            coverage_results[method] = {}
            for conf_level in confidence_levels:
                # Synthetic coverage (slightly conservative)
                coverage = conf_level + 0.02 + 0.01 * np.random.random()
                width = 0.1 / conf_level + 0.02 * np.random.random()
                
                coverage_results[method][f'coverage_{int(conf_level*100)}'] = min(coverage, 1.0)
                coverage_results[method][f'width_{int(conf_level*100)}'] = width
        
        fig = uncertainty_plotter.plot_coverage_analysis(
            coverage_results=coverage_results,
            methods=methods,
            confidence_levels=confidence_levels
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should have 2 subplots (coverage and widths)
        axes = fig.get_axes()
        assert len(axes) == 2
        
        plt.close(fig)
    
    def test_uncertainty_propagation_plot(self, uncertainty_plotter, temp_dir):
        """Test uncertainty propagation visualization."""
        np.random.seed(42)
        n_samples = 200
        
        # Input parameters
        input_samples = np.random.multivariate_normal([1.0, 2.0], [[0.5, 0.1], [0.1, 0.3]], n_samples)
        
        # Output (nonlinear function of inputs)
        output_samples = input_samples[:, 0]**2 + np.sin(input_samples[:, 1]) + \
                        0.1 * np.random.normal(size=n_samples)
        
        fig = uncertainty_plotter.plot_uncertainty_propagation(
            input_samples=input_samples,
            output_samples=output_samples,
            input_names=['param_1', 'param_2'],
            output_name='output'
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_error_bars_2d_plot(self, uncertainty_plotter, temp_dir):
        """Test 2D error bars visualization."""
        np.random.seed(42)
        n_points = 25
        
        # Random 2D points
        x = np.random.uniform(0, 1, n_points)
        y = np.random.uniform(0, 1, n_points)
        
        # Values and uncertainties
        mean_values = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        std_values = 0.1 + 0.05 * (x + y)
        
        fig = uncertainty_plotter.plot_error_bars_2d(
            x=x,
            y=y,
            mean_values=mean_values,
            std_values=std_values,
            title="Test 2D Error Bars"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check for scatter plot and error bars
        axes = fig.get_axes()
        collections = axes[0].collections
        assert len(collections) >= 1  # Scatter plot
        
        plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for visualization modules."""
    
    @pytest.mark.integration
    def test_complete_visualization_workflow(self, temp_dir):
        """Test complete visualization workflow."""
        # Generate comprehensive test data
        np.random.seed(42)
        
        # 1. PDE solution data
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y)
        solution = np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        # 2. MCMC samples
        n_samples = 1000
        mcmc_samples = np.random.multivariate_normal([1.0, -0.5], [[0.8, 0.2], [0.2, 0.6]], n_samples)
        
        # 3. Uncertainty data
        n_pred_points = 30
        x_pred = np.linspace(0, 1, n_pred_points)
        mean_pred = np.sin(2 * np.pi * x_pred)
        std_pred = 0.1 + 0.05 * x_pred
        
        # Create all plotters
        pde_plotter = PDEPlotter(style="academic")
        bayesian_plotter = BayesianPlotter(style="academic")
        uncertainty_plotter = UncertaintyPlotter(style="academic")
        
        # Generate all plot types
        plots_created = []
        
        # PDE plots
        fig1 = pde_plotter.plot_solution_field_2d(x, y, solution, title="PDE Solution")
        fig1.savefig(temp_dir / "pde_solution.png")
        plots_created.append("pde_solution.png")
        plt.close(fig1)
        
        # Bayesian plots
        fig2 = bayesian_plotter.plot_traces(
            mcmc_samples, ['param_1', 'param_2'], true_values=[1.0, -0.5]
        )
        fig2.savefig(temp_dir / "mcmc_traces.png")
        plots_created.append("mcmc_traces.png")
        plt.close(fig2)
        
        fig3 = bayesian_plotter.plot_posterior_distributions(
            mcmc_samples, ['param_1', 'param_2'], true_values=[1.0, -0.5]
        )
        fig3.savefig(temp_dir / "posteriors.png")
        plots_created.append("posteriors.png")
        plt.close(fig3)
        
        # Uncertainty plots
        fig4 = uncertainty_plotter.plot_confidence_bands(
            x_pred, mean_pred, std_pred, title="Uncertainty Bands"
        )
        fig4.savefig(temp_dir / "uncertainty_bands.png")
        plots_created.append("uncertainty_bands.png")
        plt.close(fig4)
        
        # Verify all plots were created and are non-empty
        for plot_name in plots_created:
            plot_path = temp_dir / plot_name
            assert plot_path.exists(), f"Plot {plot_name} was not created"
            assert plot_path.stat().st_size > 1000, f"Plot {plot_name} seems too small"
    
    def test_consistent_styling_across_modules(self):
        """Test that all plotters use consistent styling."""
        # Create plotters with same style
        style = "academic"
        pde_plotter = PDEPlotter(style=style)
        bayesian_plotter = BayesianPlotter(style=style)
        uncertainty_plotter = UncertaintyPlotter(style=style)
        
        # Check that all use the same style
        assert pde_plotter.style == style
        assert bayesian_plotter.style == style
        assert uncertainty_plotter.style == style
        
        # Check figure sizes are consistent
        assert pde_plotter.figure_size == bayesian_plotter.figure_size
        assert bayesian_plotter.figure_size == uncertainty_plotter.figure_size
    
    @pytest.mark.slow
    def test_memory_usage_large_plots(self, temp_dir):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        n_large = 200
        x_large = np.linspace(0, 1, n_large)
        y_large = np.linspace(0, 1, n_large)
        X_large, Y_large = np.meshgrid(x_large, y_large)
        solution_large = np.sin(np.pi * X_large) * np.sin(np.pi * Y_large)
        
        # Create large plot
        pde_plotter = PDEPlotter()
        fig = pde_plotter.plot_solution_field_2d(x_large, y_large, solution_large)
        
        current_memory = process.memory_info().rss
        memory_increase = (current_memory - initial_memory) / 1024**2  # MB
        
        # Clean up
        plt.close(fig)
        del X_large, Y_large, solution_large
        
        # Memory increase should be reasonable (< 100 MB for this size)
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f} MB too large"
    
    def test_plot_output_validation(self, temp_dir):
        """Test that plots produce valid output files."""
        # Create simple test data
        x = np.linspace(0, 1, 11)
        y = np.sin(x)
        
        pde_plotter = PDEPlotter()
        fig = pde_plotter.plot_solution_1d(x, y, title="Validation Test")
        
        # Save in multiple formats
        formats = ['png', 'pdf', 'svg']
        for fmt in formats:
            filepath = temp_dir / f"validation_test.{fmt}"
            fig.savefig(filepath, format=fmt, dpi=150)
            
            assert filepath.exists(), f"File {filepath} not created"
            
            # Basic file validation
            if fmt == 'png':
                # PNG files should start with PNG signature
                with open(filepath, 'rb') as f:
                    signature = f.read(8)
                    assert signature.startswith(b'\x89PNG\r\n\x1a\n'), "Invalid PNG signature"
            elif fmt == 'pdf':
                # PDF files should start with %PDF
                with open(filepath, 'rb') as f:
                    header = f.read(4)
                    assert header == b'%PDF', "Invalid PDF header"
        
        plt.close(fig)


class TestPlotContent:
    """Test the content and accuracy of generated plots."""
    
    def test_data_range_preservation(self):
        """Test that plots preserve data ranges correctly."""
        # Create test data with known range
        x = np.linspace(-2, 3, 50)
        y = 2 * x + 1
        
        pde_plotter = PDEPlotter()
        fig = pde_plotter.plot_solution_1d(x, y)
        
        ax = fig.get_axes()[0]
        
        # Check x-axis limits include data range
        x_lim = ax.get_xlim()
        assert x_lim[0] <= -2 and x_lim[1] >= 3
        
        # Check y-axis limits include data range
        y_lim = ax.get_ylim()
        y_min, y_max = np.min(y), np.max(y)
        assert y_lim[0] <= y_min and y_lim[1] >= y_max
        
        plt.close(fig)
    
    def test_colorbar_ranges(self, sample_2d_solution):
        """Test that colorbars have correct ranges."""
        data = sample_2d_solution
        
        pde_plotter = PDEPlotter()
        fig = pde_plotter.plot_solution_field_2d(
            data['x'], data['y'], data['solution']
        )
        
        # Find colorbar
        colorbars = [ax for ax in fig.get_axes() if ax.get_ylabel()]
        
        if colorbars:  # If colorbar exists
            cb = colorbars[-1]  # Usually the last axes
            cb_range = cb.get_ylim()
            data_range = (np.min(data['solution']), np.max(data['solution']))
            
            # Colorbar range should encompass data range
            assert cb_range[0] <= data_range[0]
            assert cb_range[1] >= data_range[1]
        
        plt.close(fig)
    
    def test_legend_presence(self):
        """Test that legends are present when expected."""
        np.random.seed(42)
        samples = np.random.normal([1, -0.5], [0.2, 0.3], (500, 2))
        
        bayesian_plotter = BayesianPlotter()
        fig = bayesian_plotter.plot_posterior_distributions(
            samples, ['param_1', 'param_2'], true_values=[1.0, -0.5]
        )
        
        # Check for legends in subplots
        axes = fig.get_axes()
        legends_found = 0
        for ax in axes:
            if ax.get_legend() is not None:
                legends_found += 1
        
        # Should have at least one legend
        assert legends_found >= 1, "No legends found in posterior plot"
        
        plt.close(fig)