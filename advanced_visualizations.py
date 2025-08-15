"""
Advanced Visualization Tools for Bayesian PDE Research

Publication-quality plotting and visualization tools for Bayesian inverse problems
in partial differential equations. Provides comprehensive visualization capabilities
for theoretical results, experimental data, uncertainty quantification, and 
comparative analysis.

Features:
- Publication-ready plots with custom scientific styling
- Interactive 3D visualizations for PDE solutions and parameter fields
- Uncertainty visualization with confidence regions and bounds
- Convergence analysis and theoretical result visualization
- Benchmark comparison plots with statistical significance
- Animation capabilities for time-dependent problems
- Multi-panel figure generation for manuscripts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import gaussian_kde, chi2
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from pathlib import Path
import json


class PublicationPlotter:
    """
    Publication-quality plotting with consistent academic styling.
    
    Provides standardized plotting functions with journal-ready formatting,
    proper typography, and consistent color schemes for scientific publications.
    """
    
    def __init__(self, style: str = 'publication', figsize_base: Tuple[float, float] = (6, 4)):
        """
        Initialize publication plotter with consistent styling.
        
        Parameters:
        -----------
        style : str
            Plotting style ('publication', 'presentation', 'poster')
        figsize_base : Tuple[float, float]
            Base figure size for single plots
        """
        self.style = style
        self.figsize_base = figsize_base
        
        # Publication-quality styling
        self.setup_publication_style()
        
        # Color schemes for different plot types
        self.color_schemes = {
            'methods': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'uncertainty': ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7'],
            'convergence': ['#08519c', '#3182bd', '#6baed6', '#9ecae1'],
            'comparison': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'],
            'sequential': 'viridis',
            'diverging': 'RdBu_r'
        }
        
        # LaTeX rendering setup
        self.setup_latex_rendering()
    
    def setup_publication_style(self):
        """Configure matplotlib for publication-quality output."""
        plt.style.use('default')  # Start with clean slate
        
        # Font and text settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times', 'Computer Modern Roman'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            
            # Line and marker settings
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'patch.linewidth': 0.5,
            
            # Axes settings
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': 'black',
            'axes.labelcolor': 'black',
            'axes.axisbelow': True,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            
            # Tick settings
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            
            # Figure settings
            'figure.facecolor': 'white',
            'figure.edgecolor': 'none',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # Legend settings
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.facecolor': 'white',
            'legend.edgecolor': 'black',
            'legend.fancybox': False,
            'legend.shadow': False
        })
    
    def setup_latex_rendering(self):
        """Setup LaTeX rendering for mathematical expressions."""
        try:
            plt.rcParams.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amsmath,amssymb,amsfonts}'
            })
        except:
            # Fallback to mathtext if LaTeX not available
            plt.rcParams.update({
                'mathtext.fontset': 'cm',
                'mathtext.rm': 'serif'
            })
    
    def create_figure(self, nrows: int = 1, ncols: int = 1, 
                     figsize: Optional[Tuple[float, float]] = None,
                     subplot_kw: Dict = None, **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create publication-ready figure with consistent styling.
        
        Parameters:
        -----------
        nrows, ncols : int
            Number of subplot rows and columns
        figsize : Tuple[float, float], optional
            Figure size (width, height) in inches
        subplot_kw : Dict, optional
            Subplot keyword arguments
            
        Returns:
        --------
        Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]
            Figure and axes objects
        """
        if figsize is None:
            width, height = self.figsize_base
            figsize = (width * ncols, height * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                subplot_kw=subplot_kw, **kwargs)
        
        # Ensure consistent spacing
        if nrows > 1 or ncols > 1:
            plt.tight_layout()
        
        return fig, axes
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   formats: List[str] = ['pdf', 'png'], **kwargs):
        """
        Save figure in multiple formats with consistent settings.
        
        Parameters:
        -----------
        fig : plt.Figure
            Figure to save
        filename : str
            Base filename (without extension)
        formats : List[str]
            File formats to save
        **kwargs : Dict
            Additional savefig arguments
        """
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            **kwargs
        }
        
        for fmt in formats:
            filepath = f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, **save_kwargs)
            print(f"Saved figure: {filepath}")


class UncertaintyVisualizer(PublicationPlotter):
    """
    Specialized visualization tools for uncertainty quantification.
    
    Provides methods for visualizing confidence regions, posterior distributions,
    concentration bounds, and other uncertainty-related quantities.
    """
    
    def plot_confidence_ellipse(self, mean: np.ndarray, cov: np.ndarray, 
                               confidence_levels: List[float] = [0.68, 0.95, 0.99],
                               ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot confidence ellipses for multivariate Gaussian distribution.
        
        Parameters:
        -----------
        mean : np.ndarray, shape (2,)
            Mean of 2D distribution
        cov : np.ndarray, shape (2, 2)
            Covariance matrix
        confidence_levels : List[float]
            Confidence levels to plot
        ax : plt.Axes, optional
            Axes to plot on
            
        Returns:
        --------
        plt.Axes
            Axes object
        """
        if ax is None:
            fig, ax = self.create_figure()
        
        # Eigendecomposition for ellipse parameters
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
        
        # Angle of rotation
        angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
        
        # Plot ellipses for each confidence level
        colors = self.color_schemes['uncertainty']
        for i, confidence in enumerate(confidence_levels):
            # Chi-squared quantile for 2D
            chi2_val = chi2.ppf(confidence, df=2)
            
            # Ellipse dimensions
            width, height = 2 * np.sqrt(eigenvals * chi2_val)
            
            # Create ellipse
            ellipse = Ellipse(mean, width, height, angle=angle,
                            facecolor=colors[i % len(colors)], 
                            alpha=0.3, edgecolor='black', linewidth=1)
            ax.add_patch(ellipse)
        
        # Plot mean
        ax.plot(mean[0], mean[1], 'ko', markersize=6, label='Mean')
        
        # Formatting
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_posterior_distribution(self, samples: np.ndarray, 
                                  true_value: Optional[np.ndarray] = None,
                                  parameter_names: Optional[List[str]] = None,
                                  **kwargs) -> plt.Figure:
        """
        Plot posterior distribution with marginals and correlations.
        
        Parameters:
        -----------
        samples : np.ndarray, shape (n_samples, n_params)
            MCMC samples from posterior
        true_value : np.ndarray, optional
            True parameter values for comparison
        parameter_names : List[str], optional
            Parameter names for labeling
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        n_params = samples.shape[1]
        
        if parameter_names is None:
            parameter_names = [f'$\\theta_{{{i+1}}}$' for i in range(n_params)]
        
        # Create subplot grid
        fig, axes = plt.subplots(n_params, n_params, 
                               figsize=(2*n_params, 2*n_params))
        
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j] if n_params > 1 else axes
                
                if i == j:
                    # Diagonal: marginal distributions
                    ax.hist(samples[:, i], bins=50, density=True, 
                           alpha=0.7, color=self.color_schemes['methods'][0])
                    
                    if true_value is not None:
                        ax.axvline(true_value[i], color='red', linestyle='--',
                                 linewidth=2, label='True value')
                    
                    ax.set_ylabel('Density')
                    if i == n_params - 1:
                        ax.set_xlabel(parameter_names[i])
                
                elif i > j:
                    # Lower triangle: scatter plots
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.5, s=1,
                             color=self.color_schemes['methods'][0])
                    
                    if true_value is not None:
                        ax.plot(true_value[j], true_value[i], 'ro', 
                               markersize=8, label='True value')
                    
                    if i == n_params - 1:
                        ax.set_xlabel(parameter_names[j])
                    if j == 0:
                        ax.set_ylabel(parameter_names[i])
                
                else:
                    # Upper triangle: correlation coefficients
                    corr = np.corrcoef(samples[:, j], samples[:, i])[0, 1]
                    ax.text(0.5, 0.5, f'$\\rho = {corr:.3f}$', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle='round', 
                                                 facecolor='white', alpha=0.8))
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def plot_concentration_bounds(self, sample_sizes: np.ndarray,
                                bounds_data: Dict[str, np.ndarray],
                                empirical_errors: Optional[np.ndarray] = None,
                                confidence_level: float = 0.95,
                                **kwargs) -> plt.Figure:
        """
        Plot concentration bounds vs sample size.
        
        Parameters:
        -----------
        sample_sizes : np.ndarray
            Sample sizes for x-axis
        bounds_data : Dict[str, np.ndarray]
            Dictionary of bound types and values
        empirical_errors : np.ndarray, optional
            Empirical errors for comparison
        confidence_level : float
            Confidence level for bounds
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        fig, ax = self.create_figure()
        
        colors = self.color_schemes['methods']
        
        # Plot different bounds
        for i, (bound_name, bound_values) in enumerate(bounds_data.items()):
            ax.loglog(sample_sizes, bound_values, 
                     color=colors[i % len(colors)], linewidth=2,
                     label=f'{bound_name.replace("_", " ").title()} Bound')
        
        # Plot empirical errors if provided
        if empirical_errors is not None:
            ax.loglog(sample_sizes, empirical_errors, 'ko-', 
                     linewidth=2, markersize=4, alpha=0.7,
                     label='Empirical Error')
        
        # Theoretical rate lines for reference
        if len(sample_sizes) > 1:
            # n^{-1/2} rate
            rate_half = bound_values[0] * (sample_sizes[0] / sample_sizes)**0.5
            ax.loglog(sample_sizes, rate_half, 'k--', alpha=0.5, 
                     label='$n^{-1/2}$ rate')
            
            # n^{-1} rate  
            rate_one = bound_values[0] * (sample_sizes[0] / sample_sizes)
            ax.loglog(sample_sizes, rate_one, 'k:', alpha=0.5,
                     label='$n^{-1}$ rate')
        
        ax.set_xlabel('Sample Size $n$')
        ax.set_ylabel('Concentration Bound')
        ax.set_title(f'Concentration Bounds ({confidence_level*100:.0f}\\% Confidence)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_uncertainty_evolution(self, iterations: np.ndarray,
                                 uncertainty_metrics: Dict[str, np.ndarray],
                                 **kwargs) -> plt.Figure:
        """
        Plot evolution of uncertainty metrics during MCMC sampling.
        
        Parameters:
        -----------
        iterations : np.ndarray
            Iteration numbers
        uncertainty_metrics : Dict[str, np.ndarray]
            Uncertainty metrics over iterations
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        n_metrics = len(uncertainty_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 2*n_metrics), 
                               sharex=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = self.color_schemes['methods']
        
        for i, (metric_name, values) in enumerate(uncertainty_metrics.items()):
            ax = axes[i]
            
            ax.plot(iterations, values, color=colors[i % len(colors)], 
                   linewidth=1.5, alpha=0.8)
            
            # Add moving average
            if len(values) > 50:
                window = min(len(values) // 10, 100)
                moving_avg = pd.Series(values).rolling(window).mean()
                ax.plot(iterations, moving_avg, color=colors[i % len(colors)], 
                       linewidth=3, alpha=0.9, label='Moving Average')
            
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_title('Uncertainty Metric Evolution')
            if i == n_metrics - 1:
                ax.set_xlabel('MCMC Iteration')
        
        plt.tight_layout()
        return fig


class BenchmarkVisualizer(PublicationPlotter):
    """
    Visualization tools for benchmark comparisons and performance analysis.
    
    Provides methods for creating comparison plots, performance profiles,
    statistical significance analysis, and method ranking visualizations.
    """
    
    def plot_method_comparison(self, methods_data: Dict[str, Dict[str, float]],
                             metrics: List[str] = None,
                             log_scale: bool = False,
                             **kwargs) -> plt.Figure:
        """
        Plot comparison of different methods across multiple metrics.
        
        Parameters:
        -----------
        methods_data : Dict[str, Dict[str, float]]
            Nested dictionary: {method_name: {metric: value}}
        metrics : List[str], optional
            Specific metrics to plot
        log_scale : bool
            Whether to use logarithmic scale
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if metrics is None:
            # Get all unique metrics
            all_metrics = set()
            for method_data in methods_data.values():
                all_metrics.update(method_data.keys())
            metrics = list(all_metrics)
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        methods = list(methods_data.keys())
        colors = self.color_schemes['comparison']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            values = []
            method_names = []
            
            for method in methods:
                if metric in methods_data[method]:
                    values.append(methods_data[method][metric])
                    method_names.append(method)
            
            # Create bar plot
            bars = ax.bar(range(len(values)), values, 
                         color=[colors[j % len(colors)] for j in range(len(values))],
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Formatting
            ax.set_xticks(range(len(method_names)))
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            
            if log_scale:
                ax.set_yscale('log')
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_performance_profile(self, methods_data: Dict[str, np.ndarray],
                               metric_name: str = 'Performance Ratio',
                               **kwargs) -> plt.Figure:
        """
        Plot performance profile for method comparison.
        
        Parameters:
        -----------
        methods_data : Dict[str, np.ndarray]
            Performance data for each method
        metric_name : str
            Name of the performance metric
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        fig, ax = self.create_figure()
        
        colors = self.color_schemes['comparison']
        
        for i, (method, data) in enumerate(methods_data.items()):
            # Sort data for performance profile
            sorted_data = np.sort(data)
            n = len(sorted_data)
            
            # Cumulative probability
            prob = np.arange(1, n+1) / n
            
            ax.plot(sorted_data, prob, color=colors[i % len(colors)],
                   linewidth=2, label=method, marker='o', markersize=3,
                   markevery=max(1, len(sorted_data)//20))
        
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Probability')
        ax.set_title('Performance Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        return fig
    
    def plot_statistical_significance(self, pairwise_tests: Dict[Tuple[str, str], float],
                                    alpha: float = 0.05,
                                    **kwargs) -> plt.Figure:
        """
        Plot statistical significance matrix for pairwise comparisons.
        
        Parameters:
        -----------
        pairwise_tests : Dict[Tuple[str, str], float]
            p-values for pairwise statistical tests
        alpha : float
            Significance level
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        # Extract unique methods
        methods = list(set([method for pair in pairwise_tests.keys() for method in pair]))
        n_methods = len(methods)
        
        # Create significance matrix
        sig_matrix = np.ones((n_methods, n_methods))
        
        for (method1, method2), p_value in pairwise_tests.items():
            i, j = methods.index(method1), methods.index(method2)
            sig_matrix[i, j] = p_value
            sig_matrix[j, i] = p_value  # Symmetric
        
        # Plot heatmap
        fig, ax = self.create_figure(figsize=(8, 8))
        
        # Custom colormap for p-values
        colors = ['red', 'orange', 'lightgray']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('pvalue', colors, N=n_bins)
        
        im = ax.imshow(sig_matrix, cmap=cmap, vmin=0, vmax=0.1, aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value', rotation=270, labelpad=15)
        
        # Add significance threshold line
        cbar.ax.axhline(alpha, color='blue', linewidth=2, linestyle='--')
        
        # Formatting
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(methods)
        ax.set_title(f'Statistical Significance Matrix ($\\alpha = {alpha}$)')
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    p_val = sig_matrix[i, j]
                    if p_val < alpha:
                        text_color = 'white'
                        text = f'{p_val:.3f}*'
                    else:
                        text_color = 'black'
                        text = f'{p_val:.3f}'
                    
                    ax.text(j, i, text, ha='center', va='center',
                           color=text_color, fontsize=8, weight='bold')
        
        return fig
    
    def plot_convergence_analysis(self, sample_sizes: np.ndarray,
                                errors_data: Dict[str, np.ndarray],
                                theoretical_rates: Dict[str, float] = None,
                                **kwargs) -> plt.Figure:
        """
        Plot convergence analysis with theoretical rates.
        
        Parameters:
        -----------
        sample_sizes : np.ndarray
            Sample sizes
        errors_data : Dict[str, np.ndarray]
            Error measurements for each method
        theoretical_rates : Dict[str, float], optional
            Theoretical convergence rates
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        fig, ax = self.create_figure()
        
        colors = self.color_schemes['convergence']
        
        for i, (method, errors) in enumerate(errors_data.items()):
            ax.loglog(sample_sizes, errors, 'o-', 
                     color=colors[i % len(colors)], linewidth=2,
                     markersize=6, label=method)
            
            # Add theoretical rate if provided
            if theoretical_rates and method in theoretical_rates:
                rate = theoretical_rates[method]
                # Compute theoretical line
                theoretical = errors[0] * (sample_sizes[0] / sample_sizes)**rate
                ax.loglog(sample_sizes, theoretical, '--',
                         color=colors[i % len(colors)], alpha=0.7,
                         label=f'{method} (rate: $n^{{-{rate}}}$)')
        
        ax.set_xlabel('Sample Size $n$')
        ax.set_ylabel('Error')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class InteractiveVisualizer:
    """
    Interactive visualization tools using Plotly for web-based displays.
    
    Provides interactive 3D plots, dynamic parameter exploration,
    and web-ready visualizations for research presentations.
    """
    
    def __init__(self):
        """Initialize interactive visualizer."""
        self.default_layout = {
            'font': {'family': 'Times New Roman', 'size': 12},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60}
        }
    
    def plot_3d_parameter_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              values: np.ndarray, title: str = "Parameter Field",
                              **kwargs) -> go.Figure:
        """
        Create interactive 3D surface plot for parameter fields.
        
        Parameters:
        -----------
        x, y, z : np.ndarray
            Coordinate arrays
        values : np.ndarray
            Parameter values at coordinates
        title : str
            Plot title
            
        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=x, y=y, z=z,
            surfacecolor=values,
            colorscale='Viridis',
            colorbar=dict(title="Parameter Value"),
            opacity=0.8
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            **self.default_layout
        )
        
        return fig
    
    def plot_uncertainty_evolution_interactive(self, iterations: np.ndarray,
                                             metrics_data: Dict[str, np.ndarray],
                                             **kwargs) -> go.Figure:
        """
        Create interactive plot of uncertainty evolution.
        
        Parameters:
        -----------
        iterations : np.ndarray
            Iteration numbers
        metrics_data : Dict[str, np.ndarray]
            Uncertainty metrics over iterations
            
        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            fig.add_trace(go.Scatter(
                x=iterations,
                y=values,
                mode='lines',
                name=metric_name.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='Iteration: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Uncertainty Metrics Evolution',
            xaxis_title='MCMC Iteration',
            yaxis_title='Metric Value',
            hovermode='x unified',
            **self.default_layout
        )
        
        return fig
    
    def plot_benchmark_radar(self, methods_data: Dict[str, Dict[str, float]],
                           **kwargs) -> go.Figure:
        """
        Create radar chart for multi-metric method comparison.
        
        Parameters:
        -----------
        methods_data : Dict[str, Dict[str, float]]
            Method performance data
            
        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        # Get all metrics
        all_metrics = set()
        for method_data in methods_data.values():
            all_metrics.update(method_data.keys())
        metrics = list(all_metrics)
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (method, data) in enumerate(methods_data.items()):
            values = [data.get(metric, 0) for metric in metrics]
            
            # Normalize values to 0-1 scale for better visualization
            max_vals = [max(methods_data[m].get(metric, 0) 
                           for m in methods_data.keys()) for metric in metrics]
            normalized_values = [v/max_v if max_v > 0 else 0 
                               for v, max_v in zip(values, max_vals)]
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=method,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Method Performance Comparison',
            **self.default_layout
        )
        
        return fig


def create_publication_figure_set(results_data: Dict[str, Any], 
                                output_dir: str = "figures") -> List[str]:
    """
    Generate complete set of publication-ready figures.
    
    Parameters:
    -----------
    results_data : Dict[str, Any]
        Comprehensive results data from experiments
    output_dir : str
        Directory to save figures
        
    Returns:
    --------
    List[str]
        List of generated figure filenames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize visualizers
    pub_plotter = PublicationPlotter()
    uncertainty_viz = UncertaintyVisualizer()
    benchmark_viz = BenchmarkVisualizer()
    
    generated_files = []
    
    # 1. Theoretical Results Figures
    if 'theoretical_results' in results_data:
        theoretical = results_data['theoretical_results']
        
        # Concentration bounds
        if 'concentration_bounds' in theoretical:
            bounds_data = theoretical['concentration_bounds']
            sample_sizes = bounds_data.get('sample_sizes', np.logspace(1, 3, 20))
            
            fig = uncertainty_viz.plot_concentration_bounds(
                sample_sizes=sample_sizes,
                bounds_data={
                    'hoeffding': bounds_data.get('hoeffding_bounds', sample_sizes**(-0.5)),
                    'bernstein': bounds_data.get('bernstein_bounds', sample_sizes**(-0.5)),
                    'mcdiarmid': bounds_data.get('mcdiarmid_bounds', sample_sizes**(-0.5))
                }
            )
            
            filename = output_path / "concentration_bounds"
            pub_plotter.save_figure(fig, filename)
            generated_files.append(str(filename))
            plt.close(fig)
        
        # Convergence rates
        if 'convergence_analysis' in theoretical:
            conv_data = theoretical['convergence_analysis']
            
            fig = benchmark_viz.plot_convergence_analysis(
                sample_sizes=conv_data.get('sample_sizes', np.logspace(1, 3, 20)),
                errors_data=conv_data.get('empirical_errors', {}),
                theoretical_rates=conv_data.get('theoretical_rates', {})
            )
            
            filename = output_path / "convergence_analysis"
            pub_plotter.save_figure(fig, filename)
            generated_files.append(str(filename))
            plt.close(fig)
    
    # 2. Benchmark Comparison Figures
    if 'benchmark_results' in results_data:
        benchmark = results_data['benchmark_results']
        
        # Method comparison
        if 'methods_comparison' in benchmark:
            fig = benchmark_viz.plot_method_comparison(
                methods_data=benchmark['methods_comparison']
            )
            
            filename = output_path / "method_comparison"
            pub_plotter.save_figure(fig, filename)
            generated_files.append(str(filename))
            plt.close(fig)
        
        # Performance profiles
        if 'performance_profiles' in benchmark:
            fig = benchmark_viz.plot_performance_profile(
                methods_data=benchmark['performance_profiles']
            )
            
            filename = output_path / "performance_profiles"
            pub_plotter.save_figure(fig, filename)
            generated_files.append(str(filename))
            plt.close(fig)
    
    # 3. Uncertainty Quantification Figures
    if 'uncertainty_analysis' in results_data:
        uncertainty = results_data['uncertainty_analysis']
        
        # Posterior distributions
        if 'posterior_samples' in uncertainty:
            samples = uncertainty['posterior_samples']
            true_params = uncertainty.get('true_parameters')
            
            fig = uncertainty_viz.plot_posterior_distribution(
                samples=samples,
                true_value=true_params
            )
            
            filename = output_path / "posterior_distribution"
            pub_plotter.save_figure(fig, filename)
            generated_files.append(str(filename))
            plt.close(fig)
        
        # Uncertainty evolution
        if 'mcmc_diagnostics' in uncertainty:
            diagnostics = uncertainty['mcmc_diagnostics']
            
            fig = uncertainty_viz.plot_uncertainty_evolution(
                iterations=diagnostics.get('iterations', np.arange(len(list(diagnostics.values())[0]))),
                uncertainty_metrics=diagnostics
            )
            
            filename = output_path / "uncertainty_evolution"
            pub_plotter.save_figure(fig, filename)
            generated_files.append(str(filename))
            plt.close(fig)
    
    print(f"Generated {len(generated_files)} publication figures in {output_dir}/")
    return generated_files


def demo_advanced_visualizations():
    """Demonstrate advanced visualization capabilities."""
    print("Advanced Visualization Demo")
    print("=" * 30)
    
    # Generate sample data
    np.random.seed(42)
    
    # Sample theoretical results
    sample_sizes = np.logspace(1, 3, 20).astype(int)
    hoeffding_bounds = 0.5 * sample_sizes**(-0.5)
    bernstein_bounds = 0.3 * sample_sizes**(-0.5)
    empirical_errors = 0.4 * sample_sizes**(-0.5) * (1 + 0.1*np.random.randn(len(sample_sizes)))
    
    # Sample benchmark data
    methods_data = {
        'Our Method': {'mse': 0.0045, 'time': 12.3, 'coverage': 94.2, 'quality': 0.891},
        'Tikhonov': {'mse': 0.0067, 'time': 2.1, 'coverage': 78.4, 'quality': 0.623},
        'EnKF': {'mse': 0.0052, 'time': 8.7, 'coverage': 87.1, 'quality': 0.754},
        'MCMC': {'mse': 0.0048, 'time': 45.6, 'coverage': 92.8, 'quality': 0.836}
    }
    
    # Sample posterior data
    n_samples, n_params = 1000, 3
    posterior_samples = np.random.multivariate_normal(
        mean=np.zeros(n_params),
        cov=np.eye(n_params) * 0.1,
        size=n_samples
    )
    true_parameters = np.array([0.1, -0.05, 0.08])
    
    # Create visualizations
    print("Creating publication plots...")
    
    # 1. Uncertainty visualizations
    uncertainty_viz = UncertaintyVisualizer()
    
    # Concentration bounds
    fig1 = uncertainty_viz.plot_concentration_bounds(
        sample_sizes=sample_sizes,
        bounds_data={
            'hoeffding': hoeffding_bounds,
            'bernstein': bernstein_bounds
        },
        empirical_errors=empirical_errors
    )
    plt.show()
    
    # Posterior distribution
    fig2 = uncertainty_viz.plot_posterior_distribution(
        samples=posterior_samples,
        true_value=true_parameters,
        parameter_names=['$\\theta_1$', '$\\theta_2$', '$\\theta_3$']
    )
    plt.show()
    
    # 2. Benchmark visualizations
    benchmark_viz = BenchmarkVisualizer()
    
    # Method comparison
    fig3 = benchmark_viz.plot_method_comparison(
        methods_data=methods_data,
        metrics=['mse', 'coverage', 'quality']
    )
    plt.show()
    
    # 3. Interactive visualizations
    print("Creating interactive plots...")
    interactive_viz = InteractiveVisualizer()
    
    # Radar chart
    radar_fig = interactive_viz.plot_benchmark_radar(methods_data)
    radar_fig.show()
    
    # Sample uncertainty evolution
    iterations = np.arange(1000)
    metrics_data = {
        'Parameter_Variance': 1.0 * np.exp(-iterations/300) + 0.1 * np.random.randn(1000) * 0.1,
        'KL_Divergence': 2.0 * np.exp(-iterations/200) + 0.05 * np.random.randn(1000) * 0.1,
        'Effective_Sample_Size': 100 * (1 - np.exp(-iterations/150)) + 10 * np.random.randn(1000)
    }
    
    evolution_fig = interactive_viz.plot_uncertainty_evolution_interactive(
        iterations=iterations,
        metrics_data=metrics_data
    )
    evolution_fig.show()
    
    print("\nAdvanced visualization demo complete!")
    print("Generated publication-quality plots with:")
    print("- Concentration bounds analysis")
    print("- Posterior distribution visualization") 
    print("- Method comparison charts")
    print("- Interactive radar plots")
    print("- Dynamic uncertainty evolution")


if __name__ == "__main__":
    demo_advanced_visualizations()