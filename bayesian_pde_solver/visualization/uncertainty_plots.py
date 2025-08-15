"""
Uncertainty Visualization

Specialized plotting functions for uncertainty quantification including
error bars, confidence bands, prediction intervals, and certified bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings

from .plotting_utils import (
    PlottingConfig, setup_matplotlib_style, create_figure_grid,
    add_colorbar, save_figure
)


class UncertaintyPlotter:
    """
    Comprehensive uncertainty visualization for Bayesian analysis.
    
    Provides methods for plotting confidence bands, prediction intervals,
    certified bounds, coverage analysis, and uncertainty propagation.
    
    Examples
    --------
    >>> plotter = UncertaintyPlotter()
    >>> fig = plotter.plot_confidence_bands(x, mean, std, observations)
    >>> fig = plotter.plot_certified_bounds(samples, bounds, true_values)
    """
    
    def __init__(self, style: str = "academic", 
                 color_scheme: str = "uncertainty",
                 figure_size: Tuple[float, float] = (10, 6)):
        """
        Initialize uncertainty plotter.
        
        Parameters
        ----------
        style : str, default="academic"
            Plotting style
        color_scheme : str, default="uncertainty"
            Color scheme for uncertainty plots
        figure_size : Tuple[float, float], default=(10, 6)
            Default figure size
        """
        self.style = style
        self.color_scheme = color_scheme
        self.figure_size = figure_size
        
        # Set up matplotlib style
        setup_matplotlib_style(style)
        
        # Uncertainty-specific colors
        self.colors = PlottingConfig.UNCERTAINTY_COLORS
        
        # Confidence level colors
        self.confidence_colors = {
            0.68: '#1f77b4',  # 1σ - blue
            0.95: '#ff7f0e',  # 2σ - orange  
            0.99: '#2ca02c'   # 3σ - green
        }
    
    def plot_confidence_bands(self, x: np.ndarray,
                             mean: np.ndarray,
                             std: np.ndarray,
                             observations: Optional[Dict[str, np.ndarray]] = None,
                             confidence_levels: List[float] = [0.68, 0.95],
                             title: str = "Uncertainty Quantification",
                             xlabel: str = "x", ylabel: str = "u(x)",
                             show_mean: bool = True,
                             **kwargs) -> plt.Figure:
        """
        Plot confidence bands with optional observations.
        
        Parameters
        ----------
        x : np.ndarray
            Independent variable values
        mean : np.ndarray
            Mean prediction
        std : np.ndarray
            Standard deviation (uncertainty)
        observations : Optional[Dict[str, np.ndarray]], default=None
            Dictionary with 'x' and 'y' for observation data
        confidence_levels : List[float], default=[0.68, 0.95]
            Confidence levels to plot
        title : str, default="Uncertainty Quantification"
            Plot title
        xlabel, ylabel : str
            Axis labels
        show_mean : bool, default=True
            Whether to show mean line
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Sort by x for proper plotting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        mean_sorted = mean[sort_idx]
        std_sorted = std[sort_idx]
        
        # Plot mean
        if show_mean:
            ax.plot(x_sorted, mean_sorted, 
                   color=self.colors['mean'], linewidth=2, 
                   label='Mean prediction', **kwargs)
        
        # Plot confidence bands
        from scipy.stats import norm
        
        for conf_level in confidence_levels:
            z_score = norm.ppf(1 - (1 - conf_level) / 2)
            
            lower = mean_sorted - z_score * std_sorted
            upper = mean_sorted + z_score * std_sorted
            
            color = self.confidence_colors.get(conf_level, '#1f77b4')
            alpha = 0.3 if conf_level == max(confidence_levels) else 0.2
            
            ax.fill_between(x_sorted, lower, upper,
                           alpha=alpha, color=color,
                           label=f'{int(conf_level*100)}% confidence')
        
        # Plot observations if provided
        if observations is not None:
            obs_x = observations.get('x', observations.get('points', []))
            obs_y = observations.get('y', observations.get('values', []))
            
            if len(obs_x) > 0 and len(obs_y) > 0:
                ax.scatter(obs_x, obs_y, 
                          color=self.colors['observations'],
                          s=30, alpha=0.8, edgecolors='black', linewidth=0.5,
                          label='Observations', zorder=5)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_ylabel(ylabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_title(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        ax.legend(fontsize=PlottingConfig.FONT_SIZES['legend'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_intervals(self, x: np.ndarray,
                                 predictions: np.ndarray,
                                 true_function: Optional[np.ndarray] = None,
                                 observations: Optional[Dict[str, np.ndarray]] = None,
                                 quantiles: List[float] = [0.05, 0.25, 0.75, 0.95],
                                 title: str = "Prediction Intervals",
                                 **kwargs) -> plt.Figure:
        """
        Plot prediction intervals from ensemble of predictions.
        
        Parameters
        ----------
        x : np.ndarray
            Independent variable values
        predictions : np.ndarray, shape (n_samples, len(x))
            Ensemble of predictions
        true_function : Optional[np.ndarray], default=None
            True function values for comparison
        observations : Optional[Dict[str, np.ndarray]], default=None
            Observation data
        quantiles : List[float], default=[0.05, 0.25, 0.75, 0.95]
            Quantiles to plot
        title : str, default="Prediction Intervals"
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Compute quantiles
        prediction_quantiles = np.percentile(predictions, 
                                           [q*100 for q in quantiles], 
                                           axis=0)
        
        mean_pred = np.mean(predictions, axis=0)
        
        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        mean_sorted = mean_pred[sort_idx]
        
        # Plot mean
        ax.plot(x_sorted, mean_sorted, 
               color=self.colors['mean'], linewidth=2,
               label='Mean prediction')
        
        # Plot prediction intervals
        colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(quantiles)//2))
        
        for i in range(len(quantiles)//2):
            lower_q = prediction_quantiles[i][sort_idx]
            upper_q = prediction_quantiles[-(i+1)][sort_idx]
            
            lower_pct = int(quantiles[i] * 100)
            upper_pct = int(quantiles[-(i+1)] * 100)
            
            ax.fill_between(x_sorted, lower_q, upper_q,
                           alpha=0.4, color=colors[i],
                           label=f'{lower_pct}-{upper_pct}% prediction interval')
        
        # Plot true function
        if true_function is not None:
            true_sorted = true_function[sort_idx]
            ax.plot(x_sorted, true_sorted, '--', 
                   color='red', linewidth=2, alpha=0.8,
                   label='True function')
        
        # Plot observations
        if observations is not None:
            obs_x = observations.get('x', observations.get('points', []))
            obs_y = observations.get('y', observations.get('values', []))
            
            if len(obs_x) > 0 and len(obs_y) > 0:
                ax.scatter(obs_x, obs_y,
                          color=self.colors['observations'],
                          s=30, alpha=0.8, edgecolors='black', linewidth=0.5,
                          label='Observations', zorder=5)
        
        ax.set_xlabel('x', fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_ylabel('Prediction', fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_title(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        ax.legend(fontsize=PlottingConfig.FONT_SIZES['legend'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_certified_bounds(self, samples: np.ndarray,
                             parameter_names: List[str],
                             certified_bounds: Dict[str, Tuple[float, float]],
                             true_values: Optional[Dict[str, float]] = None,
                             credible_intervals: Optional[Dict[float, Dict[str, Tuple[float, float]]]] = None,
                             title: str = "Certified Uncertainty Bounds",
                             **kwargs) -> plt.Figure:
        """
        Plot certified bounds comparison with empirical intervals.
        
        Parameters
        ----------
        samples : np.ndarray, shape (n_samples, n_params)
            MCMC samples
        parameter_names : List[str]
            Parameter names
        certified_bounds : Dict[str, Tuple[float, float]]
            Certified bounds for each parameter
        true_values : Optional[Dict[str, float]], default=None
            True parameter values
        credible_intervals : Optional[Dict[float, Dict[str, Tuple[float, float]]]], default=None
            Credible intervals at different levels
        title : str, default="Certified Uncertainty Bounds"
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        n_params = len(parameter_names)
        fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 6))
        
        if n_params == 1:
            axes = [axes]
        
        for i, (ax, param_name) in enumerate(zip(axes, parameter_names)):
            param_samples = samples[:, i]
            
            # Plot histogram
            ax.hist(param_samples, bins=50, density=True, alpha=0.7,
                   color='lightblue', label='Posterior samples')
            
            # Get y-limits for vertical lines
            y_max = ax.get_ylim()[1]
            
            # Plot certified bounds
            cert_lower, cert_upper = certified_bounds[param_name]
            ax.axvspan(cert_lower, cert_upper, alpha=0.2, color='red',
                      label=f'Certified bounds')
            ax.axvline(cert_lower, color='red', linestyle='-', linewidth=2)
            ax.axvline(cert_upper, color='red', linestyle='-', linewidth=2)
            
            # Plot credible intervals if provided
            if credible_intervals is not None:
                colors = ['blue', 'green', 'purple']
                for j, (conf_level, intervals) in enumerate(credible_intervals.items()):
                    if param_name in intervals:
                        ci_lower, ci_upper = intervals[param_name]
                        color = colors[j % len(colors)]
                        ax.axvline(ci_lower, color=color, linestyle='--', 
                                  alpha=0.7, linewidth=1.5)
                        ax.axvline(ci_upper, color=color, linestyle='--',
                                  alpha=0.7, linewidth=1.5)
                        ax.text(ci_lower, y_max * 0.9, f'{int(conf_level*100)}%',
                               rotation=90, color=color, fontsize=9)
            
            # Plot true value if provided
            if true_values and param_name in true_values:
                true_val = true_values[param_name]
                ax.axvline(true_val, color='darkgreen', linestyle=':', linewidth=3,
                          label=f'True value: {true_val:.3f}')
                
                # Check if true value is in certified bounds
                in_bounds = cert_lower <= true_val <= cert_upper
                coverage_text = "✓" if in_bounds else "✗"
                ax.text(true_val, y_max * 0.8, coverage_text, 
                       fontsize=16, ha='center', 
                       color='green' if in_bounds else 'red')
            
            # Formatting
            ax.set_xlabel(param_name.replace('_', ' ').title(),
                         fontsize=PlottingConfig.FONT_SIZES['axis_label'])
            ax.set_ylabel('Density',
                         fontsize=PlottingConfig.FONT_SIZES['axis_label'])
            ax.set_title(f'{param_name}',
                        fontsize=PlottingConfig.FONT_SIZES['subtitle'])
            ax.grid(True, alpha=0.3)
            
            # Add legend to first subplot
            if i == 0:
                ax.legend(fontsize=PlottingConfig.FONT_SIZES['legend'])
        
        fig.suptitle(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        plt.tight_layout()
        return fig
    
    def plot_coverage_analysis(self, coverage_results: Dict[str, Any],
                              methods: List[str],
                              confidence_levels: List[float] = [0.68, 0.95, 0.99],
                              title: str = "Coverage Analysis",
                              **kwargs) -> plt.Figure:
        """
        Plot coverage analysis for different uncertainty quantification methods.
        
        Parameters
        ----------
        coverage_results : Dict[str, Any]
            Coverage results for different methods
        methods : List[str]
            Names of UQ methods
        confidence_levels : List[float], default=[0.68, 0.95, 0.99]
            Confidence levels to analyze
        title : str, default="Coverage Analysis"
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Coverage vs Target
        x_pos = np.arange(len(confidence_levels))
        width = 0.8 / len(methods)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            if method not in coverage_results:
                continue
                
            method_results = coverage_results[method]
            coverages = [method_results.get(f'coverage_{int(cl*100)}', 0) 
                        for cl in confidence_levels]
            
            ax1.bar(x_pos + i*width, coverages, width, 
                   label=method, color=colors[i], alpha=0.8)
        
        # Perfect coverage line
        ax1.plot(x_pos, confidence_levels, 'r--', linewidth=2,
                label='Target coverage')
        
        ax1.set_xlabel('Confidence Level')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_title('Coverage vs Target')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Interval Widths
        for i, method in enumerate(methods):
            if method not in coverage_results:
                continue
                
            method_results = coverage_results[method]
            widths = [method_results.get(f'width_{int(cl*100)}', 0) 
                     for cl in confidence_levels]
            
            ax2.bar(x_pos + i*width, widths, width,
                   label=method, color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Confidence Level')
        ax2.set_ylabel('Average Interval Width')
        ax2.set_title('Interval Widths')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_propagation(self, input_samples: np.ndarray,
                                   output_samples: np.ndarray,
                                   input_names: List[str],
                                   output_name: str = "Output",
                                   title: str = "Uncertainty Propagation",
                                   **kwargs) -> plt.Figure:
        """
        Plot uncertainty propagation from inputs to outputs.
        
        Parameters
        ----------
        input_samples : np.ndarray, shape (n_samples, n_inputs)
            Input parameter samples
        output_samples : np.ndarray, shape (n_samples,)
            Output samples
        input_names : List[str]
            Input parameter names
        output_name : str, default="Output"
            Output variable name
        title : str, default="Uncertainty Propagation"
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        n_inputs = len(input_names)
        
        # Create subplot grid
        fig = plt.figure(figsize=(12, 4*((n_inputs+1)//2)))
        
        # Plot input distributions
        for i, input_name in enumerate(input_names):
            ax = plt.subplot(2, (n_inputs+1)//2 + 1, i+1)
            
            input_vals = input_samples[:, i]
            ax.hist(input_vals, bins=30, density=True, alpha=0.7,
                   color='lightblue', edgecolor='black')
            
            ax.set_xlabel(input_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Input: {input_name}')
            ax.grid(True, alpha=0.3)
        
        # Plot output distribution
        ax_output = plt.subplot(2, (n_inputs+1)//2 + 1, n_inputs+1)
        ax_output.hist(output_samples, bins=30, density=True, alpha=0.7,
                      color='lightcoral', edgecolor='black')
        
        ax_output.set_xlabel(output_name)
        ax_output.set_ylabel('Density')
        ax_output.set_title(f'Output: {output_name}')
        ax_output.grid(True, alpha=0.3)
        
        # Plot input-output relationships
        if n_inputs <= 3:  # Only for small number of inputs
            for i, input_name in enumerate(input_names):
                ax = plt.subplot(2, (n_inputs+1)//2 + 1, n_inputs+2+i)
                
                ax.scatter(input_samples[:, i], output_samples,
                          alpha=0.6, s=1, color='blue')
                
                # Add trend line
                z = np.polyfit(input_samples[:, i], output_samples, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(np.min(input_samples[:, i]),
                                    np.max(input_samples[:, i]), 100)
                ax.plot(x_trend, p(x_trend), 'r--', linewidth=2)
                
                ax.set_xlabel(input_name)
                ax.set_ylabel(output_name)
                ax.set_title(f'{input_name} vs {output_name}')
                ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        plt.tight_layout()
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_indices: Dict[str, Dict[str, float]],
                                 parameter_names: List[str],
                                 title: str = "Sensitivity Analysis",
                                 **kwargs) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Parameters
        ----------
        sensitivity_indices : Dict[str, Dict[str, float]]
            Sensitivity indices for each parameter
        parameter_names : List[str]
            Parameter names
        title : str, default="Sensitivity Analysis"
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First-order sensitivity indices
        first_order = [sensitivity_indices[param].get('first_order', 0)
                      for param in parameter_names]
        
        bars1 = ax1.bar(parameter_names, first_order, 
                       color='skyblue', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('First-order Sensitivity')
        ax1.set_title('First-order Sensitivity Indices')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, first_order):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Total sensitivity indices
        total = [sensitivity_indices[param].get('total', 0)
                for param in parameter_names]
        
        bars2 = ax2.bar(parameter_names, total,
                       color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Total Sensitivity')
        ax2.set_title('Total Sensitivity Indices')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, total):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        fig.suptitle(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        plt.tight_layout()
        return fig
    
    def plot_error_bars_2d(self, x: np.ndarray, y: np.ndarray,
                          mean_values: np.ndarray, std_values: np.ndarray,
                          title: str = "2D Error Bars",
                          **kwargs) -> plt.Figure:
        """
        Plot 2D scatter plot with error bars.
        
        Parameters
        ----------
        x, y : np.ndarray
            Coordinate arrays
        mean_values : np.ndarray
            Mean values at each point
        std_values : np.ndarray
            Standard deviations at each point
        title : str, default="2D Error Bars"
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Color by mean value
        scatter = ax.scatter(x, y, c=mean_values, cmap='viridis', s=50,
                           edgecolors='black', linewidth=0.5)
        
        # Add error bars
        ax.errorbar(x, y, yerr=std_values, fmt='none', 
                   color='black', alpha=0.5, capsize=2)
        
        # Colorbar
        cbar = add_colorbar(fig, scatter, ax, label="Mean Value")
        
        ax.set_xlabel('x', fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_ylabel('y', fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_title(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str,
                   formats: List[str] = ['png', 'pdf'],
                   dpi: int = 300) -> None:
        """Save figure in multiple formats."""
        save_figure(fig, filename, formats, dpi)