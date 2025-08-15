"""
Solution Plotting

Publication-quality visualization tools for PDE solutions including
2D contour plots, 3D surface plots, and uncertainty visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings

from .plotting_utils import (
    PlottingConfig, add_colorbar, save_figure, 
    format_scientific_notation, setup_matplotlib_style
)


class SolutionPlotter:
    """
    Comprehensive solution plotting for PDE problems.
    
    Provides methods for visualizing 1D, 2D, and 3D PDE solutions
    with uncertainty quantification and publication-quality styling.
    
    Examples
    --------
    >>> plotter = SolutionPlotter()
    >>> fig = plotter.plot_2d_solution(solution, mesh, title="Heat Distribution")
    >>> plotter.save_figure(fig, "heat_solution")
    """
    
    def __init__(self, style: str = "academic", 
                 color_scheme: str = "academic",
                 figure_size: Tuple[float, float] = (10, 6)):
        """
        Initialize solution plotter.
        
        Parameters
        ----------
        style : str, default="academic"
            Plotting style ("academic", "presentation", "minimal")
        color_scheme : str, default="academic"  
            Color scheme for plots
        figure_size : Tuple[float, float], default=(10, 6)
            Default figure size in inches
        """
        self.style = style
        self.color_scheme = color_scheme
        self.figure_size = figure_size
        
        # Set up matplotlib style
        setup_matplotlib_style(style)
        
        # Color schemes
        self.colors = PlottingConfig.get_color_palette(8, color_scheme)
        
        # Custom colormaps
        self._setup_colormaps()
    
    def _setup_colormaps(self) -> None:
        """Setup custom colormaps for different solution types."""
        # Temperature colormap (blue to red)
        temp_colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000']
        self.temp_cmap = LinearSegmentedColormap.from_list('temperature', temp_colors)
        
        # Pressure colormap (white to dark blue)
        pressure_colors = ['#FFFFFF', '#E6F3FF', '#CCE7FF', '#99D6FF', '#66C2FF', '#0080FF', '#0040FF']
        self.pressure_cmap = LinearSegmentedColormap.from_list('pressure', pressure_colors)
        
        # Velocity colormap (diverging blue-white-red)
        velocity_colors = ['#000080', '#4040FF', '#8080FF', '#FFFFFF', '#FF8080', '#FF4040', '#800000']
        self.velocity_cmap = LinearSegmentedColormap.from_list('velocity', velocity_colors)
    
    def plot_1d_solution(self, x: np.ndarray, solution: np.ndarray,
                        title: str = "1D PDE Solution",
                        xlabel: str = "x", ylabel: str = "u(x)",
                        true_solution: Optional[np.ndarray] = None,
                        uncertainty: Optional[np.ndarray] = None,
                        confidence_level: float = 0.95,
                        **kwargs) -> plt.Figure:
        """
        Plot 1D PDE solution with optional uncertainty bands.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates
        solution : np.ndarray
            Solution values (mean if uncertainty provided)
        title : str, default="1D PDE Solution"
            Plot title
        xlabel, ylabel : str
            Axis labels
        true_solution : Optional[np.ndarray], default=None
            True solution for comparison
        uncertainty : Optional[np.ndarray], default=None
            Uncertainty (standard deviation) at each point
        confidence_level : float, default=0.95
            Confidence level for uncertainty bands
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Main solution
        line_color = kwargs.get('color', self.colors[0])
        linewidth = kwargs.get('linewidth', 2)
        
        ax.plot(x, solution, color=line_color, linewidth=linewidth, 
                label='Estimated solution', **{k: v for k, v in kwargs.items() 
                                             if k not in ['color', 'linewidth']})
        
        # Uncertainty bands
        if uncertainty is not None:
            alpha = 1 - confidence_level
            z_score = abs(stats.norm.ppf(alpha / 2))  # For normal distribution
            
            lower_bound = solution - z_score * uncertainty
            upper_bound = solution + z_score * uncertainty
            
            ax.fill_between(x, lower_bound, upper_bound, 
                          alpha=0.3, color=line_color,
                          label=f'{int(confidence_level*100)}% confidence band')
        
        # True solution for comparison
        if true_solution is not None:
            ax.plot(x, true_solution, '--', color=self.colors[1], 
                   linewidth=linewidth, label='True solution')
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_ylabel(ylabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_title(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        ax.legend(fontsize=PlottingConfig.FONT_SIZES['legend'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_2d_solution(self, solution: np.ndarray, mesh: Dict[str, Any],
                        title: str = "2D PDE Solution",
                        xlabel: str = "x", ylabel: str = "y",
                        zlabel: str = "u(x,y)",
                        colormap: str = "viridis",
                        contour_levels: Optional[int] = None,
                        show_contours: bool = True,
                        **kwargs) -> plt.Figure:
        """
        Plot 2D PDE solution as contour plot.
        
        Parameters
        ----------
        solution : np.ndarray
            Solution values (flattened or 2D array)
        mesh : Dict[str, Any]
            Mesh information containing 'X', 'Y' coordinate grids
        title : str, default="2D PDE Solution"
            Plot title
        xlabel, ylabel, zlabel : str
            Axis labels
        colormap : str, default="viridis"
            Colormap name or custom colormap
        contour_levels : Optional[int], default=None
            Number of contour levels (auto-determined if None)
        show_contours : bool, default=True
            Whether to show contour lines
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Get coordinate grids
        if 'X' in mesh and 'Y' in mesh:
            X, Y = mesh['X'], mesh['Y']
        else:
            raise ValueError("Mesh must contain 'X' and 'Y' coordinate grids")
        
        # Reshape solution if needed
        if solution.ndim == 1:
            solution_2d = solution.reshape(X.shape)
        else:
            solution_2d = solution
        
        # Get colormap
        if isinstance(colormap, str):
            if hasattr(self, f'{colormap}_cmap'):
                cmap = getattr(self, f'{colormap}_cmap')
            else:
                cmap = plt.get_cmap(colormap)
        else:
            cmap = colormap
        
        # Determine contour levels
        if contour_levels is None:
            contour_levels = 20
        
        # Main contour plot
        if show_contours:
            contour = ax.contour(X, Y, solution_2d, levels=contour_levels, 
                               colors='black', alpha=0.4, linewidths=0.5)
        
        im = ax.contourf(X, Y, solution_2d, levels=contour_levels, 
                        cmap=cmap, **kwargs)
        
        # Colorbar
        cbar = add_colorbar(fig, im, ax, label=zlabel)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_ylabel(ylabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_title(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_3d_surface(self, solution: np.ndarray, mesh: Dict[str, Any],
                       title: str = "3D PDE Solution",
                       xlabel: str = "x", ylabel: str = "y", zlabel: str = "u(x,y)",
                       colormap: str = "viridis",
                       view_angle: Tuple[float, float] = (30, 45),
                       **kwargs) -> plt.Figure:
        """
        Plot 2D PDE solution as 3D surface.
        
        Parameters
        ----------
        solution : np.ndarray
            Solution values
        mesh : Dict[str, Any]
            Mesh information
        title : str, default="3D PDE Solution"
            Plot title
        xlabel, ylabel, zlabel : str
            Axis labels
        colormap : str, default="viridis"
            Colormap name
        view_angle : Tuple[float, float], default=(30, 45)
            3D viewing angle (elevation, azimuth)
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get coordinate grids
        X, Y = mesh['X'], mesh['Y']
        
        # Reshape solution if needed
        if solution.ndim == 1:
            solution_2d = solution.reshape(X.shape)
        else:
            solution_2d = solution
        
        # Surface plot
        surf = ax.plot_surface(X, Y, solution_2d, cmap=colormap,
                              alpha=0.9, **kwargs)
        
        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label(zlabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_ylabel(ylabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_zlabel(zlabel, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        ax.set_title(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        plt.tight_layout()
        return fig
    
    def plot_solution_comparison(self, solutions: Dict[str, np.ndarray],
                               mesh: Dict[str, Any],
                               titles: Optional[List[str]] = None,
                               suptitle: str = "Solution Comparison",
                               colormap: str = "viridis",
                               **kwargs) -> plt.Figure:
        """
        Compare multiple solutions side by side.
        
        Parameters
        ----------
        solutions : Dict[str, np.ndarray]
            Dictionary of solutions to compare
        mesh : Dict[str, Any]
            Mesh information
        titles : Optional[List[str]], default=None
            Custom titles for subplots
        suptitle : str, default="Solution Comparison"
            Main figure title
        colormap : str, default="viridis"
            Colormap name
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        n_solutions = len(solutions)
        
        # Determine subplot layout
        if n_solutions <= 2:
            nrows, ncols = 1, n_solutions
            figsize = (self.figure_size[0] * ncols, self.figure_size[1])
        elif n_solutions <= 4:
            nrows, ncols = 2, 2
            figsize = (self.figure_size[0], self.figure_size[1])
        else:
            ncols = int(np.ceil(np.sqrt(n_solutions)))
            nrows = int(np.ceil(n_solutions / ncols))
            figsize = (self.figure_size[0], self.figure_size[1] * nrows / 2)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_solutions == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get global colorbar limits
        all_values = np.concatenate([sol.flatten() for sol in solutions.values()])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        X, Y = mesh['X'], mesh['Y']
        
        for i, (name, solution) in enumerate(solutions.items()):
            ax = axes[i]
            
            # Reshape solution if needed
            if solution.ndim == 1:
                solution_2d = solution.reshape(X.shape)
            else:
                solution_2d = solution
            
            # Plot
            im = ax.contourf(X, Y, solution_2d, levels=20, cmap=colormap,
                           vmin=vmin, vmax=vmax, **kwargs)
            ax.contour(X, Y, solution_2d, levels=20, colors='black', 
                      alpha=0.4, linewidths=0.5)
            
            # Title
            if titles and i < len(titles):
                title = titles[i]
            else:
                title = name
            ax.set_title(title, fontsize=PlottingConfig.FONT_SIZES['subtitle'])
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(n_solutions, len(axes)):
            axes[i].set_visible(False)
        
        # Add colorbar
        if n_solutions > 1:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Solution Value", fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        
        fig.suptitle(suptitle, fontsize=PlottingConfig.FONT_SIZES['title'])
        plt.tight_layout()
        
        return fig
    
    def plot_time_evolution(self, solutions: List[np.ndarray],
                          mesh: Dict[str, Any],
                          times: List[float],
                          title: str = "Time Evolution",
                          colormap: str = "viridis",
                          animation: bool = False,
                          **kwargs) -> plt.Figure:
        """
        Plot time evolution of PDE solution.
        
        Parameters
        ----------
        solutions : List[np.ndarray]
            List of solutions at different times
        mesh : Dict[str, Any]
            Mesh information
        times : List[float]
            Time values corresponding to solutions
        title : str, default="Time Evolution"
            Plot title
        colormap : str, default="viridis"
            Colormap name
        animation : bool, default=False
            Whether to create animation (requires additional setup)
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        n_times = len(solutions)
        if n_times != len(times):
            raise ValueError("Number of solutions must match number of times")
        
        # Determine layout
        if n_times <= 3:
            nrows, ncols = 1, n_times
            figsize = (self.figure_size[0] * ncols / 2, self.figure_size[1])
        elif n_times <= 6:
            nrows, ncols = 2, 3
            figsize = (self.figure_size[0], self.figure_size[1])
        else:
            ncols = int(np.ceil(np.sqrt(n_times)))
            nrows = int(np.ceil(n_times / ncols))
            figsize = (self.figure_size[0], self.figure_size[1] * nrows / 2)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_times == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Global colorbar limits
        all_values = np.concatenate([sol.flatten() for sol in solutions])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        X, Y = mesh['X'], mesh['Y']
        
        for i, (solution, time) in enumerate(zip(solutions, times)):
            ax = axes[i]
            
            # Reshape solution if needed
            if solution.ndim == 1:
                solution_2d = solution.reshape(X.shape)
            else:
                solution_2d = solution
            
            # Plot
            im = ax.contourf(X, Y, solution_2d, levels=20, cmap=colormap,
                           vmin=vmin, vmax=vmax, **kwargs)
            
            ax.set_title(f't = {time:.3f}', fontsize=PlottingConfig.FONT_SIZES['subtitle'])
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(n_times, len(axes)):
            axes[i].set_visible(False)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Solution Value", fontsize=PlottingConfig.FONT_SIZES['axis_label'])
        
        fig.suptitle(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        plt.tight_layout()
        
        return fig
    
    def plot_residuals(self, observed: np.ndarray, predicted: np.ndarray,
                      observation_points: np.ndarray,
                      title: str = "Residual Analysis",
                      **kwargs) -> plt.Figure:
        """
        Plot residuals between observed and predicted values.
        
        Parameters
        ----------
        observed : np.ndarray
            Observed values
        predicted : np.ndarray
            Predicted values
        observation_points : np.ndarray
            Coordinates of observation points
        title : str, default="Residual Analysis"
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        residuals = observed - predicted
        
        if observation_points.shape[1] == 1:
            # 1D case
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_size[0], self.figure_size[1]*1.2))
            
            # Residuals vs position
            ax1.scatter(observation_points[:, 0], residuals, alpha=0.7, **kwargs)
            ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Position')
            ax1.grid(True, alpha=0.3)
            
            # Residual histogram
            ax2.hist(residuals, bins=20, density=True, alpha=0.7, **kwargs)
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Density')
            ax2.set_title('Residual Distribution')
            ax2.grid(True, alpha=0.3)
            
        else:
            # 2D case
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figure_size[0]*1.2, self.figure_size[1]*1.2))
            
            # Scatter plot of residuals
            scatter = ax1.scatter(observation_points[:, 0], observation_points[:, 1], 
                                c=residuals, cmap='RdBu_r', **kwargs)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title('Spatial Distribution of Residuals')
            ax1.set_aspect('equal')
            fig.colorbar(scatter, ax=ax1)
            
            # Residuals vs predicted
            ax2.scatter(predicted, residuals, alpha=0.7, **kwargs)
            ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals vs Predicted')
            ax2.grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy.stats import probplot
            probplot(residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (Normal Distribution)')
            ax3.grid(True, alpha=0.3)
            
            # Residual histogram
            ax4.hist(residuals, bins=20, density=True, alpha=0.7, **kwargs)
            ax4.set_xlabel('Residuals')
            ax4.set_ylabel('Density')
            ax4.set_title('Residual Distribution')
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=PlottingConfig.FONT_SIZES['title'])
        plt.tight_layout()
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str,
                   formats: List[str] = ['png', 'pdf'],
                   dpi: int = 300) -> None:
        """
        Save figure in multiple formats.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        filename : str
            Base filename (without extension)
        formats : List[str], default=['png', 'pdf']
            File formats to save
        dpi : int, default=300
            Resolution for raster formats
        """
        save_figure(fig, filename, formats, dpi)