"""
Plotting Utilities

Common utilities and configuration for creating publication-quality plots.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import seaborn as sns


class PlottingConfig:
    """Configuration class for consistent plotting style."""
    
    # Color palettes
    ACADEMIC_COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'tertiary': '#2ca02c',
        'quaternary': '#d62728',
        'quinary': '#9467bd',
        'senary': '#8c564b'
    }
    
    UNCERTAINTY_COLORS = {
        'mean': '#1f77b4',
        'confidence_band': '#1f77b4',
        'prediction_band': '#ff7f0e',
        'observations': '#d62728'
    }
    
    # Font configurations
    FONT_SIZES = {
        'title': 16,
        'subtitle': 14,
        'axis_label': 12,
        'tick_label': 10,
        'legend': 10,
        'annotation': 9
    }
    
    # Figure sizes (in inches)
    FIGURE_SIZES = {
        'single_column': (6, 4),
        'double_column': (12, 4),
        'square': (6, 6),
        'tall': (6, 8),
        'wide': (10, 6),
        'presentation': (10, 7)
    }
    
    # Line styles and markers
    LINE_STYLES = ['-', '--', '-.', ':']
    MARKERS = ['o', 's', '^', 'v', 'D', 'x', '+']
    
    @classmethod
    def get_color_palette(cls, n_colors: int, palette_type: str = 'academic') -> List[str]:
        """Get a color palette with specified number of colors."""
        if palette_type == 'academic':
            base_colors = list(cls.ACADEMIC_COLORS.values())
        elif palette_type == 'uncertainty':
            base_colors = list(cls.UNCERTAINTY_COLORS.values())
        else:
            base_colors = sns.color_palette("husl", n_colors)
            return base_colors
        
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            # Extend with additional colors
            extended = base_colors + sns.color_palette("husl", n_colors - len(base_colors))
            return extended
    
    @classmethod
    def get_style_cycle(cls, n_items: int) -> List[Dict[str, Any]]:
        """Get cycling styles for multiple items."""
        colors = cls.get_color_palette(n_items)
        styles = []
        
        for i in range(n_items):
            style = {
                'color': colors[i],
                'linestyle': cls.LINE_STYLES[i % len(cls.LINE_STYLES)],
                'marker': cls.MARKERS[i % len(cls.MARKERS)]
            }
            styles.append(style)
        
        return styles


def setup_matplotlib_style(style: str = 'academic') -> None:
    """
    Set up matplotlib style for publication-quality plots.
    
    Args:
        style: Style type ('academic', 'presentation', 'minimal')
    """
    if style == 'academic':
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set font to serif for academic publications
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Computer Modern Roman'],
            'font.size': PlottingConfig.FONT_SIZES['tick_label'],
            'axes.titlesize': PlottingConfig.FONT_SIZES['title'],
            'axes.labelsize': PlottingConfig.FONT_SIZES['axis_label'],
            'xtick.labelsize': PlottingConfig.FONT_SIZES['tick_label'],
            'ytick.labelsize': PlottingConfig.FONT_SIZES['tick_label'],
            'legend.fontsize': PlottingConfig.FONT_SIZES['legend'],
            'figure.titlesize': PlottingConfig.FONT_SIZES['title'],
            
            # Line and marker settings
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 0.5,
            
            # Grid and spines
            'grid.alpha': 0.3,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            # Figure settings
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-dark')
        
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': PlottingConfig.FONT_SIZES['axis_label'],
            'axes.titlesize': PlottingConfig.FONT_SIZES['title'] + 2,
            'axes.labelsize': PlottingConfig.FONT_SIZES['axis_label'] + 2,
            'xtick.labelsize': PlottingConfig.FONT_SIZES['tick_label'] + 2,
            'ytick.labelsize': PlottingConfig.FONT_SIZES['tick_label'] + 2,
            'legend.fontsize': PlottingConfig.FONT_SIZES['legend'] + 2,
            
            'lines.linewidth': 3,
            'lines.markersize': 8,
            'savefig.dpi': 150
        })
        
    elif style == 'minimal':
        plt.style.use('default')
        
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.2,
            'figure.facecolor': 'white'
        })


def create_figure_grid(n_plots: int, ncols: int = None, 
                      figsize: Tuple[float, float] = None,
                      subplot_kw: Dict[str, Any] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of subplots with automatic layout.
    
    Args:
        n_plots: Number of subplots needed
        ncols: Number of columns (auto-determined if None)
        figsize: Figure size
        subplot_kw: Subplot keyword arguments
        
    Returns:
        fig: Figure object
        axes: Array of axes objects
    """
    if ncols is None:
        ncols = min(3, n_plots)
    
    nrows = (n_plots + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * 4, nrows * 3)
    
    if subplot_kw is None:
        subplot_kw = {}
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=subplot_kw)
    
    # Handle single subplot case
    if n_plots == 1:
        axes = np.array([axes])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Hide extra subplots
    total_subplots = nrows * ncols
    if n_plots < total_subplots:
        for i in range(n_plots, total_subplots):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)
    
    return fig, axes


def add_colorbar(fig: plt.Figure, im, ax: plt.Axes, 
                label: str = "", orientation: str = "vertical",
                shrink: float = 0.8) -> mpl.colorbar.Colorbar:
    """
    Add a colorbar to a plot.
    
    Args:
        fig: Figure object
        im: Image/contour object
        ax: Axes object
        label: Colorbar label
        orientation: Colorbar orientation
        shrink: Shrink factor
        
    Returns:
        cbar: Colorbar object
    """
    if orientation == "vertical":
        cbar = fig.colorbar(im, ax=ax, shrink=shrink, aspect=20)
    else:
        cbar = fig.colorbar(im, ax=ax, shrink=shrink, aspect=20, orientation='horizontal')
    
    cbar.set_label(label, fontsize=PlottingConfig.FONT_SIZES['axis_label'])
    cbar.ax.tick_params(labelsize=PlottingConfig.FONT_SIZES['tick_label'])
    
    return cbar


def save_figure(fig: plt.Figure, filename: str, 
               formats: List[str] = ['png', 'pdf'],
               dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """
    Save figure in multiple formats.
    
    Args:
        fig: Figure object
        filename: Base filename (without extension)
        formats: List of formats to save
        dpi: Resolution for raster formats
        bbox_inches: Bbox setting for saving
    """
    for fmt in formats:
        full_filename = f"{filename}.{fmt}"
        fig.savefig(full_filename, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {full_filename}")


def format_scientific_notation(x: float, precision: int = 2) -> str:
    """Format number in scientific notation."""
    if x == 0:
        return "0"
    
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    
    if exponent == 0:
        return f"{x:.{precision}f}"
    else:
        return f"{mantissa:.{precision}f}×10$^{{{exponent}}}$"


def add_subplot_labels(axes: np.ndarray, labels: List[str] = None,
                      loc: str = "upper left", fontsize: int = None,
                      fontweight: str = "bold") -> None:
    """
    Add subplot labels (a), (b), (c), etc.
    
    Args:
        axes: Array of axes objects
        labels: Custom labels (auto-generated if None)
        loc: Label location
        fontsize: Font size
        fontweight: Font weight
    """
    if fontsize is None:
        fontsize = PlottingConfig.FONT_SIZES['annotation']
    
    axes_flat = axes.flatten() if axes.ndim > 1 else [axes]
    
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(len(axes_flat))]
    
    for ax, label in zip(axes_flat, labels):
        if ax.get_visible():
            ax.text(0.05, 0.95, label, transform=ax.transAxes,
                   fontsize=fontsize, fontweight=fontweight,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def create_custom_legend(handles: List, labels: List[str], 
                        title: str = "", ncol: int = 1,
                        loc: str = "best", frameon: bool = True) -> plt.Legend:
    """
    Create a custom legend with improved formatting.
    
    Args:
        handles: List of plot handles
        labels: List of labels
        title: Legend title
        ncol: Number of columns
        loc: Legend location
        frameon: Whether to draw frame
        
    Returns:
        legend: Legend object
    """
    legend = plt.legend(handles, labels, title=title, ncol=ncol, 
                       loc=loc, frameon=frameon,
                       fontsize=PlottingConfig.FONT_SIZES['legend'],
                       title_fontsize=PlottingConfig.FONT_SIZES['legend'])
    
    if frameon:
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.5)
    
    return legend