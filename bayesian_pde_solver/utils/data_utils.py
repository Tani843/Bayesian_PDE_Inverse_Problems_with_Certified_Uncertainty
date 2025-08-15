"""
Data Utilities

Functions for data generation, manipulation, and preprocessing
for Bayesian PDE inverse problems.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, Union
import h5py
from pathlib import Path
import warnings


def generate_synthetic_data(
    forward_solver, 
    true_parameters: Dict[str, Any],
    boundary_conditions: Dict[str, Any],
    n_observations: int = 100,
    observation_bounds: Optional[Tuple[float, ...]] = None,
    noise_std: float = 0.01,
    noise_type: str = "gaussian",
    random_seed: Optional[int] = None,
    observation_pattern: str = "random"
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic observation data for inverse problems.
    
    Parameters
    ----------
    forward_solver : object
        Forward PDE solver instance
    true_parameters : Dict[str, Any]
        True parameter values for data generation
    boundary_conditions : Dict[str, Any]
        Boundary conditions for the PDE
    n_observations : int, default=100
        Number of observation points
    observation_bounds : Optional[Tuple[float, ...]], default=None
        Bounds for observation locations (uses solver domain if None)
    noise_std : float, default=0.01
        Standard deviation of observation noise
    noise_type : str, default="gaussian"
        Type of noise ("gaussian", "laplace", "student_t")
    random_seed : Optional[int], default=None
        Random seed for reproducibility
    observation_pattern : str, default="random"
        Pattern for observation locations ("random", "grid", "boundary")
        
    Returns
    -------
    data : Dict[str, np.ndarray]
        Dictionary containing observation data:
        - 'observation_points': locations of observations
        - 'observations': noisy observation values
        - 'true_solution': true solution field
        - 'true_observations': true values at observation points
        - 'noise': noise added to observations
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Solve forward problem with true parameters
    true_solution = forward_solver.solve(true_parameters, boundary_conditions)
    
    # Generate observation points
    if observation_bounds is None:
        observation_bounds = forward_solver.domain_bounds
    
    observation_points = _generate_observation_points(
        n_observations, observation_bounds, 
        forward_solver.dimension, observation_pattern
    )
    
    # Extract true values at observation points
    true_observations = forward_solver.compute_observables(true_solution, observation_points)
    
    # Add noise
    noise = _generate_noise(len(true_observations), noise_std, noise_type)
    observations = true_observations + noise
    
    return {
        'observation_points': observation_points,
        'observations': observations,
        'true_solution': true_solution,
        'true_observations': true_observations,
        'noise': noise,
        'true_parameters': true_parameters,
        'noise_std': noise_std,
        'n_observations': n_observations
    }


def _generate_observation_points(
    n_points: int,
    bounds: Tuple[float, ...],
    dimension: int,
    pattern: str = "random"
) -> np.ndarray:
    """Generate observation point locations."""
    
    if pattern == "random":
        # Random uniform distribution
        points = np.random.uniform(
            low=bounds[::2],
            high=bounds[1::2],
            size=(n_points, dimension)
        )
        
    elif pattern == "grid":
        # Regular grid
        if dimension == 1:
            points_per_dim = n_points
            x = np.linspace(bounds[0], bounds[1], points_per_dim)
            points = x.reshape(-1, 1)
        elif dimension == 2:
            points_per_dim = int(np.sqrt(n_points))
            x = np.linspace(bounds[0], bounds[1], points_per_dim)
            y = np.linspace(bounds[2], bounds[3], points_per_dim)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack([X.ravel(), Y.ravel()])
            # Trim to exact number if needed
            points = points[:n_points]
        else:
            raise NotImplementedError("Grid pattern for 3D not implemented")
            
    elif pattern == "boundary":
        # Points on domain boundary
        if dimension == 2:
            # Distribute points along four edges
            n_per_edge = n_points // 4
            remaining = n_points % 4
            
            points_list = []
            
            # Bottom edge
            x_bottom = np.linspace(bounds[0], bounds[1], n_per_edge + (1 if remaining > 0 else 0))
            y_bottom = np.full_like(x_bottom, bounds[2])
            points_list.append(np.column_stack([x_bottom, y_bottom]))
            if remaining > 0:
                remaining -= 1
            
            # Top edge  
            x_top = np.linspace(bounds[0], bounds[1], n_per_edge + (1 if remaining > 0 else 0))
            y_top = np.full_like(x_top, bounds[3])
            points_list.append(np.column_stack([x_top, y_top]))
            if remaining > 0:
                remaining -= 1
            
            # Left edge
            y_left = np.linspace(bounds[2], bounds[3], n_per_edge + (1 if remaining > 0 else 0))
            x_left = np.full_like(y_left, bounds[0])
            points_list.append(np.column_stack([x_left, y_left]))
            if remaining > 0:
                remaining -= 1
            
            # Right edge
            y_right = np.linspace(bounds[2], bounds[3], n_per_edge + (1 if remaining > 0 else 0))
            x_right = np.full_like(y_right, bounds[1])
            points_list.append(np.column_stack([x_right, y_right]))
            
            points = np.vstack(points_list)[:n_points]
        else:
            raise NotImplementedError("Boundary pattern only implemented for 2D")
    else:
        raise ValueError(f"Unknown observation pattern: {pattern}")
    
    return points


def _generate_noise(n_points: int, noise_std: float, noise_type: str = "gaussian") -> np.ndarray:
    """Generate noise for observations."""
    
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_std, n_points)
    elif noise_type == "laplace":
        # Laplace distribution with scale parameter
        scale = noise_std / np.sqrt(2)  # Convert std to scale
        noise = np.random.laplace(0, scale, n_points)
    elif noise_type == "student_t":
        # Student's t-distribution (df=3 for heavy tails)
        df = 3
        t_std = np.sqrt(df / (df - 2))  # Standard deviation of t-distribution
        scale = noise_std / t_std
        noise = np.random.standard_t(df, n_points) * scale
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noise


def add_noise(data: np.ndarray, 
              noise_std: Union[float, np.ndarray],
              noise_type: str = "gaussian",
              relative: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise to clean data.
    
    Parameters
    ----------
    data : np.ndarray
        Clean data
    noise_std : Union[float, np.ndarray]
        Noise standard deviation (constant or spatially varying)
    noise_type : str, default="gaussian"
        Type of noise
    relative : bool, default=False
        Whether noise is relative to signal magnitude
        
    Returns
    -------
    noisy_data : np.ndarray
        Data with added noise
    noise : np.ndarray
        Noise that was added
    """
    if relative:
        effective_std = noise_std * np.abs(data)
    else:
        effective_std = noise_std
    
    if isinstance(effective_std, np.ndarray):
        # Spatially varying noise
        noise = np.array([_generate_noise(1, std, noise_type)[0] 
                         for std in effective_std])
    else:
        # Constant noise
        noise = _generate_noise(len(data), effective_std, noise_type)
    
    noisy_data = data + noise
    return noisy_data, noise


def normalize_data(data: np.ndarray, 
                  method: str = "standard",
                  return_params: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    Normalize data for better numerical conditioning.
    
    Parameters
    ----------
    data : np.ndarray
        Data to normalize
    method : str, default="standard"
        Normalization method ("standard", "minmax", "robust")
    return_params : bool, default=False
        Whether to return normalization parameters
        
    Returns
    -------
    normalized_data : np.ndarray
        Normalized data
    params : Dict[str, float], optional
        Normalization parameters if return_params=True
    """
    if method == "standard":
        # Zero mean, unit variance
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            warnings.warn("Data has zero variance, returning original data")
            normalized = data
            params = {"mean": mean, "std": 1.0}
        else:
            normalized = (data - mean) / std
            params = {"mean": mean, "std": std}
            
    elif method == "minmax":
        # Scale to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            warnings.warn("Data has zero range, returning original data")
            normalized = data
            params = {"min": min_val, "max": max_val}
        else:
            normalized = (data - min_val) / (max_val - min_val)
            params = {"min": min_val, "max": max_val}
            
    elif method == "robust":
        # Use median and MAD for robust normalization
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            warnings.warn("Data has zero MAD, using standard normalization")
            return normalize_data(data, "standard", return_params)
        normalized = (data - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for Gaussian
        params = {"median": median, "mad": mad}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if return_params:
        return normalized, params
    else:
        return normalized


def denormalize_data(normalized_data: np.ndarray,
                    params: Dict[str, float],
                    method: str = "standard") -> np.ndarray:
    """
    Reverse data normalization.
    
    Parameters
    ----------
    normalized_data : np.ndarray
        Normalized data
    params : Dict[str, float]
        Normalization parameters
    method : str, default="standard"
        Normalization method used
        
    Returns
    -------
    original_data : np.ndarray
        Denormalized data
    """
    if method == "standard":
        return normalized_data * params["std"] + params["mean"]
    elif method == "minmax":
        return normalized_data * (params["max"] - params["min"]) + params["min"]
    elif method == "robust":
        return normalized_data * (1.4826 * params["mad"]) + params["median"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def split_data(observation_points: np.ndarray,
              observations: np.ndarray,
              split_ratio: float = 0.8,
              split_type: str = "random",
              random_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Split data into training and validation sets.
    
    Parameters
    ----------
    observation_points : np.ndarray
        Observation point coordinates
    observations : np.ndarray  
        Observation values
    split_ratio : float, default=0.8
        Fraction of data for training
    split_type : str, default="random"
        How to split data ("random", "spatial", "alternating")
    random_seed : Optional[int], default=None
        Random seed for reproducible splits
        
    Returns
    -------
    data_split : Dict[str, np.ndarray]
        Dictionary with train/validation data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_total = len(observations)
    n_train = int(n_total * split_ratio)
    
    if split_type == "random":
        # Random permutation
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
    elif split_type == "spatial":
        # Split based on spatial location (for 1D/2D)
        if observation_points.shape[1] == 1:
            # 1D: split at midpoint
            sorted_indices = np.argsort(observation_points[:, 0])
            train_indices = sorted_indices[:n_train]
            val_indices = sorted_indices[n_train:]
        elif observation_points.shape[1] == 2:
            # 2D: split along x-axis
            sorted_indices = np.argsort(observation_points[:, 0])
            train_indices = sorted_indices[:n_train]  
            val_indices = sorted_indices[n_train:]
        else:
            raise NotImplementedError("Spatial split only implemented for 1D/2D")
            
    elif split_type == "alternating":
        # Alternating pattern
        train_indices = np.arange(0, n_total, 2)[:n_train]
        val_indices = np.arange(1, n_total, 2)
        # Adjust if we need more training points
        if len(train_indices) < n_train:
            remaining = n_train - len(train_indices)
            extra_indices = val_indices[:remaining]
            train_indices = np.concatenate([train_indices, extra_indices])
            val_indices = val_indices[remaining:]
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    return {
        'train_points': observation_points[train_indices],
        'train_observations': observations[train_indices],
        'train_indices': train_indices,
        'val_points': observation_points[val_indices],
        'val_observations': observations[val_indices], 
        'val_indices': val_indices
    }


def load_data(file_path: Union[str, Path],
             format: str = "auto") -> Dict[str, np.ndarray]:
    """
    Load observation data from file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to data file
    format : str, default="auto"
        File format ("auto", "hdf5", "npz", "csv")
        
    Returns
    -------
    data : Dict[str, np.ndarray]
        Loaded data dictionary
    """
    file_path = Path(file_path)
    
    if format == "auto":
        format = file_path.suffix.lower()
    
    if format in [".h5", ".hdf5", "hdf5"]:
        with h5py.File(file_path, 'r') as f:
            data = {key: f[key][:] for key in f.keys()}
    elif format in [".npz", "npz"]:
        data_file = np.load(file_path)
        data = {key: data_file[key] for key in data_file.files}
    elif format in [".csv", "csv"]:
        import pandas as pd
        df = pd.read_csv(file_path)
        data = {col: df[col].values for col in df.columns}
    else:
        raise ValueError(f"Unsupported file format: {format}")
    
    return data


def save_data(data: Dict[str, np.ndarray],
             file_path: Union[str, Path],
             format: str = "auto",
             compression: bool = True) -> None:
    """
    Save observation data to file.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Data dictionary to save
    file_path : Union[str, Path] 
        Output file path
    format : str, default="auto"
        File format ("auto", "hdf5", "npz", "csv")
    compression : bool, default=True
        Whether to use compression
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "auto":
        format = file_path.suffix.lower()
    
    if format in [".h5", ".hdf5", "hdf5"]:
        compression_opts = "gzip" if compression else None
        with h5py.File(file_path, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value, compression=compression_opts)
    elif format in [".npz", "npz"]:
        if compression:
            np.savez_compressed(file_path, **data)
        else:
            np.savez(file_path, **data)
    elif format in [".csv", "csv"]:
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {format}")


def compute_data_statistics(observations: np.ndarray,
                           observation_points: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute basic statistics for observation data.
    
    Parameters
    ----------
    observations : np.ndarray
        Observation values
    observation_points : Optional[np.ndarray], default=None
        Observation locations
        
    Returns
    -------
    stats : Dict[str, float]
        Data statistics
    """
    stats = {
        'n_observations': len(observations),
        'mean': np.mean(observations),
        'median': np.median(observations),
        'std': np.std(observations, ddof=1),
        'min': np.min(observations),
        'max': np.max(observations),
        'range': np.max(observations) - np.min(observations),
        'skewness': float(stats.skew(observations)) if len(observations) > 2 else np.nan,
        'kurtosis': float(stats.kurtosis(observations)) if len(observations) > 3 else np.nan
    }
    
    # Spatial statistics if locations provided
    if observation_points is not None:
        if observation_points.shape[1] >= 1:
            stats['x_extent'] = np.max(observation_points[:, 0]) - np.min(observation_points[:, 0])
        if observation_points.shape[1] >= 2:
            stats['y_extent'] = np.max(observation_points[:, 1]) - np.min(observation_points[:, 1])
            stats['area_coverage'] = stats['x_extent'] * stats['y_extent']
    
    return stats