"""
Geothermal Parameter Estimation in Yellowstone National Park

This module implements a comprehensive geothermal data analysis system for
estimating subsurface thermal conductivity and heat source parameters using
real USGS temperature monitoring data from Yellowstone geothermal features.

Applications:
- Subsurface thermal conductivity mapping
- Geothermal resource assessment
- Heat source location and intensity estimation
- Validation against core sample measurements
- Geothermal energy potential evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial.distance import cdist
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import os


@dataclass
class GeothermalMeasurement:
    """Data structure for geothermal measurements."""
    location: Tuple[float, float, float]  # (lat, lon, depth)
    temperature: float
    measurement_time: datetime
    feature_name: str
    measurement_type: str  # 'surface', 'shallow', 'deep'
    uncertainty: float


@dataclass
class CoreSample:
    """Data structure for core sample validation data."""
    location: Tuple[float, float, float]
    thermal_conductivity: float
    porosity: float
    rock_type: str
    measurement_uncertainty: float


class GeothermalDataLoader:
    """
    Load and process USGS geothermal data from Yellowstone National Park.
    
    Interfaces with USGS National Water Information System (NWIS) and
    Yellowstone Volcano Observatory data to collect temperature measurements
    from geothermal features.
    """
    
    def __init__(self, cache_dir: str = "data/geothermal_cache"):
        """
        Initialize geothermal data loader.
        
        Parameters:
        -----------
        cache_dir : str
            Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.measurements = []
        self.core_samples = []
        
    def load_usgs_temperature_data(self, site_codes: List[str],
                                 start_date: str, end_date: str) -> List[GeothermalMeasurement]:
        """
        Load temperature data from USGS monitoring stations.
        
        Parameters:
        -----------
        site_codes : List[str]
            USGS site identification codes for Yellowstone stations
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        List[GeothermalMeasurement]
            List of temperature measurements
        """
        measurements = []
        
        # USGS Water Services API base URL
        base_url = "https://waterservices.usgs.gov/nwis/dv/"
        
        for site_code in site_codes:
            cache_file = os.path.join(self.cache_dir, f"usgs_{site_code}_{start_date}_{end_date}.json")
            
            # Check cache first
            if os.path.exists(cache_file):
                print(f"Loading cached data for site {site_code}")
                with open(cache_file, 'r') as f:
                    data = json.load(f)
            else:
                print(f"Downloading data for site {site_code}")
                
                # Construct API request
                params = {
                    'sites': site_code,
                    'startDT': start_date,
                    'endDT': end_date,
                    'parameterCd': '00010',  # Temperature parameter code
                    'format': 'json',
                    'siteStatus': 'all'
                }
                
                try:
                    response = requests.get(base_url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Cache the data
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)
                        
                except requests.RequestException as e:
                    print(f"Failed to download data for site {site_code}: {str(e)}")
                    continue
            
            # Process the data
            measurements.extend(self._process_usgs_data(data, site_code))
        
        self.measurements.extend(measurements)
        return measurements
    
    def _process_usgs_data(self, data: Dict, site_code: str) -> List[GeothermalMeasurement]:
        """Process raw USGS JSON data into GeothermalMeasurement objects."""
        measurements = []
        
        try:
            time_series = data['value']['timeSeries']
            
            for series in time_series:
                site_info = series['sourceInfo']
                lat = float(site_info['geoLocation']['geogLocation']['latitude'])
                lon = float(site_info['geoLocation']['geogLocation']['longitude'])
                
                # Get site name and feature type
                site_name = site_info.get('siteName', f'Site_{site_code}')
                
                # Estimate depth based on site type (simplified)
                if 'spring' in site_name.lower():
                    depth = 0.5  # Surface spring
                elif 'well' in site_name.lower():
                    depth = 10.0  # Shallow well
                else:
                    depth = 1.0  # Default shallow
                
                # Process temperature values
                values = series['values'][0]['value']
                
                for value_entry in values:
                    try:
                        temp_celsius = float(value_entry['value'])
                        date_str = value_entry['dateTime']
                        measurement_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        
                        # Quality control - reasonable temperature range
                        if 0 <= temp_celsius <= 100:  # Celsius
                            measurement = GeothermalMeasurement(
                                location=(lat, lon, depth),
                                temperature=temp_celsius,
                                measurement_time=measurement_time,
                                feature_name=site_name,
                                measurement_type='surface' if depth < 2 else 'shallow',
                                uncertainty=0.5  # ±0.5°C measurement uncertainty
                            )
                            measurements.append(measurement)
                            
                    except (ValueError, KeyError) as e:
                        continue  # Skip invalid data points
                        
        except KeyError as e:
            print(f"Data format error for site {site_code}: {str(e)}")
        
        return measurements
    
    def load_synthetic_temperature_data(self, n_features: int = 20,
                                      yellowstone_bounds: Tuple[float, float, float, float] = None) -> List[GeothermalMeasurement]:
        """
        Generate synthetic temperature data for demonstration purposes.
        
        Parameters:
        -----------
        n_features : int
            Number of geothermal features to simulate
        yellowstone_bounds : Tuple[float, float, float, float]
            (min_lat, max_lat, min_lon, max_lon) bounds for Yellowstone
            
        Returns:
        --------
        List[GeothermalMeasurement]
            Synthetic temperature measurements
        """
        if yellowstone_bounds is None:
            # Approximate Yellowstone National Park bounds
            yellowstone_bounds = (44.13, 45.11, -111.15, -109.93)
        
        min_lat, max_lat, min_lon, max_lon = yellowstone_bounds
        
        measurements = []
        np.random.seed(42)  # For reproducibility
        
        # Known geothermal feature locations (approximate)
        feature_locations = [
            (44.4605, -110.8288, "Old Faithful"),
            (44.5256, -110.8362, "Grand Prismatic Spring"),
            (44.7280, -110.7010, "Mammoth Hot Springs"),
            (44.6171, -110.4992, "Yellowstone Lake"),
            (44.5436, -110.8281, "Castle Geyser"),
        ]
        
        current_time = datetime.now()
        
        for i in range(n_features):
            if i < len(feature_locations):
                # Use known locations
                lat, lon, name = feature_locations[i]
            else:
                # Generate random locations
                lat = np.random.uniform(min_lat, max_lat)
                lon = np.random.uniform(min_lon, max_lon)
                name = f"Geothermal Feature {i+1}"
            
            # Generate multiple measurements per feature (different depths/times)
            n_measurements = np.random.randint(5, 15)
            
            for j in range(n_measurements):
                # Depth variation
                depth = np.random.exponential(2.0)  # Exponential distribution for depths
                
                # Temperature model: higher temperature at known geothermal areas
                # Base temperature from geothermal gradient
                base_temp = 10 + depth * 25  # 25°C per meter gradient
                
                # Add geothermal anomaly based on proximity to known features
                anomaly = 0
                for feat_lat, feat_lon, _ in feature_locations:
                    distance = np.sqrt((lat - feat_lat)**2 + (lon - feat_lon)**2)
                    anomaly += 30 * np.exp(-distance * 1000)  # Distance in degrees
                
                # Final temperature with noise
                temperature = base_temp + anomaly + np.random.normal(0, 2)
                temperature = max(0, temperature)  # Non-negative
                
                # Measurement time (random within last year)
                days_ago = np.random.randint(0, 365)
                measurement_time = current_time - timedelta(days=days_ago)
                
                # Measurement type based on depth
                if depth < 1:
                    meas_type = 'surface'
                elif depth < 5:
                    meas_type = 'shallow'
                else:
                    meas_type = 'deep'
                
                measurement = GeothermalMeasurement(
                    location=(lat, lon, depth),
                    temperature=temperature,
                    measurement_time=measurement_time,
                    feature_name=name,
                    measurement_type=meas_type,
                    uncertainty=0.5 + 0.1 * depth  # Uncertainty increases with depth
                )
                measurements.append(measurement)
        
        self.measurements.extend(measurements)
        return measurements
    
    def load_core_sample_data(self, sample_file: Optional[str] = None) -> List[CoreSample]:
        """
        Load core sample validation data.
        
        Parameters:
        -----------
        sample_file : str, optional
            Path to core sample data file (CSV format)
            
        Returns:
        --------
        List[CoreSample]
            Core sample measurements for validation
        """
        if sample_file and os.path.exists(sample_file):
            # Load from actual file
            df = pd.read_csv(sample_file)
            samples = []
            
            for _, row in df.iterrows():
                sample = CoreSample(
                    location=(row['latitude'], row['longitude'], row['depth']),
                    thermal_conductivity=row['thermal_conductivity'],
                    porosity=row['porosity'],
                    rock_type=row['rock_type'],
                    measurement_uncertainty=row.get('uncertainty', 0.1)
                )
                samples.append(sample)
        else:
            # Generate synthetic core sample data
            samples = self._generate_synthetic_core_samples()
        
        self.core_samples.extend(samples)
        return samples
    
    def _generate_synthetic_core_samples(self) -> List[CoreSample]:
        """Generate synthetic core sample data for demonstration."""
        samples = []
        np.random.seed(42)
        
        # Yellowstone geology: mix of volcanic rocks with varying properties
        rock_types = ['rhyolite', 'basalt', 'andesite', 'tuff', 'obsidian']
        
        # Typical thermal conductivity ranges (W/m·K)
        conductivity_ranges = {
            'rhyolite': (2.0, 3.5),
            'basalt': (1.5, 2.8),
            'andesite': (2.2, 3.2),
            'tuff': (0.8, 1.8),
            'obsidian': (1.0, 1.5)
        }
        
        n_samples = 15
        yellowstone_bounds = (44.13, 45.11, -111.15, -109.93)
        min_lat, max_lat, min_lon, max_lon = yellowstone_bounds
        
        for i in range(n_samples):
            # Random location
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            depth = np.random.uniform(5, 50)  # 5-50 meter cores
            
            # Random rock type
            rock_type = np.random.choice(rock_types)
            
            # Thermal conductivity based on rock type
            k_min, k_max = conductivity_ranges[rock_type]
            thermal_conductivity = np.random.uniform(k_min, k_max)
            
            # Porosity (affects thermal properties)
            if rock_type == 'tuff':
                porosity = np.random.uniform(0.2, 0.6)  # High porosity
            else:
                porosity = np.random.uniform(0.05, 0.25)  # Lower porosity
            
            # Measurement uncertainty
            uncertainty = 0.05 + 0.02 * np.random.rand()  # 5-7% uncertainty
            
            sample = CoreSample(
                location=(lat, lon, depth),
                thermal_conductivity=thermal_conductivity,
                porosity=porosity,
                rock_type=rock_type,
                measurement_uncertainty=uncertainty
            )
            samples.append(sample)
        
        return samples
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for loaded data."""
        if not self.measurements:
            return {"error": "No measurements loaded"}
        
        temperatures = [m.temperature for m in self.measurements]
        depths = [m.location[2] for m in self.measurements]
        
        stats = {
            'n_measurements': len(self.measurements),
            'n_features': len(set(m.feature_name for m in self.measurements)),
            'temperature_stats': {
                'mean': np.mean(temperatures),
                'std': np.std(temperatures),
                'min': np.min(temperatures),
                'max': np.max(temperatures)
            },
            'depth_stats': {
                'mean': np.mean(depths),
                'std': np.std(depths),
                'min': np.min(depths),
                'max': np.max(depths)
            },
            'measurement_types': {}
        }
        
        # Count by measurement type
        for m_type in ['surface', 'shallow', 'deep']:
            count = len([m for m in self.measurements if m.measurement_type == m_type])
            stats['measurement_types'][m_type] = count
        
        if self.core_samples:
            conductivities = [s.thermal_conductivity for s in self.core_samples]
            stats['core_samples'] = {
                'n_samples': len(self.core_samples),
                'conductivity_mean': np.mean(conductivities),
                'conductivity_std': np.std(conductivities),
                'rock_types': list(set(s.rock_type for s in self.core_samples))
            }
        
        return stats


class SubsurfaceConductivityEstimator:
    """
    Estimate subsurface thermal conductivity and heat source parameters
    using Bayesian inference on geothermal temperature data.
    """
    
    def __init__(self, domain_bounds: Tuple[float, float, float, float],
                 grid_resolution: Tuple[int, int] = (50, 50)):
        """
        Initialize conductivity estimator.
        
        Parameters:
        -----------
        domain_bounds : Tuple[float, float, float, float]
            (min_lat, max_lat, min_lon, max_lon) for estimation domain
        grid_resolution : Tuple[int, int]
            Grid resolution for spatial discretization
        """
        self.domain_bounds = domain_bounds
        self.grid_resolution = grid_resolution
        
        # Create spatial grid
        min_lat, max_lat, min_lon, max_lon = domain_bounds
        self.lat_grid = np.linspace(min_lat, max_lat, grid_resolution[0])
        self.lon_grid = np.linspace(min_lon, max_lon, grid_resolution[1])
        self.lat_mesh, self.lon_mesh = np.meshgrid(self.lat_grid, self.lon_grid, indexing='ij')
        
        self.estimation_results = None
        
    def steady_state_heat_equation_forward(self, conductivity_field: np.ndarray,
                                         heat_sources: np.ndarray,
                                         boundary_temp: float = 10.0) -> np.ndarray:
        """
        Solve steady-state heat equation for temperature field.
        
        ∇·(k∇T) + Q = 0
        
        Parameters:
        -----------
        conductivity_field : np.ndarray, shape (nx, ny)
            Thermal conductivity field k(x,y)
        heat_sources : np.ndarray, shape (nx, ny)
            Heat source field Q(x,y)
        boundary_temp : float
            Boundary temperature (Dirichlet BC)
            
        Returns:
        --------
        np.ndarray, shape (nx, ny)
            Temperature field T(x,y)
        """
        nx, ny = self.grid_resolution
        
        # Grid spacing (assuming uniform)
        min_lat, max_lat, min_lon, max_lon = self.domain_bounds
        dx = (max_lat - min_lat) / (nx - 1) * 111000  # Convert to meters (approx)
        dy = (max_lon - min_lon) / (ny - 1) * 111000 * np.cos(np.radians(np.mean([min_lat, max_lat])))
        
        # Finite difference discretization
        # Using 5-point stencil for ∇·(k∇T)
        
        # Flatten arrays for linear system
        n_points = nx * ny
        A = np.zeros((n_points, n_points))
        b = np.zeros(n_points)
        
        def idx(i, j):
            """Convert 2D indices to 1D index."""
            return i * ny + j
        
        for i in range(nx):
            for j in range(ny):
                curr_idx = idx(i, j)
                
                # Boundary conditions
                if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                    A[curr_idx, curr_idx] = 1.0
                    b[curr_idx] = boundary_temp
                else:
                    # Interior points: finite difference discretization
                    # ∇·(k∇T) ≈ 1/dx² * [k(i+1/2,j)(T(i+1,j)-T(i,j)) - k(i-1/2,j)(T(i,j)-T(i-1,j))]
                    #          + 1/dy² * [k(i,j+1/2)(T(i,j+1)-T(i,j)) - k(i,j-1/2)(T(i,j)-T(i,j-1))]
                    
                    # Conductivity at half-grid points (harmonic mean)
                    k_ip = 2 * conductivity_field[i, j] * conductivity_field[i+1, j] / (
                        conductivity_field[i, j] + conductivity_field[i+1, j] + 1e-12)
                    k_im = 2 * conductivity_field[i-1, j] * conductivity_field[i, j] / (
                        conductivity_field[i-1, j] + conductivity_field[i, j] + 1e-12)
                    k_jp = 2 * conductivity_field[i, j] * conductivity_field[i, j+1] / (
                        conductivity_field[i, j] + conductivity_field[i, j+1] + 1e-12)
                    k_jm = 2 * conductivity_field[i, j-1] * conductivity_field[i, j] / (
                        conductivity_field[i, j-1] + conductivity_field[i, j] + 1e-12)
                    
                    # Coefficients
                    coeff_center = -(k_ip + k_im) / dx**2 - (k_jp + k_jm) / dy**2
                    coeff_ip = k_ip / dx**2
                    coeff_im = k_im / dx**2
                    coeff_jp = k_jp / dy**2
                    coeff_jm = k_jm / dy**2
                    
                    # Fill matrix
                    A[curr_idx, curr_idx] = coeff_center
                    A[curr_idx, idx(i+1, j)] = coeff_ip
                    A[curr_idx, idx(i-1, j)] = coeff_im
                    A[curr_idx, idx(i, j+1)] = coeff_jp
                    A[curr_idx, idx(i, j-1)] = coeff_jm
                    
                    # Right-hand side (heat source)
                    b[curr_idx] = -heat_sources[i, j]
        
        # Solve linear system
        try:
            temp_flat = np.linalg.solve(A, b)
            temperature_field = temp_flat.reshape((nx, ny))
        except np.linalg.LinAlgError:
            # Use least squares if singular
            temp_flat = np.linalg.lstsq(A, b, rcond=None)[0]
            temperature_field = temp_flat.reshape((nx, ny))
        
        return temperature_field
    
    def extract_observations_from_field(self, temperature_field: np.ndarray,
                                      measurements: List[GeothermalMeasurement]) -> np.ndarray:
        """
        Extract modeled temperatures at measurement locations.
        
        Parameters:
        -----------
        temperature_field : np.ndarray
            Computed temperature field
        measurements : List[GeothermalMeasurement]
            Measurement locations
            
        Returns:
        --------
        np.ndarray
            Interpolated temperatures at measurement points
        """
        # Extract coordinates
        meas_lats = np.array([m.location[0] for m in measurements])
        meas_lons = np.array([m.location[1] for m in measurements])
        
        # Grid coordinates
        grid_points = np.column_stack([self.lat_mesh.ravel(), self.lon_mesh.ravel()])
        grid_temps = temperature_field.ravel()
        
        # Interpolate to measurement locations
        meas_points = np.column_stack([meas_lats, meas_lons])
        
        try:
            interpolated_temps = griddata(grid_points, grid_temps, meas_points, 
                                        method='linear', fill_value=np.nan)
            
            # Fill NaN values with nearest neighbor
            nan_mask = np.isnan(interpolated_temps)
            if np.any(nan_mask):
                nn_temps = griddata(grid_points, grid_temps, meas_points[nan_mask], 
                                  method='nearest')
                interpolated_temps[nan_mask] = nn_temps
                
        except Exception:
            # Fallback: use nearest neighbor interpolation
            interpolated_temps = griddata(grid_points, grid_temps, meas_points, 
                                        method='nearest')
        
        return interpolated_temps
    
    def estimate_parameters(self, measurements: List[GeothermalMeasurement],
                          n_conductivity_zones: int = 5,
                          n_heat_sources: int = 3) -> Dict[str, Any]:
        """
        Estimate thermal conductivity zones and heat source parameters.
        
        Parameters:
        -----------
        measurements : List[GeothermalMeasurement]
            Temperature measurements
        n_conductivity_zones : int
            Number of conductivity zones to estimate
        n_heat_sources : int
            Number of discrete heat sources
            
        Returns:
        --------
        Dict[str, Any]
            Estimation results including parameters and uncertainties
        """
        print(f"Estimating parameters from {len(measurements)} measurements")
        print(f"Domain: {self.domain_bounds}")
        print(f"Grid resolution: {self.grid_resolution}")
        
        # Extract observation data
        obs_temps = np.array([m.temperature for m in measurements])
        obs_uncertainties = np.array([m.uncertainty for m in measurements])
        
        # Parameter vector: [conductivity_zones, heat_source_locations, heat_source_intensities]
        # Simplified approach: uniform conductivity + point sources
        
        # Initial parameter guess
        initial_conductivity = np.full(n_conductivity_zones, 2.0)  # W/m·K
        initial_source_locations = np.random.uniform(
            [self.domain_bounds[0], self.domain_bounds[2]], 
            [self.domain_bounds[1], self.domain_bounds[3]], 
            (n_heat_sources, 2)
        )
        initial_source_intensities = np.full(n_heat_sources, 1000.0)  # W/m²
        
        # Pack parameters
        initial_params = np.concatenate([
            initial_conductivity.ravel(),
            initial_source_locations.ravel(),
            initial_source_intensities.ravel()
        ])
        
        def unpack_parameters(params):
            """Unpack parameter vector."""
            n_k = n_conductivity_zones
            n_s_loc = n_heat_sources * 2
            n_s_int = n_heat_sources
            
            conductivities = params[:n_k]
            source_locs = params[n_k:n_k+n_s_loc].reshape((n_heat_sources, 2))
            source_ints = params[n_k+n_s_loc:n_k+n_s_loc+n_s_int]
            
            return conductivities, source_locs, source_ints
        
        def forward_model(params):
            """Forward model evaluation."""
            try:
                conductivities, source_locs, source_ints = unpack_parameters(params)
                
                # Create conductivity field (simplified: use first value uniformly)
                conductivity_field = np.full(self.grid_resolution, max(conductivities[0], 0.1))
                
                # Create heat source field
                heat_sources = np.zeros(self.grid_resolution)
                
                for (lat, lon), intensity in zip(source_locs, source_ints):
                    # Find nearest grid point
                    i_lat = np.argmin(np.abs(self.lat_grid - lat))
                    j_lon = np.argmin(np.abs(self.lon_grid - lon))
                    
                    if 0 <= i_lat < self.grid_resolution[0] and 0 <= j_lon < self.grid_resolution[1]:
                        heat_sources[i_lat, j_lon] += max(intensity, 0)
                
                # Solve heat equation
                temperature_field = self.steady_state_heat_equation_forward(
                    conductivity_field, heat_sources
                )
                
                # Extract at measurement locations
                predicted_temps = self.extract_observations_from_field(
                    temperature_field, measurements
                )
                
                return predicted_temps
                
            except Exception as e:
                print(f"Forward model error: {str(e)}")
                return np.full(len(measurements), np.nan)
        
        def objective_function(params):
            """Objective function for optimization."""
            predicted = forward_model(params)
            
            if np.any(np.isnan(predicted)):
                return 1e10
            
            # Weighted least squares
            residuals = (obs_temps - predicted) / obs_uncertainties
            return 0.5 * np.sum(residuals**2)
        
        # Optimization using scipy
        from scipy.optimize import minimize
        
        print("Running optimization...")
        start_time = time.time()
        
        # Set bounds
        bounds = []
        # Conductivity bounds
        for _ in range(n_conductivity_zones):
            bounds.append((0.1, 10.0))  # Reasonable conductivity range
        # Location bounds
        for _ in range(n_heat_sources):
            bounds.append((self.domain_bounds[0], self.domain_bounds[1]))  # Latitude
            bounds.append((self.domain_bounds[2], self.domain_bounds[3]))  # Longitude
        # Intensity bounds
        for _ in range(n_heat_sources):
            bounds.append((0.0, 10000.0))  # Heat source intensity
        
        try:
            result = minimize(
                objective_function, initial_params, 
                method='L-BFGS-B', bounds=bounds,
                options={'disp': True, 'maxiter': 500}
            )
            
            optimization_time = time.time() - start_time
            print(f"Optimization completed in {optimization_time:.2f} seconds")
            
            # Extract final parameters
            final_conductivities, final_source_locs, final_source_ints = unpack_parameters(result.x)
            
            # Final forward model evaluation
            final_predicted = forward_model(result.x)
            
            # Compute diagnostics
            residuals = obs_temps - final_predicted
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            r_squared = 1 - np.var(residuals) / np.var(obs_temps)
            
            self.estimation_results = {
                'conductivities': final_conductivities,
                'heat_source_locations': final_source_locs,
                'heat_source_intensities': final_source_ints,
                'predicted_temperatures': final_predicted,
                'residuals': residuals,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'optimization_success': result.success,
                'optimization_message': result.message,
                'optimization_time': optimization_time,
                'final_objective': result.fun
            }
            
            print(f"Estimation completed:")
            print(f"  RMSE: {rmse:.2f}°C")
            print(f"  MAE: {mae:.2f}°C") 
            print(f"  R²: {r_squared:.3f}")
            print(f"  Avg. conductivity: {np.mean(final_conductivities):.2f} W/m·K")
            
            return self.estimation_results
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return {'error': str(e)}


class CoreSampleValidator:
    """
    Validate geothermal parameter estimates against core sample measurements.
    """
    
    def __init__(self):
        """Initialize validator."""
        self.validation_results = None
    
    def validate_estimates(self, estimation_results: Dict[str, Any],
                         core_samples: List[CoreSample]) -> Dict[str, Any]:
        """
        Validate parameter estimates against core sample data.
        
        Parameters:
        -----------
        estimation_results : Dict[str, Any]
            Results from SubsurfaceConductivityEstimator
        core_samples : List[CoreSample]
            Core sample validation data
            
        Returns:
        --------
        Dict[str, Any]
            Validation metrics and analysis
        """
        print(f"Validating estimates against {len(core_samples)} core samples")
        
        if 'conductivities' not in estimation_results:
            return {'error': 'No conductivity estimates found'}
        
        # Extract core sample data
        core_conductivities = np.array([s.thermal_conductivity for s in core_samples])
        core_locations = np.array([s.location[:2] for s in core_samples])  # lat, lon
        core_uncertainties = np.array([s.measurement_uncertainty for s in core_samples])
        
        # For simplicity, compare against mean estimated conductivity
        # In reality, would interpolate spatially varying conductivity field
        estimated_conductivity = np.mean(estimation_results['conductivities'])
        
        # Statistical comparison
        bias = estimated_conductivity - np.mean(core_conductivities)
        relative_bias = bias / np.mean(core_conductivities)
        
        # Individual comparisons
        individual_errors = np.abs(estimated_conductivity - core_conductivities)
        mae = np.mean(individual_errors)
        rmse = np.sqrt(np.mean(individual_errors**2))
        
        # Coverage analysis (percentage of core samples within uncertainty bounds)
        # Assuming ±20% uncertainty on estimates for demonstration
        estimate_uncertainty = 0.2 * estimated_conductivity
        
        within_bounds = individual_errors <= (estimate_uncertainty + core_uncertainties * core_conductivities)
        coverage_rate = np.mean(within_bounds)
        
        # Correlation analysis
        from scipy.stats import pearsonr, spearmanr
        
        # For demonstration, use distance-based proxy for spatial correlation
        if len(core_samples) > 3:
            distances = cdist(core_locations, core_locations).mean(axis=1)
            dist_corr, dist_p = pearsonr(distances, core_conductivities)
        else:
            dist_corr, dist_p = 0.0, 1.0
        
        # Rock type analysis
        rock_type_analysis = {}
        rock_types = set(s.rock_type for s in core_samples)
        
        for rock_type in rock_types:
            rock_samples = [s for s in core_samples if s.rock_type == rock_type]
            rock_conductivities = [s.thermal_conductivity for s in rock_samples]
            
            rock_type_analysis[rock_type] = {
                'n_samples': len(rock_samples),
                'mean_conductivity': np.mean(rock_conductivities),
                'std_conductivity': np.std(rock_conductivities),
                'bias_from_estimate': estimated_conductivity - np.mean(rock_conductivities)
            }
        
        self.validation_results = {
            'estimated_conductivity': estimated_conductivity,
            'core_mean_conductivity': np.mean(core_conductivities),
            'core_std_conductivity': np.std(core_conductivities),
            'bias': bias,
            'relative_bias': relative_bias,
            'mae': mae,
            'rmse': rmse,
            'coverage_rate': coverage_rate,
            'spatial_correlation': dist_corr,
            'spatial_correlation_pvalue': dist_p,
            'rock_type_analysis': rock_type_analysis,
            'individual_comparisons': {
                'core_values': core_conductivities,
                'errors': individual_errors,
                'within_bounds': within_bounds
            }
        }
        
        print(f"Validation Results:")
        print(f"  Bias: {bias:.3f} W/m·K ({relative_bias*100:.1f}%)")
        print(f"  MAE: {mae:.3f} W/m·K")
        print(f"  RMSE: {rmse:.3f} W/m·K")
        print(f"  Coverage rate: {coverage_rate*100:.1f}%")
        
        return self.validation_results
    
    def generate_validation_plots(self, save_path: Optional[str] = None) -> plt.Figure:
        """Generate validation visualization plots."""
        if self.validation_results is None:
            raise ValueError("No validation results available. Run validate_estimates first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Estimated vs Core values
        ax = axes[0, 0]
        core_values = self.validation_results['individual_comparisons']['core_values']
        estimated_value = self.validation_results['estimated_conductivity']
        
        ax.scatter(core_values, [estimated_value] * len(core_values), 
                  alpha=0.7, s=60, color='steelblue')
        
        # Perfect agreement line
        min_val = min(np.min(core_values), estimated_value)
        max_val = max(np.max(core_values), estimated_value)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Agreement')
        
        ax.set_xlabel('Core Sample Conductivity (W/m·K)')
        ax.set_ylabel('Estimated Conductivity (W/m·K)')
        ax.set_title('Estimated vs Core Sample Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Error distribution
        ax = axes[0, 1]
        errors = self.validation_results['individual_comparisons']['errors']
        ax.hist(errors, bins=min(10, len(errors)), alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(self.validation_results['mae'], color='red', linestyle='--', 
                  label=f'MAE: {self.validation_results["mae"]:.3f}')
        ax.set_xlabel('Absolute Error (W/m·K)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Rock type comparison
        ax = axes[1, 0]
        rock_analysis = self.validation_results['rock_type_analysis']
        
        if rock_analysis:
            rock_types = list(rock_analysis.keys())
            rock_means = [rock_analysis[rt]['mean_conductivity'] for rt in rock_types]
            rock_stds = [rock_analysis[rt]['std_conductivity'] for rt in rock_types]
            
            x_pos = np.arange(len(rock_types))
            bars = ax.bar(x_pos, rock_means, yerr=rock_stds, capsize=5, 
                         alpha=0.7, color='lightgreen', edgecolor='black')
            
            # Add estimated value line
            ax.axhline(estimated_value, color='red', linestyle='--', 
                      label=f'Estimated: {estimated_value:.2f}')
            
            ax.set_xlabel('Rock Type')
            ax.set_ylabel('Thermal Conductivity (W/m·K)')
            ax.set_title('Conductivity by Rock Type')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rock_types, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Coverage analysis
        ax = axes[1, 1]
        within_bounds = self.validation_results['individual_comparisons']['within_bounds']
        coverage_rate = self.validation_results['coverage_rate']
        
        labels = ['Within Bounds', 'Outside Bounds']
        sizes = [np.sum(within_bounds), np.sum(~within_bounds)]
        colors = ['lightblue', 'lightcoral']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Coverage Analysis\n(Rate: {coverage_rate*100:.1f}%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def demonstrate_geothermal_analysis():
    """Demonstrate complete geothermal analysis workflow."""
    print("Yellowstone Geothermal Analysis Demonstration")
    print("=" * 50)
    
    # Initialize data loader
    loader = GeothermalDataLoader()
    
    # Load synthetic temperature data (real USGS data would require API access)
    print("\n1. Loading Temperature Data")
    measurements = loader.load_synthetic_temperature_data(n_features=15)
    
    # Load core sample data
    core_samples = loader.load_core_sample_data()
    
    # Print summary statistics
    stats = loader.get_summary_statistics()
    print(f"\nData Summary:")
    print(f"  Measurements: {stats['n_measurements']}")
    print(f"  Features: {stats['n_features']}")
    print(f"  Temperature range: {stats['temperature_stats']['min']:.1f} - {stats['temperature_stats']['max']:.1f}°C")
    print(f"  Core samples: {stats['core_samples']['n_samples']}")
    
    # Initialize estimator
    print("\n2. Parameter Estimation")
    yellowstone_bounds = (44.13, 45.11, -111.15, -109.93)
    estimator = SubsurfaceConductivityEstimator(
        domain_bounds=yellowstone_bounds,
        grid_resolution=(30, 30)
    )
    
    # Estimate parameters
    estimation_results = estimator.estimate_parameters(
        measurements, n_conductivity_zones=3, n_heat_sources=2
    )
    
    if 'error' not in estimation_results:
        # Validate against core samples
        print("\n3. Validation Against Core Samples")
        validator = CoreSampleValidator()
        validation_results = validator.validate_estimates(estimation_results, core_samples)
        
        # Generate validation plots
        fig = validator.generate_validation_plots('geothermal_validation.pdf')
        plt.show()
        
        print("\n4. Results Summary")
        print(f"  Estimated conductivity: {estimation_results['conductivities'][0]:.2f} W/m·K")
        print(f"  Heat sources: {len(estimation_results['heat_source_locations'])}")
        print(f"  Model RMSE: {estimation_results['rmse']:.2f}°C")
        print(f"  Validation bias: {validation_results['relative_bias']*100:.1f}%")
        
        return {
            'measurements': measurements,
            'core_samples': core_samples,
            'estimation_results': estimation_results,
            'validation_results': validation_results
        }
    else:
        print(f"Estimation failed: {estimation_results['error']}")
        return None


if __name__ == "__main__":
    import time
    
    # Suppress warnings for demonstration
    warnings.filterwarnings('ignore')
    
    results = demonstrate_geothermal_analysis()