"""
Finite Difference PDE Solver

Implements finite difference methods for solving PDEs on structured grids.
Supports 1D and 2D problems with various boundary conditions.
"""

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Any, Tuple, Callable, Optional
from .forward_solver import ForwardSolver


class FiniteDifferenceSolver(ForwardSolver):
    """
    Finite difference solver for PDEs on structured grids.
    """
    
    def __init__(self, domain_bounds: Tuple[float, ...], 
                 mesh_size: Tuple[int, ...],
                 pde_type: str = "elliptic",
                 scheme: str = "central"):
        """
        Initialize finite difference solver.
        
        Args:
            domain_bounds: Domain boundaries (x_min, x_max, y_min, y_max, ...)
            mesh_size: Number of grid points (nx, ny, ...)
            pde_type: Type of PDE
            scheme: Finite difference scheme ("central", "upwind", "backward")
        """
        self.scheme = scheme
        super().__init__(domain_bounds, mesh_size, pde_type)
    
    def _setup_mesh(self) -> None:
        """Setup structured grid mesh."""
        if self.dimension == 1:
            x_min, x_max = self.domain_bounds
            nx = self.mesh_size[0]
            self.x = np.linspace(x_min, x_max, nx)
            self.dx = (x_max - x_min) / (nx - 1)
            self.dof_coordinates = self.x.reshape(-1, 1)
            self.mesh = {"x": self.x, "dx": self.dx}
            
        elif self.dimension == 2:
            x_min, x_max, y_min, y_max = self.domain_bounds
            nx, ny = self.mesh_size
            self.x = np.linspace(x_min, x_max, nx)
            self.y = np.linspace(y_min, y_max, ny)
            self.dx = (x_max - x_min) / (nx - 1)
            self.dy = (y_max - y_min) / (ny - 1)
            
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            self.dof_coordinates = np.column_stack([X.ravel(), Y.ravel()])
            self.mesh = {"x": self.x, "y": self.y, "dx": self.dx, "dy": self.dy,
                        "X": X, "Y": Y}
        else:
            raise NotImplementedError("3D finite difference not implemented")
    
    def assemble_system(self, parameters: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        Assemble finite difference system.
        
        Args:
            parameters: PDE parameters (diffusion, reaction, source, etc.)
            
        Returns:
            A: System matrix
            b: Right-hand side vector
        """
        if self.dimension == 1:
            return self._assemble_1d(parameters)
        elif self.dimension == 2:
            return self._assemble_2d(parameters)
        else:
            raise NotImplementedError("3D assembly not implemented")
    
    def _assemble_1d(self, parameters: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """Assemble 1D finite difference system."""
        nx = self.mesh_size[0]
        dx = self.mesh["dx"]
        
        # Get parameters
        diffusion = parameters.get("diffusion", 1.0)
        reaction = parameters.get("reaction", 0.0)
        source = parameters.get("source", 0.0)
        
        # Handle spatially varying parameters
        if callable(diffusion):
            diff_vals = np.array([diffusion(x) for x in self.x])
        elif isinstance(diffusion, np.ndarray):
            diff_vals = diffusion
        else:
            diff_vals = np.full(nx, diffusion)
            
        if callable(reaction):
            react_vals = np.array([reaction(x) for x in self.x])
        elif isinstance(reaction, np.ndarray):
            react_vals = reaction
        else:
            react_vals = np.full(nx, reaction)
            
        if callable(source):
            source_vals = np.array([source(x) for x in self.x])
        elif isinstance(source, np.ndarray):
            source_vals = source
        else:
            source_vals = np.full(nx, source)
        
        # Assemble matrix
        A = sp.lil_matrix((nx, nx))
        b = source_vals.copy()
        
        for i in range(1, nx - 1):
            # Central difference for diffusion term: -d/dx(D du/dx)
            if self.scheme == "central":
                D_left = 0.5 * (diff_vals[i] + diff_vals[i-1])
                D_right = 0.5 * (diff_vals[i] + diff_vals[i+1])
                
                A[i, i-1] = -D_left / dx**2
                A[i, i] = (D_left + D_right) / dx**2 + react_vals[i]
                A[i, i+1] = -D_right / dx**2
            else:
                raise NotImplementedError(f"Scheme {self.scheme} not implemented for 1D")
        
        return A.tocsc(), b
    
    def _assemble_2d(self, parameters: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """Assemble 2D finite difference system."""
        nx, ny = self.mesh_size
        dx, dy = self.mesh["dx"], self.mesh["dy"]
        X, Y = self.mesh["X"], self.mesh["Y"]
        
        # Get parameters
        diffusion = parameters.get("diffusion", 1.0)
        reaction = parameters.get("reaction", 0.0)
        source = parameters.get("source", 0.0)
        
        # Handle spatially varying parameters
        if callable(diffusion):
            diff_vals = np.array([[diffusion(x, y) for y in self.y] for x in self.x])
        elif isinstance(diffusion, np.ndarray):
            diff_vals = diffusion
        else:
            diff_vals = np.full((nx, ny), diffusion)
            
        if callable(reaction):
            react_vals = np.array([[reaction(x, y) for y in self.y] for x in self.x])
        elif isinstance(reaction, np.ndarray):
            react_vals = reaction
        else:
            react_vals = np.full((nx, ny), reaction)
            
        if callable(source):
            source_vals = np.array([[source(x, y) for y in self.y] for x in self.x])
        elif isinstance(source, np.ndarray):
            source_vals = source
        else:
            source_vals = np.full((nx, ny), source)
        
        # Total number of unknowns
        n_total = nx * ny
        A = sp.lil_matrix((n_total, n_total))
        b = source_vals.ravel()
        
        # Map 2D indices to 1D
        def idx(i, j):
            return i * ny + j
        
        # Interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                k = idx(i, j)
                
                # 5-point stencil for Laplacian
                D_center = diff_vals[i, j]
                
                # x-direction
                A[k, idx(i-1, j)] = -D_center / dx**2
                A[k, idx(i+1, j)] = -D_center / dx**2
                
                # y-direction  
                A[k, idx(i, j-1)] = -D_center / dy**2
                A[k, idx(i, j+1)] = -D_center / dy**2
                
                # Diagonal
                A[k, k] = 2 * D_center * (1/dx**2 + 1/dy**2) + react_vals[i, j]
        
        return A.tocsc(), b
    
    def apply_boundary_conditions(self, A: sp.csc_matrix, b: np.ndarray,
                                boundary_conditions: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """Apply boundary conditions to finite difference system."""
        A_bc = A.tolil()
        b_bc = b.copy()
        
        if self.dimension == 1:
            return self._apply_bc_1d(A_bc, b_bc, boundary_conditions)
        elif self.dimension == 2:
            return self._apply_bc_2d(A_bc, b_bc, boundary_conditions)
        else:
            raise NotImplementedError("3D boundary conditions not implemented")
    
    def _apply_bc_1d(self, A: sp.lil_matrix, b: np.ndarray,
                     boundary_conditions: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """Apply 1D boundary conditions."""
        nx = self.mesh_size[0]
        
        # Left boundary (x = x_min)
        left_bc = boundary_conditions.get("left", {"type": "dirichlet", "value": 0.0})
        if left_bc["type"] == "dirichlet":
            A[0, :] = 0
            A[0, 0] = 1
            b[0] = left_bc["value"]
        elif left_bc["type"] == "neumann":
            dx = self.mesh["dx"]
            A[0, :] = 0
            A[0, 0] = -1/dx
            A[0, 1] = 1/dx
            b[0] = left_bc["value"]
        
        # Right boundary (x = x_max)
        right_bc = boundary_conditions.get("right", {"type": "dirichlet", "value": 0.0})
        if right_bc["type"] == "dirichlet":
            A[nx-1, :] = 0
            A[nx-1, nx-1] = 1
            b[nx-1] = right_bc["value"]
        elif right_bc["type"] == "neumann":
            dx = self.mesh["dx"]
            A[nx-1, :] = 0
            A[nx-1, nx-2] = -1/dx
            A[nx-1, nx-1] = 1/dx
            b[nx-1] = right_bc["value"]
        
        return A.tocsc(), b
    
    def _apply_bc_2d(self, A: sp.lil_matrix, b: np.ndarray,
                     boundary_conditions: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """Apply 2D boundary conditions."""
        nx, ny = self.mesh_size
        
        def idx(i, j):
            return i * ny + j
        
        # Get boundary conditions
        left_bc = boundary_conditions.get("left", {"type": "dirichlet", "value": 0.0})
        right_bc = boundary_conditions.get("right", {"type": "dirichlet", "value": 0.0})
        bottom_bc = boundary_conditions.get("bottom", {"type": "dirichlet", "value": 0.0})
        top_bc = boundary_conditions.get("top", {"type": "dirichlet", "value": 0.0})
        
        # Left boundary (i = 0)
        for j in range(ny):
            k = idx(0, j)
            if left_bc["type"] == "dirichlet":
                A[k, :] = 0
                A[k, k] = 1
                if callable(left_bc["value"]):
                    b[k] = left_bc["value"](self.y[j])
                else:
                    b[k] = left_bc["value"]
        
        # Right boundary (i = nx-1)
        for j in range(ny):
            k = idx(nx-1, j)
            if right_bc["type"] == "dirichlet":
                A[k, :] = 0
                A[k, k] = 1
                if callable(right_bc["value"]):
                    b[k] = right_bc["value"](self.y[j])
                else:
                    b[k] = right_bc["value"]
        
        # Bottom boundary (j = 0)
        for i in range(nx):
            k = idx(i, 0)
            if bottom_bc["type"] == "dirichlet":
                A[k, :] = 0
                A[k, k] = 1
                if callable(bottom_bc["value"]):
                    b[k] = bottom_bc["value"](self.x[i])
                else:
                    b[k] = bottom_bc["value"]
        
        # Top boundary (j = ny-1)
        for i in range(nx):
            k = idx(i, ny-1)
            if top_bc["type"] == "dirichlet":
                A[k, :] = 0
                A[k, k] = 1
                if callable(top_bc["value"]):
                    b[k] = top_bc["value"](self.x[i])
                else:
                    b[k] = top_bc["value"]
        
        return A.tocsc(), b
    
    def _interpolate_solution(self, solution: np.ndarray, 
                             points: np.ndarray) -> np.ndarray:
        """Interpolate solution at arbitrary points using regular grid interpolator."""
        if self.dimension == 1:
            from scipy.interpolate import interp1d
            interp = interp1d(self.x, solution, kind='linear', 
                            bounds_error=False, fill_value=0.0)
            return interp(points.ravel())
            
        elif self.dimension == 2:
            nx, ny = self.mesh_size
            solution_2d = solution.reshape(nx, ny)
            interp = RegularGridInterpolator((self.x, self.y), solution_2d,
                                           method='linear', bounds_error=False,
                                           fill_value=0.0)
            return interp(points)
        else:
            raise NotImplementedError("3D interpolation not implemented")
    
    def _compute_gradient_1d(self, solution: np.ndarray) -> np.ndarray:
        """Compute 1D gradient using finite differences."""
        dx = self.mesh["dx"]
        gradient = np.zeros_like(solution)
        
        # Central differences for interior points
        gradient[1:-1] = (solution[2:] - solution[:-2]) / (2 * dx)
        
        # Forward/backward differences for boundaries
        gradient[0] = (solution[1] - solution[0]) / dx
        gradient[-1] = (solution[-1] - solution[-2]) / dx
        
        return gradient
    
    def _compute_gradient_2d(self, solution: np.ndarray) -> np.ndarray:
        """Compute 2D gradient using finite differences."""
        nx, ny = self.mesh_size
        dx, dy = self.mesh["dx"], self.mesh["dy"]
        
        solution_2d = solution.reshape(nx, ny)
        grad_x = np.zeros_like(solution_2d)
        grad_y = np.zeros_like(solution_2d)
        
        # x-direction gradient
        grad_x[1:-1, :] = (solution_2d[2:, :] - solution_2d[:-2, :]) / (2 * dx)
        grad_x[0, :] = (solution_2d[1, :] - solution_2d[0, :]) / dx
        grad_x[-1, :] = (solution_2d[-1, :] - solution_2d[-2, :]) / dx
        
        # y-direction gradient
        grad_y[:, 1:-1] = (solution_2d[:, 2:] - solution_2d[:, :-2]) / (2 * dy)
        grad_y[:, 0] = (solution_2d[:, 1] - solution_2d[:, 0]) / dy
        grad_y[:, -1] = (solution_2d[:, -1] - solution_2d[:, -2]) / dy
        
        return np.stack([grad_x.ravel(), grad_y.ravel()], axis=1)