"""
Forward PDE Solver Base Class

Abstract base class for solving forward problems in partial differential equations.
Provides common interface for finite element and finite difference methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Callable
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class ForwardSolver(ABC):
    """
    Abstract base class for forward PDE solvers.
    """
    
    def __init__(self, domain_bounds: Tuple[float, ...], 
                 mesh_size: Tuple[int, ...],
                 pde_type: str = "elliptic"):
        """
        Initialize forward solver.
        
        Args:
            domain_bounds: Tuple of (min, max) for each spatial dimension
            mesh_size: Number of grid points in each dimension
            pde_type: Type of PDE ("elliptic", "parabolic", "hyperbolic")
        """
        self.domain_bounds = domain_bounds
        self.mesh_size = mesh_size
        self.pde_type = pde_type
        self.dimension = len(domain_bounds) // 2
        self.mesh = None
        self.dof_coordinates = None
        self._setup_mesh()
    
    @abstractmethod
    def _setup_mesh(self) -> None:
        """Setup computational mesh."""
        pass
    
    @abstractmethod
    def assemble_system(self, parameters: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        Assemble the linear system Ax = b.
        
        Args:
            parameters: Dictionary of PDE parameters
            
        Returns:
            A: System matrix
            b: Right-hand side vector
        """
        pass
    
    @abstractmethod
    def apply_boundary_conditions(self, A: sp.csc_matrix, b: np.ndarray,
                                boundary_conditions: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        Apply boundary conditions to the system.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            boundary_conditions: Boundary condition specifications
            
        Returns:
            A_bc: System matrix with boundary conditions
            b_bc: RHS vector with boundary conditions
        """
        pass
    
    def solve(self, parameters: Dict[str, Any], 
              boundary_conditions: Dict[str, Any],
              solver_options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Solve the forward PDE problem.
        
        Args:
            parameters: PDE parameters
            boundary_conditions: Boundary conditions
            solver_options: Linear solver options
            
        Returns:
            solution: Solution vector at mesh points
        """
        # Assemble system
        A, b = self.assemble_system(parameters)
        
        # Apply boundary conditions
        A_bc, b_bc = self.apply_boundary_conditions(A, b, boundary_conditions)
        
        # Solve linear system
        if solver_options is None:
            solver_options = {"method": "spsolve"}
            
        if solver_options.get("method", "spsolve") == "spsolve":
            solution = spsolve(A_bc, b_bc)
        else:
            from scipy.sparse.linalg import cg, gmres
            solver_method = solver_options.get("method", "cg")
            if solver_method == "cg":
                solution, info = cg(A_bc, b_bc, 
                                  tol=solver_options.get("tol", 1e-8),
                                  maxiter=solver_options.get("maxiter", 1000))
            elif solver_method == "gmres":
                solution, info = gmres(A_bc, b_bc,
                                     tol=solver_options.get("tol", 1e-8),
                                     maxiter=solver_options.get("maxiter", 1000))
            else:
                raise ValueError(f"Unknown solver method: {solver_method}")
                
            if info != 0:
                raise RuntimeError(f"Linear solver failed with info: {info}")
        
        return solution
    
    def compute_observables(self, solution: np.ndarray, 
                           observation_points: np.ndarray) -> np.ndarray:
        """
        Compute observable quantities at specified points.
        
        Args:
            solution: PDE solution
            observation_points: Points where to evaluate observables
            
        Returns:
            observables: Observable values
        """
        return self._interpolate_solution(solution, observation_points)
    
    @abstractmethod
    def _interpolate_solution(self, solution: np.ndarray, 
                             points: np.ndarray) -> np.ndarray:
        """Interpolate solution at arbitrary points."""
        pass
    
    def compute_gradient(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute gradient of the solution.
        
        Args:
            solution: PDE solution
            
        Returns:
            gradient: Gradient field
        """
        if self.dimension == 1:
            return self._compute_gradient_1d(solution)
        elif self.dimension == 2:
            return self._compute_gradient_2d(solution)
        else:
            raise NotImplementedError("3D gradient computation not implemented")
    
    @abstractmethod
    def _compute_gradient_1d(self, solution: np.ndarray) -> np.ndarray:
        """Compute 1D gradient."""
        pass
    
    @abstractmethod
    def _compute_gradient_2d(self, solution: np.ndarray) -> np.ndarray:
        """Compute 2D gradient."""
        pass
    
    def get_mesh_info(self) -> Dict[str, Any]:
        """Return mesh information."""
        return {
            "domain_bounds": self.domain_bounds,
            "mesh_size": self.mesh_size,
            "dimension": self.dimension,
            "num_dofs": len(self.dof_coordinates) if self.dof_coordinates is not None else 0,
            "coordinates": self.dof_coordinates
        }