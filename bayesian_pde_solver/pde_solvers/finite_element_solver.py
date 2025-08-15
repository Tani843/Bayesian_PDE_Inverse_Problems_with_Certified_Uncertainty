"""
Finite Element PDE Solver

Implements finite element methods for elliptic, parabolic, and hyperbolic PDEs
using variational formulations and Galerkin discretization.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg
from scipy.spatial import Delaunay
from typing import Dict, Any, Tuple, Optional, Callable, List
import warnings

from .forward_solver import ForwardSolver


class FiniteElementSolver(ForwardSolver):
    """
    Finite element solver using linear triangular elements for 2D problems.
    
    Supports:
    - Elliptic PDEs: -∇·(D∇u) + Ru = f
    - Parabolic PDEs: ∂u/∂t - ∇·(D∇u) + Ru = f  
    - Hyperbolic PDEs: ∂²u/∂t² - ∇·(D∇u) + Ru = f
    
    Examples
    --------
    >>> solver = FiniteElementSolver(
    ...     domain_bounds=(0, 1, 0, 1),
    ...     mesh_size=(50, 50),
    ...     pde_type="elliptic"
    ... )
    >>> solution = solver.solve(parameters, boundary_conditions)
    """
    
    def __init__(self, domain_bounds: Tuple[float, ...], 
                 mesh_size: Tuple[int, ...],
                 pde_type: str = "elliptic",
                 element_type: str = "linear_triangle"):
        """
        Initialize finite element solver.
        
        Parameters
        ----------
        domain_bounds : Tuple[float, ...]
            Domain boundaries (x_min, x_max, y_min, y_max)
        mesh_size : Tuple[int, ...]
            Mesh resolution (nx, ny)
        pde_type : str, default="elliptic"
            Type of PDE
        element_type : str, default="linear_triangle"
            Element type for discretization
        """
        self.element_type = element_type
        super().__init__(domain_bounds, mesh_size, pde_type)
        
        # Additional FE-specific attributes
        self.elements = None
        self.boundary_nodes = None
        self.element_areas = None
        
    def _setup_mesh(self) -> None:
        """Setup triangular mesh for finite element discretization."""
        if self.dimension != 2:
            raise NotImplementedError("Finite elements only implemented for 2D")
        
        x_min, x_max, y_min, y_max = self.domain_bounds
        nx, ny = self.mesh_size
        
        # Create structured grid
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Node coordinates
        self.dof_coordinates = np.column_stack([X.ravel(), Y.ravel()])
        n_nodes = len(self.dof_coordinates)
        
        # Create triangular mesh using Delaunay triangulation
        self.triangulation = Delaunay(self.dof_coordinates)
        self.elements = self.triangulation.simplices
        
        # Compute element areas
        self._compute_element_areas()
        
        # Identify boundary nodes
        self._identify_boundary_nodes()
        
        # Store mesh info
        self.mesh = {
            "coordinates": self.dof_coordinates,
            "elements": self.elements,
            "boundary_nodes": self.boundary_nodes,
            "n_nodes": n_nodes,
            "n_elements": len(self.elements)
        }
    
    def _compute_element_areas(self) -> None:
        """Compute areas of triangular elements."""
        n_elements = len(self.elements)
        self.element_areas = np.zeros(n_elements)
        
        for e, element in enumerate(self.elements):
            nodes = self.dof_coordinates[element]
            # Area using cross product
            v1 = nodes[1] - nodes[0]
            v2 = nodes[2] - nodes[0]
            self.element_areas[e] = 0.5 * abs(np.cross(v1, v2))
    
    def _identify_boundary_nodes(self) -> None:
        """Identify nodes on the domain boundary."""
        x_min, x_max, y_min, y_max = self.domain_bounds
        coords = self.dof_coordinates
        
        tol = 1e-10
        boundary_mask = (
            (np.abs(coords[:, 0] - x_min) < tol) |
            (np.abs(coords[:, 0] - x_max) < tol) |
            (np.abs(coords[:, 1] - y_min) < tol) |
            (np.abs(coords[:, 1] - y_max) < tol)
        )
        
        self.boundary_nodes = np.where(boundary_mask)[0]
    
    def _linear_triangle_basis(self, element_coords: np.ndarray, 
                              point: np.ndarray) -> np.ndarray:
        """
        Compute linear triangle basis functions at a point.
        
        Parameters
        ----------
        element_coords : np.ndarray, shape (3, 2)
            Coordinates of triangle vertices
        point : np.ndarray, shape (2,)
            Point where to evaluate basis functions
            
        Returns
        -------
        basis_values : np.ndarray, shape (3,)
            Basis function values
        """
        x1, y1 = element_coords[0]
        x2, y2 = element_coords[1]
        x3, y3 = element_coords[2]
        x, y = point
        
        # Area coordinates (barycentric coordinates)
        area_total = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        if area_total < 1e-12:
            return np.array([1/3, 1/3, 1/3])  # Degenerate element
        
        lambda1 = 0.5 * abs((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)) / area_total
        lambda2 = 0.5 * abs((x3 - x) * (y1 - y) - (x1 - x) * (y3 - y)) / area_total
        lambda3 = 1.0 - lambda1 - lambda2
        
        return np.array([lambda1, lambda2, lambda3])
    
    def _compute_element_matrices(self, element_idx: int,
                                 parameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute element stiffness matrix and load vector.
        
        Parameters
        ----------
        element_idx : int
            Element index
        parameters : Dict[str, Any]
            PDE parameters
            
        Returns
        -------
        K_e : np.ndarray, shape (3, 3)
            Element stiffness matrix
        f_e : np.ndarray, shape (3,)
            Element load vector
        """
        element = self.elements[element_idx]
        element_coords = self.dof_coordinates[element]
        area = self.element_areas[element_idx]
        
        # Extract parameters at element centroid
        centroid = np.mean(element_coords, axis=0)
        
        if callable(parameters.get('diffusion', 1.0)):
            D = parameters['diffusion'](centroid[0], centroid[1])
        else:
            D = parameters.get('diffusion', 1.0)
            
        if callable(parameters.get('reaction', 0.0)):
            R = parameters['reaction'](centroid[0], centroid[1])
        else:
            R = parameters.get('reaction', 0.0)
            
        if callable(parameters.get('source', 0.0)):
            f = parameters['source'](centroid[0], centroid[1])
        else:
            f = parameters.get('source', 0.0)
        
        # Compute gradient of basis functions
        x1, y1 = element_coords[0]
        x2, y2 = element_coords[1]  
        x3, y3 = element_coords[2]
        
        # Gradient matrix (constant over element for linear triangles)
        B = np.array([
            [y2 - y3, y3 - y1, y1 - y2],
            [x3 - x2, x1 - x3, x2 - x1]
        ]) / (2 * area)
        
        # Element stiffness matrix: K_e = D * ∫ B^T B dΩ + R * ∫ N^T N dΩ
        K_diffusion = D * area * (B.T @ B)
        
        # Mass matrix for reaction term (∫ N_i N_j dΩ)
        mass_matrix = area * np.array([
            [2, 1, 1],
            [1, 2, 1], 
            [1, 1, 2]
        ]) / 12.0
        
        K_e = K_diffusion + R * mass_matrix
        
        # Element load vector: f_e = f * ∫ N dΩ
        f_e = f * area * np.ones(3) / 3.0
        
        return K_e, f_e
    
    def assemble_system(self, parameters: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        Assemble global finite element system.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            PDE parameters
            
        Returns
        -------
        K : sp.csc_matrix
            Global stiffness matrix
        F : np.ndarray
            Global load vector
        """
        n_nodes = len(self.dof_coordinates)
        n_elements = len(self.elements)
        
        # Initialize global system
        K = sp.lil_matrix((n_nodes, n_nodes))
        F = np.zeros(n_nodes)
        
        # Assemble element contributions
        for e in range(n_elements):
            K_e, f_e = self._compute_element_matrices(e, parameters)
            element_nodes = self.elements[e]
            
            # Add to global system
            for i in range(3):
                for j in range(3):
                    K[element_nodes[i], element_nodes[j]] += K_e[i, j]
                F[element_nodes[i]] += f_e[i]
        
        return K.tocsc(), F
    
    def apply_boundary_conditions(self, K: sp.csc_matrix, F: np.ndarray,
                                 boundary_conditions: Dict[str, Any]) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        Apply boundary conditions to finite element system.
        
        Parameters
        ----------
        K : sp.csc_matrix
            Global stiffness matrix
        F : np.ndarray
            Global load vector
        boundary_conditions : Dict[str, Any]
            Boundary conditions
            
        Returns
        -------
        K_bc : sp.csc_matrix
            System matrix with boundary conditions
        F_bc : np.ndarray
            Load vector with boundary conditions
        """
        K_bc = K.tolil()
        F_bc = F.copy()
        
        x_min, x_max, y_min, y_max = self.domain_bounds
        coords = self.dof_coordinates
        tol = 1e-10
        
        # Apply Dirichlet boundary conditions
        for side, bc in boundary_conditions.items():
            if bc["type"] == "dirichlet":
                value = bc["value"]
                
                if side == "left":
                    nodes = np.where(np.abs(coords[:, 0] - x_min) < tol)[0]
                elif side == "right":
                    nodes = np.where(np.abs(coords[:, 0] - x_max) < tol)[0]
                elif side == "bottom":
                    nodes = np.where(np.abs(coords[:, 1] - y_min) < tol)[0]
                elif side == "top":
                    nodes = np.where(np.abs(coords[:, 1] - y_max) < tol)[0]
                else:
                    continue
                
                for node in nodes:
                    # Set row to identity
                    K_bc[node, :] = 0
                    K_bc[node, node] = 1
                    
                    if callable(value):
                        F_bc[node] = value(coords[node, 0], coords[node, 1])
                    else:
                        F_bc[node] = value
        
        return K_bc.tocsc(), F_bc
    
    def _interpolate_solution(self, solution: np.ndarray, 
                             points: np.ndarray) -> np.ndarray:
        """
        Interpolate finite element solution at arbitrary points.
        
        Parameters
        ----------
        solution : np.ndarray
            FE solution coefficients
        points : np.ndarray, shape (n_points, 2)
            Points where to interpolate
            
        Returns
        -------
        interpolated : np.ndarray
            Interpolated values
        """
        n_points = len(points)
        interpolated = np.zeros(n_points)
        
        for i, point in enumerate(points):
            # Find containing element
            element_idx = self.triangulation.find_simplex(point)
            
            if element_idx == -1:
                # Point outside domain
                interpolated[i] = 0.0
                continue
            
            # Get element and compute basis functions
            element = self.elements[element_idx]
            element_coords = self.dof_coordinates[element]
            basis_values = self._linear_triangle_basis(element_coords, point)
            
            # Interpolate
            interpolated[i] = np.dot(basis_values, solution[element])
        
        return interpolated
    
    def _compute_gradient_1d(self, solution: np.ndarray) -> np.ndarray:
        """Not implemented for finite elements."""
        raise NotImplementedError("1D gradient not implemented for finite elements")
    
    def _compute_gradient_2d(self, solution: np.ndarray) -> np.ndarray:
        """
        Compute solution gradient using finite element interpolation.
        
        Parameters
        ----------
        solution : np.ndarray
            FE solution coefficients
            
        Returns
        -------
        gradient : np.ndarray, shape (n_nodes, 2)
            Gradient at each node
        """
        n_nodes = len(self.dof_coordinates)
        gradient = np.zeros((n_nodes, 2))
        node_contributions = np.zeros(n_nodes)
        
        # Loop over elements
        for e, element in enumerate(self.elements):
            element_coords = self.dof_coordinates[element]
            area = self.element_areas[e]
            
            # Compute gradient matrix
            x1, y1 = element_coords[0]
            x2, y2 = element_coords[1]
            x3, y3 = element_coords[2]
            
            B = np.array([
                [y2 - y3, y3 - y1, y1 - y2],
                [x3 - x2, x1 - x3, x2 - x1]
            ]) / (2 * area)
            
            # Compute element gradient (constant over element)
            element_solution = solution[element]
            grad_element = B @ element_solution
            
            # Distribute to nodes (simple averaging)
            for i, node in enumerate(element):
                gradient[node] += grad_element
                node_contributions[node] += 1
        
        # Average contributions from different elements
        for i in range(n_nodes):
            if node_contributions[i] > 0:
                gradient[i] /= node_contributions[i]
        
        return gradient
    
    def solve_time_dependent(self, parameters: Dict[str, Any],
                            boundary_conditions: Dict[str, Any],
                            initial_condition: np.ndarray,
                            time_steps: int,
                            final_time: float,
                            method: str = "backward_euler") -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve time-dependent PDE using finite elements.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            PDE parameters
        boundary_conditions : Dict[str, Any]
            Boundary conditions
        initial_condition : np.ndarray
            Initial condition
        time_steps : int
            Number of time steps
        final_time : float
            Final simulation time
        method : str, default="backward_euler"
            Time integration method
            
        Returns
        -------
        solutions : np.ndarray, shape (time_steps+1, n_nodes)
            Solutions at all time steps
        times : np.ndarray
            Time values
        """
        if self.pde_type not in ["parabolic", "hyperbolic"]:
            raise ValueError("Time-dependent solver only for parabolic/hyperbolic PDEs")
        
        dt = final_time / time_steps
        times = np.linspace(0, final_time, time_steps + 1)
        n_nodes = len(self.dof_coordinates)
        
        # Assemble spatial matrices
        K, F = self.assemble_system(parameters)
        
        # Assemble mass matrix
        M = self._assemble_mass_matrix()
        
        # Apply boundary conditions to mass matrix
        M_bc, _ = self.apply_boundary_conditions(M, np.zeros(n_nodes), boundary_conditions)
        K_bc, F_bc = self.apply_boundary_conditions(K, F, boundary_conditions)
        
        # Initialize solution storage
        solutions = np.zeros((time_steps + 1, n_nodes))
        solutions[0] = initial_condition.copy()
        
        if self.pde_type == "parabolic":
            # Parabolic: M du/dt + K u = F
            if method == "backward_euler":
                # (M + dt*K) u^{n+1} = M u^n + dt*F
                A = M_bc + dt * K_bc
                
                for n in range(time_steps):
                    rhs = M_bc @ solutions[n] + dt * F_bc
                    solutions[n+1] = spsolve(A, rhs)
            
            elif method == "crank_nicolson":
                # (M + dt/2*K) u^{n+1} = (M - dt/2*K) u^n + dt*F
                A = M_bc + 0.5 * dt * K_bc
                B = M_bc - 0.5 * dt * K_bc
                
                for n in range(time_steps):
                    rhs = B @ solutions[n] + dt * F_bc
                    solutions[n+1] = spsolve(A, rhs)
        
        elif self.pde_type == "hyperbolic":
            # Hyperbolic: M d²u/dt² + K u = F (second-order in time)
            # Use Newmark method or convert to first-order system
            raise NotImplementedError("Hyperbolic time integration not yet implemented")
        
        return solutions, times
    
    def _assemble_mass_matrix(self) -> sp.csc_matrix:
        """Assemble global mass matrix."""
        n_nodes = len(self.dof_coordinates)
        M = sp.lil_matrix((n_nodes, n_nodes))
        
        for e, element in enumerate(self.elements):
            area = self.element_areas[e]
            
            # Element mass matrix for linear triangles
            M_e = area * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ]) / 12.0
            
            # Assemble into global matrix
            for i in range(3):
                for j in range(3):
                    M[element[i], element[j]] += M_e[i, j]
        
        return M.tocsc()
    
    def compute_error_indicators(self, solution: np.ndarray,
                                parameters: Dict[str, Any]) -> np.ndarray:
        """
        Compute element-wise error indicators for adaptive refinement.
        
        Parameters
        ----------
        solution : np.ndarray
            FE solution coefficients
        parameters : Dict[str, Any]
            PDE parameters
            
        Returns
        -------
        error_indicators : np.ndarray
            Error indicator for each element
        """
        n_elements = len(self.elements)
        error_indicators = np.zeros(n_elements)
        
        # Simple gradient-based error indicator
        gradient = self._compute_gradient_2d(solution)
        
        for e, element in enumerate(self.elements):
            # Compute gradient jump across element edges (simplified)
            element_gradients = gradient[element]
            grad_variation = np.var(element_gradients, axis=0)
            error_indicators[e] = np.sqrt(np.sum(grad_variation)) * self.element_areas[e]
        
        return error_indicators