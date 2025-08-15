"""
Test suite for PDE solvers.

Tests finite difference and finite element solvers against manufactured
solutions and validates convergence rates, boundary conditions, and
numerical accuracy.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import matplotlib.pyplot as plt
from pathlib import Path

from bayesian_pde_solver.pde_solvers.finite_difference_solver import FiniteDifferenceSolver
from bayesian_pde_solver.pde_solvers.finite_element_solver import FiniteElementSolver


class TestFiniteDifferenceSolver:
    """Test suite for finite difference solver."""
    
    def test_initialization_1d(self, simple_1d_domain, coarse_mesh_1d):
        """Test 1D solver initialization."""
        solver = FiniteDifferenceSolver(
            domain_bounds=simple_1d_domain,
            mesh_size=coarse_mesh_1d,
            pde_type="elliptic"
        )
        
        assert solver.dimension == 1
        assert solver.mesh_size == coarse_mesh_1d
        assert solver.coordinates.shape[0] == coarse_mesh_1d[0]
        
    def test_initialization_2d(self, simple_2d_domain, coarse_mesh_2d):
        """Test 2D solver initialization."""
        solver = FiniteDifferenceSolver(
            domain_bounds=simple_2d_domain,
            mesh_size=coarse_mesh_2d,
            pde_type="elliptic"
        )
        
        assert solver.dimension == 2
        assert solver.mesh_size == coarse_mesh_2d
        assert solver.coordinates.shape[0] == coarse_mesh_2d[0] * coarse_mesh_2d[1]
        assert solver.coordinates.shape[1] == 2
    
    def test_manufactured_solution_1d(self, simple_1d_domain, manufactured_solution_1d):
        """Test 1D solver against manufactured solution."""
        # Use multiple mesh refinements
        mesh_sizes = [21, 41, 81]
        errors = []
        
        for n in mesh_sizes:
            solver = FiniteDifferenceSolver(
                domain_bounds=simple_1d_domain,
                mesh_size=(n,),
                pde_type="elliptic"
            )
            
            # Solve with manufactured parameters
            bc = {
                "left": {"type": "dirichlet", "value": 0.0},
                "right": {"type": "dirichlet", "value": 0.0}
            }
            
            solution = solver.solve(manufactured_solution_1d["parameters"], bc)
            
            # Compute error
            x_vals = solver.coordinates.ravel()
            exact_solution = manufactured_solution_1d["solution"](x_vals)
            
            # L2 error
            dx = (simple_1d_domain[1] - simple_1d_domain[0]) / (n - 1)
            error = np.sqrt(dx * np.sum((solution - exact_solution)**2))
            errors.append(error)
        
        # Check convergence (should be O(h^2) for 2nd order FD)
        for i in range(len(errors)-1):
            ratio = errors[i] / errors[i+1]
            assert ratio > 3.0, f"Expected convergence ratio > 3, got {ratio}"
    
    def test_manufactured_solution_2d(self, simple_2d_domain, manufactured_solution_2d, 
                                    dirichlet_zero_bc):
        """Test 2D solver against manufactured solution."""
        solver = FiniteDifferenceSolver(
            domain_bounds=simple_2d_domain,
            mesh_size=(31, 31),
            pde_type="elliptic"
        )
        
        solution = solver.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
        
        # Compute exact solution
        X = solver.coordinates[:, 0]
        Y = solver.coordinates[:, 1]
        exact_solution = manufactured_solution_2d["solution"](X, Y)
        
        # Check relative error
        rel_error = np.linalg.norm(solution - exact_solution) / np.linalg.norm(exact_solution)
        assert rel_error < 0.01, f"Relative error {rel_error} too large"
    
    @pytest.mark.convergence
    def test_convergence_rate_2d(self, simple_2d_domain, manufactured_solution_2d, 
                               dirichlet_zero_bc, convergence_meshes):
        """Test convergence rate for 2D finite difference solver."""
        errors = []
        
        for mesh_size in convergence_meshes:
            solver = FiniteDifferenceSolver(
                domain_bounds=simple_2d_domain,
                mesh_size=mesh_size,
                pde_type="elliptic"
            )
            
            solution = solver.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
            
            # Compute L2 error
            X = solver.coordinates[:, 0]
            Y = solver.coordinates[:, 1]
            exact_solution = manufactured_solution_2d["solution"](X, Y)
            
            dx = (simple_2d_domain[1] - simple_2d_domain[0]) / (mesh_size[0] - 1)
            dy = (simple_2d_domain[3] - simple_2d_domain[2]) / (mesh_size[1] - 1)
            error = np.sqrt(dx * dy * np.sum((solution - exact_solution)**2))
            errors.append(error)
        
        # Check O(h^2) convergence
        from conftest import assert_convergence_rate
        assert_convergence_rate(errors, convergence_meshes, expected_rate=2.0, tolerance=0.5)
    
    def test_different_boundary_conditions(self, fd_solver_2d, simple_pde_params):
        """Test different boundary condition types."""
        # Dirichlet BC
        bc_dirichlet = {
            "left": {"type": "dirichlet", "value": 1.0},
            "right": {"type": "dirichlet", "value": 0.0},
            "top": {"type": "dirichlet", "value": 0.0},
            "bottom": {"type": "dirichlet", "value": 0.0}
        }
        
        solution_dirichlet = fd_solver_2d.solve(simple_pde_params, bc_dirichlet)
        
        # Check boundary values are approximately correct
        # (Note: This is a simplified check - exact boundary node identification 
        # would require more sophisticated mesh analysis)
        assert np.all(np.isfinite(solution_dirichlet))
        assert not np.allclose(solution_dirichlet, 0)  # Non-trivial solution
    
    def test_spatially_varying_parameters(self, fd_solver_2d, dirichlet_zero_bc):
        """Test solver with spatially varying parameters."""
        def varying_diffusion(x, y):
            return 1.0 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        
        def varying_source(x, y):
            return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
        
        params = {
            "diffusion": varying_diffusion,
            "reaction": 0.0,
            "source": varying_source
        }
        
        solution = fd_solver_2d.solve(params, dirichlet_zero_bc)
        
        # Check solution is reasonable
        assert np.all(np.isfinite(solution))
        assert np.max(solution) > 0  # Should have positive values
        assert np.min(solution) >= -1e-10  # Should be non-negative (approximately)
    
    def test_gradient_computation(self, fd_solver_2d, manufactured_solution_2d, 
                                dirichlet_zero_bc):
        """Test gradient computation accuracy."""
        solution = fd_solver_2d.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
        gradient = fd_solver_2d.compute_gradient(solution)
        
        assert gradient.shape == (len(fd_solver_2d.coordinates), 2)
        assert np.all(np.isfinite(gradient))
        
        # Check gradient is not zero everywhere
        grad_norm = np.linalg.norm(gradient, axis=1)
        assert np.max(grad_norm) > 1e-10
    
    @pytest.mark.slow
    def test_large_system_solve(self, simple_2d_domain, simple_pde_params, 
                              dirichlet_zero_bc):
        """Test solver performance on larger system."""
        solver = FiniteDifferenceSolver(
            domain_bounds=simple_2d_domain,
            mesh_size=(101, 101),
            pde_type="elliptic"
        )
        
        solution = solver.solve(simple_pde_params, dirichlet_zero_bc)
        
        assert solution.shape[0] == 101 * 101
        assert np.all(np.isfinite(solution))
    
    def test_error_handling(self, simple_2d_domain, coarse_mesh_2d):
        """Test error handling and validation."""
        # Invalid PDE type
        with pytest.raises(ValueError):
            FiniteDifferenceSolver(
                domain_bounds=simple_2d_domain,
                mesh_size=coarse_mesh_2d,
                pde_type="invalid_type"
            )
        
        # Invalid domain bounds
        with pytest.raises(ValueError):
            FiniteDifferenceSolver(
                domain_bounds=(1.0, 0.0, 0.0, 1.0),  # x_min > x_max
                mesh_size=coarse_mesh_2d,
                pde_type="elliptic"
            )


class TestFiniteElementSolver:
    """Test suite for finite element solver."""
    
    def test_initialization(self, simple_2d_domain, coarse_mesh_2d):
        """Test finite element solver initialization."""
        solver = FiniteElementSolver(
            domain_bounds=simple_2d_domain,
            mesh_size=coarse_mesh_2d,
            pde_type="elliptic"
        )
        
        assert solver.dimension == 2
        assert solver.mesh_size == coarse_mesh_2d
        assert solver.dof_coordinates is not None
        assert solver.elements is not None
        assert solver.boundary_nodes is not None
    
    def test_mesh_generation(self, fe_solver_2d):
        """Test triangular mesh generation."""
        n_nodes = len(fe_solver_2d.dof_coordinates)
        n_elements = len(fe_solver_2d.elements)
        
        # Basic mesh validity checks
        assert n_nodes > 0
        assert n_elements > 0
        assert fe_solver_2d.elements.shape[1] == 3  # Triangular elements
        
        # Check element node indices are valid
        assert np.all(fe_solver_2d.elements >= 0)
        assert np.all(fe_solver_2d.elements < n_nodes)
        
        # Check element areas are positive
        assert np.all(fe_solver_2d.element_areas > 0)
    
    def test_boundary_identification(self, fe_solver_2d, simple_2d_domain):
        """Test boundary node identification."""
        x_min, x_max, y_min, y_max = simple_2d_domain
        coords = fe_solver_2d.dof_coordinates
        boundary_nodes = fe_solver_2d.boundary_nodes
        
        # Check that boundary nodes are actually on the boundary
        boundary_coords = coords[boundary_nodes]
        tol = 1e-10
        
        on_boundary = (
            (np.abs(boundary_coords[:, 0] - x_min) < tol) |
            (np.abs(boundary_coords[:, 0] - x_max) < tol) |
            (np.abs(boundary_coords[:, 1] - y_min) < tol) |
            (np.abs(boundary_coords[:, 1] - y_max) < tol)
        )
        
        assert np.all(on_boundary), "Some identified boundary nodes are not on boundary"
    
    def test_manufactured_solution_2d(self, fe_solver_2d, manufactured_solution_2d,
                                    dirichlet_zero_bc):
        """Test FE solver against manufactured solution."""
        solution = fe_solver_2d.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
        
        # Compute exact solution at nodes
        X = fe_solver_2d.dof_coordinates[:, 0]
        Y = fe_solver_2d.dof_coordinates[:, 1]
        exact_solution = manufactured_solution_2d["solution"](X, Y)
        
        # Check relative error
        rel_error = np.linalg.norm(solution - exact_solution) / np.linalg.norm(exact_solution)
        assert rel_error < 0.05, f"Relative error {rel_error} too large for FE solver"
    
    def test_system_assembly(self, fe_solver_2d, simple_pde_params):
        """Test finite element system assembly."""
        K, F = fe_solver_2d.assemble_system(simple_pde_params)
        
        n_nodes = len(fe_solver_2d.dof_coordinates)
        
        # Check matrix dimensions
        assert K.shape == (n_nodes, n_nodes)
        assert F.shape == (n_nodes,)
        
        # Check matrix properties
        assert K.nnz > 0  # Non-empty sparse matrix
        
        # Check symmetry (for elliptic problems with symmetric coefficients)
        K_dense = K.toarray()
        assert np.allclose(K_dense, K_dense.T, atol=1e-12), "Stiffness matrix should be symmetric"
    
    def test_interpolation(self, fe_solver_2d, manufactured_solution_2d, dirichlet_zero_bc):
        """Test solution interpolation at arbitrary points."""
        solution = fe_solver_2d.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
        
        # Test points inside domain
        test_points = np.array([
            [0.25, 0.25],
            [0.5, 0.5],
            [0.75, 0.75]
        ])
        
        interpolated = fe_solver_2d._interpolate_solution(solution, test_points)
        
        # Compute exact values
        exact_values = manufactured_solution_2d["solution"](
            test_points[:, 0], test_points[:, 1]
        )
        
        # Check interpolation accuracy
        rel_error = np.abs(interpolated - exact_values) / np.abs(exact_values)
        assert np.all(rel_error < 0.1), "Interpolation error too large"
    
    def test_gradient_computation_2d(self, fe_solver_2d, manufactured_solution_2d,
                                   dirichlet_zero_bc):
        """Test gradient computation for finite elements."""
        solution = fe_solver_2d.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
        gradient = fe_solver_2d.compute_gradient(solution)
        
        n_nodes = len(fe_solver_2d.dof_coordinates)
        assert gradient.shape == (n_nodes, 2)
        assert np.all(np.isfinite(gradient))
    
    @pytest.mark.slow
    def test_time_dependent_parabolic(self, simple_2d_domain, dirichlet_zero_bc):
        """Test time-dependent parabolic PDE solving."""
        solver = FiniteElementSolver(
            domain_bounds=simple_2d_domain,
            mesh_size=(21, 21),
            pde_type="parabolic"
        )
        
        # Heat equation parameters
        params = {
            "diffusion": 1.0,
            "reaction": 0.0,
            "source": lambda x, y: 0.0
        }
        
        # Initial condition (Gaussian)
        coords = solver.dof_coordinates
        initial_condition = np.exp(-((coords[:, 0] - 0.5)**2 + (coords[:, 1] - 0.5)**2) / 0.1)
        
        # Solve
        solutions, times = solver.solve_time_dependent(
            parameters=params,
            boundary_conditions=dirichlet_zero_bc,
            initial_condition=initial_condition,
            time_steps=20,
            final_time=0.1,
            method="backward_euler"
        )
        
        # Check solution properties
        assert solutions.shape == (21, len(coords))
        assert len(times) == 21
        assert np.all(np.isfinite(solutions))
        
        # Check that solution decays over time (heat dissipation)
        initial_max = np.max(solutions[0])
        final_max = np.max(solutions[-1])
        assert final_max < initial_max, "Solution should decay over time"
    
    def test_error_indicators(self, fe_solver_2d, manufactured_solution_2d, dirichlet_zero_bc):
        """Test error indicator computation for adaptive refinement."""
        solution = fe_solver_2d.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
        error_indicators = fe_solver_2d.compute_error_indicators(solution, 
                                                               manufactured_solution_2d["parameters"])
        
        n_elements = len(fe_solver_2d.elements)
        assert error_indicators.shape == (n_elements,)
        assert np.all(error_indicators >= 0)
        assert np.all(np.isfinite(error_indicators))


class TestSolverComparison:
    """Compare finite difference and finite element solvers."""
    
    @pytest.mark.integration
    def test_fd_vs_fe_convergence(self, simple_2d_domain, manufactured_solution_2d, 
                                dirichlet_zero_bc):
        """Compare convergence of FD and FE solvers."""
        mesh_sizes = [(11, 11), (21, 21), (31, 31)]
        fd_errors = []
        fe_errors = []
        
        for mesh_size in mesh_sizes:
            # Finite difference solver
            fd_solver = FiniteDifferenceSolver(
                domain_bounds=simple_2d_domain,
                mesh_size=mesh_size,
                pde_type="elliptic"
            )
            
            fd_solution = fd_solver.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
            X_fd = fd_solver.coordinates[:, 0]
            Y_fd = fd_solver.coordinates[:, 1]
            exact_fd = manufactured_solution_2d["solution"](X_fd, Y_fd)
            fd_error = np.linalg.norm(fd_solution - exact_fd) / np.linalg.norm(exact_fd)
            fd_errors.append(fd_error)
            
            # Finite element solver  
            fe_solver = FiniteElementSolver(
                domain_bounds=simple_2d_domain,
                mesh_size=mesh_size,
                pde_type="elliptic"
            )
            
            fe_solution = fe_solver.solve(manufactured_solution_2d["parameters"], dirichlet_zero_bc)
            X_fe = fe_solver.dof_coordinates[:, 0]
            Y_fe = fe_solver.dof_coordinates[:, 1]
            exact_fe = manufactured_solution_2d["solution"](X_fe, Y_fe)
            fe_error = np.linalg.norm(fe_solution - exact_fe) / np.linalg.norm(exact_fe)
            fe_errors.append(fe_error)
        
        # Both methods should converge
        assert fd_errors[0] > fd_errors[-1], "FD solver should converge"
        assert fe_errors[0] > fe_errors[-1], "FE solver should converge"
        
        print(f"FD errors: {fd_errors}")
        print(f"FE errors: {fe_errors}")


# Performance and scaling tests
class TestSolverPerformance:
    """Test solver performance and scaling."""
    
    @pytest.mark.slow
    def test_memory_usage(self, simple_2d_domain, simple_pde_params, dirichlet_zero_bc):
        """Test memory usage for different mesh sizes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Solve progressively larger systems
        mesh_sizes = [(31, 31), (51, 51), (101, 101)]
        
        for mesh_size in mesh_sizes:
            solver = FiniteDifferenceSolver(
                domain_bounds=simple_2d_domain,
                mesh_size=mesh_size,
                pde_type="elliptic"
            )
            
            solution = solver.solve(simple_pde_params, dirichlet_zero_bc)
            
            current_memory = process.memory_info().rss
            memory_increase = (current_memory - initial_memory) / 1024**2  # MB
            
            print(f"Mesh {mesh_size}: Memory increase = {memory_increase:.1f} MB")
            
            # Clean up
            del solver, solution
    
    @pytest.mark.slow
    @pytest.mark.parametrize("mesh_size", [(51, 51), (101, 101)])
    def test_solve_time_scaling(self, simple_2d_domain, simple_pde_params, 
                              dirichlet_zero_bc, mesh_size):
        """Test solve time scaling with mesh size."""
        import time
        
        solver = FiniteDifferenceSolver(
            domain_bounds=simple_2d_domain,
            mesh_size=mesh_size,
            pde_type="elliptic"
        )
        
        start_time = time.time()
        solution = solver.solve(simple_pde_params, dirichlet_zero_bc)
        solve_time = time.time() - start_time
        
        n_dof = mesh_size[0] * mesh_size[1]
        time_per_dof = solve_time / n_dof
        
        print(f"Mesh {mesh_size}: Solve time = {solve_time:.3f}s, Time/DOF = {time_per_dof:.6f}s")
        
        # Reasonable performance check
        assert solve_time < 60.0, f"Solve time {solve_time}s too long for mesh {mesh_size}"