"""
PDE Solvers Module

Forward solvers for various types of partial differential equations using
finite element and finite difference methods.
"""

from .forward_solver import ForwardSolver
from .finite_element_solver import FiniteElementSolver
from .finite_difference_solver import FiniteDifferenceSolver
from .boundary_conditions import BoundaryCondition, DirichletBC, NeumannBC, RobinBC
from .pde_problems import EllipticProblem, ParabolicProblem, HyperbolicProblem

__all__ = [
    "ForwardSolver",
    "FiniteElementSolver", 
    "FiniteDifferenceSolver",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC", 
    "RobinBC",
    "EllipticProblem",
    "ParabolicProblem",
    "HyperbolicProblem"
]