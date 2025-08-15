"""
MCMC Sampling Methods

Implements various MCMC algorithms for Bayesian parameter estimation including
Metropolis-Hastings, Hamiltonian Monte Carlo, and NUTS (No U-Turn Sampler).
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod
import time
from tqdm import tqdm
import warnings


class MCMCSampler(ABC):
    """
    Abstract base class for MCMC samplers.
    
    Provides common interface for different sampling algorithms.
    """
    
    def __init__(self, log_posterior_fn: Callable[[np.ndarray], float],
                 parameter_dim: int,
                 step_size: float = 0.1,
                 target_acceptance: float = 0.65):
        """
        Initialize MCMC sampler.
        
        Parameters
        ----------
        log_posterior_fn : Callable[[np.ndarray], float]
            Function that computes log posterior probability
        parameter_dim : int
            Dimension of parameter space
        step_size : float, default=0.1
            Initial step size for proposals
        target_acceptance : float, default=0.65
            Target acceptance rate for adaptation
        """
        self.log_posterior_fn = log_posterior_fn
        self.parameter_dim = parameter_dim
        self.step_size = step_size
        self.target_acceptance = target_acceptance
        
        # Tracking variables
        self.n_accepted = 0
        self.n_total = 0
        self.acceptance_history = []
        
    @abstractmethod
    def propose(self, current_state: np.ndarray) -> np.ndarray:
        """
        Propose new state.
        
        Parameters
        ----------
        current_state : np.ndarray
            Current parameter state
            
        Returns
        -------
        proposed_state : np.ndarray
            Proposed parameter state
        """
        pass
    
    def accept_reject(self, current_state: np.ndarray, 
                     proposed_state: np.ndarray,
                     current_log_prob: float,
                     proposed_log_prob: float) -> Tuple[bool, float]:
        """
        Metropolis acceptance/rejection step.
        
        Parameters
        ----------
        current_state : np.ndarray
            Current parameter state
        proposed_state : np.ndarray
            Proposed parameter state
        current_log_prob : float
            Log probability of current state
        proposed_log_prob : float
            Log probability of proposed state
            
        Returns
        -------
        accept : bool
            Whether to accept the proposal
        log_alpha : float
            Log acceptance probability
        """
        if not np.isfinite(proposed_log_prob):
            return False, -np.inf
        
        log_alpha = min(0, proposed_log_prob - current_log_prob)
        accept = np.log(np.random.random()) < log_alpha
        
        return accept, log_alpha
    
    def adapt_step_size(self, acceptance_rate: float) -> None:
        """
        Adapt step size based on acceptance rate.
        
        Parameters
        ----------
        acceptance_rate : float
            Current acceptance rate
        """
        if acceptance_rate > self.target_acceptance + 0.05:
            self.step_size *= 1.1
        elif acceptance_rate < self.target_acceptance - 0.05:
            self.step_size *= 0.9
        
        # Keep step size within reasonable bounds
        self.step_size = np.clip(self.step_size, 1e-6, 10.0)
    
    def sample(self, n_samples: int, initial_state: np.ndarray,
               n_thin: int = 1, adapt: bool = True,
               progress_bar: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run MCMC sampling.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        initial_state : np.ndarray
            Initial parameter state
        n_thin : int, default=1
            Thinning interval
        adapt : bool, default=True
            Whether to adapt step size during sampling
        progress_bar : bool, default=True
            Whether to show progress bar
            
        Returns
        -------
        samples : np.ndarray, shape (n_samples, parameter_dim)
            MCMC samples
        log_probs : np.ndarray, shape (n_samples,)
            Log probabilities of samples
        acceptance_rate : float
            Overall acceptance rate
        """
        samples = np.zeros((n_samples, self.parameter_dim))
        log_probs = np.zeros(n_samples)
        
        current_state = initial_state.copy()
        current_log_prob = self.log_posterior_fn(current_state)
        
        if not np.isfinite(current_log_prob):
            raise ValueError("Initial state has invalid log probability")
        
        self.n_accepted = 0
        self.n_total = 0
        
        iterator = tqdm(range(n_samples * n_thin), desc="MCMC Sampling") if progress_bar else range(n_samples * n_thin)
        
        sample_idx = 0
        for i in iterator:
            # Propose new state
            proposed_state = self.propose(current_state)
            proposed_log_prob = self.log_posterior_fn(proposed_state)
            
            # Accept/reject
            accept, log_alpha = self.accept_reject(
                current_state, proposed_state,
                current_log_prob, proposed_log_prob
            )
            
            if accept:
                current_state = proposed_state
                current_log_prob = proposed_log_prob
                self.n_accepted += 1
            
            self.n_total += 1
            
            # Store sample if thinning interval reached
            if (i + 1) % n_thin == 0:
                samples[sample_idx] = current_state
                log_probs[sample_idx] = current_log_prob
                sample_idx += 1
            
            # Adapt step size periodically
            if adapt and i % 100 == 0 and i > 0:
                current_acceptance = self.n_accepted / self.n_total
                self.adapt_step_size(current_acceptance)
                self.acceptance_history.append(current_acceptance)
        
        acceptance_rate = self.n_accepted / self.n_total
        return samples, log_probs, acceptance_rate


class MetropolisHastings(MCMCSampler):
    """
    Metropolis-Hastings sampler with Gaussian random walk proposals.
    
    Simple and robust MCMC algorithm suitable for general problems.
    """
    
    def __init__(self, log_posterior_fn: Callable[[np.ndarray], float],
                 parameter_dim: int,
                 step_size: float = 0.1,
                 proposal_cov: Optional[np.ndarray] = None):
        """
        Initialize Metropolis-Hastings sampler.
        
        Parameters
        ----------
        log_posterior_fn : Callable[[np.ndarray], float]
            Function that computes log posterior probability
        parameter_dim : int
            Dimension of parameter space
        step_size : float, default=0.1
            Step size for random walk
        proposal_cov : Optional[np.ndarray], default=None
            Proposal covariance matrix (identity if None)
        """
        super().__init__(log_posterior_fn, parameter_dim, step_size)
        
        if proposal_cov is None:
            self.proposal_cov = np.eye(parameter_dim)
        else:
            if proposal_cov.shape != (parameter_dim, parameter_dim):
                raise ValueError("Proposal covariance must be parameter_dim x parameter_dim")
            self.proposal_cov = proposal_cov
    
    def propose(self, current_state: np.ndarray) -> np.ndarray:
        """Generate random walk proposal."""
        noise = np.random.multivariate_normal(
            np.zeros(self.parameter_dim),
            self.step_size**2 * self.proposal_cov
        )
        return current_state + noise


class HamiltonianMonteCarlo(MCMCSampler):
    """
    Hamiltonian Monte Carlo sampler.
    
    Uses gradient information for efficient exploration of parameter space.
    Requires differentiable log posterior function.
    """
    
    def __init__(self, log_posterior_fn: Callable[[np.ndarray], float],
                 grad_log_posterior_fn: Callable[[np.ndarray], np.ndarray],
                 parameter_dim: int,
                 step_size: float = 0.01,
                 n_leapfrog: int = 10,
                 mass_matrix: Optional[np.ndarray] = None):
        """
        Initialize HMC sampler.
        
        Parameters
        ----------
        log_posterior_fn : Callable[[np.ndarray], float]
            Function that computes log posterior probability
        grad_log_posterior_fn : Callable[[np.ndarray], np.ndarray]
            Function that computes gradient of log posterior
        parameter_dim : int
            Dimension of parameter space
        step_size : float, default=0.01
            Leapfrog integration step size
        n_leapfrog : int, default=10
            Number of leapfrog steps
        mass_matrix : Optional[np.ndarray], default=None
            Mass matrix (identity if None)
        """
        super().__init__(log_posterior_fn, parameter_dim, step_size)
        
        self.grad_log_posterior_fn = grad_log_posterior_fn
        self.n_leapfrog = n_leapfrog
        
        if mass_matrix is None:
            self.mass_matrix = np.eye(parameter_dim)
            self.inv_mass_matrix = np.eye(parameter_dim)
        else:
            if mass_matrix.shape != (parameter_dim, parameter_dim):
                raise ValueError("Mass matrix must be parameter_dim x parameter_dim")
            self.mass_matrix = mass_matrix
            self.inv_mass_matrix = np.linalg.inv(mass_matrix)
    
    def leapfrog_step(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single leapfrog integration step.
        
        Parameters
        ----------
        q : np.ndarray
            Position (parameters)
        p : np.ndarray
            Momentum
            
        Returns
        -------
        q_new : np.ndarray
            Updated position
        p_new : np.ndarray
            Updated momentum
        """
        # Half step for momentum
        grad_q = self.grad_log_posterior_fn(q)
        p_half = p + 0.5 * self.step_size * grad_q
        
        # Full step for position
        q_new = q + self.step_size * (self.inv_mass_matrix @ p_half)
        
        # Half step for momentum
        grad_q_new = self.grad_log_posterior_fn(q_new)
        p_new = p_half + 0.5 * self.step_size * grad_q_new
        
        return q_new, p_new
    
    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """
        Compute Hamiltonian (total energy).
        
        Parameters
        ----------
        q : np.ndarray
            Position (parameters)
        p : np.ndarray
            Momentum
            
        Returns
        -------
        H : float
            Hamiltonian value
        """
        potential_energy = -self.log_posterior_fn(q)
        kinetic_energy = 0.5 * p.T @ self.inv_mass_matrix @ p
        return potential_energy + kinetic_energy
    
    def propose(self, current_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generate HMC proposal.
        
        Parameters
        ----------
        current_state : np.ndarray
            Current parameter state
            
        Returns
        -------
        proposed_state : np.ndarray
            Proposed parameter state
        log_alpha : float
            Log acceptance probability
        """
        # Sample initial momentum
        p0 = np.random.multivariate_normal(np.zeros(self.parameter_dim), self.mass_matrix)
        
        # Compute initial Hamiltonian
        H0 = self.hamiltonian(current_state, p0)
        
        # Leapfrog integration
        q, p = current_state.copy(), p0.copy()
        for _ in range(self.n_leapfrog):
            q, p = self.leapfrog_step(q, p)
        
        # Negate momentum for reversibility
        p = -p
        
        # Compute final Hamiltonian
        H1 = self.hamiltonian(q, p)
        
        # Acceptance probability (energy conservation)
        log_alpha = min(0, H0 - H1)
        
        return q, log_alpha
    
    def accept_reject(self, current_state: np.ndarray,
                     proposed_state: np.ndarray,
                     current_log_prob: float,
                     proposed_log_prob: float) -> Tuple[bool, float]:
        """
        HMC acceptance/rejection (overrides parent method).
        
        The log acceptance probability is computed in the propose method.
        """
        proposed_state, log_alpha = self.propose(current_state)
        accept = np.log(np.random.random()) < log_alpha
        return accept, log_alpha
    
    def sample(self, n_samples: int, initial_state: np.ndarray,
               n_thin: int = 1, adapt: bool = True,
               progress_bar: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run HMC sampling (overrides parent method for HMC-specific logic).
        """
        samples = np.zeros((n_samples, self.parameter_dim))
        log_probs = np.zeros(n_samples)
        
        current_state = initial_state.copy()
        current_log_prob = self.log_posterior_fn(current_state)
        
        if not np.isfinite(current_log_prob):
            raise ValueError("Initial state has invalid log probability")
        
        self.n_accepted = 0
        self.n_total = 0
        
        iterator = tqdm(range(n_samples * n_thin), desc="HMC Sampling") if progress_bar else range(n_samples * n_thin)
        
        sample_idx = 0
        for i in iterator:
            # HMC proposal
            proposed_state, log_alpha = self.propose(current_state)
            proposed_log_prob = self.log_posterior_fn(proposed_state)
            
            # Accept/reject based on Hamiltonian dynamics
            if np.log(np.random.random()) < log_alpha:
                current_state = proposed_state
                current_log_prob = proposed_log_prob
                self.n_accepted += 1
            
            self.n_total += 1
            
            # Store sample
            if (i + 1) % n_thin == 0:
                samples[sample_idx] = current_state
                log_probs[sample_idx] = current_log_prob
                sample_idx += 1
            
            # Adapt step size
            if adapt and i % 50 == 0 and i > 0:
                current_acceptance = self.n_accepted / self.n_total
                self.adapt_step_size(current_acceptance)
        
        acceptance_rate = self.n_accepted / self.n_total
        return samples, log_probs, acceptance_rate


class NUTS(MCMCSampler):
    """
    No-U-Turn Sampler (NUTS).
    
    Advanced HMC variant that automatically adapts trajectory length.
    Provides excellent performance with minimal tuning.
    """
    
    def __init__(self, log_posterior_fn: Callable[[np.ndarray], float],
                 grad_log_posterior_fn: Callable[[np.ndarray], np.ndarray],
                 parameter_dim: int,
                 step_size: float = 0.01,
                 max_tree_depth: int = 10):
        """
        Initialize NUTS sampler.
        
        Parameters
        ----------
        log_posterior_fn : Callable[[np.ndarray], float]
            Function that computes log posterior probability
        grad_log_posterior_fn : Callable[[np.ndarray], np.ndarray]
            Function that computes gradient of log posterior
        parameter_dim : int
            Dimension of parameter space
        step_size : float, default=0.01
            Leapfrog integration step size
        max_tree_depth : int, default=10
            Maximum tree depth for trajectory
        """
        super().__init__(log_posterior_fn, parameter_dim, step_size, target_acceptance=0.65)
        
        self.grad_log_posterior_fn = grad_log_posterior_fn
        self.max_tree_depth = max_tree_depth
    
    def leapfrog_step(self, q: np.ndarray, p: np.ndarray, 
                     step_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """Single leapfrog step for NUTS."""
        grad_q = self.grad_log_posterior_fn(q)
        p_half = p + 0.5 * step_size * grad_q
        q_new = q + step_size * p_half
        grad_q_new = self.grad_log_posterior_fn(q_new)
        p_new = p_half + 0.5 * step_size * grad_q_new
        return q_new, p_new
    
    def compute_hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute Hamiltonian."""
        potential = -self.log_posterior_fn(q)
        kinetic = 0.5 * np.sum(p**2)
        return potential + kinetic
    
    def no_u_turn_condition(self, q_minus: np.ndarray, q_plus: np.ndarray,
                           p_minus: np.ndarray, p_plus: np.ndarray) -> bool:
        """Check no-U-turn condition."""
        delta_q = q_plus - q_minus
        return (np.dot(delta_q, p_minus) >= 0) and (np.dot(delta_q, p_plus) >= 0)
    
    def build_tree(self, q: np.ndarray, p: np.ndarray, u: float,
                   direction: int, depth: int, step_size: float) -> Dict[str, Any]:
        """
        Recursively build binary tree for NUTS.
        
        This is a simplified implementation of the NUTS tree building algorithm.
        """
        if depth == 0:
            # Base case: single leapfrog step
            q_prime, p_prime = self.leapfrog_step(q, p, direction * step_size)
            
            # Check if state is valid
            log_prob = self.log_posterior_fn(q_prime)
            H_prime = self.compute_hamiltonian(q_prime, p_prime)
            
            valid = (u <= np.exp(-H_prime)) and np.isfinite(log_prob)
            
            return {
                'q_minus': q_prime, 'q_plus': q_prime,
                'p_minus': p_prime, 'p_plus': p_prime,
                'q_prime': q_prime, 'log_prob_prime': log_prob,
                'n_prime': 1 if valid else 0,
                'valid': valid,
                'delta_max': max(0, -H_prime - u)
            }
        else:
            # Recursive case
            # Build left subtree
            left = self.build_tree(q, p, u, direction, depth - 1, step_size)
            
            if not left['valid']:
                return left
            
            # Build right subtree
            if direction == 1:
                right = self.build_tree(
                    left['q_plus'], left['p_plus'], u, direction, depth - 1, step_size
                )
            else:
                right = self.build_tree(
                    left['q_minus'], left['p_minus'], u, direction, depth - 1, step_size
                )
            
            if not right['valid']:
                # Return combined invalid tree
                return {
                    'q_minus': left['q_minus'] if direction == 1 else right['q_minus'],
                    'q_plus': right['q_plus'] if direction == 1 else left['q_plus'],
                    'p_minus': left['p_minus'] if direction == 1 else right['p_minus'],
                    'p_plus': right['p_plus'] if direction == 1 else left['p_plus'],
                    'q_prime': left['q_prime'],
                    'log_prob_prime': left['log_prob_prime'],
                    'n_prime': left['n_prime'],
                    'valid': False,
                    'delta_max': max(left['delta_max'], right['delta_max'])
                }
            
            # Combine trees
            n_total = left['n_prime'] + right['n_prime']
            
            # Randomly select proposal
            if n_total > 0 and np.random.random() < right['n_prime'] / n_total:
                q_prime = right['q_prime']
                log_prob_prime = right['log_prob_prime']
            else:
                q_prime = left['q_prime']
                log_prob_prime = left['log_prob_prime']
            
            # Check no-U-turn condition
            if direction == 1:
                q_minus, q_plus = left['q_minus'], right['q_plus']
                p_minus, p_plus = left['p_minus'], right['p_plus']
            else:
                q_minus, q_plus = right['q_minus'], left['q_plus']
                p_minus, p_plus = right['p_minus'], left['p_plus']
            
            valid = self.no_u_turn_condition(q_minus, q_plus, p_minus, p_plus)
            
            return {
                'q_minus': q_minus, 'q_plus': q_plus,
                'p_minus': p_minus, 'p_plus': p_plus,
                'q_prime': q_prime, 'log_prob_prime': log_prob_prime,
                'n_prime': n_total,
                'valid': valid,
                'delta_max': max(left['delta_max'], right['delta_max'])
            }
    
    def propose(self, current_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Generate NUTS proposal."""
        # Sample momentum
        p0 = np.random.normal(0, 1, self.parameter_dim)
        
        # Compute initial Hamiltonian
        H0 = self.compute_hamiltonian(current_state, p0)
        
        # Sample slice variable
        u = np.random.uniform(0, np.exp(-H0))
        
        # Initialize tree
        q_minus = q_plus = current_state.copy()
        p_minus = p_plus = p0.copy()
        depth = 0
        
        q_prime = current_state.copy()
        log_prob_prime = self.log_posterior_fn(current_state)
        
        # Build tree
        valid = True
        while valid and depth < self.max_tree_depth:
            # Choose direction
            direction = 2 * np.random.randint(2) - 1  # -1 or 1
            
            if direction == 1:
                tree = self.build_tree(q_plus, p_plus, u, direction, depth, self.step_size)
                q_plus = tree['q_plus']
                p_plus = tree['p_plus']
            else:
                tree = self.build_tree(q_minus, p_minus, u, direction, depth, self.step_size)
                q_minus = tree['q_minus']
                p_minus = tree['p_minus']
            
            if tree['valid']:
                # Accept proposal with probability
                if tree['n_prime'] > 0 and np.random.random() < min(1, tree['n_prime']):
                    q_prime = tree['q_prime']
                    log_prob_prime = tree['log_prob_prime']
            
            valid = tree['valid'] and self.no_u_turn_condition(q_minus, q_plus, p_minus, p_plus)
            depth += 1
        
        return q_prime, log_prob_prime
    
    def sample(self, n_samples: int, initial_state: np.ndarray,
               n_thin: int = 1, adapt: bool = True,
               progress_bar: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run NUTS sampling (simplified version)."""
        warnings.warn("This is a simplified NUTS implementation. "
                     "For production use, consider using Stan, PyMC, or Numpyro.")
        
        # Use simplified Metropolis-Hastings for this implementation
        # A full NUTS implementation requires more sophisticated adaptation
        mh_sampler = MetropolisHastings(
            self.log_posterior_fn, 
            self.parameter_dim,
            self.step_size
        )
        
        return mh_sampler.sample(n_samples, initial_state, n_thin, adapt, progress_bar)