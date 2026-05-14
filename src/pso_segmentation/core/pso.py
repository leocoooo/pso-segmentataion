"""Particle Swarm Optimization (PSO) implementation.

This module provides a minimal, robust PSO algorithm for continuous optimization.
The algorithm maximizes the objective function by iteratively updating particle
positions and velocities based on cognitive and social components.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

# Type alias for cleaner code
NDArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass
class PSO_Result:
    """Result of PSO optimization run.

    Attributes
    ----------
    best_position : NDArray
        The best position found (shape: (n_dim,))
    best_fitness : float
        The fitness value at best position
    n_iterations : int
        Number of iterations performed
    converged : bool
        Whether convergence was achieved
    history : list[dict[str, Any]]
        Optimization history with iteration, best_fitness, best_position
    """

    best_position: NDArray
    best_fitness: float
    n_iterations: int
    converged: bool
    history: list[dict[str, Any]] = field(default_factory=list)


class PSO:
    """Particle Swarm Optimization algorithm for continuous optimization.

    This implementation uses the standard PSO with inertia weight, cognitive
    and social coefficients. It maximizes the objective function.

    Parameters
    ----------
    objective_func : Callable[[NDArray], float]
        Function to maximize. Takes array of shape (n_dim,) and returns float.
    n_dim : int
        Dimensionality of the optimization problem.
    pop_size : int, default=30
        Number of particles in the swarm.
    max_iter : int, default=100
        Maximum number of iterations.
    lb : NDArray, optional
        Lower bounds for each dimension. If None, uses -inf.
    ub : NDArray, optional
        Upper bounds for each dimension. If None, uses +inf.
    w : float, default=0.7
        Inertia weight. Controls influence of previous velocity.
    c1 : float, default=1.5
        Cognitive coefficient (attraction to personal best).
    c2 : float, default=1.5
        Social coefficient (attraction to global best).
    seed : int, optional
        Random seed for reproducibility.
    track_history : bool, default=True
        Whether to track optimization history.

    Attributes
    ----------
    particles : NDArray
        Current particle positions (pop_size, n_dim)
    velocities : NDArray
        Current particle velocities (pop_size, n_dim)
    personal_best : NDArray
        Best position for each particle (pop_size, n_dim)
    personal_best_fitness : NDArray
        Best fitness for each particle (pop_size,)
    best_position : NDArray
        Global best position found
    best_fitness : float
        Global best fitness found
    history : list[dict[str, Any]]
        Optimization history if track_history=True

    Examples
    --------
    >>> def sphere(x: NDArray) -> float:
    ...     return -np.sum(x ** 2)  # Maximize negative sphere
    >>> pso = PSO(sphere, n_dim=3, max_iter=50)
    >>> result = pso.run()
    >>> print(f"Best fitness: {result.best_fitness}")
    >>> print(f"Best position: {result.best_position}")
    """

    particles: NDArray
    velocities: NDArray
    personal_best: NDArray
    personal_best_fitness: NDArray
    best_position: NDArray
    best_fitness: float
    lb: NDArray
    ub: NDArray
    history: list[dict[str, Any]]

    def __init__(
        self,
        objective_func: Callable[[NDArray], float],
        n_dim: int,
        pop_size: int = 30,
        max_iter: int = 100,
        lb: NDArray | None = None,
        ub: NDArray | None = None,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int | None = None,
        track_history: bool = True,
    ) -> None:
        """Initialize PSO optimizer."""
        self.objective_func = objective_func
        self.n_dim = n_dim
        self.pop_size = max(1, pop_size)
        self.max_iter = max(0, max_iter)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.track_history = track_history

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Set bounds
        if lb is None:
            self.lb = np.full(n_dim, -np.inf, dtype=np.float64)
        else:
            self.lb = np.asarray(lb, dtype=np.float64)

        if ub is None:
            self.ub = np.full(n_dim, np.inf, dtype=np.float64)
        else:
            self.ub = np.asarray(ub, dtype=np.float64)

        # Validate bounds
        if np.any(self.lb >= self.ub):
            msg = "Lower bounds must be strictly less than upper bounds"
            raise ValueError(msg)

        # Initialize particles and velocities
        self.particles = self._initialize_particles()
        self.velocities = self._initialize_velocities()

        # Track personal best
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(self.pop_size, -np.inf, dtype=np.float64)

        # Track global best
        self.best_position = np.zeros(n_dim, dtype=np.float64)
        self.best_fitness = -np.inf

        # History
        self.history = []

    def _initialize_particles(self) -> NDArray:
        """Initialize particles uniformly within bounds."""
        if np.any(np.isinf(self.lb)) or np.any(np.isinf(self.ub)):
            # If bounds are infinite, use standard normal
            particles: NDArray = np.random.randn(self.pop_size, self.n_dim).astype(np.float64)
        else:
            # Uniform initialization within bounds
            particles = np.random.uniform(
                self.lb, self.ub, size=(self.pop_size, self.n_dim)
            ).astype(np.float64)
        return particles

    def _initialize_velocities(self) -> NDArray:
        """Initialize velocities uniformly in [-1, 1] per dimension."""
        velocities: NDArray = np.random.uniform(-1, 1, size=(self.pop_size, self.n_dim)).astype(
            np.float64
        )
        return velocities

    def _clip_to_bounds(self, particles: NDArray) -> NDArray:
        """Clip particles to specified bounds."""
        clipped: NDArray = np.clip(particles, self.lb, self.ub)
        return clipped

    def _evaluate(self, particles: NDArray) -> NDArray:
        """Evaluate objective function for all particles.

        Parameters
        ----------
        particles : NDArray
            Particle positions (pop_size, n_dim)

        Returns
        -------
        NDArray
            Fitness values (pop_size,)
        """
        fitness: list[float] = [self.objective_func(cast(NDArray, p)) for p in particles]
        return np.array(fitness, dtype=np.float64)

    def run(self) -> PSO_Result:
        """Run PSO optimization.

        Returns
        -------
        PSO_Result
            Optimization result with best position, fitness, and history
        """
        # Evaluate initial positions
        fitness = self._evaluate(self.particles)
        self.personal_best_fitness = fitness.copy()
        self.personal_best = self.particles.copy()

        # Find initial global best
        best_idx = int(np.argmax(fitness))
        self.best_position = self.particles[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        # Iterate
        for iteration in range(self.max_iter):
            # Update velocities and positions
            r1 = np.random.uniform(0, 1, size=(self.pop_size, self.n_dim))
            r2 = np.random.uniform(0, 1, size=(self.pop_size, self.n_dim))

            cognitive = self.c1 * r1 * (self.personal_best - self.particles)
            social = self.c2 * r2 * (self.best_position - self.particles)

            self.velocities = self.w * self.velocities + cognitive + social

            self.particles = self.particles + self.velocities
            self.particles = self._clip_to_bounds(self.particles)

            # Evaluate new positions
            fitness = self._evaluate(self.particles)

            # Update personal best
            improved = fitness > self.personal_best_fitness
            self.personal_best[improved] = self.particles[improved]
            self.personal_best_fitness[improved] = fitness[improved]

            # Update global best
            current_best_idx = int(np.argmax(fitness))
            current_best_fitness = float(fitness[current_best_idx])

            if current_best_fitness > self.best_fitness:
                self.best_position = self.particles[current_best_idx].copy()
                self.best_fitness = current_best_fitness

            # Track history
            if self.track_history:
                self.history.append(
                    {
                        "iteration": iteration,
                        "best_fitness": self.best_fitness,
                        "best_position": self.best_position.copy().tolist(),
                        "mean_fitness": float(np.mean(fitness)),
                        "std_fitness": float(np.std(fitness)),
                    }
                )

        # Return result
        converged = self.max_iter > 0
        return PSO_Result(
            best_position=self.best_position.copy(),
            best_fitness=self.best_fitness,
            n_iterations=self.max_iter,
            converged=converged,
            history=self.history.copy(),
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Get optimization history.

        Returns
        -------
        list[dict[str, Any]]
            History with iteration number, best fitness, and statistics
        """
        return self.history.copy()
