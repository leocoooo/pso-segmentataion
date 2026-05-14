"""Unit tests for PSO module."""

import numpy as np
import pytest

from pso_segmentation.core import PSO, PSO_Result


class TestPSOInitialization:
    """Test PSO initialization."""

    def test_pso_init_basic(self) -> None:
        """Test basic PSO initialization."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=3, pop_size=10, max_iter=5)
        assert pso.n_dim == 3
        assert pso.pop_size == 10
        assert pso.max_iter == 5
        assert pso.particles.shape == (10, 3)

    def test_pso_init_with_bounds(self) -> None:
        """Test PSO initialization with bounds."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        lb = np.array([0.0, -10.0, -5.0])
        ub = np.array([10.0, 10.0, 5.0])

        pso = PSO(sphere, n_dim=3, lb=lb, ub=ub)
        assert np.allclose(pso.lb, lb)
        assert np.allclose(pso.ub, ub)

    def test_pso_init_invalid_bounds(self) -> None:
        """Test PSO raises error for invalid bounds."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        lb = np.array([10.0, 10.0])
        ub = np.array([0.0, 5.0])

        with pytest.raises(
            ValueError, match="Lower bounds must be strictly less than upper bounds"
        ):
            PSO(sphere, n_dim=2, lb=lb, ub=ub)

    def test_pso_init_edge_case_pop_size_zero(self) -> None:
        """Test PSO with pop_size=0 becomes pop_size=1."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, pop_size=0)
        assert pso.pop_size == 1

    def test_pso_init_edge_case_max_iter_zero(self) -> None:
        """Test PSO with max_iter=0."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, max_iter=0)
        assert pso.max_iter == 0


class TestPSOExecution:
    """Test PSO execution."""

    def test_pso_run_returns_result(self) -> None:
        """Test PSO.run() returns PSO_Result."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=3, pop_size=20, max_iter=50, seed=42)
        result = pso.run()

        assert isinstance(result, PSO_Result)
        assert result.best_position.shape == (3,)
        assert isinstance(result.best_fitness, float)
        assert result.n_iterations == 50

    def test_pso_converges_simple_sphere(self) -> None:
        """Test PSO converges on simple sphere function."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, pop_size=30, max_iter=100, seed=42)
        result = pso.run()

        # Sphere minimum is at (0, 0) with fitness 0
        # After optimization, should be closer to 0
        assert result.best_fitness > -10.0  # Better than random point at ±3

    def test_pso_respects_bounds(self) -> None:
        """Test PSO respects boundary constraints."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        lb = np.array([-5.0, -10.0])
        ub = np.array([5.0, 10.0])

        pso = PSO(sphere, n_dim=2, pop_size=20, max_iter=50, lb=lb, ub=ub, seed=42)
        result = pso.run()

        assert np.all(result.best_position >= lb)
        assert np.all(result.best_position <= ub)

    def test_pso_reproducibility_with_seed(self) -> None:
        """Test PSO produces same result with same seed."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso1 = PSO(sphere, n_dim=3, pop_size=20, max_iter=50, seed=123)
        result1 = pso1.run()

        pso2 = PSO(sphere, n_dim=3, pop_size=20, max_iter=50, seed=123)
        result2 = pso2.run()

        assert np.allclose(result1.best_position, result2.best_position)
        assert np.isclose(result1.best_fitness, result2.best_fitness)

    def test_pso_different_results_different_seeds(self) -> None:
        """Test PSO produces different results with different seeds."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso1 = PSO(sphere, n_dim=3, pop_size=20, max_iter=50, seed=111)
        result1 = pso1.run()

        pso2 = PSO(sphere, n_dim=3, pop_size=20, max_iter=50, seed=222)
        result2 = pso2.run()

        # At least positions should be different
        assert not np.allclose(result1.best_position, result2.best_position)

    def test_pso_edge_case_single_dimension(self) -> None:
        """Test PSO with n_dim=1."""

        def f(x: np.ndarray) -> float:
            return -((x[0] - 5.0) ** 2)

        pso = PSO(f, n_dim=1, pop_size=20, max_iter=50, seed=42)
        result = pso.run()

        # Optimum at x=5
        assert 4.0 < result.best_position[0] < 6.0

    def test_pso_edge_case_single_particle(self) -> None:
        """Test PSO with pop_size=1."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, pop_size=1, max_iter=30, seed=42)
        result = pso.run()

        assert result.best_position.shape == (2,)
        assert isinstance(result.best_fitness, float)

    def test_pso_edge_case_zero_iterations(self) -> None:
        """Test PSO with max_iter=0."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, pop_size=10, max_iter=0)
        result = pso.run()

        assert result.n_iterations == 0
        assert result.converged is False  # max_iter=0 means no iterations performed


class TestPSOHistory:
    """Test PSO history tracking."""

    def test_pso_history_tracked(self) -> None:
        """Test PSO tracks history during optimization."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, pop_size=10, max_iter=20, track_history=True, seed=42)
        result = pso.run()

        assert len(result.history) == 20
        assert all("iteration" in h for h in result.history)
        assert all("best_fitness" in h for h in result.history)
        assert all("best_position" in h for h in result.history)

    def test_pso_history_not_tracked(self) -> None:
        """Test PSO doesn't track history if disabled."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, pop_size=10, max_iter=20, track_history=False, seed=42)
        result = pso.run()

        assert len(result.history) == 0

    def test_pso_history_fitness_improves(self) -> None:
        """Test PSO history shows non-decreasing best fitness."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=2, pop_size=20, max_iter=50, seed=42)
        result = pso.run()

        fitness_values = [h["best_fitness"] for h in result.history]
        # Check monotonicity: best fitness should never decrease
        assert all(
            fitness_values[i] <= fitness_values[i + 1] for i in range(len(fitness_values) - 1)
        )


class TestPSOBounds:
    """Test PSO boundary handling."""

    def test_pso_particles_stay_in_bounds(self) -> None:
        """Test all particles remain within bounds during optimization."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        lb = np.array([-2.0, -3.0])
        ub = np.array([2.0, 3.0])

        pso = PSO(sphere, n_dim=2, pop_size=30, max_iter=100, lb=lb, ub=ub, seed=42)
        pso.run()

        # Check internal particles stayed in bounds
        assert np.all(pso.particles >= lb)
        assert np.all(pso.particles <= ub)

    def test_pso_asymmetric_bounds(self) -> None:
        """Test PSO with asymmetric bounds."""

        def f(x: np.ndarray) -> float:
            return -((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

        lb = np.array([-10.0, -100.0])
        ub = np.array([100.0, 10.0])

        pso = PSO(f, n_dim=2, pop_size=30, max_iter=100, lb=lb, ub=ub, seed=42)
        result = pso.run()

        assert np.all(result.best_position >= lb)
        assert np.all(result.best_position <= ub)


class TestPSOParameters:
    """Test PSO with different hyperparameters."""

    def test_pso_different_inertia_weights(self) -> None:
        """Test PSO with different inertia weights."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        results = []
        for w in [0.4, 0.7, 0.9]:
            pso = PSO(
                sphere,
                n_dim=2,
                pop_size=20,
                max_iter=50,
                w=w,
                seed=42,
            )
            results.append(pso.run().best_fitness)

        # All should converge reasonably
        assert all(r > -5.0 for r in results)

    def test_pso_different_cognitive_social_weights(self) -> None:
        """Test PSO with different cognitive/social coefficients."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso1 = PSO(sphere, n_dim=2, pop_size=20, max_iter=50, c1=0.5, c2=2.5, seed=42)
        result1 = pso1.run()

        pso2 = PSO(sphere, n_dim=2, pop_size=20, max_iter=50, c1=2.5, c2=0.5, seed=42)
        result2 = pso2.run()

        # Both should converge but potentially differently
        assert isinstance(result1.best_fitness, float)
        assert isinstance(result2.best_fitness, float)


class TestPSOEdgeCases:
    """Test PSO edge cases and corner scenarios."""

    def test_pso_constant_function(self) -> None:
        """Test PSO with constant objective function."""

        def const(_x: np.ndarray) -> float:
            return 5.0

        pso = PSO(const, n_dim=2, pop_size=10, max_iter=20)
        result = pso.run()

        assert result.best_fitness == 5.0

    def test_pso_large_dimensional(self) -> None:
        """Test PSO with high-dimensional problem."""

        def sphere(x: np.ndarray) -> float:
            return -np.sum(x**2)

        pso = PSO(sphere, n_dim=50, pop_size=50, max_iter=100, seed=42)
        result = pso.run()

        assert result.best_position.shape == (50,)

    def test_pso_negative_fitness(self) -> None:
        """Test PSO handles negative fitness values."""

        def f(x: np.ndarray) -> float:
            return -1000.0 - np.sum(x**2)

        pso = PSO(f, n_dim=2, pop_size=20, max_iter=50, seed=42)
        result = pso.run()

        assert result.best_fitness < 0
        assert result.best_fitness > -1005.0
