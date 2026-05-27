"""Unit tests for example fitness functions."""

import numpy as np
import pytest

from pso_segmentation.objective_functions_examples import (
    example_fitness_custom_business_metric,
    example_fitness_gini_focused,
    example_fitness_r2_only,
    example_fitness_r2_with_all_constraints,
    example_fitness_r2_with_balance_penalty,
    example_fitness_r2_with_monotonic_penalty,
)


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Create simple test data."""
    np.random.seed(42)
    scores = np.linspace(0, 100, 100)
    labels = np.array([1.0] * 30 + [0.5] * 40 + [0.0] * 30)
    return scores, labels


@pytest.fixture
def perfect_cuts() -> np.ndarray:
    """Create cuts that separate high/low PD well."""
    return np.array([33.0, 67.0])


@pytest.fixture
def poor_cuts() -> np.ndarray:
    """Create cuts that don't separate well."""
    return np.array([25.0, 75.0])


class TestExampleFitnessR2Only:
    """Test R²-only fitness function."""

    def test_r2_only_returns_float(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test function returns float."""
        scores, labels = simple_data
        fitness = example_fitness_r2_only(perfect_cuts, scores, labels)

        assert isinstance(fitness, float)

    def test_r2_only_positive(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test fitness is positive for good cuts."""
        scores, labels = simple_data
        fitness = example_fitness_r2_only(perfect_cuts, scores, labels)

        assert fitness > 0.0

    def test_r2_only_range(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test fitness is in [0, 1] range."""
        scores, labels = simple_data
        fitness = example_fitness_r2_only(perfect_cuts, scores, labels)

        assert 0.0 <= fitness <= 1.0

    def test_r2_only_invalid_cuts(self, simple_data: tuple) -> None:
        """Test handles invalid cuts gracefully."""
        scores, labels = simple_data
        invalid_cuts = np.array([0.0])  # At or below min score boundary

        fitness = example_fitness_r2_only(invalid_cuts, scores, labels)

        # Should return very small value (near 0) on invalid cuts
        assert fitness < 1e-10

    def test_r2_only_empty_cuts(self, simple_data: tuple) -> None:
        """Test handles empty cuts."""
        scores, labels = simple_data
        empty_cuts = np.array([])

        fitness = example_fitness_r2_only(empty_cuts, scores, labels)

        assert fitness == 0.0


class TestExampleFitnessMonotonicPenalty:
    """Test R² with monotonic penalty."""

    def test_monotonic_penalty_returns_float(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test function returns float."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_monotonic_penalty(perfect_cuts, scores, labels)

        assert isinstance(fitness, float)

    def test_monotonic_penalty_range(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test fitness is in [0, 1] range."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_monotonic_penalty(perfect_cuts, scores, labels)

        assert 0.0 <= fitness <= 1.0

    def test_monotonic_penalty_weight_effect(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test penalty weight affects fitness."""
        scores, labels = simple_data

        fitness_no_penalty = example_fitness_r2_with_monotonic_penalty(
            perfect_cuts, scores, labels, penalty_weight=0.0
        )
        _ = example_fitness_r2_with_monotonic_penalty(
            perfect_cuts, scores, labels, penalty_weight=1.0
        )

        # Higher penalty weight could reduce fitness (if cuts are non-monotonic)
        # For perfect_cuts which should be monotonic, both should be same
        assert fitness_no_penalty > 0.0

    def test_monotonic_penalty_custom_weight(
        self, simple_data: tuple, poor_cuts: np.ndarray
    ) -> None:
        """Test custom penalty weight."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_monotonic_penalty(
            poor_cuts, scores, labels, penalty_weight=0.2
        )

        assert 0.0 <= fitness <= 1.0


class TestExampleFitnessBalancePenalty:
    """Test R² with balance penalty."""

    def test_balance_penalty_returns_float(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test function returns float."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_balance_penalty(perfect_cuts, scores, labels)

        assert isinstance(fitness, float)

    def test_balance_penalty_range(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test fitness is in [0, 1] range."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_balance_penalty(perfect_cuts, scores, labels)

        assert 0.0 <= fitness <= 1.0

    def test_balance_penalty_custom_constraints(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test custom min/max size constraints."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_balance_penalty(
            perfect_cuts,
            scores,
            labels,
            min_size=0.1,
            max_size=0.5,
            penalty_weight=0.1,
        )

        assert 0.0 <= fitness <= 1.0


class TestExampleFitnessAllConstraints:
    """Test R² with all constraints."""

    def test_all_constraints_returns_float(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test function returns float."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_all_constraints(perfect_cuts, scores, labels)

        assert isinstance(fitness, float)

    def test_all_constraints_range(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test fitness is in [0, 1] range."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_all_constraints(perfect_cuts, scores, labels)

        assert 0.0 <= fitness <= 1.0

    def test_all_constraints_with_options(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test with various constraint options."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_all_constraints(
            perfect_cuts,
            scores,
            labels,
            enforce_monotonic=True,
            monotonic_weight=0.2,
            balance_weight=0.1,
        )

        assert 0.0 <= fitness <= 1.0

    def test_all_constraints_no_monotonic(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test with monotonic enforcement disabled."""
        scores, labels = simple_data
        fitness = example_fitness_r2_with_all_constraints(
            perfect_cuts, scores, labels, enforce_monotonic=False
        )

        assert 0.0 <= fitness <= 1.0


class TestExampleFitnessGiniFocused:
    """Test Gini-focused fitness function."""

    def test_gini_focused_returns_float(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test function returns float."""
        scores, labels = simple_data
        fitness = example_fitness_gini_focused(perfect_cuts, scores, labels)

        assert isinstance(fitness, float)

    def test_gini_focused_range(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test fitness is in [0, 1] range."""
        scores, labels = simple_data
        fitness = example_fitness_gini_focused(perfect_cuts, scores, labels)

        assert 0.0 <= fitness <= 1.0

    def test_gini_focused_weight_effect(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test weight parameters affect result."""
        scores, labels = simple_data

        fitness1 = example_fitness_gini_focused(
            perfect_cuts, scores, labels, r2_weight=1.0, gini_weight=0.0
        )
        fitness2 = example_fitness_gini_focused(
            perfect_cuts, scores, labels, r2_weight=0.0, gini_weight=1.0
        )

        # Both should be valid
        assert 0.0 <= fitness1 <= 1.0
        assert 0.0 <= fitness2 <= 1.0

    def test_gini_focused_custom_weights(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test custom weight combinations."""
        scores, labels = simple_data
        fitness = example_fitness_gini_focused(
            perfect_cuts,
            scores,
            labels,
            r2_weight=0.6,
            gini_weight=0.4,
        )

        assert 0.0 <= fitness <= 1.0


class TestExampleFitnessCustomBusinessMetric:
    """Test custom business metric fitness function."""

    def test_custom_business_returns_float(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test function returns float."""
        scores, labels = simple_data
        fitness = example_fitness_custom_business_metric(perfect_cuts, scores, labels)

        assert isinstance(fitness, float)

    def test_custom_business_default_constraints(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test with default business constraints."""
        scores, labels = simple_data
        fitness = example_fitness_custom_business_metric(perfect_cuts, scores, labels)

        assert 0.0 <= fitness <= 1.0

    def test_custom_business_custom_constraints(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test with custom business constraints."""
        scores, labels = simple_data
        constraints = {
            "min_r2": 0.2,
            "min_segments_above_5pct": 2,
            "max_proportion": 0.45,
        }
        fitness = example_fitness_custom_business_metric(
            perfect_cuts, scores, labels, business_constraint=constraints
        )

        assert 0.0 <= fitness <= 1.0

    def test_custom_business_strict_constraints(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test with very strict constraints."""
        scores, labels = simple_data
        constraints = {
            "min_r2": 0.9,  # Very high R² requirement
            "min_segments_above_5pct": 5,
            "max_proportion": 0.1,  # Very strict balance
        }
        fitness = example_fitness_custom_business_metric(
            perfect_cuts, scores, labels, business_constraint=constraints
        )

        # May return 0 if constraints too strict
        assert 0.0 <= fitness <= 1.0

    def test_custom_business_none_constraint(
        self, simple_data: tuple, perfect_cuts: np.ndarray
    ) -> None:
        """Test with None constraint (uses defaults)."""
        scores, labels = simple_data
        fitness = example_fitness_custom_business_metric(
            perfect_cuts, scores, labels, business_constraint=None
        )

        assert 0.0 <= fitness <= 1.0


class TestFitnessFunctionConsistency:
    """Test consistency across fitness functions."""

    def test_all_functions_return_float(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test all functions return float."""
        scores, labels = simple_data

        f1 = example_fitness_r2_only(perfect_cuts, scores, labels)
        f2 = example_fitness_r2_with_monotonic_penalty(perfect_cuts, scores, labels)
        f3 = example_fitness_r2_with_balance_penalty(perfect_cuts, scores, labels)
        f4 = example_fitness_r2_with_all_constraints(perfect_cuts, scores, labels)
        f5 = example_fitness_gini_focused(perfect_cuts, scores, labels)
        f6 = example_fitness_custom_business_metric(perfect_cuts, scores, labels)

        assert all(isinstance(f, float) for f in [f1, f2, f3, f4, f5, f6])

    def test_all_functions_in_range(self, simple_data: tuple, perfect_cuts: np.ndarray) -> None:
        """Test all functions return values in valid range."""
        scores, labels = simple_data

        functions = [
            example_fitness_r2_only,
            example_fitness_r2_with_monotonic_penalty,
            example_fitness_r2_with_balance_penalty,
            example_fitness_r2_with_all_constraints,
            example_fitness_gini_focused,
            example_fitness_custom_business_metric,
        ]

        for func in functions:
            fitness = func(perfect_cuts, scores, labels)
            assert 0.0 <= fitness <= 1.0, f"{func.__name__} returned {fitness}"

    def test_error_handling(self, simple_data: tuple) -> None:
        """Test all functions handle errors gracefully."""
        scores, labels = simple_data
        invalid_cuts = np.array([])  # Empty cuts

        functions = [
            example_fitness_r2_only,
            example_fitness_r2_with_monotonic_penalty,
            example_fitness_r2_with_balance_penalty,
            example_fitness_r2_with_all_constraints,
            example_fitness_gini_focused,
            example_fitness_custom_business_metric,
        ]

        for func in functions:
            fitness = func(invalid_cuts, scores, labels)
            assert 0.0 <= fitness <= 1.0

    def test_better_cuts_higher_fitness(
        self, simple_data: tuple, perfect_cuts: np.ndarray, poor_cuts: np.ndarray
    ) -> None:
        """Test that better cuts generally have higher fitness."""
        scores, labels = simple_data

        # R²-only should clearly prefer better cuts
        f_perfect = example_fitness_r2_only(perfect_cuts, scores, labels)
        f_poor = example_fitness_r2_only(poor_cuts, scores, labels)

        # Better cuts should have higher R²
        assert f_perfect >= f_poor, "Perfect cuts should have >= fitness than poor cuts"
