"""Unit tests for functional API module."""

from __future__ import annotations

import numpy as np
import pytest

from pso_segmentation.api import segment_scores
from pso_segmentation.objective_functions_examples import (
    example_fitness_custom_business_metric,
    example_fitness_gini_focused,
    example_fitness_r2_only,
    example_fitness_r2_with_balance_penalty,
    example_fitness_r2_with_monotonic_penalty,
)
from pso_segmentation.optimizer import OptimizerConfig
from pso_segmentation.segmentation.metrics import SegmentationResult


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple synthetic data for testing."""
    np.random.seed(42)
    n = 500
    scores = np.linspace(0, 1, n)
    labels = (scores > 0.5).astype(float)
    return scores, labels


@pytest.fixture
def complex_data() -> tuple[np.ndarray, np.ndarray]:
    """More complex synthetic data with noise."""
    np.random.seed(123)
    n = 1000
    scores = np.random.rand(n)
    labels = np.random.binomial(1, scores, n).astype(float)
    return scores, labels


class TestSegmentScoresBasic:
    """Tests for basic segment_scores functionality."""

    def test_segment_scores_returns_segmentation_result(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that segment_scores returns SegmentationResult."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        assert isinstance(result, SegmentationResult)

    def test_segment_scores_has_expected_attributes(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that result has all required attributes."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        assert hasattr(result, "r2")
        assert hasattr(result, "n_segments")
        assert hasattr(result, "segment_proportions")
        assert hasattr(result, "pd_by_segment")
        assert hasattr(result, "segment_sizes")
        assert hasattr(result, "h_inter")
        assert hasattr(result, "h_intra")

    def test_segment_scores_r2_in_valid_range(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that R² is in valid range [0, 1]."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        assert 0.0 <= result.r2 <= 1.0

    def test_segment_scores_produces_segments(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that segmentation produces > 1 segment."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        assert result.n_segments > 1

    def test_segment_scores_proportions_sum_to_one(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that segment proportions sum to 1."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        total = float(result.segment_proportions.sum())
        assert abs(total - 1.0) < 1e-6


class TestSegmentScoresWithConfig:
    """Tests for segment_scores with custom configuration."""

    def test_segment_scores_with_custom_config(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test segment_scores with custom OptimizerConfig."""
        scores, labels = simple_data
        config = OptimizerConfig(pop_size=30, max_iter=50)

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness, config)
        assert isinstance(result, SegmentationResult)
        assert result.n_segments >= 2

    def test_segment_scores_with_different_fitness_functions(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test segment_scores with various fitness functions."""
        scores, labels = simple_data
        fitness_funcs_unbound = [
            example_fitness_r2_only,
            example_fitness_r2_with_monotonic_penalty,
            example_fitness_r2_with_balance_penalty,
        ]
        for fitness_func_unbound in fitness_funcs_unbound:

            def fitness(cuts: np.ndarray, func: object = fitness_func_unbound) -> float:
                return func(cuts, scores, labels)  # type: ignore[misc]

            result = segment_scores(scores, labels, fitness)
            assert isinstance(result, SegmentationResult)
            assert 0.0 <= result.r2 <= 1.0

    def test_segment_scores_with_monotonic_constraint(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test segment_scores with monotonic constraint enabled."""
        scores, labels = simple_data
        config = OptimizerConfig(enforce_monotonic=True)

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness, config)
        # Check monotonicity
        assert result.is_monotonic_increasing() or result.is_monotonic_decreasing()


class TestSegmentScoresInputValidation:
    """Tests for input validation in segment_scores."""

    def test_segment_scores_rejects_mismatched_lengths(
        self,
    ) -> None:
        """Test that mismatched score/label lengths raise ValueError."""
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 1])

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        with pytest.raises(ValueError, match="Mismatched lengths"):
            segment_scores(scores, labels, fitness)

    def test_segment_scores_accepts_different_dtypes(
        self,
    ) -> None:
        """Test that segment_scores converts different dtypes."""
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        labels = np.array([0, 1, 1], dtype=np.int32)

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        assert isinstance(result, SegmentationResult)


class TestSegmentScoresWithDifferentData:
    """Tests with different data characteristics."""

    def test_segment_scores_with_complex_data(
        self, complex_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test segment_scores with more complex data."""
        scores, labels = complex_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        assert isinstance(result, SegmentationResult)
        assert result.n_segments >= 2
        assert 0.0 <= result.r2 <= 1.0

    def test_segment_scores_with_gini_metric(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test segment_scores with Gini-based fitness."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_gini_focused(cuts, scores, labels)

        result = segment_scores(scores, labels, fitness)
        assert isinstance(result, SegmentationResult)

    def test_segment_scores_with_business_constraints(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test segment_scores with business constraint fitness."""
        scores, labels = simple_data
        constraints = {
            "min_r2": 0.3,
            "min_segments_above_5pct": 3,
            "max_proportion": 0.4,
        }

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_custom_business_metric(
                cuts, scores, labels, business_constraint=constraints
            )

        result = segment_scores(scores, labels, fitness)
        assert isinstance(result, SegmentationResult)


class TestSegmentScoresDefaultConfig:
    """Tests for default configuration behavior."""

    def test_segment_scores_without_config_uses_defaults(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that segment_scores uses reasonable defaults."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        # Call without config
        result1 = segment_scores(scores, labels, fitness)
        # Call with explicit defaults
        config = OptimizerConfig()
        result2 = segment_scores(scores, labels, fitness, config)

        # Both should produce valid results
        assert isinstance(result1, SegmentationResult)
        assert isinstance(result2, SegmentationResult)
        assert result1.r2 >= 0.0
        assert result2.r2 >= 0.0


class TestSegmentScoresRepeatable:
    """Tests for result consistency and behavior."""

    def test_segment_scores_produces_consistent_results(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that results are stable (same seed → same results)."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        np.random.seed(999)
        result1 = segment_scores(scores, labels, fitness)
        np.random.seed(999)
        result2 = segment_scores(scores, labels, fitness)

        # Both should have same R² (or very close, due to floating point)
        assert abs(result1.r2 - result2.r2) < 1e-10

    def test_segment_scores_different_seeds_different_results(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that different seeds can produce different results."""
        scores, labels = simple_data

        def fitness(cuts: np.ndarray) -> float:
            return example_fitness_r2_only(cuts, scores, labels)

        np.random.seed(111)
        result1 = segment_scores(scores, labels, fitness, OptimizerConfig(max_iter=10))
        np.random.seed(222)
        result2 = segment_scores(scores, labels, fitness, OptimizerConfig(max_iter=10))

        # Different seeds may produce different results
        # (not guaranteed, but likely with small max_iter)
        # Just verify both are valid
        assert result1.r2 >= 0.0
        assert result2.r2 >= 0.0
