"""Unit tests for segmentation module."""

import numpy as np
import pytest

from pso_segmentation.segmentation import (
    SegmentationResult,
    check_segment_stability,
    compute_metrics,
    get_segment_assignments,
    validate_cuts,
    validate_segmentation,
)


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_compute_metrics_simple(self) -> None:
        """Test compute_metrics with simple data."""
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        cuts = np.array([25.0])

        result = compute_metrics(scores, labels, cuts)

        assert isinstance(result, SegmentationResult)
        assert result.n_segments == 2
        assert len(result.pd_by_segment) == 2
        assert len(result.segment_sizes) == 2

    def test_compute_metrics_multiple_cuts(self) -> None:
        """Test compute_metrics with multiple cuts."""
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        labels = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cuts = np.array([25.0, 50.0, 75.0])

        result = compute_metrics(scores, labels, cuts)

        assert result.n_segments == 4
        assert len(result.segment_sizes) == 4
        assert np.sum(result.segment_sizes) == 8

    def test_compute_metrics_r2_calculation(self) -> None:
        """Test R² is correctly calculated."""
        # Create data with clear separation
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        labels = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        cuts = np.array([4.5])

        result = compute_metrics(scores, labels, cuts)

        # With perfect separation, R² should be high
        assert result.r2 > 0.8

    def test_compute_metrics_segment_proportions(self) -> None:
        """Test segment proportions sum to 1."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        cuts = np.array([2.5, 4.0])

        result = compute_metrics(scores, labels, cuts)

        assert np.isclose(np.sum(result.segment_proportions), 1.0)

    def test_compute_metrics_pd_range(self) -> None:
        """Test PD values are in [0, 1]."""
        scores = np.random.uniform(0, 100, 100)
        labels = np.random.binomial(1, 0.3, 100)
        cuts = np.array([25.0, 50.0, 75.0])

        result = compute_metrics(scores, labels, cuts)

        assert np.all(result.pd_by_segment >= 0.0)
        assert np.all(result.pd_by_segment <= 1.0)

    def test_compute_metrics_list_cuts(self) -> None:
        """Test compute_metrics accepts list of cuts."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        cuts = [2.5]  # List instead of array

        result = compute_metrics(scores, labels, cuts)

        assert result.n_segments == 2

    def test_compute_metrics_duplicate_cuts(self) -> None:
        """Test compute_metrics handles duplicate cuts."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        cuts = np.array([2.5, 2.5, 4.0])

        result = compute_metrics(scores, labels, cuts)

        assert result.n_segments == 3  # Duplicates removed

    def test_compute_metrics_invalid_scores_shape(self) -> None:
        """Test compute_metrics raises error for 2D scores."""
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([1.0, 0.0])
        cuts = np.array([2.5])

        with pytest.raises(ValueError, match="scores must be 1D"):
            compute_metrics(scores, labels, cuts)

    def test_compute_metrics_mismatched_lengths(self) -> None:
        """Test compute_metrics raises error for mismatched lengths."""
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.0, 0.0])  # Wrong length
        cuts = np.array([2.5])

        with pytest.raises(ValueError, match="same length"):
            compute_metrics(scores, labels, cuts)

    def test_compute_metrics_empty_cuts(self) -> None:
        """Test compute_metrics raises error for empty cuts."""
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.0, 0.0, 0.0])
        cuts = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            compute_metrics(scores, labels, cuts)


class TestGetSegmentAssignments:
    """Test get_segment_assignments function."""

    def test_get_segment_assignments_simple(self) -> None:
        """Test get_segment_assignments with simple data."""
        scores = np.array([10.0, 25.0, 50.0, 75.0])
        cuts = np.array([30.0, 70.0])

        assignments = get_segment_assignments(scores, cuts)

        expected = np.array([0, 0, 1, 2])
        assert np.array_equal(assignments, expected)

    def test_get_segment_assignments_list_cuts(self) -> None:
        """Test get_segment_assignments accepts list of cuts."""
        scores = np.array([10.0, 25.0, 50.0])
        cuts = [30.0]  # List

        assignments = get_segment_assignments(scores, cuts)

        assert len(assignments) == 3

    def test_get_segment_assignments_single_cut(self) -> None:
        """Test get_segment_assignments with single cut."""
        scores = np.array([1.0, 2.0, 3.5, 4.0, 5.0])
        cuts = np.array([3.0])

        assignments = get_segment_assignments(scores, cuts)

        assert np.all(assignments[scores <= 3.0] == 0)
        assert np.all(assignments[scores > 3.0] == 1)


class TestSegmentationResult:
    """Test SegmentationResult dataclass methods."""

    @pytest.fixture
    def sample_result(self) -> SegmentationResult:
        """Create a sample SegmentationResult for testing."""
        return SegmentationResult(
            r2=0.75,
            n_segments=4,
            pd_by_segment=np.array([0.1, 0.2, 0.3, 0.4]),
            segment_sizes=np.array([100, 150, 120, 130]),
            segment_proportions=np.array([0.2, 0.3, 0.24, 0.26]),
            h_inter=0.05,
            h_intra=0.15,
        )

    def test_is_monotonic_increasing(self, sample_result: SegmentationResult) -> None:
        """Test is_monotonic_increasing method."""
        assert sample_result.is_monotonic_increasing() is True

    def test_is_monotonic_increasing_false(self) -> None:
        """Test is_monotonic_increasing returns False for non-monotonic."""
        non_monotonic = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.4, 0.2]),  # Not monotonic
            segment_sizes=np.array([100, 150, 120]),
            segment_proportions=np.array([0.25, 0.375, 0.375]),
            h_inter=0.05,
            h_intra=0.15,
        )
        assert non_monotonic.is_monotonic_increasing() is False

    def test_is_monotonic_decreasing(self) -> None:
        """Test is_monotonic_decreasing method."""
        decreasing = SegmentationResult(
            r2=0.75,
            n_segments=4,
            pd_by_segment=np.array([0.4, 0.3, 0.2, 0.1]),
            segment_sizes=np.array([100, 150, 120, 80]),
            segment_proportions=np.array([0.25, 0.375, 0.30, 0.20]),
            h_inter=0.05,
            h_intra=0.15,
        )
        assert decreasing.is_monotonic_decreasing() is True

    def test_max_segment_proportion(self, sample_result: SegmentationResult) -> None:
        """Test max_segment_proportion method."""
        assert np.isclose(sample_result.max_segment_proportion(), 0.30)

    def test_min_segment_proportion(self, sample_result: SegmentationResult) -> None:
        """Test min_segment_proportion method."""
        assert np.isclose(sample_result.min_segment_proportion(), 0.2)

    def test_is_balanced(self, sample_result: SegmentationResult) -> None:
        """Test is_balanced method."""
        # All segments between 0.05 and 0.30? min is 0.20 which is in range
        assert sample_result.is_balanced() is True

    def test_is_balanced_too_small(self) -> None:
        """Test is_balanced returns False for segments too small."""
        unbalanced = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([10, 100, 100]),
            segment_proportions=np.array([0.05, 0.5, 0.45]),  # max > 0.30
            h_inter=0.05,
            h_intra=0.15,
        )
        assert unbalanced.is_balanced() is False

    def test_is_monotonic_single_segment(self) -> None:
        """Test is_monotonic with single segment returns True."""
        result = SegmentationResult(
            r2=0.0,
            n_segments=1,
            pd_by_segment=np.array([0.2]),
            segment_sizes=np.array([100]),
            segment_proportions=np.array([1.0]),
            h_inter=0.0,
            h_intra=0.16,
        )
        assert result.is_monotonic_increasing() is True
        assert result.is_monotonic_decreasing() is True


class TestValidateCuts:
    """Test validate_cuts function."""

    def test_validate_cuts_valid(self) -> None:
        """Test validate_cuts with valid cuts."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cuts = np.array([2.0, 4.0])

        is_valid, msg = validate_cuts(cuts, scores)

        assert is_valid is True

    def test_validate_cuts_below_min(self) -> None:
        """Test validate_cuts fails when cut <= min score."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cuts = np.array([1.0])  # Equal to min

        is_valid, msg = validate_cuts(cuts, scores)

        assert is_valid is False
        assert "must be >" in msg

    def test_validate_cuts_above_max(self) -> None:
        """Test validate_cuts fails when cut >= max score."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cuts = np.array([5.0])  # Equal to max

        is_valid, msg = validate_cuts(cuts, scores)

        assert is_valid is False
        assert "must be <" in msg

    def test_validate_cuts_duplicate(self) -> None:
        """Test validate_cuts detects duplicates."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cuts = np.array([2.5, 2.5, 4.0])

        is_valid, msg = validate_cuts(cuts, scores)

        assert is_valid is False
        assert "Duplicate" in msg

    def test_validate_cuts_duplicate_allowed(self) -> None:
        """Test validate_cuts with allow_duplicate=True."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cuts = np.array([2.5, 2.5, 4.0])

        is_valid, msg = validate_cuts(cuts, scores, allow_duplicate=True)

        assert is_valid is True

    def test_validate_cuts_empty(self) -> None:
        """Test validate_cuts fails with empty cuts."""
        scores = np.array([1.0, 2.0, 3.0])
        cuts = np.array([])

        is_valid, msg = validate_cuts(cuts, scores)

        assert is_valid is False
        assert "empty" in msg


class TestValidateSegmentation:
    """Test validate_segmentation function."""

    @pytest.fixture
    def valid_result(self) -> SegmentationResult:
        """Create a valid SegmentationResult."""
        return SegmentationResult(
            r2=0.75,
            n_segments=4,
            pd_by_segment=np.array([0.1, 0.2, 0.25, 0.3]),
            segment_sizes=np.array([60, 80, 80, 80]),
            segment_proportions=np.array([0.20, 0.27, 0.27, 0.26]),  # All in range
            h_inter=0.05,
            h_intra=0.15,
        )

    def test_validate_segmentation_valid(self, valid_result: SegmentationResult) -> None:
        """Test validate_segmentation with valid result."""
        is_valid, msg = validate_segmentation(valid_result)

        assert is_valid is True

    def test_validate_segmentation_too_small(self) -> None:
        """Test validate_segmentation fails for too small segment."""
        result = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([1, 100, 100]),
            segment_proportions=np.array([0.005, 0.5, 0.495]),  # min < 0.05
            h_inter=0.05,
            h_intra=0.15,
        )
        is_valid, msg = validate_segmentation(result)

        assert is_valid is False
        assert "Minimum" in msg

    def test_validate_segmentation_too_large(self) -> None:
        """Test validate_segmentation fails for too large segment."""
        result = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([200, 100, 100]),
            segment_proportions=np.array([0.5, 0.25, 0.25]),  # max > 0.30
            h_inter=0.05,
            h_intra=0.15,
        )
        is_valid, msg = validate_segmentation(result)

        assert is_valid is False
        assert "Maximum" in msg

    def test_validate_segmentation_not_monotonic(self) -> None:
        """Test validate_segmentation fails for non-monotonic."""
        result = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.3, 0.2]),  # Not monotonic
            segment_sizes=np.array([100, 100, 100]),
            segment_proportions=np.array([0.33, 0.33, 0.34]),
            h_inter=0.05,
            h_intra=0.15,
        )
        is_valid, msg = validate_segmentation(result, monotonic=True)

        assert is_valid is False

    def test_validate_segmentation_custom_constraints(self) -> None:
        """Test validate_segmentation with custom constraints."""
        result = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([30, 40, 30]),
            segment_proportions=np.array([0.30, 0.40, 0.30]),
            h_inter=0.05,
            h_intra=0.15,
        )
        # Stricter constraints - max 0.35 fails because result has 0.40
        is_valid, msg = validate_segmentation(result, min_segment_size=0.20, max_segment_size=0.35)

        assert is_valid is False


class TestCheckSegmentStability:
    """Test check_segment_stability function."""

    def test_check_segment_stability_similar(self) -> None:
        """Test check_segment_stability with similar results."""
        result1 = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([60, 70, 70]),
            segment_proportions=np.array([0.30, 0.35, 0.35]),
            h_inter=0.05,
            h_intra=0.15,
        )
        result2 = SegmentationResult(
            r2=0.76,  # Very close
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([60, 70, 70]),
            segment_proportions=np.array([0.30, 0.35, 0.35]),
            h_inter=0.05,
            h_intra=0.15,
        )

        is_stable, diff = check_segment_stability(result1, result2)

        assert is_stable is True
        assert diff < 0.05

    def test_check_segment_stability_different(self) -> None:
        """Test check_segment_stability with different results."""
        result1 = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([60, 70, 70]),
            segment_proportions=np.array([0.30, 0.35, 0.35]),
            h_inter=0.05,
            h_intra=0.15,
        )
        result2 = SegmentationResult(
            r2=0.60,  # Significantly different
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([60, 70, 70]),
            segment_proportions=np.array([0.30, 0.35, 0.35]),
            h_inter=0.05,
            h_intra=0.15,
        )

        is_stable, diff = check_segment_stability(result1, result2)

        assert is_stable is False
        assert diff > 0.05

    def test_check_segment_stability_custom_tolerance(self) -> None:
        """Test check_segment_stability with custom tolerance."""
        result1 = SegmentationResult(
            r2=0.75,
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([60, 70, 70]),
            segment_proportions=np.array([0.30, 0.35, 0.35]),
            h_inter=0.05,
            h_intra=0.15,
        )
        result2 = SegmentationResult(
            r2=0.72,  # Diff = 0.03
            n_segments=3,
            pd_by_segment=np.array([0.1, 0.2, 0.3]),
            segment_sizes=np.array([60, 70, 70]),
            segment_proportions=np.array([0.30, 0.35, 0.35]),
            h_inter=0.05,
            h_intra=0.15,
        )

        # Stricter tolerance
        is_stable, diff = check_segment_stability(result1, result2, tolerance=0.01)

        assert is_stable is False

        # Looser tolerance
        is_stable, diff = check_segment_stability(result1, result2, tolerance=0.05)

        assert is_stable is True
