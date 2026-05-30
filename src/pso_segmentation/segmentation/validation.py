"""Validation functions for segmentation boundaries.

This module provides functions to validate segment boundaries (cuts) and
check constraints on segmentation results.
"""

from __future__ import annotations

import numpy as np

from .metrics import (
    DEFAULT_MAX_SEGMENT_SIZE,
    DEFAULT_MIN_SEGMENT_SIZE,
    NDArray,
    SegmentationResult,
)


def validate_cuts(
    cuts: NDArray | list[float],
    scores: NDArray,
    allow_duplicate: bool = False,
) -> tuple[bool, str]:
    """Validate that cuts are appropriate for the given scores.

    Parameters
    ----------
    cuts : NDArray or list[float]
        Segment boundaries to validate
    scores : NDArray
        Continuous scores (for min/max reference)
    allow_duplicate : bool, default=False
        If False, raise error on duplicate cuts

    Returns
    -------
    tuple[bool, str]
        (is_valid, message) - Boolean validity and descriptive message

    Examples
    --------
    >>> scores = np.array([10, 20, 30, 40, 50])
    >>> cuts = np.array([15, 25, 35])
    >>> is_valid, msg = validate_cuts(cuts, scores)
    >>> print(is_valid)  # True
    """
    cuts_arr: NDArray = np.asarray(cuts, dtype=np.float64).flatten()
    scores_arr: NDArray = np.asarray(scores, dtype=np.float64)

    if len(cuts_arr) == 0:
        return False, "Cuts array is empty"

    score_min = float(scores_arr.min())
    score_max = float(scores_arr.max())

    # Check if cuts are strictly within bounds
    cuts_sorted = np.sort(cuts_arr)

    if float(cuts_sorted[0]) <= score_min:
        return (
            False,
            f"First cut ({cuts_sorted[0]:.6f}) must be > min score ({score_min:.6f})",
        )

    if float(cuts_sorted[-1]) >= score_max:
        return (
            False,
            f"Last cut ({cuts_sorted[-1]:.6f}) must be < max score ({score_max:.6f})",
        )

    # Check for duplicates
    if not allow_duplicate:
        unique_cuts = np.unique(cuts_sorted)
        if len(unique_cuts) != len(cuts_sorted):
            return False, "Duplicate cuts detected"

    return True, "Cuts are valid"


def validate_segmentation(
    result: SegmentationResult,
    min_segment_size: float = DEFAULT_MIN_SEGMENT_SIZE,
    max_segment_size: float = DEFAULT_MAX_SEGMENT_SIZE,
    monotonic: bool = True,
) -> tuple[bool, str]:
    """Validate segmentation result against business constraints.

    Parameters
    ----------
    result : SegmentationResult
        Segmentation result to validate
    min_segment_size : float, default=0.05
        Minimum allowed segment proportion
    max_segment_size : float, default=0.30
        Maximum allowed segment proportion
    monotonic : bool, default=True
        Whether to require monotonicity (increasing target mean with score)

    Returns
    -------
    tuple[bool, str]
        (is_valid, message) - Boolean validity and descriptive message

    Examples
    --------
    >>> result = compute_metrics(scores, labels, cuts)
    >>> is_valid, msg = validate_segmentation(result)
    >>> if not is_valid:
    ...     print(f"Validation failed: {msg}")
    """
    # Check segment size constraints
    min_prop = result.min_segment_proportion()
    max_prop = result.max_segment_proportion()

    if min_prop < min_segment_size:
        return (
            False,
            f"Minimum segment proportion ({min_prop:.4f}) < {min_segment_size:.4f}",
        )

    if max_prop > max_segment_size:
        return (
            False,
            f"Maximum segment proportion ({max_prop:.4f}) > {max_segment_size:.4f}",
        )

    # Check monotonicity if required
    if monotonic and not result.is_monotonic_increasing():
        return False, "Segmentation is not monotonic increasing (target mean should increase)"

    return True, "Segmentation is valid"


def check_segment_stability(
    result1: SegmentationResult,
    result2: SegmentationResult,
    tolerance: float = 0.05,
) -> tuple[bool, float]:
    """Check if two segmentation results are similar (stability check).

    Useful for comparing different optimization runs or parameter settings.

    Parameters
    ----------
    result1 : SegmentationResult
        First segmentation result
    result2 : SegmentationResult
        Second segmentation result
    tolerance : float, default=0.05
        Maximum allowed difference in R² (5%)

    Returns
    -------
    tuple[bool, float]
        (is_stable, r2_difference) - Stability flag and R² difference

    Examples
    --------
    >>> result1 = compute_metrics(scores, labels, cuts1)
    >>> result2 = compute_metrics(scores, labels, cuts2)
    >>> is_stable, diff = check_segment_stability(result1, result2)
    >>> print(f"Stable: {is_stable}, Difference: {diff:.4f}")
    """
    r2_diff = abs(result1.r2 - result2.r2)
    is_stable = r2_diff <= tolerance

    return is_stable, float(r2_diff)
