"""Core computation functions for segmentation metrics.

This module provides functions to compute segmentation metrics from scores,
labels, and segment boundaries.
"""

from __future__ import annotations

import numpy as np

from .metrics import (
    EPSILON,
    NDArray,
    SegmentationResult,
)


def compute_metrics(
    scores: NDArray,
    labels: NDArray,
    cuts: NDArray | list[float],
) -> SegmentationResult:
    """Compute segmentation metrics from scores, labels, and cut boundaries.

    This function evaluates a segmentation defined by cut boundaries. It computes
    R² (variance explained), within and between-group variances, segment sizes,
    and proportions.

    Parameters
    ----------
    scores : NDArray
        Continuous scores to segment (shape: (n_samples,))
    labels : NDArray
        Binary or continuous target variable (shape: (n_samples,))
    cuts : NDArray or list[float]
        Segment boundaries. For k segments, provide k-1 cuts.
        Example: cuts=[30, 50, 70] creates segments (-inf,30], (30,50], (50,70], (70,inf)

    Returns
    -------
    SegmentationResult
        Comprehensive segmentation metrics

    Examples
    --------
    >>> scores = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    >>> labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> cuts = np.array([30, 70])
    >>> result = compute_metrics(scores, labels, cuts)
    >>> print(f"R²: {result.r2:.3f}")
    >>> print(f"Segments: {result.n_segments}")
    """
    # Ensure inputs are numpy arrays with correct dtype
    scores_arr: NDArray = np.asarray(scores, dtype=np.float64)
    labels_arr: NDArray = np.asarray(labels, dtype=np.float64)
    cuts_arr: NDArray = np.asarray(cuts, dtype=np.float64).flatten()

    # Validate inputs
    if scores_arr.ndim != 1:
        msg = f"scores must be 1D array, got shape {scores_arr.shape}"
        raise ValueError(msg)
    if labels_arr.ndim != 1:
        msg = f"labels must be 1D array, got shape {labels_arr.shape}"
        raise ValueError(msg)
    if len(scores_arr) != len(labels_arr):
        msg = (
            f"scores and labels must have same length, got {len(scores_arr)} and {len(labels_arr)}"
        )
        raise ValueError(msg)
    if len(cuts_arr) == 0:
        msg = "cuts cannot be empty"
        raise ValueError(msg)

    # Sort cuts and remove duplicates
    cuts_sorted = np.unique(cuts_arr)

    # Assign each observation to a segment using digitize
    # digitize returns bin index (1-based), so segments are 1, 2, ..., k+1
    segment_indices_int = np.digitize(scores_arr, cuts_sorted)

    # Get unique segments and inverse indices
    unique_segments, inverse_indices = np.unique(segment_indices_int, return_inverse=True)
    n_segments = len(unique_segments)

    # Compute segment statistics
    segment_sizes_arr: NDArray = np.bincount(inverse_indices).astype(np.float64)
    total_n = len(scores_arr)

    # Segment proportions
    segment_proportions_arr: NDArray = segment_sizes_arr / (total_n + EPSILON)

    # Target variable (PD) per segment
    sum_labels: NDArray = np.bincount(inverse_indices, weights=labels_arr).astype(np.float64)
    pd_by_segment_arr: NDArray = sum_labels / (segment_sizes_arr + EPSILON)

    # Global mean of target
    global_mean = np.mean(labels_arr)

    # Compute R² and variances
    # H_inter: between-group variance (sum of n_i * (pd_i - global_pd)²)
    h_inter: float = float(np.sum(segment_sizes_arr * (pd_by_segment_arr - global_mean) ** 2))

    # H_intra: within-group variance
    # Expand pd values to match original observations
    pd_expanded: NDArray = pd_by_segment_arr[inverse_indices]
    h_intra: float = float(np.sum((labels_arr - pd_expanded) ** 2))

    # R²: ratio of explained variance
    total_variance = h_inter + h_intra
    r2: float = float(h_inter / (total_variance + EPSILON))

    sorted_indices = np.argsort(pd_by_segment_arr)
    sorted_proportions = segment_proportions_arr[sorted_indices]
    cumsum_proportions = np.cumsum(sorted_proportions)
    gini: float = float(
        np.clip(
            1.0 - 2.0 * np.sum(sorted_proportions * (1.0 - cumsum_proportions)),
            0.0,
            1.0,
        )
    )

    bad_counts: NDArray = sum_labels
    good_counts: NDArray = segment_sizes_arr - bad_counts
    total_bad = float(np.sum(bad_counts))
    total_good = float(np.sum(good_counts))
    if total_bad <= EPSILON or total_good <= EPSILON:
        ks = 0.0
    else:
        cumulative_bad = np.cumsum(bad_counts) / total_bad
        cumulative_good = np.cumsum(good_counts) / total_good
        ks = float(np.max(np.abs(cumulative_bad - cumulative_good)))

    return SegmentationResult(
        r2=r2,
        n_segments=n_segments,
        pd_by_segment=pd_by_segment_arr,
        segment_sizes=segment_sizes_arr,
        segment_proportions=segment_proportions_arr,
        h_inter=h_inter,
        h_intra=h_intra,
        gini=gini,
        ks=ks,
    )


def get_segment_assignments(
    scores: NDArray,
    cuts: NDArray | list[float],
) -> NDArray:
    """Assign scores to segments based on cut boundaries.

    Parameters
    ----------
    scores : NDArray
        Continuous scores (shape: (n_samples,))
    cuts : NDArray or list[float]
        Segment boundaries

    Returns
    -------
    NDArray
        Segment assignment for each score (0-indexed) (shape: (n_samples,))

    Examples
    --------
    >>> scores = np.array([10, 25, 50, 75, 100])
    >>> cuts = np.array([30, 70])
    >>> segments = get_segment_assignments(scores, cuts)
    >>> print(segments)  # [0, 0, 1, 2, 2]
    """
    scores_arr: NDArray = np.asarray(scores, dtype=np.float64)
    cuts_arr: NDArray = np.asarray(cuts, dtype=np.float64).flatten()

    cuts_sorted = np.unique(cuts_arr)
    # Use right=True to get correct 0-indexed segments
    # With right=True: scores <= cuts[0] -> 0, cuts[0] < scores <= cuts[1] -> 1, etc.
    segment_indices_int = np.digitize(scores_arr, cuts_sorted, right=True)

    return segment_indices_int.astype(np.float64)
