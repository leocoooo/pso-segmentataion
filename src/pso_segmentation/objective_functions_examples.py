"""Example fitness functions for segmentation optimization.

This module provides pedagogical examples of fitness functions that can be
used with SegmentationOptimizer. Users can copy and adapt these examples
to implement custom business metrics.

Each example demonstrates different approaches:
- Simple R² maximization (generic)
- R² with constraint penalties (generic templates)
- Business-specific metrics
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pso_segmentation.segmentation import (
    compute_metrics,
)

# Type alias
NDArray = np.ndarray[Any, np.dtype[np.float64]]


def example_fitness_r2_only(
    cuts: NDArray,
    scores: NDArray,
    labels: NDArray,
) -> float:
    """Example: Maximize R² without constraints.

    Simple baseline: just maximize the variance explained by segmentation.
    No penalty for constraint violations.

    Parameters
    ----------
    cuts : NDArray
        Segment cut boundaries (shape: (n_cuts,))
    scores : NDArray
        Continuous scores (shape: (n_samples,))
    labels : NDArray
        Target variable (shape: (n_samples,))

    Returns
    -------
    float
        R² value (higher is better, range [0, 1])

    Example
    -------
    >>> scores = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    >>> labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> cuts = np.array([30, 70])
    >>> fitness = example_fitness_r2_only(cuts, scores, labels)
    """
    try:
        result = compute_metrics(scores, labels, cuts)
        return float(result.r2)
    except (ValueError, RuntimeError):
        return 0.0


def example_fitness_r2_with_monotonic_penalty(
    cuts: NDArray,
    scores: NDArray,
    labels: NDArray,
    penalty_weight: float = 0.5,
) -> float:
    """Example: R² with penalty for non-monotonic segmentation.

    Maximize R² while penalizing violations of monotonic increasing target mean.
    Useful when the target mean should increase with the score.

    Parameters
    ----------
    cuts : NDArray
        Segment cut boundaries
    scores : NDArray
        Continuous scores
    labels : NDArray
        Target variable
    penalty_weight : float, default=0.5
        Weight for monotonicity penalty (0-1).
        Higher weight = stricter enforcement.

    Returns
    -------
    float
        Penalized fitness (range [0, 1])

    Example
    -------
    >>> optimizer.fit(scores, labels,
    ...    lambda cuts: example_fitness_r2_with_monotonic_penalty(
    ...        cuts, scores, labels, penalty_weight=0.3))
    """
    try:
        result = compute_metrics(scores, labels, cuts)
        r2 = float(result.r2)

        # Check monotonicity
        is_monotonic = result.is_monotonic_increasing()

        # Apply penalty if not monotonic
        if not is_monotonic:
            # Penalty decreases fitness
            penalty = penalty_weight
            return r2 * (1.0 - penalty)

        return r2
    except (ValueError, RuntimeError):
        return 0.0


def example_fitness_r2_with_balance_penalty(
    cuts: NDArray,
    scores: NDArray,
    labels: NDArray,
    min_size: float = 0.05,
    max_size: float = 0.30,
    penalty_weight: float = 0.3,
) -> float:
    """Example: R² with penalty for imbalanced segments.

    Maximize R² while encouraging balanced segment sizes.
    Useful when you want segments to be roughly equal in size.

    Parameters
    ----------
    cuts : NDArray
        Segment cut boundaries
    scores : NDArray
        Continuous scores
    labels : NDArray
        Target variable
    min_size : float, default=0.05
        Minimum segment proportion
    max_size : float, default=0.30
        Maximum segment proportion
    penalty_weight : float, default=0.3
        Weight for balance penalty (0-1)

    Returns
    -------
    float
        Penalized fitness (range [0, 1])

    Example
    -------
    >>> optimizer.fit(scores, labels,
    ...    lambda cuts: example_fitness_r2_with_balance_penalty(
    ...        cuts, scores, labels, penalty_weight=0.2))
    """
    try:
        result = compute_metrics(scores, labels, cuts)
        r2 = float(result.r2)

        # Check balance
        is_balanced = result.is_balanced(min_size, max_size)

        if not is_balanced:
            penalty = penalty_weight
            return r2 * (1.0 - penalty)

        return r2
    except (ValueError, RuntimeError):
        return 0.0


def example_fitness_r2_with_all_constraints(
    cuts: NDArray,
    scores: NDArray,
    labels: NDArray,
    min_size: float = 0.05,
    max_size: float = 0.30,
    enforce_monotonic: bool = True,
    monotonic_weight: float = 0.3,
    balance_weight: float = 0.2,
) -> float:
    """Example: R² with multiple constraint penalties.

    Maximize R² while enforcing both monotonicity and balance constraints.
    Most practical approach for business segmentation.

    Parameters
    ----------
    cuts : NDArray
        Segment cut boundaries
    scores : NDArray
        Continuous scores
    labels : NDArray
        Target variable
    min_size : float, default=0.05
        Minimum segment proportion
    max_size : float, default=0.30
        Maximum segment proportion
    enforce_monotonic : bool, default=True
        Whether to enforce monotonic increasing constraint
    monotonic_weight : float, default=0.3
        Weight for monotonicity penalty
    balance_weight : float, default=0.2
        Weight for balance penalty

    Returns
    -------
    float
        Penalized fitness (range [0, 1])

    Example
    -------
    >>> optimizer = SegmentationOptimizer(OptimizerConfig(max_iter=100))
    >>> optimizer.fit(scores, labels,
    ...    lambda cuts: example_fitness_r2_with_all_constraints(
    ...        cuts, scores, labels,
    ...        monotonic_weight=0.25,
    ...        balance_weight=0.15))
    """
    try:
        result = compute_metrics(scores, labels, cuts)
        r2 = float(result.r2)

        total_penalty = 0.0

        # Monotonicity penalty
        if enforce_monotonic and not result.is_monotonic_increasing():
            total_penalty += monotonic_weight

        # Balance penalty
        if not result.is_balanced(min_size, max_size):
            total_penalty += balance_weight

        return r2 * (1.0 - total_penalty)
    except (ValueError, RuntimeError):
        return 0.0


def example_fitness_gini_focused(
    cuts: NDArray,
    scores: NDArray,
    labels: NDArray,
    r2_weight: float = 0.5,
    gini_weight: float = 0.5,
) -> float:
    """Example: Weighted combination of R² and Gini coefficient.

    Combines R² (predictive power) with Gini coefficient (concentration).
    Useful when you want both good predictions and risk concentration.

    The Gini coefficient measures inequality: 0 = uniform, 1 = all in one segment.

    Parameters
    ----------
    cuts : NDArray
        Segment cut boundaries
    scores : NDArray
        Continuous scores
    labels : NDArray
        Target variable (Gini is most interpretable for binary targets)
    r2_weight : float, default=0.5
        Weight for R² component
    gini_weight : float, default=0.5
        Weight for Gini component (1 - Gini for maximization)

    Returns
    -------
    float
        Weighted fitness (range [0, 1])

    Example
    -------
    >>> optimizer.fit(scores, labels,
    ...    lambda cuts: example_fitness_gini_focused(
    ...        cuts, scores, labels,
    ...        r2_weight=0.6, gini_weight=0.4))

    Notes
    -----
    Gini coefficient is calculated as:
    Gini = 1 - 2 * sum(segment_proportion_i * (1 - cumsum_target_i))
    where segment_proportion is the share of population in each segment
    and cumsum_target is the cumulative target mean up to that segment.
    """
    try:
        result = compute_metrics(scores, labels, cuts)
        r2 = float(result.r2)

        # Calculate Gini coefficient from segment proportions
        # Sorted by segment index (which should be sorted by target mean if monotonic)
        proportions = result.segment_proportions
        target_mean_values = result.target_mean_by_segment

        # Sort by target mean for Gini calculation
        sorted_indices = np.argsort(target_mean_values)
        sorted_proportions = proportions[sorted_indices]

        # Cumulative proportion of population
        cumsum_proportions = np.cumsum(sorted_proportions)

        # Gini coefficient (Lorenz curve based)
        # This is a simplified Gini for target concentration
        gini = 1.0 - 2.0 * np.sum(sorted_proportions * (1.0 - cumsum_proportions))
        gini = np.clip(gini, 0.0, 1.0)

        # Combine metrics: we want HIGH R² and HIGH GINI (concentration)
        # GINI is already 0-1, so use it directly
        combined = r2_weight * r2 + gini_weight * gini

        return float(combined)
    except (ValueError, RuntimeError):
        return 0.0


def example_fitness_custom_business_metric(
    cuts: NDArray,
    scores: NDArray,
    labels: NDArray,
    business_constraint: dict[str, float] | None = None,
) -> float:
    """Example: Custom business-specific fitness function.

    Demonstrates how to implement domain-specific metrics.
    This example optimizes for:
    - High R² (predictive power)
    - Monotonic increasing target mean (risk ordering)
    - At least 3 segments with >5% population (market presence)
    - No segment with >40% population (concentration risk)

    Parameters
    ----------
    cuts : NDArray
        Segment cut boundaries
    scores : NDArray
        Continuous scores
    labels : NDArray
        Target variable
    business_constraint : dict, optional
        Custom constraints dict with keys like:
        - 'min_r2': minimum acceptable R² (default: 0.3)
        - 'min_segments_above_5pct': min segments >5% (default: 3)
        - 'max_proportion': max prop for one segment (default: 0.4)

    Returns
    -------
    float
        Business-adjusted fitness (range [0, 1])

    Example
    -------
    >>> constraints = {
    ...     'min_r2': 0.35,
    ...     'min_segments_above_5pct': 4,
    ...     'max_proportion': 0.35
    ... }
    >>> optimizer.fit(scores, labels,
    ...    lambda cuts: example_fitness_custom_business_metric(
    ...        cuts, scores, labels, business_constraint=constraints))
    """
    # Default constraints
    if business_constraint is None:
        business_constraint = {}

    min_r2 = business_constraint.get("min_r2", 0.3)
    min_segments_5pct = business_constraint.get("min_segments_above_5pct", 3)
    max_proportion = business_constraint.get("max_proportion", 0.4)

    try:
        result = compute_metrics(scores, labels, cuts)
        r2 = float(result.r2)

        # Hard constraint: R² must exceed minimum
        if r2 < min_r2:
            return 0.0

        # Hard constraint: monotonicity required
        if not result.is_monotonic_increasing():
            return 0.0

        # Soft constraint: count segments above 5%
        segments_above_5pct = np.sum(result.segment_proportions > 0.05)
        if segments_above_5pct < min_segments_5pct:
            # Penalize but don't reject
            penalty = 0.5 * (1.0 - segments_above_5pct / min_segments_5pct)
            r2 = r2 * (1.0 - penalty)

        # Soft constraint: max concentration
        max_prop = float(result.segment_proportions.max().item())
        if max_prop > max_proportion:
            # Penalize concentration
            excess = float(max_prop - max_proportion)
            penalty = float(0.3 * excess / (1.0 - max_proportion))  # type: ignore[assignment]
            r2 = r2 * (1.0 - penalty)

        return r2
    except (ValueError, RuntimeError):
        return 0.0
