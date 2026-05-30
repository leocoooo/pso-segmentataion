"""Objective function builders for PSO segmentation.

This module provides the standard way to build objective functions accepted by
``SegmentationOptimizer`` and ``segment_scores``. Users can combine a base
metric with built-in or custom penalties.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pso_segmentation.segmentation import SegmentationResult, compute_metrics

NDArray = np.ndarray[Any, np.dtype[np.float64]]
MetricName = Literal["r2", "gini", "ks", "h_inter", "h_intra"]
ObjectiveFunc = Callable[[NDArray], float]
MetricFunc = Callable[["ObjectiveContext"], float]
PenaltyFunc = Callable[["ObjectiveContext"], float]


@dataclass(frozen=True)
class ObjectiveContext:
    """Data available to custom objective metrics and penalties."""

    cuts: NDArray
    scores: NDArray
    labels: NDArray
    result: SegmentationResult


def _metric_value(metric: MetricName | MetricFunc, context: ObjectiveContext) -> float:
    """Extract the base metric value from an objective context."""
    if callable(metric):
        return float(metric(context))
    return float(getattr(context.result, metric))


def make_objective(
    scores: NDArray | list[float],
    labels: NDArray | list[float],
    metric: MetricName | MetricFunc = "r2",
    penalties: Iterable[PenaltyFunc] | None = None,
    invalid_score: float = 0.0,
) -> ObjectiveFunc:
    """Build an objective function with optional penalties.

    Parameters
    ----------
    scores : NDArray or list[float]
        Continuous values to segment.
    labels : NDArray or list[float]
        Target values aligned with scores.
    metric : MetricName or callable, default="r2"
        Base metric to maximize. Built-in names are ``"r2"``, ``"gini"``,
        ``"ks"``, ``"h_inter"``, and ``"h_intra"``. A callable receives an
        ``ObjectiveContext`` and returns a scalar score.
    penalties : iterable of callable, optional
        Penalty functions. Each receives an ``ObjectiveContext`` and returns a
        non-negative amount subtracted from the base metric.
    invalid_score : float, default=0.0
        Score returned when metrics cannot be computed or the final score is
        not finite.

    Returns
    -------
    ObjectiveFunc
        Function with signature ``objective(cuts) -> float``.
    """
    scores_arr: NDArray = np.asarray(scores, dtype=np.float64)
    labels_arr: NDArray = np.asarray(labels, dtype=np.float64)
    penalty_funcs = list(penalties or [])

    def objective(cuts: NDArray) -> float:
        cuts_arr: NDArray = np.asarray(cuts, dtype=np.float64).flatten()
        try:
            result = compute_metrics(scores_arr, labels_arr, cuts_arr)
            context = ObjectiveContext(
                cuts=cuts_arr,
                scores=scores_arr,
                labels=labels_arr,
                result=result,
            )
            score = _metric_value(metric, context)
            penalty = sum(float(penalty_func(context)) for penalty_func in penalty_funcs)
            objective_value = float(score - penalty)
        except (ValueError, RuntimeError, FloatingPointError):
            return float(invalid_score)

        if not np.isfinite(objective_value):
            return float(invalid_score)
        return objective_value

    return objective


def monotonic_penalty(
    weight: float,
    direction: Literal["increasing", "decreasing"] = "increasing",
    tolerance: float = 0.0,
) -> PenaltyFunc:
    """Create a penalty for non-monotonic segment target means.

    The returned penalty is proportional to the total monotonicity violation.
    ``weight`` controls how strongly the violation is penalized.
    """
    if direction not in {"increasing", "decreasing"}:
        msg = "direction must be either 'increasing' or 'decreasing'"
        raise ValueError(msg)

    def penalty(context: ObjectiveContext) -> float:
        target_means = context.result.target_mean_by_segment
        if len(target_means) <= 1:
            return 0.0

        diffs = np.diff(target_means)
        if direction == "increasing":
            violations = np.maximum(-(diffs + tolerance), 0.0)
        else:
            violations = np.maximum(diffs - tolerance, 0.0)
        return float(weight * np.sum(violations))

    return penalty


def segment_size_penalty(
    weight: float,
    min_size: float | None = None,
    max_size: float | None = None,
) -> PenaltyFunc:
    """Create a penalty for segments outside size bounds."""

    def penalty(context: ObjectiveContext) -> float:
        proportions = context.result.segment_proportions
        penalty_value = 0.0
        if min_size is not None:
            penalty_value += float(np.sum(np.maximum(min_size - proportions, 0.0)))
        if max_size is not None:
            penalty_value += float(np.sum(np.maximum(proportions - max_size, 0.0)))
        return float(weight * penalty_value)

    return penalty


def empty_segment_penalty(weight: float = 1.0) -> PenaltyFunc:
    """Create a penalty for cuts that produce fewer segments than requested."""

    def penalty(context: ObjectiveContext) -> float:
        expected_segments = len(context.cuts) + 1
        missing_segments = max(0, expected_segments - context.result.n_segments)
        return float(weight * missing_segments)

    return penalty
