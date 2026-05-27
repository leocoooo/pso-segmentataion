"""Model selection utilities for PSO segmentation.

This module provides a thin orchestration layer on top of
SegmentationOptimizer. It selects the number of segments by fitting one
optimizer per candidate segment count, then choosing the best valid result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Literal

import numpy as np

from pso_segmentation.objective_functions_examples import example_fitness_r2_only
from pso_segmentation.optimizer import OptimizerConfig, SegmentationOptimizer
from pso_segmentation.segmentation import SegmentationResult, validate_segmentation

NDArray = np.ndarray[Any, np.dtype[np.float64]]
ObjectiveFunc = Callable[[NDArray], float]
ParamSet = dict[str, Any]
ParamGrid = Mapping[str, Iterable[Any]]
ObjectiveFactory = Callable[[NDArray, NDArray, int, ParamSet], ObjectiveFunc]
SelectionMetric = Literal["r2", "gini", "ks"]


@dataclass
class SegmentCandidate:
    """Result for one candidate number of segments."""

    n_segments: int
    params: ParamSet
    optimizer: SegmentationOptimizer
    metrics: SegmentationResult
    cuts: NDArray
    selection_score: float
    valid: bool
    validation_message: str


@dataclass
class SegmentSelectionResult:
    """Result of selecting the number of segments over a candidate range."""

    best_candidate: SegmentCandidate
    candidates: list[SegmentCandidate]

    @property
    def best_n_segments(self) -> int:
        """Number of segments selected as best."""
        return self.best_candidate.n_segments

    @property
    def best_optimizer(self) -> SegmentationOptimizer:
        """Fitted optimizer for the best candidate."""
        return self.best_candidate.optimizer

    @property
    def best_metrics(self) -> SegmentationResult:
        """Metrics for the best candidate."""
        return self.best_candidate.metrics

    @property
    def valid_candidates(self) -> list[SegmentCandidate]:
        """Candidates that satisfy the configured validation constraints."""
        return [candidate for candidate in self.candidates if candidate.valid]


def _default_objective_factory(
    scores: NDArray,
    labels: NDArray,
    _n_segments: int,
    _params: ParamSet,
) -> ObjectiveFunc:
    """Build the default R2-only objective."""

    def objective(cuts: NDArray) -> float:
        return example_fitness_r2_only(cuts, scores, labels)

    return objective


def _normalize_segment_range(segment_range: Iterable[int] | tuple[int, int]) -> list[int]:
    """Normalize user input to a sorted list of unique segment counts."""
    if isinstance(segment_range, tuple) and len(segment_range) == 2:
        start, stop = segment_range
        values = list(range(start, stop + 1))
    else:
        values = list(segment_range)

    normalized = sorted({int(value) for value in values})
    if not normalized:
        msg = "segment_range must contain at least one segment count"
        raise ValueError(msg)
    if normalized[0] < 2:
        msg = "segment_range values must be >= 2"
        raise ValueError(msg)
    return normalized


def _expand_param_grid(param_grid: ParamGrid | None) -> list[ParamSet]:
    """Expand a parameter grid into concrete parameter combinations."""
    if param_grid is None:
        return [{}]
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values_by_key = [list(param_grid[key]) for key in keys]
    empty_keys = [key for key, values in zip(keys, values_by_key, strict=True) if not values]
    if empty_keys:
        msg = f"param_grid values must be non-empty, got empty values for {empty_keys}"
        raise ValueError(msg)

    return [dict(zip(keys, values, strict=True)) for values in product(*values_by_key)]


def _metric_score(metrics: SegmentationResult, metric: SelectionMetric) -> float:
    """Extract a scalar selection score from segmentation metrics."""
    return float(getattr(metrics, metric))


def select_n_segments(
    scores: NDArray | list[float],
    labels: NDArray | list[float],
    segment_range: Iterable[int] | tuple[int, int],
    objective_factory: ObjectiveFactory | None = None,
    base_config: OptimizerConfig | None = None,
    param_grid: ParamGrid | None = None,
    selection_metric: SelectionMetric = "r2",
    selection_func: Callable[[SegmentCandidate], float] | None = None,
    require_valid: bool = True,
) -> SegmentSelectionResult:
    """Select the best number of segments over a candidate range.

    Parameters
    ----------
    scores : NDArray or list[float]
        Continuous scores to segment.
    labels : NDArray or list[float]
        Target labels aligned with scores.
    segment_range : iterable[int] or tuple[int, int]
        Candidate segment counts. A tuple ``(3, 7)`` is interpreted as the
        inclusive range ``3, 4, 5, 6, 7``.
    objective_factory : callable, optional
        Factory with signature ``factory(scores, labels, n_segments, params) -> objective``.
        The returned objective must have signature ``objective(cuts) -> float``.
        If omitted, the selector maximizes R2 only.
    base_config : OptimizerConfig, optional
        Base PSO configuration. ``n_segments`` is overwritten for each candidate.
    param_grid : mapping[str, iterable], optional
        Optional grid of objective parameters. Each combination is passed to
        ``objective_factory`` as ``params``. If omitted, a single empty parameter
        set is used.
    selection_metric : {"r2", "gini", "ks"}, default="r2"
        Metric used to select the best candidate when ``selection_func`` is not
        provided.
    selection_func : callable, optional
        Custom final scoring function with signature ``selection_func(candidate)``.
        Use this for business-level model selection criteria.
    require_valid : bool, default=True
        If True, only candidates that pass ``validate_segmentation`` can be
        selected. If no candidate is valid, raises RuntimeError.

    Returns
    -------
    SegmentSelectionResult
        Best candidate plus all fitted candidates.
    """
    scores_arr: NDArray = np.asarray(scores, dtype=np.float64)
    labels_arr: NDArray = np.asarray(labels, dtype=np.float64)

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

    segment_counts = _normalize_segment_range(segment_range)
    param_sets = _expand_param_grid(param_grid)
    config_template = base_config or OptimizerConfig()
    factory = objective_factory or _default_objective_factory

    candidates: list[SegmentCandidate] = []
    for n_segments in segment_counts:
        for params in param_sets:
            config = replace(config_template, n_segments=n_segments)
            optimizer = SegmentationOptimizer(config)
            objective = factory(scores_arr, labels_arr, n_segments, params.copy())
            optimizer.fit(scores_arr, labels_arr, objective)

            metrics = optimizer.get_metrics()
            cuts = optimizer.get_cuts()
            valid, validation_message = validate_segmentation(
                metrics,
                min_segment_size=config.min_segment_size,
                max_segment_size=config.max_segment_size,
                monotonic=config.enforce_monotonic,
            )
            candidate = SegmentCandidate(
                n_segments=n_segments,
                params=params.copy(),
                optimizer=optimizer,
                metrics=metrics,
                cuts=cuts,
                selection_score=0.0,
                valid=valid,
                validation_message=validation_message,
            )
            if selection_func is None:
                candidate.selection_score = _metric_score(metrics, selection_metric)
            else:
                candidate.selection_score = float(selection_func(candidate))
            candidates.append(candidate)

    selectable = (
        [candidate for candidate in candidates if candidate.valid] if require_valid else candidates
    )
    if not selectable:
        msg = "No valid segmentation found for the provided segment_range"
        raise RuntimeError(msg)

    best_candidate = max(selectable, key=lambda candidate: candidate.selection_score)
    return SegmentSelectionResult(best_candidate=best_candidate, candidates=candidates)
