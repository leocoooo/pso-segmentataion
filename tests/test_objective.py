"""Unit tests for objective function builders."""

from __future__ import annotations

import numpy as np
import pytest

from pso_segmentation import (
    ObjectiveContext,
    OptimizerConfig,
    SegmentationOptimizer,
    empty_segment_penalty,
    make_objective,
    monotonic_penalty,
    segment_size_penalty,
    select_n_segments,
)
from pso_segmentation.segmentation import compute_metrics


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic segmentation data."""
    scores = np.linspace(0.0, 1.0, 120)
    labels = (scores > 0.5).astype(float)
    return scores, labels


def test_make_objective_matches_named_metric(simple_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The default objective should maximize the selected metric."""
    scores, labels = simple_data
    cuts = np.array([0.5])
    objective = make_objective(scores, labels, metric="r2")

    assert objective(cuts) == pytest.approx(compute_metrics(scores, labels, cuts).r2)


def test_make_objective_accepts_custom_metric(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Users should be able to provide a custom base metric."""
    scores, labels = simple_data
    cuts = np.array([0.5])
    objective = make_objective(
        scores,
        labels,
        metric=lambda context: context.result.r2 + 0.25,
    )

    assert objective(cuts) == pytest.approx(compute_metrics(scores, labels, cuts).r2 + 0.25)


def test_make_objective_returns_invalid_score_on_invalid_cuts(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Invalid cuts should return the configured fallback score."""
    scores, labels = simple_data
    objective = make_objective(scores, labels, invalid_score=-1.0)

    assert objective(np.array([])) == -1.0


def test_make_objective_accepts_custom_penalty(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Users should be able to plug in arbitrary penalties."""
    scores, labels = simple_data
    cuts = np.array([0.5])

    def custom_penalty(context: ObjectiveContext) -> float:
        assert context.cuts.shape == (1,)
        return 0.2

    objective = make_objective(scores, labels, penalties=[custom_penalty])
    expected = compute_metrics(scores, labels, cuts).r2 - 0.2

    assert objective(cuts) == pytest.approx(expected)


def test_monotonic_penalty_uses_weight() -> None:
    """Increasing the weight should increase the monotonicity penalty."""
    scores = np.arange(6, dtype=float)
    labels = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    cuts = np.array([1.5, 3.5])

    no_penalty = make_objective(scores, labels, penalties=[])
    light_penalty = make_objective(scores, labels, penalties=[monotonic_penalty(weight=0.1)])
    heavy_penalty = make_objective(scores, labels, penalties=[monotonic_penalty(weight=0.5)])

    assert light_penalty(cuts) < no_penalty(cuts)
    assert heavy_penalty(cuts) < light_penalty(cuts)


def test_monotonic_penalty_supports_decreasing_direction() -> None:
    """The monotonicity direction should be configurable."""
    scores = np.arange(6, dtype=float)
    labels = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    cuts = np.array([1.5, 3.5])

    increasing = make_objective(scores, labels, penalties=[monotonic_penalty(0.5)])
    decreasing = make_objective(
        scores,
        labels,
        penalties=[monotonic_penalty(0.5, direction="decreasing")],
    )

    assert decreasing(cuts) > increasing(cuts)


def test_monotonic_penalty_rejects_invalid_direction() -> None:
    """Invalid monotonicity directions should fail early."""
    with pytest.raises(ValueError, match="direction"):
        monotonic_penalty(0.5, direction="flat")  # type: ignore[arg-type]


def test_segment_size_penalty_uses_bounds_and_weight(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Segment size penalties should be controlled by min/max bounds and weight."""
    scores, labels = simple_data
    cuts = np.array([0.9])

    no_penalty = make_objective(scores, labels)
    penalized = make_objective(
        scores,
        labels,
        penalties=[segment_size_penalty(weight=1.0, min_size=0.2, max_size=0.8)],
    )

    assert penalized(cuts) < no_penalty(cuts)


def test_empty_segment_penalty_penalizes_missing_segments() -> None:
    """Duplicate cuts should be penalized when they reduce the realized segment count."""
    scores = np.linspace(0.0, 1.0, 20)
    labels = (scores > 0.5).astype(float)
    cuts = np.array([0.5, 0.5])

    no_penalty = make_objective(scores, labels)
    penalized = make_objective(scores, labels, penalties=[empty_segment_penalty(weight=0.5)])

    assert penalized(cuts) < no_penalty(cuts)


def test_objective_can_be_used_with_optimizer(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The built objective should match SegmentationOptimizer.fit."""
    scores, labels = simple_data
    objective = make_objective(scores, labels, penalties=[segment_size_penalty(0.1)])
    optimizer = SegmentationOptimizer(OptimizerConfig(n_segments=3, pop_size=8, max_iter=5, seed=1))

    optimizer.fit(scores, labels, objective)

    assert optimizer.get_metrics().n_segments >= 2


def test_objective_can_be_used_with_select_n_segments_param_grid(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Penalty weights should be easy to tune through select_n_segments grids."""
    scores, labels = simple_data

    def objective_factory(
        scores_arr: np.ndarray,
        labels_arr: np.ndarray,
        _n_segments: int,
        params: dict[str, float],
    ):
        return make_objective(
            scores_arr,
            labels_arr,
            penalties=[
                monotonic_penalty(weight=params["monotonic_weight"]),
                segment_size_penalty(
                    weight=params["size_weight"],
                    min_size=params["min_size"],
                    max_size=params["max_size"],
                ),
            ],
        )

    result = select_n_segments(
        scores,
        labels,
        (2, 3),
        base_config=OptimizerConfig(pop_size=8, max_iter=5, seed=2),
        objective_factory=objective_factory,
        param_grid={
            "monotonic_weight": [0.0, 0.2],
            "size_weight": [0.0, 0.1],
            "min_size": [0.05],
            "max_size": [0.9],
        },
    )

    assert len(result.candidates) == 8
    assert result.best_candidate.params["monotonic_weight"] in {0.0, 0.2}
