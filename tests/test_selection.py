"""Unit tests for segment-count selection."""

from __future__ import annotations

import numpy as np
import pytest

from pso_segmentation import SegmentSelectionResult, select_n_segments
from pso_segmentation.optimizer import OptimizerConfig
from pso_segmentation.segmentation import compute_metrics


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic segmentation data."""
    scores = np.linspace(0.0, 1.0, 120)
    labels = (scores > 0.5).astype(float)
    return scores, labels


def test_select_n_segments_accepts_inclusive_tuple_range(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """A tuple range should be interpreted as inclusive."""
    scores, labels = simple_data
    config = OptimizerConfig(pop_size=8, max_iter=5, seed=42)

    result = select_n_segments(scores, labels, (2, 4), base_config=config)

    assert isinstance(result, SegmentSelectionResult)
    assert [candidate.n_segments for candidate in result.candidates] == [2, 3, 4]
    assert result.best_n_segments in {2, 3, 4}
    assert result.best_metrics is result.best_candidate.metrics
    assert result.best_optimizer is result.best_candidate.optimizer


def test_select_n_segments_accepts_custom_objective_factory(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The selector should build one objective per candidate segment count."""
    scores, labels = simple_data
    seen_n_segments: list[int] = []

    def objective_factory(
        scores_arr: np.ndarray,
        labels_arr: np.ndarray,
        n_segments: int,
        params: dict[str, float],
    ):
        seen_n_segments.append(n_segments)
        assert params == {}

        def objective(cuts: np.ndarray) -> float:
            return compute_metrics(scores_arr, labels_arr, cuts).r2

        return objective

    config = OptimizerConfig(pop_size=8, max_iter=5, seed=7)
    result = select_n_segments(
        scores,
        labels,
        [2, 4],
        objective_factory=objective_factory,
        base_config=config,
    )

    assert seen_n_segments == [2, 4]
    assert len(result.candidates) == 2


def test_select_n_segments_expands_param_grid(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The selector should evaluate every segment/parameter combination."""
    scores, labels = simple_data
    seen_params: list[dict[str, float]] = []

    def objective_factory(
        scores_arr: np.ndarray,
        labels_arr: np.ndarray,
        _n_segments: int,
        params: dict[str, float],
    ):
        seen_params.append(params)

        def objective(cuts: np.ndarray) -> float:
            base_score = compute_metrics(scores_arr, labels_arr, cuts).r2
            return base_score - params["penalty"]

        return objective

    config = OptimizerConfig(pop_size=8, max_iter=5, seed=13)
    result = select_n_segments(
        scores,
        labels,
        [2, 3],
        objective_factory=objective_factory,
        base_config=config,
        param_grid={"penalty": [0.0, 0.1], "weight": [1.0, 2.0]},
    )

    assert len(result.candidates) == 8
    assert {tuple(sorted(candidate.params.items())) for candidate in result.candidates} == {
        (("penalty", 0.0), ("weight", 1.0)),
        (("penalty", 0.0), ("weight", 2.0)),
        (("penalty", 0.1), ("weight", 1.0)),
        (("penalty", 0.1), ("weight", 2.0)),
    }
    assert len(seen_params) == 8


def test_select_n_segments_rejects_empty_param_grid_values(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Grid values must be non-empty to avoid silent no-op searches."""
    scores, labels = simple_data

    with pytest.raises(ValueError, match="non-empty"):
        select_n_segments(scores, labels, [2], param_grid={"penalty": []})


def test_select_n_segments_can_use_custom_selection_func(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """A custom selection function should control the final best candidate."""
    scores, labels = simple_data
    config = OptimizerConfig(pop_size=8, max_iter=5, seed=11)

    result = select_n_segments(
        scores,
        labels,
        (2, 4),
        base_config=config,
        selection_func=lambda candidate: -float(candidate.n_segments),
    )

    assert result.best_n_segments == 2


def test_select_n_segments_rejects_invalid_range(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Segment counts must be at least two."""
    scores, labels = simple_data

    with pytest.raises(ValueError, match=">= 2"):
        select_n_segments(scores, labels, [1])


def test_select_n_segments_accepts_validation_func(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """A custom validation function should control candidate validity."""
    scores, labels = simple_data
    config = OptimizerConfig(pop_size=8, max_iter=5, seed=42)

    result = select_n_segments(
        scores,
        labels,
        (2, 4),
        base_config=config,
        validation_func=lambda candidate: (
            candidate.n_segments == 3,
            "accepted" if candidate.n_segments == 3 else "only 3 segments accepted",
        ),
    )

    assert result.best_n_segments == 3
    assert [candidate.n_segments for candidate in result.valid_candidates] == [3]
    assert all(candidate.validation_message for candidate in result.candidates)


def test_select_n_segments_requires_valid_candidate_when_validation_func_rejects_all(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The selector should fail clearly when a custom validator rejects every candidate."""
    scores, labels = simple_data
    config = OptimizerConfig(pop_size=8, max_iter=5, seed=42)

    with pytest.raises(RuntimeError, match="No valid segmentation"):
        select_n_segments(
            scores,
            labels,
            (2, 3),
            base_config=config,
            validation_func=lambda _candidate: (False, "rejected"),
        )
