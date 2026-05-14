"""Unit tests for optimizer module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pso_segmentation.optimizer import OptimizerConfig, SegmentationOptimizer


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Create simple test data."""
    np.random.seed(42)
    scores = np.linspace(0, 100, 100)
    labels = np.array([1.0] * 30 + [0.5] * 40 + [0.0] * 30)
    return scores, labels


@pytest.fixture
def simple_objective_func(scores: np.ndarray, labels: np.ndarray) -> callable:
    """Create a simple fitness function."""

    def fitness(cuts: np.ndarray) -> float:
        from pso_segmentation.segmentation import compute_metrics

        result = compute_metrics(scores, labels, cuts)
        return result.r2

    return fitness


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OptimizerConfig()

        assert config.n_segments == 4
        assert config.pop_size == 30
        assert config.max_iter == 100
        assert config.w == 0.7
        assert config.c1 == 1.5
        assert config.c2 == 1.5
        assert config.min_segment_size == 0.05
        assert config.max_segment_size == 0.30
        assert config.enforce_monotonic is True
        assert config.track_history is True
        assert config.seed is None

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = OptimizerConfig(
            n_segments=5,
            pop_size=50,
            max_iter=200,
            w=0.8,
            seed=123,
        )

        assert config.n_segments == 5
        assert config.pop_size == 50
        assert config.max_iter == 200
        assert config.w == 0.8
        assert config.seed == 123

    def test_config_negative_values(self) -> None:
        """Test config with invalid negative values."""
        # Note: OptimizerConfig doesn't validate, so this just tests assignment
        config = OptimizerConfig(n_segments=-1)
        assert config.n_segments == -1


class TestSegmentationOptimizer:
    """Test SegmentationOptimizer class."""

    def test_initialization(self) -> None:
        """Test optimizer initialization."""
        optimizer = SegmentationOptimizer()

        assert optimizer.config.n_segments == 4
        assert not optimizer._fitted

    def test_initialization_with_config(self) -> None:
        """Test optimizer initialization with custom config."""
        config = OptimizerConfig(n_segments=5, pop_size=20)
        optimizer = SegmentationOptimizer(config)

        assert optimizer.config.n_segments == 5
        assert optimizer.config.pop_size == 20

    def test_fit_basic(self, simple_data: tuple) -> None:
        """Test basic fit operation."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)
        optimizer.fit(scores, labels, objective)

        assert optimizer._fitted is True
        assert optimizer._cuts is not None
        assert len(optimizer._cuts) == 2

    def test_fit_mismatched_lengths(self, simple_data: tuple) -> None:
        """Test fit with mismatched scores/labels."""
        scores, labels = simple_data
        labels_wrong = labels[:-10]

        def objective(_: np.ndarray) -> float:
            return 0.5

        optimizer = SegmentationOptimizer()

        with pytest.raises(ValueError, match="same length"):
            optimizer.fit(scores, labels_wrong, objective)

    def test_get_cuts_before_fit(self) -> None:
        """Test get_cuts raises error before fit."""
        optimizer = SegmentationOptimizer()

        with pytest.raises(RuntimeError, match="Must call fit"):
            optimizer.get_cuts()

    def test_get_cuts_after_fit(self, simple_data: tuple) -> None:
        """Test get_cuts returns correct cuts."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)
        optimizer.fit(scores, labels, objective)

        cuts = optimizer.get_cuts()

        assert len(cuts) == 2
        assert np.all(cuts[:-1] <= cuts[1:])  # Sorted

    def test_get_segments_before_fit(self) -> None:
        """Test get_segments raises error before fit."""
        optimizer = SegmentationOptimizer()

        with pytest.raises(RuntimeError, match="Must call fit"):
            optimizer.get_segments()

    def test_get_segments_after_fit(self, simple_data: tuple) -> None:
        """Test get_segments returns assignments."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)
        optimizer.fit(scores, labels, objective)

        segments = optimizer.get_segments()

        assert len(segments) == len(scores)
        assert np.all(segments >= 0)

    def test_get_metrics_before_fit(self) -> None:
        """Test get_metrics raises error before fit."""
        optimizer = SegmentationOptimizer()

        with pytest.raises(RuntimeError, match="Must call fit"):
            optimizer.get_metrics()

    def test_get_metrics_after_fit(self, simple_data: tuple) -> None:
        """Test get_metrics returns segmentation result."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)
        optimizer.fit(scores, labels, objective)

        metrics = optimizer.get_metrics()

        assert metrics.r2 >= 0.0
        assert metrics.r2 <= 1.0
        assert metrics.n_segments == 3

    def test_summary_before_fit(self) -> None:
        """Test summary raises error before fit."""
        optimizer = SegmentationOptimizer()

        with pytest.raises(RuntimeError, match="Must call fit"):
            optimizer.summary()

    def test_summary_after_fit(self, simple_data: tuple) -> None:
        """Test summary returns formatted string."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)
        optimizer.fit(scores, labels, objective)

        summary = optimizer.summary()

        assert isinstance(summary, str)
        assert "R²" in summary
        assert "Segment" in summary
        assert "Cut" in summary

    def test_to_json_before_fit(self) -> None:
        """Test to_json raises error before fit."""
        optimizer = SegmentationOptimizer()

        with pytest.raises(RuntimeError, match="Must call fit"):
            optimizer.to_json()

    def test_to_json_returns_string(self, simple_data: tuple) -> None:
        """Test to_json returns valid JSON string."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)
        optimizer.fit(scores, labels, objective)

        json_str = optimizer.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "config" in data
        assert "cuts" in data
        assert "r2" in data

    def test_to_json_saves_file(self, simple_data: tuple) -> None:
        """Test to_json saves to file."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)
        optimizer.fit(scores, labels, objective)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "optimizer.json"
            optimizer.to_json(str(filepath))

            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert "config" in data

    def test_from_json_string(self, simple_data: tuple) -> None:
        """Test from_json with JSON string."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer1 = SegmentationOptimizer(config)
        optimizer1.fit(scores, labels, objective)

        json_str = optimizer1.to_json()
        optimizer2 = SegmentationOptimizer.from_json(json_str=json_str)

        assert optimizer2._fitted is True
        np.testing.assert_array_almost_equal(optimizer1.get_cuts(), optimizer2.get_cuts())

    def test_from_json_file(self, simple_data: tuple) -> None:
        """Test from_json with file path."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer1 = SegmentationOptimizer(config)
        optimizer1.fit(scores, labels, objective)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "optimizer.json"
            optimizer1.to_json(str(filepath))
            optimizer2 = SegmentationOptimizer.from_json(str(filepath))

        assert optimizer2._fitted is True
        np.testing.assert_array_almost_equal(optimizer1.get_cuts(), optimizer2.get_cuts())

    def test_from_json_both_args_raises(self) -> None:
        """Test from_json raises with both args provided."""
        with pytest.raises(ValueError, match="Provide either"):
            SegmentationOptimizer.from_json(filepath="test.json", json_str='{"test": "value"}')

    def test_from_json_no_args_raises(self) -> None:
        """Test from_json raises with no args."""
        with pytest.raises(ValueError, match="Provide either"):
            SegmentationOptimizer.from_json()

    def test_method_chaining(self, simple_data: tuple) -> None:
        """Test fit returns self for method chaining."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        optimizer = SegmentationOptimizer(OptimizerConfig(n_segments=3, max_iter=10))

        result = optimizer.fit(scores, labels, objective)

        assert result is optimizer

    def test_get_cuts_returns_copy(self, simple_data: tuple) -> None:
        """Test get_cuts returns a copy, not reference."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        optimizer = SegmentationOptimizer(OptimizerConfig(n_segments=3, max_iter=10))
        optimizer.fit(scores, labels, objective)

        cuts1 = optimizer.get_cuts()
        cuts2 = optimizer.get_cuts()

        # Modify one copy
        cuts1[0] = 999

        # Verify other is unchanged
        assert cuts2[0] != 999

    def test_fit_with_list_inputs(self, simple_data: tuple) -> None:
        """Test fit with list inputs instead of arrays."""
        scores, labels = simple_data

        def objective(cuts: np.ndarray) -> float:
            from pso_segmentation.segmentation import compute_metrics

            result = compute_metrics(scores, labels, cuts)
            return result.r2

        config = OptimizerConfig(n_segments=3, max_iter=10)
        optimizer = SegmentationOptimizer(config)

        # Convert to lists
        optimizer.fit(scores.tolist(), labels.tolist(), objective)

        assert optimizer._fitted is True
        assert optimizer._cuts is not None

    def test_config_stored_correctly(self) -> None:
        """Test that config is stored correctly."""
        config = OptimizerConfig(
            n_segments=5,
            pop_size=40,
            min_segment_size=0.1,
        )
        optimizer = SegmentationOptimizer(config)

        assert optimizer.config.n_segments == 5
        assert optimizer.config.pop_size == 40
        assert optimizer.config.min_segment_size == 0.1
