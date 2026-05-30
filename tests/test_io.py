"""Unit tests for IO module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pso_segmentation.io import (
    export_metrics_to_json,
    export_segmentation_to_csv,
    import_segmentation_from_csv,
    load_optimizer_state,
    save_optimizer_state,
)
from pso_segmentation.optimizer import OptimizerConfig, SegmentationOptimizer
from pso_segmentation.segmentation.metrics import SegmentationResult


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample segmentation data."""
    np.random.seed(42)
    scores = np.linspace(0, 1, 200)
    labels = (scores > 0.5).astype(float)
    cuts = np.array([0.25, 0.5, 0.75])
    segments = np.digitize(scores, cuts, right=True)
    return scores, labels, segments, cuts


@pytest.fixture
def sample_result(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> SegmentationResult:
    """Sample segmentation result."""
    scores, labels, _, _ = sample_data
    # Create a simple result object
    result = SegmentationResult(
        r2=0.65,
        n_segments=4,
        pd_by_segment=np.array([0.1, 0.3, 0.6, 0.9]),
        segment_sizes=np.array([50, 50, 50, 50]),
        segment_proportions=np.array([0.25, 0.25, 0.25, 0.25]),
        h_inter=0.12,
        h_intra=0.08,
    )
    return result


class TestExportSegmentationToCsv:
    """Tests for CSV export functionality."""

    def test_export_creates_three_files(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that export creates cuts, data, and metrics CSV files."""
        scores, labels, segments, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)
            assert "cuts" in result
            assert "data" in result
            assert "metrics" in result
            assert Path(result["cuts"]).exists()
            assert Path(result["data"]).exists()
            assert Path(result["metrics"]).exists()

    def test_export_cuts_csv_format(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test cuts CSV has correct format."""
        scores, labels, segments, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)
            cuts_df = pd.read_csv(result["cuts"])
            assert "cut_index" in cuts_df.columns
            assert "cut_value" in cuts_df.columns
            assert len(cuts_df) == len(cuts)
            assert np.allclose(cuts_df["cut_value"].values, cuts)

    def test_export_data_csv_format(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test data CSV has correct format."""
        scores, labels, segments, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)
            data_df = pd.read_csv(result["data"])
            assert "score" in data_df.columns
            assert "label" in data_df.columns
            assert "segment" in data_df.columns
            assert len(data_df) == len(scores)

    def test_export_metrics_csv_has_expected_columns(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test metrics CSV has expected columns."""
        scores, labels, segments, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)
            metrics_df = pd.read_csv(result["metrics"])
            expected_cols = {
                "segment",
                "n_observations",
                "proportion",
                "target_mean",
                "pd_rate",
                "min_score",
                "max_score",
            }
            assert expected_cols.issubset(metrics_df.columns)

    def test_export_without_segments_auto_computes(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that segments are auto-computed if not provided."""
        scores, labels, _, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_segmentation_to_csv(cuts, scores, labels, None, tmpdir)
            data_df = pd.read_csv(result["data"])
            assert "segment" in data_df.columns
            assert len(data_df) == len(scores)

    def test_export_rejects_mismatched_lengths(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that mismatched score/label lengths raise ValueError."""
        _, _, _, cuts = sample_data
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 1])
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="Mismatched lengths"),
        ):
            export_segmentation_to_csv(cuts, scores, labels, None, tmpdir)


class TestImportSegmentationFromCsv:
    """Tests for CSV import functionality."""

    def test_import_loads_data_correctly(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that import loads data correctly."""
        scores, labels, segments, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)
            data_path = Path(tmpdir) / "segmented_data.csv"
            cuts_path = Path(tmpdir) / "cuts.csv"

            loaded_scores, loaded_labels, loaded_segments, loaded_cuts = (
                import_segmentation_from_csv(data_path, cuts_path)
            )

            assert np.allclose(loaded_scores, scores)
            assert np.allclose(loaded_labels, labels)
            assert np.array_equal(loaded_segments, segments)
            assert np.allclose(loaded_cuts, cuts)

    def test_import_without_cuts(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that import works without cuts CSV."""
        scores, labels, segments, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)
            data_path = Path(tmpdir) / "segmented_data.csv"

            loaded_scores, loaded_labels, loaded_segments, loaded_cuts = (
                import_segmentation_from_csv(data_path)
            )

            assert np.allclose(loaded_scores, scores)
            assert np.allclose(loaded_labels, labels)
            assert np.array_equal(loaded_segments, segments)
            assert len(loaded_cuts) == 0

    def test_import_missing_file_raises_error(self) -> None:
        """Test that importing missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            import_segmentation_from_csv("nonexistent_file.csv")

    def test_import_missing_required_column_raises_error(self) -> None:
        """Test that missing required column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with missing 'segment' column
            bad_csv = Path(tmpdir) / "bad_data.csv"
            bad_df = pd.DataFrame({"score": [0.1, 0.2], "label": [0, 1]})
            bad_df.to_csv(bad_csv, index=False)

            with pytest.raises(ValueError, match="Missing required columns"):
                import_segmentation_from_csv(bad_csv)

    def test_import_export_roundtrip(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test roundtrip: export then import."""
        scores, labels, segments, cuts = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export
            export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)

            # Import
            data_path = Path(tmpdir) / "segmented_data.csv"
            cuts_path = Path(tmpdir) / "cuts.csv"
            loaded_scores, loaded_labels, loaded_segments, loaded_cuts = (
                import_segmentation_from_csv(data_path, cuts_path)
            )

            # Verify
            assert np.allclose(loaded_scores, scores)
            assert np.allclose(loaded_labels, labels)
            assert np.array_equal(loaded_segments, segments)
            assert np.allclose(loaded_cuts, cuts)


class TestSaveLoadOptimizerState:
    """Tests for optimizer serialization."""

    def test_save_creates_file(self) -> None:
        """Test that save_optimizer_state creates pickle file."""
        config = OptimizerConfig(max_iter=10)
        optimizer = SegmentationOptimizer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "optimizer.pkl"
            save_optimizer_state(optimizer, filepath)
            assert filepath.exists()

    def test_load_restores_state(self) -> None:
        """Test that load_optimizer_state restores state correctly."""
        config = OptimizerConfig(pop_size=25, max_iter=15)
        optimizer = SegmentationOptimizer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "optimizer.pkl"
            save_optimizer_state(optimizer, filepath)
            loaded = load_optimizer_state(filepath)

            assert isinstance(loaded, SegmentationOptimizer)
            assert loaded.config.pop_size == 25
            assert loaded.config.max_iter == 15

    def test_load_missing_file_raises_error(self) -> None:
        """Test that loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_optimizer_state("nonexistent.pkl")

    def test_save_load_roundtrip(self) -> None:
        """Test roundtrip: save then load."""
        config = OptimizerConfig(n_segments=5, pop_size=40, max_iter=50)
        optimizer1 = SegmentationOptimizer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "optimizer.pkl"
            save_optimizer_state(optimizer1, filepath)
            optimizer2 = load_optimizer_state(filepath)

            assert optimizer2.config.n_segments == 5
            assert optimizer2.config.pop_size == 40
            assert optimizer2.config.max_iter == 50


class TestExportMetricsToJson:
    """Tests for JSON metrics export."""

    def test_export_creates_json_file(self, sample_result: SegmentationResult) -> None:
        """Test that export creates JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.json"
            export_metrics_to_json(sample_result, filepath)
            assert filepath.exists()

    def test_export_json_has_required_keys(self, sample_result: SegmentationResult) -> None:
        """Test that exported JSON has required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.json"
            export_metrics_to_json(sample_result, filepath)

            with open(filepath) as f:
                data = json.load(f)

            expected_keys = {
                "r2",
                "n_segments",
                "h_inter",
                "h_intra",
                "segment_proportions",
                "pd_by_segment",
                "segment_sizes",
            }
            assert expected_keys.issubset(data.keys())

    def test_export_json_values_correct(self, sample_result: SegmentationResult) -> None:
        """Test that exported JSON values are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.json"
            export_metrics_to_json(sample_result, filepath)

            with open(filepath) as f:
                data = json.load(f)

            assert float(data["r2"]) == sample_result.r2
            assert int(data["n_segments"]) == sample_result.n_segments
            assert np.allclose(
                data["segment_proportions"], sample_result.segment_proportions.tolist()
            )

    def test_export_creates_parent_directories(self, sample_result: SegmentationResult) -> None:
        """Test that export creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir1" / "subdir2" / "metrics.json"
            result_path = export_metrics_to_json(sample_result, filepath)
            assert Path(result_path).exists()
            assert Path(result_path).parent.parent.exists()


class TestIOIntegration:
    """Integration tests combining multiple IO functions."""

    def test_full_export_import_workflow(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test complete export/import workflow."""
        scores, labels, segments, cuts = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export
            files = export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)

            # Import
            data_path = Path(files["data"])
            cuts_path = Path(files["cuts"])
            loaded_scores, loaded_labels, loaded_segments, loaded_cuts = (
                import_segmentation_from_csv(data_path, cuts_path)
            )

            # Verify data integrity
            assert len(loaded_scores) == len(scores)
            assert np.allclose(loaded_scores, scores)
            assert np.allclose(loaded_labels, labels)

    def test_export_then_save_optimizer(
        self, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test exporting data and saving optimizer together."""
        scores, labels, segments, cuts = sample_data
        config = OptimizerConfig(max_iter=10)
        optimizer = SegmentationOptimizer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export data
            export_segmentation_to_csv(cuts, scores, labels, segments, tmpdir)

            # Save optimizer
            optimizer_path = Path(tmpdir) / "optimizer.pkl"
            save_optimizer_state(optimizer, optimizer_path)

            # Verify both exist
            assert (Path(tmpdir) / "segmented_data.csv").exists()
            assert optimizer_path.exists()

            # Load and verify
            loaded_optimizer = load_optimizer_state(optimizer_path)
            assert isinstance(loaded_optimizer, SegmentationOptimizer)
