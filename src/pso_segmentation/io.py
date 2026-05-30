"""Input/Output utilities for segmentation results.

This module provides functionality to export and import segmentation results
to/from various formats (CSV, pickle) for persistence and analysis.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pso_segmentation.optimizer import SegmentationOptimizer
from pso_segmentation.segmentation.metrics import SegmentationResult

# Type aliases for type hints
type NDArray = np.ndarray[Any, np.dtype[np.float64]]
type NDArrayInt = np.ndarray[Any, np.dtype[np.int32]]


def export_segmentation_to_csv(
    cuts: NDArray,
    scores: NDArray,
    labels: NDArray,
    segment_assignments: NDArray | None = None,
    output_dir: str | Path = ".",
) -> dict[str, str]:
    """Export segmentation results to CSV files.

    Saves segmentation cuts, data with assignments, and segment-level metrics
    to CSV files. The metrics file includes ``target_mean``.

    Parameters
    ----------
    cuts : NDArray
        Cut boundaries for segmentation (shape: (n_cuts,))
    scores : NDArray
        Original scores/predictions (shape: (n_samples,))
    labels : NDArray
        Original target values (shape: (n_samples,))
    segment_assignments : NDArray | None, optional
        Segment assignment for each observation (shape: (n_samples,))
        If None, computed from cuts and scores. Default: None
    output_dir : str | Path, optional
        Directory to save CSV files. Default: "."

    Returns
    -------
    dict[str, str]
        Mapping of file description to file path
        Keys: "cuts", "data", "metrics"

    Raises
    ------
    ValueError
        If scores and labels have mismatched lengths

    Examples
    --------
    >>> from pso_segmentation import (
    ...     OptimizerConfig,
    ...     SegmentationOptimizer,
    ...     example_fitness_r2_only,
    ... )
    >>> import numpy as np
    >>> scores = np.random.rand(1000)
    >>> labels = np.random.binomial(1, 0.3, 1000)
    >>> optimizer = SegmentationOptimizer(OptimizerConfig(seed=42))
    >>> optimizer.fit(
    ...     scores,
    ...     labels,
    ...     lambda cuts: example_fitness_r2_only(cuts, scores, labels),
    ... )
    >>> files = export_segmentation_to_csv(
    ...     optimizer.get_cuts(),
    ...     scores,
    ...     labels,
    ...     optimizer.get_segments(),
    ... )
    >>> print(files)
    {'cuts': './cuts.csv', 'data': './segmented_data.csv', ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    cuts = np.asarray(cuts, dtype=np.float64)

    if scores.shape[0] != labels.shape[0]:
        msg = f"Mismatched lengths: scores ({scores.shape[0]}) vs labels ({labels.shape[0]})"
        raise ValueError(msg)

    # Compute segment assignments if not provided
    segment_assignments_typed: NDArrayInt
    if segment_assignments is None:
        segment_assignments_typed = np.digitize(scores, cuts, right=True).astype(np.int32)
    else:
        segment_assignments_typed = np.asarray(segment_assignments, dtype=np.int32)

    # Export cuts
    cuts_df = pd.DataFrame({"cut_index": np.arange(len(cuts)), "cut_value": cuts})
    cuts_path = output_dir / "cuts.csv"
    cuts_df.to_csv(cuts_path, index=False)

    # Export segmented data
    data_df = pd.DataFrame(
        {
            "score": scores,
            "label": labels,
            "segment": segment_assignments_typed,
        }
    )
    data_path = output_dir / "segmented_data.csv"
    data_df.to_csv(data_path, index=False)

    # Compute and export metrics per segment
    n_segments = int(segment_assignments_typed.max()) + 1
    segment_metrics = []
    for seg in range(n_segments):
        mask = segment_assignments_typed == seg
        n_obs_val: int = int(mask.sum())
        if n_obs_val == 0:
            continue
        proportion = n_obs_val / len(scores)
        labels_segment = labels[mask]
        target_mean = float(labels_segment.mean()) if n_obs_val > 0 else 0.0
        min_score = float(scores[mask].min())
        max_score = float(scores[mask].max())

        segment_metrics.append(
            {
                "segment": seg,
                "n_observations": n_obs_val,
                "proportion": float(proportion),
                "target_mean": target_mean,
                "min_score": min_score,
                "max_score": max_score,
            }
        )

    metrics_df = pd.DataFrame(segment_metrics)
    metrics_path = output_dir / "segment_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return {
        "cuts": str(cuts_path),
        "data": str(data_path),
        "metrics": str(metrics_path),
    }


def import_segmentation_from_csv(
    data_csv: str | Path,
    cuts_csv: str | Path | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Import segmentation results from CSV files.

    Loads segmented data and optionally cut boundaries from CSV.

    Parameters
    ----------
    data_csv : str | Path
        Path to CSV file with columns: score, label, segment
    cuts_csv : str | Path | None, optional
        Path to CSV file with cut boundaries. If None, no cuts loaded.
        Default: None

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        (scores, labels, segments, cuts)
        - scores: Original scores (shape: (n_samples,))
        - labels: Original target values (shape: (n_samples,))
        - segments: Segment assignments (shape: (n_samples,))
        - cuts: Cut boundaries (shape: (n_cuts,)) or empty if not provided

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist
    ValueError
        If CSV file missing required columns

    Examples
    --------
    >>> scores, labels, segments, cuts = import_segmentation_from_csv(
    ...     "segmented_data.csv", "cuts.csv")
    >>> print(f"Loaded {len(scores)} observations with {len(cuts)} cuts")
    """
    data_csv = Path(data_csv)
    if not data_csv.exists():
        msg = f"Data CSV not found: {data_csv}"
        raise FileNotFoundError(msg)

    data_df = pd.read_csv(data_csv)
    required_cols = {"score", "label", "segment"}
    if not required_cols.issubset(data_df.columns):
        missing = required_cols - set(data_df.columns)
        msg = f"Missing required columns in data CSV: {missing}"
        raise ValueError(msg)

    scores = data_df["score"].values.astype(np.float64)
    labels = data_df["label"].values.astype(np.float64)
    segments = data_df["segment"].values.astype(np.int32)

    cuts = np.array([], dtype=np.float64)
    if cuts_csv is not None:
        cuts_csv = Path(cuts_csv)
        if cuts_csv.exists():
            cuts_df = pd.read_csv(cuts_csv)
            if "cut_value" in cuts_df.columns:
                cuts = cuts_df["cut_value"].values.astype(np.float64)
            else:
                msg = "Cuts CSV missing 'cut_value' column"
                raise ValueError(msg)

    return scores, labels, segments, cuts


def save_optimizer_state(
    optimizer: SegmentationOptimizer,
    filepath: str | Path,
) -> None:
    """Save optimizer state to pickle file.

    Serializes complete optimizer object including configuration and results
    for later restoration.

    Parameters
    ----------
    optimizer : SegmentationOptimizer
        Optimizer object to serialize
    filepath : str | Path
        Path where pickle file will be saved

    Raises
    ------
    IOError
        If file cannot be written

    Examples
    --------
    >>> optimizer = SegmentationOptimizer()
    >>> # ... fit optimizer ...
    >>> save_optimizer_state(optimizer, "optimizer.pkl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(optimizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        msg = f"Failed to save optimizer state to {filepath}: {e}"
        raise OSError(msg) from e


def load_optimizer_state(filepath: str | Path) -> SegmentationOptimizer:
    """Load optimizer state from pickle file.

    Restores previously serialized optimizer object with all configuration
    and results intact.

    Parameters
    ----------
    filepath : str | Path
        Path to pickle file

    Returns
    -------
    SegmentationOptimizer
        Restored optimizer object

    Raises
    ------
    FileNotFoundError
        If pickle file does not exist
    IOError
        If file cannot be read or deserialized

    Examples
    --------
    >>> optimizer = load_optimizer_state("optimizer.pkl")
    >>> result = optimizer.get_metrics()
    """
    filepath = Path(filepath)
    if not filepath.exists():
        msg = f"Optimizer pickle file not found: {filepath}"
        raise FileNotFoundError(msg)

    try:
        with open(filepath, "rb") as f:
            optimizer_obj: Any = pickle.load(f)
    except Exception as e:
        msg = f"Failed to load optimizer state from {filepath}: {e}"
        raise OSError(msg) from e

    if not isinstance(optimizer_obj, SegmentationOptimizer):
        msg = f"Loaded object is not SegmentationOptimizer, got {type(optimizer_obj)}"
        warnings.warn(msg, stacklevel=2)

    return cast(SegmentationOptimizer, optimizer_obj)


def export_metrics_to_json(
    result: SegmentationResult,
    filepath: str | Path,
) -> str:
    """Export segmentation metrics to JSON file.

    Saves key metrics in human-readable JSON format.

    Parameters
    ----------
    result : SegmentationResult
        Segmentation result with metrics
    filepath : str | Path
        Path where JSON file will be saved

    Returns
    -------
    str
        Path to saved JSON file

    Examples
    --------
    >>> # After segmentation
    >>> export_metrics_to_json(result, "metrics.json")
    """
    import json

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "r2": float(result.r2),
        "n_segments": int(result.n_segments),
        "h_inter": float(result.h_inter),
        "h_intra": float(result.h_intra),
        "segment_proportions": result.segment_proportions.tolist(),
        "target_mean_by_segment": result.target_mean_by_segment.tolist(),
        "segment_sizes": result.segment_sizes.tolist(),
    }

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

    return str(filepath)
