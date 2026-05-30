"""Segmentation optimizer using PSO.

This module provides the main OO API for PSO-based segmentation optimization.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from pso_segmentation.core import PSO, PSO_Result
from pso_segmentation.segmentation import (
    SegmentationResult,
    compute_metrics,
    get_segment_assignments,
)

# Type alias for type hints
NDArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass
class OptimizerConfig:
    """Configuration for segmentation optimizer.

    Parameters
    ----------
    n_segments : int, default=4
        Number of segments to create (determines dimension of optimization)
    pop_size : int, default=30
        PSO population size
    max_iter : int, default=100
        Maximum PSO iterations
    w : float, default=0.7
        PSO inertia weight
    c1 : float, default=1.5
        PSO cognitive coefficient
    c2 : float, default=1.5
        PSO social coefficient
    track_history : bool, default=True
        Track optimization history
    seed : int | None, default=None
        Random seed for reproducibility
    """

    n_segments: int = 4
    pop_size: int = 30
    max_iter: int = 100
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    track_history: bool = True
    seed: int | None = None


class SegmentationOptimizer:
    """PSO-based segmentation optimizer.

    Main API for optimizing score segmentation using Particle Swarm Optimization.
    Finds optimal cut boundaries to maximize a user-defined objective function.

    Examples
    --------
    >>> optimizer = SegmentationOptimizer(OptimizerConfig(n_segments=4))
    >>> optimizer.fit(scores, labels, fitness_func)
    >>> cuts = optimizer.get_cuts()
    >>> result = optimizer.get_metrics()
    >>> print(f"R²: {result.r2:.3f}")
    """

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        """Initialize optimizer with configuration.

        Parameters
        ----------
        config : OptimizerConfig, optional
            Configuration object. If None, uses default OptimizerConfig()
        """
        self.config = config or OptimizerConfig()
        self._cuts: NDArray | None = None
        self._result: SegmentationResult | None = None
        self._pso_result: PSO_Result | None = None
        self._scores: NDArray | None = None
        self._labels: NDArray | None = None
        self._fitted = False

    def fit(
        self,
        scores: NDArray | list[float],
        labels: NDArray | list[float],
        objective_func: Callable[[NDArray], float],
    ) -> SegmentationOptimizer:
        """Fit segmentation optimizer to data.

        Runs PSO to find optimal cut boundaries that maximize the fitness function.

        Parameters
        ----------
        scores : NDArray or list[float]
            Continuous scores to segment (shape: (n_samples,))
        labels : NDArray or list[float]
            Target variable (binary or continuous) (shape: (n_samples,))
        objective_func : callable
            Fitness function with signature: func(cuts: NDArray) -> float
            Should return a value to MAXIMIZE (higher is better).
            Function receives k-1 cuts for k segments.

        Returns
        -------
        SegmentationOptimizer
            Returns self for method chaining

        Raises
        ------
        ValueError
            If scores and labels have different lengths
        """
        # Convert to numpy arrays
        self._scores = np.asarray(scores, dtype=np.float64)
        self._labels = np.asarray(labels, dtype=np.float64)

        if len(self._scores) != len(self._labels):
            msg = (
                f"scores and labels must have same length, "
                f"got {len(self._scores)} and {len(self._labels)}"
            )
            raise ValueError(msg)

        # Define bounds for PSO (cuts must be within score range)
        score_min = float(np.min(self._scores))
        score_max = float(np.max(self._scores))
        lb = np.full(self.config.n_segments - 1, score_min + 1e-6)
        ub = np.full(self.config.n_segments - 1, score_max - 1e-6)

        # Run PSO optimization
        pso = PSO(
            objective_func=objective_func,
            n_dim=self.config.n_segments - 1,
            pop_size=self.config.pop_size,
            max_iter=self.config.max_iter,
            lb=lb,
            ub=ub,
            w=self.config.w,
            c1=self.config.c1,
            c2=self.config.c2,
            seed=self.config.seed,
            track_history=self.config.track_history,
        )
        self._pso_result = pso.run()

        # Extract and sort cuts
        self._cuts = np.sort(self._pso_result.best_position)

        # Compute final segmentation metrics
        self._result = compute_metrics(self._scores, self._labels, self._cuts)

        self._fitted = True
        return self

    def get_cuts(self) -> NDArray:
        """Get optimal cut boundaries.

        Returns
        -------
        NDArray
            Sorted cut boundaries (shape: (n_segments-1,))

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self._fitted or self._cuts is None:
            msg = "Must call fit() before get_cuts()"
            raise RuntimeError(msg)
        return self._cuts.copy()

    def get_segments(self) -> NDArray:
        """Get segment assignments for each score.

        Returns
        -------
        NDArray
            Segment assignment for each score (0-indexed) (shape: (n_samples,))

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self._fitted or self._cuts is None or self._scores is None:
            msg = "Must call fit() before get_segments()"
            raise RuntimeError(msg)
        return get_segment_assignments(self._scores, self._cuts)

    def get_metrics(self) -> SegmentationResult:
        """Get segmentation metrics.

        Returns
        -------
        SegmentationResult
            Comprehensive segmentation metrics including R², target means, sizes, etc.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self._fitted or self._result is None:
            msg = "Must call fit() before get_metrics()"
            raise RuntimeError(msg)
        return self._result

    def get_history(self) -> list[dict[str, Any]]:
        """Get optimization history from the underlying PSO run.

        Returns
        -------
        list[dict[str, Any]]
            Iteration history with best fitness and position snapshots.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self._fitted or self._pso_result is None:
            msg = "Must call fit() before get_history()"
            raise RuntimeError(msg)
        return self._pso_result.history.copy()

    def summary(self) -> str:
        """Get human-readable summary of results.

        Returns
        -------
        str
            Formatted summary including R², segments, and target means by segment.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self._fitted:
            msg = "Must call fit() before summary()"
            raise RuntimeError(msg)

        metrics = self.get_metrics()
        cuts = self.get_cuts()

        # Format summary
        lines = [
            "=" * 60,
            "SEGMENTATION OPTIMIZER RESULTS",
            "=" * 60,
            "Target mean = segment average of labels (PD for binary targets)",
            f"R² (Variance Explained): {metrics.r2:.4f}",
            f"Number of Segments: {metrics.n_segments}",
            "",
            "Cut Boundaries:",
        ]

        for i, cut in enumerate(cuts):
            lines.append(f"  Cut {i + 1}: {cut:.6f}")

        lines.extend(
            [
                "",
                "Segment Statistics:",
            ]
        )

        for i in range(metrics.n_segments):
            segment_pct = metrics.segment_proportions[i] * 100
            target_pct = metrics.pd_by_segment[i] * 100
            lines.append(
                f"  Segment {i}: Target mean={target_pct:.2f}%, "
                f"Size={segment_pct:.2f}% (n={int(metrics.segment_sizes[i])})"
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_json(self, filepath: str | None = None) -> str:
        """Serialize optimizer state to JSON.

        Parameters
        ----------
        filepath : str, optional
            Path to save JSON file. If None, returns JSON string.

        Returns
        -------
        str
            JSON string representation

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self._fitted:
            msg = "Must call fit() before to_json()"
            raise RuntimeError(msg)

        # Prepare data for JSON serialization
        data = {
            "config": asdict(self.config),
            "cuts": self._cuts.tolist() if self._cuts is not None else None,
            "r2": float(self._result.r2) if self._result else None,
            "n_segments": self._result.n_segments if self._result else None,
            "pd_by_segment": (self._result.pd_by_segment.tolist() if self._result else None),
            "target_mean_by_segment": (
                self._result.pd_by_segment.tolist() if self._result else None
            ),
            "segment_sizes": (self._result.segment_sizes.tolist() if self._result else None),
            "segment_proportions": (
                self._result.segment_proportions.tolist() if self._result else None
            ),
            "h_inter": float(self._result.h_inter) if self._result else None,
            "h_intra": float(self._result.h_intra) if self._result else None,
        }

        json_str = json.dumps(data, indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(
        cls, filepath: str | None = None, json_str: str | None = None
    ) -> SegmentationOptimizer:
        """Deserialize optimizer state from JSON.

        Parameters
        ----------
        filepath : str, optional
            Path to JSON file to load
        json_str : str, optional
            JSON string to deserialize

        Returns
        -------
        SegmentationOptimizer
            Optimizer instance with loaded state

        Raises
        ------
        ValueError
            If neither filepath nor json_str provided, or both provided
        """
        if (filepath is None and json_str is None) or (
            filepath is not None and json_str is not None
        ):
            msg = "Provide either filepath or json_str, not both"
            raise ValueError(msg)

        # Load JSON
        if filepath:
            with open(filepath) as f:
                data = json.load(f)
        else:
            assert json_str is not None  # Guaranteed by check above
            data = json.loads(json_str)

        # Reconstruct config
        config_keys = {field.name for field in fields(OptimizerConfig)}
        config_dict = {
            key: value for key, value in data.get("config", {}).items() if key in config_keys
        }
        config = OptimizerConfig(**config_dict)

        # Create optimizer instance
        optimizer = cls(config)

        # Restore state
        optimizer._cuts = np.array(data["cuts"], dtype=np.float64) if data.get("cuts") else None
        optimizer._fitted = optimizer._cuts is not None

        # Restore segmentation result if present
        if (
            all(k in data for k in ["r2", "n_segments", "pd_by_segment"])
            and data.get("r2") is not None
        ):
            optimizer._result = SegmentationResult(
                r2=data["r2"],
                n_segments=data["n_segments"],
                pd_by_segment=np.array(data["pd_by_segment"], dtype=np.float64),
                segment_sizes=np.array(data["segment_sizes"], dtype=np.float64),
                segment_proportions=np.array(data["segment_proportions"], dtype=np.float64),
                h_inter=data["h_inter"],
                h_intra=data["h_intra"],
            )

        return optimizer
