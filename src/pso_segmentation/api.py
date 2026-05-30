"""Simple functional API for PSO-based segmentation.

This module provides a lightweight functional interface for quick segmentation tasks.
For more advanced use cases, see SegmentationOptimizer in the optimizer module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pso_segmentation.optimizer import OptimizerConfig, SegmentationOptimizer
from pso_segmentation.segmentation.metrics import SegmentationResult

# Type alias for NDArray with float64 dtype
NDArray = np.ndarray[Any, np.dtype[np.float64]]


def segment_scores(
    scores: NDArray,
    labels: NDArray,
    objective_func: Callable[[NDArray], float],
    config: OptimizerConfig | None = None,
) -> SegmentationResult:
    """Quick segmentation using PSO optimization.

    Lightweight functional wrapper around SegmentationOptimizer for simple
    segmentation tasks. For advanced configuration and result inspection,
    use SegmentationOptimizer directly.

    Parameters
    ----------
    scores : NDArray
        Array of continuous values to segment (shape: (n_samples,))
        Example: risk scores, probabilities, or any continuous signal
    labels : NDArray
        Target variable aligned with scores (shape: (n_samples,))
        Used to compute metrics (R², segment means, etc.)
    objective_func : Callable[[NDArray], float]
        Fitness function to maximize during optimization
        Input: Cut values (1D array)
        Output: Scalar fitness score (higher is better)
        Examples: example_fitness_r2_only, example_fitness_custom_business_metric
    config : OptimizerConfig | None, optional
        PSO configuration. If None, uses sensible defaults:
        - pop_size=30
        - max_iter=100
        - w, c1, c2: standard PSO parameters
        Default: None

    Returns
    -------
    SegmentationResult
        Segmentation metrics and segment assignments
        Attributes:
        - r2: Variance explained by segmentation
        - n_segments: Number of segments created
        - segment_proportions: Share of population per segment
        - pd_by_segment: Segment mean of the target (PD if target is binary)
        - segment_sizes: Count of observations per segment
        - h_inter, h_intra: Between/within-group heterogeneity

    Raises
    ------
    ValueError
        If scores and labels have mismatched lengths
    RuntimeError
        If PSO optimization fails to converge

    Examples
    --------
    >>> from pso_segmentation import segment_scores, example_fitness_r2_only
    >>> import numpy as np
    >>> scores = np.random.rand(1000)
    >>> labels = np.random.binomial(1, 0.3, 1000)
    >>> result = segment_scores(scores, labels, example_fitness_r2_only)
    >>> print(f"R²: {result.r2:.3f}, Segments: {result.n_segments}")

    >>> # With custom config
    >>> from pso_segmentation import OptimizerConfig
    >>> config = OptimizerConfig(pop_size=100, max_iter=200)
    >>> result = segment_scores(scores, labels, example_fitness_r2_only, config)

    Notes
    -----
    - Use example_fitness_* functions from pso_segmentation.objective_functions_examples
    - For business constraints, use example_fitness_custom_business_metric as a template
    - PSO is stochastic; results vary slightly across runs
    - Larger pop_size and max_iter → better results but slower convergence
    """
    # Validate inputs
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    if scores.shape[0] != labels.shape[0]:
        msg = f"Mismatched lengths: scores ({scores.shape[0]}) vs labels ({labels.shape[0]})"
        raise ValueError(msg)

    # Use default config if not provided
    if config is None:
        config = OptimizerConfig()

    # Run optimization
    optimizer = SegmentationOptimizer(config)
    optimizer.fit(scores, labels, objective_func)

    # Return metrics directly
    return optimizer.get_metrics()
