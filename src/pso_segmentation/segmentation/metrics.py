"""Segmentation metrics and data structures.

This module defines the data structures and constants for segmentation results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Type alias
NDArray = np.ndarray[Any, np.dtype[np.float64]]

# Default constants for segmentation
DEFAULT_MIN_SEGMENT_SIZE: float = 0.05  # 5% of population minimum per segment
DEFAULT_MAX_SEGMENT_SIZE: float = 0.30  # 30% of population maximum per segment
EPSILON: float = 1e-10  # Small value to avoid division by zero


@dataclass
class SegmentationResult:
    """Result of segmentation evaluation.

    Attributes
    ----------
    r2 : float
        R² of segmentation (between-group variance / total variance)
    n_segments : int
        Number of segments created
    pd_by_segment : NDArray
        Probability of default (or target variable) per segment (shape: (n_segments,))
    segment_sizes : NDArray
        Number of observations per segment (shape: (n_segments,))
    segment_proportions : NDArray
        Proportion of population per segment (shape: (n_segments,))
    h_inter : float
        Inter-segment heterogeneity (between-group variance)
    h_intra : float
        Intra-segment homogeneity (within-group variance)
    gini : float, default=0.0
        Gini-style concentration score computed from segment default rates
    ks : float, default=0.0
        Kolmogorov-Smirnov statistic between cumulative bad and good rates

    Examples
    --------
    >>> result = SegmentationResult(
    ...     r2=0.85,
    ...     n_segments=5,
    ...     pd_by_segment=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    ...     segment_sizes=np.array([200, 300, 250, 150, 100]),
    ...     segment_proportions=np.array([0.2, 0.3, 0.25, 0.15, 0.1]),
    ...     h_inter=0.15,
    ...     h_intra=0.85,
    ... )
    """

    r2: float
    n_segments: int
    pd_by_segment: NDArray
    segment_sizes: NDArray
    segment_proportions: NDArray
    h_inter: float
    h_intra: float
    gini: float = 0.0
    ks: float = 0.0

    def is_monotonic_increasing(self) -> bool:
        """Check if target variable (PD) is strictly increasing across segments.

        This assumes scores are sorted ascending, so PD should also increase.

        Returns
        -------
        bool
            True if PD values are strictly increasing
        """
        if self.n_segments <= 1:
            return True
        diffs = np.diff(self.pd_by_segment)
        return bool(np.all(diffs > -EPSILON))  # Allow small floating-point errors

    def is_monotonic_decreasing(self) -> bool:
        """Check if target variable (PD) is strictly decreasing across segments.

        This assumes higher scores should have lower PD (risk), so PD decreases.

        Returns
        -------
        bool
            True if PD values are strictly decreasing
        """
        if self.n_segments <= 1:
            return True
        diffs = np.diff(self.pd_by_segment)
        return bool(np.all(diffs < EPSILON))  # Allow small floating-point errors

    def max_segment_proportion(self) -> float:
        """Get maximum segment proportion.

        Returns
        -------
        float
            Largest segment proportion (0 to 1)
        """
        return float(self.segment_proportions.max())

    def min_segment_proportion(self) -> float:
        """Get minimum segment proportion.

        Returns
        -------
        float
            Smallest segment proportion (0 to 1)
        """
        return float(self.segment_proportions.min())

    def is_balanced(
        self,
        min_size: float = DEFAULT_MIN_SEGMENT_SIZE,
        max_size: float = DEFAULT_MAX_SEGMENT_SIZE,
    ) -> bool:
        """Check if all segments meet size constraints.

        Parameters
        ----------
        min_size : float, default=0.05
            Minimum allowed segment proportion
        max_size : float, default=0.30
            Maximum allowed segment proportion

        Returns
        -------
        bool
            True if all segments are within size bounds
        """
        return bool(
            np.all(self.segment_proportions >= min_size)
            and np.all(self.segment_proportions <= max_size)
        )
