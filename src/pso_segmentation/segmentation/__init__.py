"""Segmentation module for PSO optimization.

Provides core functionality for segmentation: metrics computation,
validation, and constraint checking.
"""

from .computation import compute_metrics, get_segment_assignments
from .metrics import (
    DEFAULT_MAX_SEGMENT_SIZE,
    DEFAULT_MIN_SEGMENT_SIZE,
    EPSILON,
    SegmentationResult,
)
from .validation import (
    check_segment_stability,
    validate_cuts,
    validate_segmentation,
)

__all__ = [
    "SegmentationResult",
    "compute_metrics",
    "get_segment_assignments",
    "validate_cuts",
    "validate_segmentation",
    "check_segment_stability",
    "DEFAULT_MIN_SEGMENT_SIZE",
    "DEFAULT_MAX_SEGMENT_SIZE",
    "EPSILON",
]
