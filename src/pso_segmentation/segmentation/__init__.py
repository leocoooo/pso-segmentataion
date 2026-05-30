"""Segmentation module for PSO optimization.

Provides core functionality for segmentation: metrics computation,
boundary validation, and stability checks.
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
)

__all__ = [
    "SegmentationResult",
    "compute_metrics",
    "get_segment_assignments",
    "validate_cuts",
    "check_segment_stability",
    "DEFAULT_MIN_SEGMENT_SIZE",
    "DEFAULT_MAX_SEGMENT_SIZE",
    "EPSILON",
]
