"""pso-segmentation package.

A robust, professional-grade Python package for segmentation optimization
using Particle Swarm Optimization (PSO).

Version: 0.1.0
"""

from pso_segmentation.api import segment_scores
from pso_segmentation.examples import (
    example_fitness_custom_business_metric,
    example_fitness_gini_focused,
    example_fitness_r2_only,
    example_fitness_r2_with_all_constraints,
    example_fitness_r2_with_balance_penalty,
    example_fitness_r2_with_monotonic_penalty,
)
from pso_segmentation.io import (
    export_metrics_to_json,
    export_segmentation_to_csv,
    import_segmentation_from_csv,
    load_optimizer_state,
    save_optimizer_state,
)
from pso_segmentation.optimizer import OptimizerConfig, SegmentationOptimizer
from pso_segmentation.segmentation import SegmentationResult
from pso_segmentation.segmentation.computation import compute_metrics
from pso_segmentation.segmentation.validation import validate_cuts
from pso_segmentation.selection import (
    SegmentCandidate,
    SegmentSelectionResult,
    select_n_segments,
)

__version__ = "0.1.0"
__author__ = "Léo Colin"
__email__ = "leocolin7002@gmail.com"

__all__ = [
    "segment_scores",
    "SegmentationOptimizer",
    "OptimizerConfig",
    "SegmentationResult",
    "SegmentCandidate",
    "SegmentSelectionResult",
    "select_n_segments",
    "example_fitness_r2_only",
    "example_fitness_r2_with_monotonic_penalty",
    "example_fitness_r2_with_balance_penalty",
    "example_fitness_r2_with_all_constraints",
    "example_fitness_gini_focused",
    "example_fitness_custom_business_metric",
    "compute_metrics",
    "validate_cuts",
    "export_segmentation_to_csv",
    "import_segmentation_from_csv",
    "save_optimizer_state",
    "load_optimizer_state",
    "export_metrics_to_json",
]
