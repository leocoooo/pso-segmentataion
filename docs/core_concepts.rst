===============
Core Concepts
===============

Particle Swarm Optimization (PSO)
=================================

Particle Swarm Optimization is a metaheuristic algorithm inspired by social behavior of bird flocking. It's particularly useful for optimization problems where:

- The solution space is continuous
- The objective function is expensive to compute
- Traditional gradient-based methods are unsuitable

**How PSO Works:**

1. A population (swarm) of candidate solutions (particles) explores the solution space
2. Each particle moves through the space, influenced by:
   - Its current velocity (inertia)
   - Its best position so far (cognitive/memory)
   - The best position found by the swarm (social/collaboration)
3. Over iterations, particles converge toward the global optimum

---

Segmentation Problem
====================

Segmentation divides a population into homogeneous groups based on a continuous variable.

**Generic terminology:**

- **Scores**: Continuous variable to segment (model score, signal, risk score, etc.)
- **Labels**: Target values used to evaluate the segmentation (optional but typical)
- **Cuts**: Boundaries that define segment thresholds
- **Segments**: The resulting groups

**Credit scoring example (optional):**

- Scores are predicted probabilities of default (0-1)
- Labels are observed defaults (0 or 1)
- Segment-level target means correspond to PD by segment

**Example:**

.. code-block:: text

   Scores:  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
   Cuts:    [0.3, 0.6]
   
   Segment 0: scores ≤ 0.3  → [0.1, 0.2, 0.3]
   Segment 1: 0.3 < scores ≤ 0.6 → [0.4, 0.5, 0.6]
   Segment 2: scores > 0.6   → [0.7, 0.8, 0.9]

---

Fitness Functions
=================

A fitness function quantifies how good a segmentation is. The PSO optimizer maximizes this function.

**Common Metrics:**

1. **R² (Coefficient of Determination)**
   - Measures explained variance of labels by segments
   - Range: [0, 1], higher is better
   - Formula: R² = 1 - SS_res / SS_tot

2. **H_inter (Between-group Homogeneity)**
   - Measures separation between segments
   - Higher is better for distinct segments

3. **H_intra (Within-group Homogeneity)**
   - Measures uniformity within segments
   - Higher is better for cohesive segments

4. **Gini Coefficient**
   - Measures inequality/concentration
   - Useful for risk distribution analysis

**Example Fitness Function:**

.. code-block:: python

   def fitness_r2_only(cuts, scores, labels):
       """Maximize R² without constraints."""
       result = compute_metrics(cuts, scores, labels)
       return result.r2

---

Constraints in Segmentation
============================

Real-world segmentation often requires constraints:

1. **Monotonic Increasing/Decreasing**
   - Target mean should increase (or decrease) monotonically across segments
   - Common in risk-based segmentation

2. **Balanced Segments**
   - Segments should have roughly equal size
   - Avoids very large or very small groups

3. **Size Boundaries**
   - Minimum segment size (e.g., 5% of population)
   - Maximum segment size (e.g., 30% of population)

Constraints are typically implemented by penalizing the fitness function.

---

API Design
==========

pso-segmentation provides two complementary APIs:

**Functional API** (Simple & Quick)

.. code-block:: python

   from pso_segmentation import segment_scores

   result = segment_scores(scores, labels, fitness_func)

- Single function call
- Default configuration
- Minimal setup
- Good for quick experiments

**Object-Oriented API** (Flexible & Advanced)

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig

   config = OptimizerConfig(...)
   optimizer = SegmentationOptimizer(config)
   optimizer.fit(scores, labels, fitness_func)

- Full configuration control
- Method chaining support
- State management
- Serialization/deserialization
- Good for production workflows

---

Data Structures
===============

**SegmentationResult** - Immutable dataclass storing results

.. code-block:: python

   @dataclass
   class SegmentationResult:
       r2: float                          # Coefficient of determination
       n_segments: int                    # Number of segments
    pd_by_segment: NDArray             # Segment mean of the target (PD for binary targets)
       segment_sizes: NDArray             # Count per segment
       segment_proportions: NDArray       # Proportion per segment
       h_inter: float                     # Between-group homogeneity
       h_intra: float                     # Within-group homogeneity

**OptimizerConfig** - Configuration dataclass

.. code-block:: python

   @dataclass
   class OptimizerConfig:
       n_segments: int = 5
       pop_size: int = 50
       max_iter: int = 500
       w: float = 0.7
       c1: float = 1.5
       c2: float = 1.5
       track_history: bool = True
       seed: int | None = None

---

Type System
===========

pso-segmentation uses strict type hints (PEP 604, 563):

.. code-block:: python

   # Type aliases
   type NDArray = np.ndarray[Any, np.dtype[np.float64]]
   type Objective = Callable[[NDArray], float]

   # Full annotations
   def segment_scores(
       scores: NDArray,
       labels: NDArray,
       objective_func: Objective,
       config: OptimizerConfig | None = None,
   ) -> SegmentationResult:
       ...

All code passes ``mypy --strict`` validation.

---

Next Steps
==========

- See :doc:`examples` for practical use cases
- Check :doc:`api/index` for API details
- Explore :doc:`advanced` for custom fitness functions
