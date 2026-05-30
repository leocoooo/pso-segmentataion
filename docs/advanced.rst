===============
Advanced Usage
===============

Custom Fitness Functions
========================

The recommended entry point is ``make_objective``. It builds the
``objective(cuts) -> float`` callable expected by the optimizer.

.. code-block:: python

   from pso_segmentation import make_objective, monotonic_penalty, segment_size_penalty

   objective = make_objective(
       scores,
       labels,
       metric="r2",
       penalties=[
           monotonic_penalty(weight=0.3),
           segment_size_penalty(min_size=0.05, max_size=0.4, weight=0.2),
       ],
   )

Weights are ordinary arguments, so they can be tuned manually or passed from a
``param_grid`` in ``select_n_segments``.

.. code-block:: python

   def objective_factory(scores, labels, n_segments, params):
       return make_objective(
           scores,
           labels,
           metric="r2",
           penalties=[
               monotonic_penalty(weight=params["monotonic_weight"]),
               segment_size_penalty(
                   min_size=params["min_size"],
                   max_size=params["max_size"],
                   weight=params["size_weight"],
               ),
           ],
       )

Custom penalties are regular callables receiving an ``ObjectiveContext``:

.. code-block:: python

   def min_event_count_penalty(context):
       event_counts = context.result.target_mean_by_segment * context.result.segment_sizes
       return 0.5 if event_counts.min() < 20 else 0.0

   objective = make_objective(
       scores,
       labels,
       metric="r2",
       penalties=[min_event_count_penalty],
   )

**Design Principles**

A fitness function should:
1. Accept cuts as numpy array of segment boundaries
2. Return a float representing fitness (higher is better)
3. Handle edge cases gracefully (invalid cuts, empty segments)
4. Balance exploration vs. constraint satisfaction

**Template**

.. code-block:: python

   def my_fitness_function(cuts: NDArray) -> float:
       """
       Custom fitness function for PSO segmentation.

       Parameters
       ----------
       cuts : NDArray
           Array of n_segments-1 cut values defining segment boundaries
       Returns
       -------
       float
           Fitness value. Higher values indicate better segmentation.
           Must handle invalid cuts by returning 0.0 or negative value.
       """
       from pso_segmentation.segmentation.computation import compute_metrics
       from pso_segmentation.segmentation.validation import validate_cuts

       # Validate cuts (critical!)
       if not validate_cuts(cuts, scores):
           return 0.0

       # Compute metrics
       result = compute_metrics(scores, labels, cuts)

       # Base fitness
       fitness = result.r2

       # Add penalties for violations
       fitness -= penalty_for_violations(result)

       return fitness

---

Constraint Implementation Patterns
===================================

**Monotonicity Constraint**

Enforce target mean increasing across segments (PD for binary targets):

.. code-block:: python

   def penalty_monotonic(target_mean_by_segment, weight=0.2):
       """Penalize violation of monotonic increasing target mean."""
       penalty = 0.0
       for i in range(len(target_mean_by_segment) - 1):
           if target_mean_by_segment[i] > target_mean_by_segment[i+1]:
               penalty += weight * (target_mean_by_segment[i] - target_mean_by_segment[i+1])
       return penalty

**Size Balance Constraint**

Penalize unequal segment sizes:

.. code-block:: python

   def penalty_size_imbalance(segment_proportions, weight=0.1):
       """Penalize segments that are too large or small."""
       expected = 1.0 / len(segment_proportions)
       penalty = 0.0
       for prop in segment_proportions:
           deviation = abs(prop - expected)
           penalty += weight * deviation
       return penalty

**Minimum Size Constraint**

Ensure segments meet minimum size:

.. code-block:: python

   def penalty_minimum_size(segment_proportions, min_size=0.05, weight=1.0):
       """Penalize segments below minimum size."""
       penalty = 0.0
       for prop in segment_proportions:
           if prop < min_size:
               penalty += weight * (min_size - prop)
       return penalty

---

Multi-Objective Optimization
=============================

Combine multiple objectives with weighted aggregation:

.. code-block:: python

   def weighted_multi_objective(cuts, scores, labels, weights):
       """
       Multi-objective fitness combining R², Gini, and monotonicity.

       Parameters
       ----------
       cuts : NDArray
           Segment boundaries
       scores : NDArray
           Risk scores
    labels : NDArray
        Target labels
       weights : dict
           Objective weights: {'r2': 0.5, 'gini': 0.3, 'monotonic': 0.2}

       Returns
       -------
       float
           Weighted fitness value
       """
       from pso_segmentation.segmentation.computation import compute_metrics
       from pso_segmentation.segmentation.validation import validate_cuts

       if not validate_cuts(cuts, scores):
           return 0.0

       result = compute_metrics(scores, labels, cuts)

       # Normalize components to [0, 1]
       r2_component = result.r2  # Already in [0, 1]
       gini_component = result.gini / 0.5  # Normalize
       target_mean = result.target_mean_by_segment
       monotonic_component = all(
           target_mean[i] <= target_mean[i+1] for i in range(len(target_mean)-1)
       ) and 1.0 or 0.0

       # Weighted sum
       fitness = (
           weights['r2'] * r2_component +
           weights['gini'] * gini_component +
           weights['monotonic'] * monotonic_component
       )

       return fitness

---

Advanced PSO Tuning
====================

**Parameter Effects**

- **w (inertia)**: Higher = more exploration, lower = more exploitation
- **c1 (cognitive)**: Controls attraction to particle's best position
- **c2 (social)**: Controls attraction to swarm's best position
- **pop_size**: Larger = better exploration, slower convergence
- **max_iter**: More iterations for better convergence

**Recommended Configurations**

For quick exploration:

.. code-block:: python

   config = OptimizerConfig(
       n_segments=5,
       pop_size=30,
       max_iter=100,
       w=0.9,      # High inertia for exploration
       c1=2.0,
       c2=2.0,
   )

For fine convergence:

.. code-block:: python

   config = OptimizerConfig(
       n_segments=5,
       pop_size=100,
       max_iter=500,
       w=0.4,      # Low inertia for exploitation
       c1=1.5,
       c2=1.5,
   )

---

Analyzing Optimization History
================================

When ``track_history=True``, access optimization history:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig

   config = OptimizerConfig(track_history=True)
   optimizer = SegmentationOptimizer(config)
   optimizer.fit(scores, labels, fitness_func)

   # Access history
   history = optimizer.get_history()  # List of best fitness per iteration

   # Plot convergence
   import matplotlib.pyplot as plt
   plt.plot(history)
   plt.xlabel('Iteration')
   plt.ylabel('Best Fitness')
   plt.title('PSO Convergence')
   plt.show()

---

Reproducibility
================

Ensure reproducible results with seed control:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig

   config = OptimizerConfig(seed=42)
   optimizer = SegmentationOptimizer(config)

   optimizer.fit(scores, labels, fitness_func)
   result1 = optimizer.get_metrics()

   # Same seed produces identical results
   config2 = OptimizerConfig(seed=42)
   optimizer2 = SegmentationOptimizer(config2)
   optimizer2.fit(scores, labels, fitness_func)
   result2 = optimizer2.get_metrics()

   assert result1.r2 == result2.r2

---

Integration with Existing Workflows
====================================

**In Scikit-learn Pipelines**

.. code-block:: python

   from sklearn.base import BaseEstimator, TransformerMixin
   from pso_segmentation import SegmentationOptimizer, OptimizerConfig

   class PSO_Segmenter(BaseEstimator, TransformerMixin):
       def __init__(self, n_segments=5, seed=None):
           self.n_segments = n_segments
           self.seed = seed

       def fit(self, X, y):
           config = OptimizerConfig(n_segments=self.n_segments, seed=self.seed)
           self.optimizer_ = SegmentationOptimizer(config)
           self.optimizer_.fit(X, y, fitness_func)
           return self

       def transform(self, X):
           cuts = self.optimizer_.get_cuts()
           segments = np.digitize(X, cuts)
           return segments.reshape(-1, 1)

**Integration with Pandas**

.. code-block:: python

   import pandas as pd
   from pso_segmentation import segment_scores, example_fitness_r2_only

   df = pd.read_csv("customers.csv")

   result = segment_scores(
       df['score'].values,
       df['target'].values,
       lambda cuts: example_fitness_r2_only(cuts, df['score'].values, df['target'].values)
   )

   df['segment'] = pd.cut(df['score'], bins=np.concatenate([[-np.inf], result.cuts, [np.inf]]))

---

Performance Optimization
========================

For large datasets (>100K samples):

.. code-block:: python

   # Use smaller population and fewer iterations first
   config = OptimizerConfig(
       n_segments=5,
       pop_size=50,     # Start small
       max_iter=200,
       w=0.7,
   )

   optimizer = SegmentationOptimizer(config)
   optimizer.fit(scores, labels, fitness_func)

   # Refine with warm start if needed
   # (requires custom PSO implementation extension)

---

Troubleshooting
===============

**Fitness function returns NaN/inf**

Check for division by zero or invalid cuts:

.. code-block:: python

   def safe_fitness(cuts, scores, labels):
       from pso_segmentation.segmentation.validation import validate_cuts
       if not validate_cuts(cuts, scores):
           return 0.0
       # ... rest of implementation

**Segmentation boundaries seem random**

Increase max_iter or adjust PSO parameters:

.. code-block:: python

   config = OptimizerConfig(
       max_iter=1000,    # Increase iterations
       pop_size=100,     # Increase population
       w=0.7,            # Balanced exploration/exploitation
   )

**Slow convergence**

Use smaller pop_size or fewer segments:

.. code-block:: python

   config = OptimizerConfig(
       n_segments=3,     # Start with fewer segments
       pop_size=50,      # Smaller population
       max_iter=200,
   )

---

See :doc:`examples` for practical demonstrations and :doc:`api/index` for complete API documentation.
