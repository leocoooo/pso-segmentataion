========
Examples
========

This page contains practical examples demonstrating common use cases for pso-segmentation.

---

Basic Segmentation
===================

**Functional API - Simplest Approach**

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_r2_only
   import numpy as np

   # Generate sample data
   np.random.seed(42)
   scores = np.random.rand(500)
   labels = (scores > 0.6).astype(float) + np.random.normal(0, 0.1, 500)
   labels = np.clip(labels, 0, 1) > 0.5

   # Run segmentation with R² only
   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_r2_only(cuts, scores, labels)
   )

   print(f"R² Score: {result.r2:.4f}")
   print(f"Number of segments: {result.n_segments}")
   print(f"PD by segment: {result.pd_by_segment}")

---

Segmentation with Constraints
==============================

**Monotonic Default Rate**

Enforce that default rate (PD) increases monotonically across segments:

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_r2_with_monotonic_penalty
   import numpy as np

   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_r2_with_monotonic_penalty(cuts, scores, labels)
   )

   # PD should now be monotonically increasing
   print(f"PD by segment: {result.pd_by_segment}")

**Balanced Segments**

Enforce roughly equal segment sizes:

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_r2_with_balance_penalty
   import numpy as np

   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_r2_with_balance_penalty(cuts, scores, labels)
   )

   print(f"Segment sizes: {result.segment_sizes}")
   print(f"Segment proportions: {result.segment_proportions}")

**Combined Constraints**

Use all constraints together:

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_r2_with_all_constraints
   import numpy as np

   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_r2_with_all_constraints(cuts, scores, labels)
   )

---

Using the OO API
=================

For more control, use the ``SegmentationOptimizer`` class:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig
   from pso_segmentation import example_fitness_r2_with_all_constraints
   import numpy as np

   # Configure optimizer
   config = OptimizerConfig(
       n_segments=4,
       pop_size=100,
       max_iter=500,
       w=0.7,
       c1=1.5,
       c2=1.5,
       enforce_monotonic=False,  # Handled in fitness function
       track_history=True,
       seed=42,
   )

   # Create optimizer
   optimizer = SegmentationOptimizer(config)

   # Define fitness function
   def fitness_func(cuts):
       return example_fitness_r2_with_all_constraints(cuts, scores, labels)

   # Fit
   optimizer.fit(scores, labels, fitness_func)

   # Get results
   result = optimizer.get_metrics()
   print(optimizer.summary())

---

Exporting Results
==================

**Save to CSV Files**

.. code-block:: python

   from pso_segmentation import export_segmentation_to_csv
   import os

   files = export_segmentation_to_csv(
       cuts=optimizer.get_cuts(),
       scores=scores,
       labels=labels,
       output_dir="./results"
   )

   for file in files:
       print(f"Created: {file}")

   # Files created:
   # - results/cuts.csv
   # - results/segmented_data.csv
   # - results/metrics.csv

**Load from CSV Files**

.. code-block:: python

   from pso_segmentation import import_segmentation_from_csv

   scores_loaded, labels_loaded, segments_loaded, cuts_loaded = \
       import_segmentation_from_csv(
           data_csv="results/segmented_data.csv",
           cuts_csv="results/cuts.csv"
       )

**Save Optimizer State**

.. code-block:: python

   from pso_segmentation import save_optimizer_state, load_optimizer_state

   # Save
   save_optimizer_state(optimizer, "optimizer_state.pkl")

   # Load
   optimizer_loaded = load_optimizer_state("optimizer_state.pkl")
   print(optimizer_loaded.summary())

---

Custom Fitness Functions
=========================

Create your own fitness function combining multiple objectives:

.. code-block:: python

   def custom_fitness(cuts, scores, labels):
       """
       Custom fitness: Maximize R² while enforcing constraints.

       Parameters
       ----------
       cuts : NDArray
           Segment boundaries
       scores : NDArray
           Credit scores
       labels : NDArray
           Default indicators

       Returns
       -------
       float
           Fitness value (higher is better)
       """
       from pso_segmentation.segmentation.computation import compute_metrics
       from pso_segmentation.segmentation.validation import validate_cuts

       # Validate cuts
       if not validate_cuts(cuts, scores):
           return 0.0

       # Compute metrics
       result = compute_metrics(cuts, scores, labels)

       # Base fitness: R²
       fitness = result.r2

       # Penalty for violated constraints
       # Monotonicity penalty
       pd = result.pd_by_segment
       if not all(pd[i] <= pd[i+1] for i in range(len(pd)-1)):
           fitness -= 0.2

       # Size balance penalty
       proportions = result.segment_proportions
       expected_proportion = 1.0 / len(proportions)
       size_penalty = sum(abs(p - expected_proportion) for p in proportions)
       fitness -= 0.1 * size_penalty

       return fitness

   # Use custom fitness
   result = segment_scores(scores, labels, custom_fitness)

---

Gini-Based Segmentation
=======================

Focus on Gini coefficient instead of R²:

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_gini_focused
   import numpy as np

   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_gini_focused(cuts, scores, labels)
   )

   print(f"Best Gini: {result.gini:.4f}")

---

Business Metric Optimization
=============================

Optimize for custom business metrics:

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_custom_business_metric
   import numpy as np

   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_custom_business_metric(cuts, scores, labels)
   )

   # This example combines Kolmogorov-Smirnov, Gini, and population divergence

---

Accessing Detailed Results
===========================

The ``SegmentationResult`` dataclass contains detailed metrics:

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_r2_only

   result = segment_scores(scores, labels, fitness_func)

   # Access all metrics
   print(f"R² (explained variance): {result.r2:.4f}")
   print(f"Number of segments: {result.n_segments}")
   print(f"PD by segment: {result.pd_by_segment}")
   print(f"Segment sizes: {result.segment_sizes}")
   print(f"Segment proportions: {result.segment_proportions}")
   print(f"H_inter (between-group): {result.h_inter:.4f}")
   print(f"H_intra (within-group): {result.h_intra:.4f}")

---

Performance Tuning
===================

For faster results, adjust PSO parameters:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig

   config = OptimizerConfig(
       n_segments=5,
       pop_size=30,        # Smaller population
       max_iter=100,       # Fewer iterations
       w=0.9,              # Higher inertia (explore more)
       c1=2.0,             # Higher cognitive weight
       c2=2.0,             # Higher social weight
       seed=42,
   )

   optimizer = SegmentationOptimizer(config)
   optimizer.fit(scores, labels, fitness_func)

---

Multi-Run Analysis
===================

Run multiple times and compare results:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig
   from pso_segmentation import example_fitness_r2_only
   import numpy as np

   results = []
   for run in range(5):
       config = OptimizerConfig(n_segments=5, seed=run)
       optimizer = SegmentationOptimizer(config)
       optimizer.fit(scores, labels, 
                     lambda cuts: example_fitness_r2_only(cuts, scores, labels))
       results.append(optimizer.get_metrics())

   # Compare
   for i, res in enumerate(results):
       print(f"Run {i}: R² = {res.r2:.4f}")

   # Find best
   best = max(results, key=lambda x: x.r2)
   print(f"Best R²: {best.r2:.4f}")

---

See :doc:`advanced` for more sophisticated examples and :doc:`api/index` for complete API reference.
