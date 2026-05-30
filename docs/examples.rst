========
Examples
========

This page contains practical examples demonstrating common use cases for pso-segmentation.

For a full end-to-end PD segmentation walkthrough (including objective function design),
see ``notebooks/01_business_use_case.ipynb``.

---

Basic Segmentation
===================

**Functional API - Simplest Approach**

.. code-block:: python

   from pso_segmentation import make_objective, segment_scores
   import numpy as np

   # Generate sample data
   np.random.seed(42)
   scores = np.random.rand(500)
   labels = (scores > 0.6).astype(float) + np.random.normal(0, 0.1, 500)
   labels = np.clip(labels, 0, 1) > 0.5

   # Run segmentation with R² only
   objective = make_objective(scores, labels, metric="r2")
   result = segment_scores(scores, labels, objective)

   print(f"R² Score: {result.r2:.4f}")
   print(f"Number of segments: {result.n_segments}")
   print(f"Target mean by segment: {result.target_mean_by_segment}")

---

Segmentation with Constraints
==============================

**Monotonic Default Rate**

Enforce that target mean (PD for binary targets) increases monotonically across segments:

.. code-block:: python

   from pso_segmentation import make_objective, monotonic_penalty, segment_scores
   import numpy as np

   objective = make_objective(
       scores,
       labels,
       metric="r2",
       penalties=[monotonic_penalty(weight=0.3)],
   )
   result = segment_scores(scores, labels, objective)

   # Target mean should now be monotonically increasing
   print(f"Target mean by segment: {result.target_mean_by_segment}")

**Balanced Segments**

Enforce roughly equal segment sizes:

.. code-block:: python

   from pso_segmentation import make_objective, segment_size_penalty, segment_scores
   import numpy as np

   objective = make_objective(
       scores,
       labels,
       metric="r2",
       penalties=[segment_size_penalty(min_size=0.05, max_size=0.4, weight=0.2)],
   )
   result = segment_scores(scores, labels, objective)

   print(f"Segment sizes: {result.segment_sizes}")
   print(f"Segment proportions: {result.segment_proportions}")

**Combined Constraints**

Use all constraints together:

.. code-block:: python

   from pso_segmentation import make_objective, monotonic_penalty, segment_size_penalty, segment_scores
   import numpy as np

   objective = make_objective(
       scores,
       labels,
       metric="r2",
       penalties=[
           monotonic_penalty(weight=0.3),
           segment_size_penalty(min_size=0.05, max_size=0.4, weight=0.2),
       ],
   )
   result = segment_scores(scores, labels, objective)

---

Using the OO API
=================

For more control, use the ``SegmentationOptimizer`` class:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig
   from pso_segmentation import make_objective, monotonic_penalty, segment_size_penalty
   import numpy as np

   # Configure optimizer
   config = OptimizerConfig(
       n_segments=4,
       pop_size=100,
       max_iter=500,
       w=0.7,
       c1=1.5,
       c2=1.5,
       track_history=True,
       seed=42,
   )

   # Create optimizer
   optimizer = SegmentationOptimizer(config)

   # Build objective function
   objective = make_objective(
       scores,
       labels,
       metric="r2",
       penalties=[
           monotonic_penalty(weight=0.3),
           segment_size_penalty(min_size=0.05, max_size=0.4, weight=0.2),
       ],
   )

   # Fit
   optimizer.fit(scores, labels, objective)

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

Create your own penalty and plug it into ``make_objective``:

.. code-block:: python

   from pso_segmentation import make_objective

   def min_event_count_penalty(context):
       event_counts = context.result.target_mean_by_segment * context.result.segment_sizes
       return 0.5 if event_counts.min() < 20 else 0.0

   # Use custom fitness
   objective = make_objective(scores, labels, metric="r2", penalties=[min_event_count_penalty])
   result = segment_scores(scores, labels, objective)

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
   print(f"Target mean by segment: {result.target_mean_by_segment}")
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
