=======================
Getting Started
=======================

Installation
============

**Requirements:** Python 3.12+

Install via pip:

.. code-block:: bash

   pip install pso-segmentation

From source:

.. code-block:: bash

   git clone <repository-url>
   cd pso-segmentation
   pip install -e ".[dev]"

---

Basic Usage
===========

The simplest way to use pso-segmentation is with the functional API:

.. code-block:: python

   from pso_segmentation import make_objective, segment_scores
   import numpy as np

   # Prepare your data
   scores = np.random.rand(1000)
   labels = np.random.binomial(1, scores)

   # Build the objective function
   objective = make_objective(scores, labels, metric="r2")

   # Run segmentation
   result = segment_scores(scores, labels, objective)

   # Access results
   print(f"Best R²: {result.r2:.4f}")
   print(f"Number of segments: {result.n_segments}")
   print(f"Target mean by segment: {result.target_mean_by_segment}")

---

Object-Oriented API
====================

For more control, use the ``SegmentationOptimizer`` class:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig
   from pso_segmentation import make_objective, monotonic_penalty, segment_size_penalty
   import numpy as np

   # Prepare data
   scores = np.random.rand(1000)
   labels = np.random.binomial(1, scores)

   # Configure optimizer
   config = OptimizerConfig(
       n_segments=4,
       pop_size=50,
       max_iter=200,
       w=0.7,
       c1=1.5,
       c2=1.5,
   )

   # Create optimizer
   optimizer = SegmentationOptimizer(config)

   # Build an objective with user-chosen penalty weights
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
   print(optimizer.summary())

---

Export Results
==============

Save your results for analysis or sharing:

.. code-block:: python

   from pso_segmentation import export_segmentation_to_csv, save_optimizer_state

   # Export to CSV files
   files = export_segmentation_to_csv(
       cuts=optimizer.get_cuts(),
       scores=scores,
       labels=labels,
       output_dir="./results"
   )

   print(f"Exported: {files}")

   # Save optimizer state
   save_optimizer_state(optimizer, "optimizer.pkl")

---

Load Results
============

Reload previously saved results:

.. code-block:: python

   from pso_segmentation import import_segmentation_from_csv, load_optimizer_state

   # Load from CSV
   scores, labels, segments, cuts = import_segmentation_from_csv(
       data_csv="results/segmented_data.csv",
       cuts_csv="results/cuts.csv"
   )

   # Load optimizer
   optimizer = load_optimizer_state("optimizer.pkl")

---

Configuration Options
======================

``OptimizerConfig`` parameters:

.. code-block:: python

   config = OptimizerConfig(
       # Segmentation parameters
       n_segments=5,                      # Number of segments

       # PSO parameters
       pop_size=50,                       # Population size
       max_iter=500,                      # Maximum iterations
       w=0.7,                             # Inertia weight
       c1=1.5,                            # Cognitive parameter
       c2=1.5,                            # Social parameter

       track_history=True,                # Track optimization history
       seed=None,                         # Random seed (None = random)
   )

---

Objective Functions
===================

The recommended way to create an objective is ``make_objective``:

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

Custom penalties are regular callables receiving an ``ObjectiveContext``:

.. code-block:: python

   def min_events_penalty(context):
       event_counts = context.result.target_mean_by_segment * context.result.segment_sizes
       return 0.5 if event_counts.min() < 20 else 0.0

   objective = make_objective(scores, labels, penalties=[min_events_penalty])

Legacy Example Fitness Functions
================================

The package includes 6 example fitness functions:

1. **example_fitness_r2_only** - Pure R² maximization
2. **example_fitness_r2_with_monotonic_penalty** - R² + monotonic constraint
3. **example_fitness_r2_with_balance_penalty** - R² + balanced segments
4. **example_fitness_r2_with_all_constraints** - Combined constraints
5. **example_fitness_gini_focused** - Gini-based metric
6. **example_fitness_custom_business_metric** - Custom KS/Gini/divergence

Use them as templates for your custom fitness functions:

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_r2_with_monotonic_penalty

   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_r2_with_monotonic_penalty(cuts, scores, labels)
   )

---

Next Steps
==========

- Explore :doc:`examples` for detailed use cases
- Learn :doc:`core_concepts` for background
- Check :doc:`api/index` for complete API reference
- See :doc:`advanced` for custom fitness functions
