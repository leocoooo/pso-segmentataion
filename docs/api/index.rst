===============
API Reference
===============

.. toctree::
   :maxdepth: 2

   pso_core
   segmentation
   optimizer
   api_functional
   io

---

Main Modules
============

The pso-segmentation package is organized into several modules:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Purpose
   * - ``pso_segmentation.core.pso``
     - Custom PSO engine implementation
   * - ``pso_segmentation.segmentation``
     - Segmentation logic and metrics
   * - ``pso_segmentation.optimizer``
     - High-level OO interface
   * - ``pso_segmentation.api``
     - Functional API wrapper
   * - ``pso_segmentation.io``
     - Export/import and serialization
   * - ``pso_segmentation.examples``
     - Example fitness functions

---

Quick Module Overview
=====================

**Core PSO** (:doc:`pso_core`)

The custom Particle Swarm Optimization engine:

.. code-block:: python

   from pso_segmentation.core.pso import PSO

   pso = PSO(
       objective_func=fitness,
       n_dimensions=4,
       bounds=[(0, 1)] * 4,
       pop_size=50,
       max_iter=500,
   )
   result = pso.run()

**Segmentation** (:doc:`segmentation`)

Utilities for computing segmentation metrics:

.. code-block:: python

   from pso_segmentation.segmentation.computation import compute_metrics
   from pso_segmentation.segmentation.metrics import SegmentationResult

   result = compute_metrics(cuts, scores, labels)

**Optimizer** (:doc:`optimizer`)

High-level OO interface:

.. code-block:: python

   from pso_segmentation import SegmentationOptimizer, OptimizerConfig

   config = OptimizerConfig(n_segments=5)
   optimizer = SegmentationOptimizer(config)
   optimizer.fit(scores, labels, fitness)

**Functional API** (:doc:`api_functional`)

Simple one-liner interface:

.. code-block:: python

   from pso_segmentation import segment_scores

   result = segment_scores(scores, labels, fitness)

**IO** (:doc:`io`)

Export and serialization:

.. code-block:: python

   from pso_segmentation import export_segmentation_to_csv, save_optimizer_state

   export_segmentation_to_csv(cuts, scores, labels)
   save_optimizer_state(optimizer, "state.pkl")

---

See individual module pages for complete API documentation.
