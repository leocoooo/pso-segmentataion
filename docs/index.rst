=====================================
PSO Segmentation - Documentation
=====================================

**pso-segmentation** is a robust, professional-grade Python package for continuous-variable segmentation using Particle Swarm Optimization (PSO). It combines a custom PSO engine with flexible APIs for easy integration into analytics workflows (credit scoring is a common example, not a requirement).

.. image:: https://img.shields.io/badge/python-3.12+-blue
   :alt: Python 3.12+

.. image:: https://img.shields.io/badge/coverage-97%25-brightgreen
   :alt: Test Coverage 97%

.. image:: https://img.shields.io/badge/type_checking-strict-blue
   :alt: Type Checking (mypy --strict)

---

**Quick Links:**

- :doc:`Installation & Quick Start <getting_started>`
- :doc:`API Reference <api/index>`
- :doc:`Examples <examples>`
- :doc:`Advanced Usage <advanced>`

---

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :numbered:

   getting_started
   core_concepts
   api/index
   examples
   advanced
   contributing
   faq

---

Features
========

✨ **Core Features:**

- **Custom PSO Engine** - Lightweight, no dependencies on scikit-opt
- **Dual API** - Object-oriented interface + simple functional API
- **Type Safety** - Full mypy --strict compliance
- **Comprehensive Testing** - 150 tests, 97% coverage
- **Production Ready** - Clean architecture, comprehensive documentation

🎯 **Key Components:**

- **PSO Core** - Configurable particle swarm optimizer with history tracking
- **Segmentation Logic** - Flexible segmentation with R², homogeneity, and balance metrics
- **Example Fitness Functions** - 6 pedagogical fitness function implementations
- **IO Module** - CSV export/import, pickle serialization, JSON metrics
- **Optimizer Class** - High-level object-oriented interface
- **Segment Scores** - Simple functional API for quick segmentation

---

Installation
============

Install via pip:

.. code-block:: bash

   pip install pso-segmentation

Or install from source:

.. code-block:: bash

   git clone <repository-url>
   cd pso-segmentation
   pip install -e .

---

Quick Example
=============

.. code-block:: python

   from pso_segmentation import segment_scores, example_fitness_r2_only
   import numpy as np

   # Your data
   scores = np.linspace(0, 1, 500)
   labels = (scores > 0.5).astype(float)

   # Simple functional API
   result = segment_scores(
       scores, labels,
       lambda cuts: example_fitness_r2_only(cuts, scores, labels)
   )

   # Access results
   print(f"R²: {result.r2:.3f}")
   print(f"Segments: {result.n_segments}")
   print(f"Target mean by segment: {result.target_mean_by_segment}")

For more detailed examples, see :doc:`examples`.

---

Project Statistics
==================

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Metric
     - Value
   * - Total Tests
     - 150/150 (100%)
   * - Coverage
     - 97%
   * - Lines of Code
     - ~1,900
   * - Type Errors
     - 0
   * - Lint Errors
     - 0

---

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

---

License
=======

MIT License - see LICENSE file for details.

---

Contributing
============

See :doc:`contributing` for contribution guidelines.
