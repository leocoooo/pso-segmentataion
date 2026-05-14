============
Contributing
============

Thank you for interest in contributing to pso-segmentation! This guide outlines how to contribute code, documentation, and improvements.

---

Getting Started
===============

1. **Fork and Clone**

   .. code-block:: bash

      git clone https://github.com/yourusername/pso-segmentation.git
      cd pso-segmentation

2. **Create Virtual Environment**

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Windows: venv\Scripts\activate
      pip install -e ".[dev]"

3. **Verify Setup**

   .. code-block:: bash

      pytest tests/
      mypy src/ --strict
      ruff check src/

---

Development Workflow
====================

**Branch Convention**

- Feature: `feature/description` (e.g., `feature/custom-fitness`)
- Bug fix: `bugfix/issue-name` (e.g., `bugfix/monotonic-constraint`)
- Documentation: `docs/topic` (e.g., `docs/advanced-examples`)

**Commit Messages**

Follow conventional commits:

.. code-block:: text

   feat(module): Short description
   fix(optimizer): Correct boundary handling
   docs(api): Add fitness function examples
   test(pso): Increase coverage for edge cases
   refactor(io): Simplify serialization logic

---

Code Standards
==============

**Type Hints**

All code must have full type hints passing `mypy --strict`:

.. code-block:: python

   from __future__ import annotations

   import numpy as np
   from typing import Callable
   from pso_segmentation.segmentation.metrics import NDArray

   def my_function(
       data: NDArray,
       callback: Callable[[float], None] | None = None,
   ) -> NDArray:
       """Process data with optional callback."""
       ...

**Style Guide**

- Follow PEP 8
- Line length: 100 characters (ruff configured)
- Use double quotes for strings
- Use `ruff format` for automatic formatting

**Docstrings**

Use NumPy style docstrings:

.. code-block:: python

   def segment_scores(
       scores: NDArray,
       labels: NDArray,
       objective_func: Callable[[NDArray], float],
   ) -> SegmentationResult:
       """
       Segment scores using PSO optimization.

       Parameters
       ----------
       scores : NDArray
           Risk scores (shape: (n_samples,))
       labels : NDArray
           Binary labels (shape: (n_samples,))
       objective_func : Callable
           Fitness function mapping cuts to float

       Returns
       -------
       SegmentationResult
           Segmentation results with cuts, metrics, and statistics

       Raises
       ------
       ValueError
           If scores or labels are invalid

       Examples
       --------
       >>> scores = np.array([0.1, 0.5, 0.9])
       >>> labels = np.array([0, 1, 1])
       >>> result = segment_scores(scores, labels, fitness)
       >>> print(result.r2)
       """

---

Testing Requirements
====================

**Minimum Coverage: 95%**

.. code-block:: bash

   # Run tests with coverage
   pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

   # View detailed coverage
   open htmlcov/index.html

**Test File Convention**

- Located in `tests/`
- Named `test_*.py`
- Use pytest fixtures for common setup
- Target 95%+ coverage

**Example Test**

.. code-block:: python

   import pytest
   from pso_segmentation import SegmentationOptimizer, OptimizerConfig
   import numpy as np

   def test_optimizer_fit():
       """Test basic optimizer fitting."""
       scores = np.linspace(0, 1, 100)
       labels = (scores > 0.5).astype(float)

       config = OptimizerConfig(n_segments=3, max_iter=50)
       optimizer = SegmentationOptimizer(config)

       def fitness(cuts):
           from pso_segmentation import example_fitness_r2_only
           return example_fitness_r2_only(cuts, scores, labels)

       optimizer.fit(scores, labels, fitness)
       result = optimizer.get_metrics()

       assert 0 <= result.r2 <= 1
       assert result.n_segments == 3

---

Quality Gates
=============

All PRs must pass:

1. **Tests**

   .. code-block:: bash

      pytest tests/ -q --tb=short

   ✅ 100% pass rate required

2. **Type Checking**

   .. code-block:: bash

      mypy src/ --strict

   ✅ Zero errors required

3. **Linting**

   .. code-block:: bash

      ruff check src/ tests/

   ✅ All checks pass

4. **Formatting**

   .. code-block:: bash

      ruff format src/ tests/

   ✅ All files formatted (auto-check)

**Run All Checks**

.. code-block:: bash

   pytest tests/ -q && mypy src/ --strict && ruff check src/ tests/ && ruff format src/ tests/ --check

---

Adding Features
===============

**Process**

1. Create feature branch: `git checkout -b feature/my-feature`
2. Implement with full type hints and docstrings
3. Write tests achieving 95%+ coverage
4. Pass all quality gates
5. Update documentation if needed
6. Submit PR with clear description

**Example: New Fitness Function**

Create `src/pso_segmentation/examples_new.py`:

.. code-block:: python

   from __future__ import annotations

   from pso_segmentation.segmentation.computation import compute_metrics
   from pso_segmentation.segmentation.validation import validate_cuts
   from pso_segmentation.segmentation.metrics import NDArray

   def example_fitness_custom_metric(
       cuts: NDArray,
       scores: NDArray,
       labels: NDArray,
   ) -> float:
       """
       New fitness function for demonstration.

       Parameters
       ----------
       cuts : NDArray
           Segment boundaries
       scores : NDArray
           Risk scores
       labels : NDArray
           Default labels

       Returns
       -------
       float
           Fitness value
       """
       if not validate_cuts(cuts, scores):
           return 0.0

       result = compute_metrics(cuts, scores, labels)
       return result.r2

Create `tests/test_new_fitness.py`:

.. code-block:: python

   import numpy as np
   from pso_segmentation.examples import example_fitness_custom_metric

   def test_custom_metric():
       """Test new fitness function."""
       scores = np.array([0.1, 0.5, 0.9])
       labels = np.array([0, 1, 1])
       cuts = np.array([0.3, 0.7])

       fitness = example_fitness_custom_metric(cuts, scores, labels)

       assert isinstance(fitness, float)
       assert fitness >= 0

---

Documentation
==============

**Update Docs When**

- Adding new public API
- Changing parameter names/defaults
- Adding new examples
- Documenting limitations

**Documentation Structure**

- API docs: Auto-generated from docstrings via Sphinx
- Examples: `docs/examples.rst`
- Advanced: `docs/advanced.rst`
- FAQ: `docs/faq.rst`

**Build and Preview**

.. code-block:: bash

   cd docs
   sphinx-build -b html source _build/html
   # Open _build/html/index.html in browser

---

Bug Reports
===========

**What to Include**

1. Python version, OS, package version
2. Minimal reproducible example
3. Full error traceback
4. Expected vs actual behavior

**Example Issue**

.. code-block:: text

   Title: ValueError when using negative scores

   Environment:
   - Python 3.12.1
   - pso-segmentation 0.1.0
   - Windows 11

   Reproduction:
   ```python
   from pso_segmentation import segment_scores
   import numpy as np

   scores = np.array([-0.1, 0.5, 1.1])  # Out of [0,1] range
   labels = np.array([0, 1, 1])
   result = segment_scores(scores, labels, lambda cuts: 0.5)
   ```

   Error: ValueError: ...
   Expected: Should handle or validate input ranges gracefully

---

Pull Requests
=============

**PR Checklist**

- [ ] Tests pass (100%)
- [ ] Type checks pass (mypy --strict)
- [ ] Code formatted (ruff format)
- [ ] Linting passes (ruff check)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow convention
- [ ] No breaking changes (or documented)

**PR Description Template**

.. code-block:: markdown

   **Description**
   Brief summary of changes

   **Motivation**
   Why this change is needed

   **Testing**
   - [ ] New tests added
   - [ ] All tests pass
   - [ ] Coverage maintained (95%+)

   **Checklist**
   - [ ] Follows code standards
   - [ ] Type hints complete
   - [ ] Docstrings added
   - [ ] Related issues linked (#123)

---

Code Review
===========

**What Reviewers Look For**

1. Code quality and style compliance
2. Test coverage and edge case handling
3. Type safety and correctness
4. Documentation completeness
5. Performance implications
6. Breaking changes

**Feedback Process**

- Respond to all comments
- Push updates to same branch
- Re-request review after changes
- Resolve conversations when complete

---

Release Process
===============

Versioning follows Semantic Versioning (semver):

.. code-block:: text

   MAJOR.MINOR.PATCH (e.g., 0.1.0)
   
   MAJOR: Breaking changes
   MINOR: New features (backward compatible)
   PATCH: Bug fixes

**Release Checklist**

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Ensure all tests pass
4. Create release commit
5. Tag with version
6. Build distribution: `python -m build`
7. Upload to PyPI

---

Questions?
==========

- Check existing issues and discussions
- Read :doc:`faq`
- Review :doc:`api/index`
- Ask in project discussions

Thank you for contributing! 🎉
