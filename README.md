# pso-segmentation

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`pso-segmentation` is a PSO-based package for building interpretable score segmentation models.

## Overview

The package gives you a compact way to:

- optimize cut points on continuous scores
- enforce business constraints such as monotonicity and segment size
- select the number of segments with a dedicated helper
- export results and persist optimizer state
- compare candidates with a business-specific selection function

## Installation

```bash
pip install pso-segmentation
```

For development and documentation work:

```bash
pip install -e ".[dev,docs]"
```

## Quick Start

### Functional API

```python
import numpy as np
from pso_segmentation import example_fitness_r2_only, segment_scores

scores = np.random.uniform(0, 100, 1000)
labels = np.random.binomial(1, 0.15, 1000)

result = segment_scores(
    scores,
    labels,
    lambda cuts: example_fitness_r2_only(cuts, scores, labels),
)

print(f"R2: {result.r2:.3f}")
print(f"Segments: {result.n_segments}")
```

### Object-Oriented API

```python
import numpy as np
from pso_segmentation import (
    OptimizerConfig,
    SegmentationOptimizer,
    example_fitness_r2_with_all_constraints,
)

scores = np.random.uniform(0, 100, 1000)
labels = np.random.binomial(1, 0.15, 1000)

config = OptimizerConfig(n_segments=5, pop_size=50, max_iter=100, seed=42)
optimizer = SegmentationOptimizer(config)
optimizer.fit(
    scores,
    labels,
    lambda cuts: example_fitness_r2_with_all_constraints(cuts, scores, labels),
)

print(optimizer.summary())
print(optimizer.get_metrics())
```

### Selecting the number of segments

```python
from pso_segmentation import select_n_segments

selection = select_n_segments(
    scores,
    labels,
    segment_range=(3, 7),
    selection_metric="r2",
)

print(selection.best_candidate.n_segments)
print(selection.best_candidate.cuts)
```

## Documentation

The full user guide lives in the `docs/` folder and the business-oriented walkthrough is in
`notebooks/`.

## Development

Run the test and quality checks from the repository root:

```bash
pytest
ruff check .
ruff format .
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and contribution rules.

## Status

Version 0.1.0 is the current alpha release line. The package API is stable enough for
experimentation, notebooks, and internal use, while production release work continues.
