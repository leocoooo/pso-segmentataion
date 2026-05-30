# pso-segmentation

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`pso-segmentation` is a PSO-based package for building interpretable segmentations on any continuous variable.

## Overview

The package gives you a compact way to:

- optimize cut points on continuous variables
- encode business-specific constraints inside custom objective functions
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
from pso_segmentation import make_objective, segment_scores

scores = np.random.uniform(0, 100, 1000)
labels = np.random.binomial(1, 0.15, 1000)  # target (binary here)
objective = make_objective(scores, labels, metric="r2")

result = segment_scores(scores, labels, objective)

print(f"R2: {result.r2:.3f}")
print(f"Segments: {result.n_segments}")
```

### Object-Oriented API

```python
import numpy as np
from pso_segmentation import (
    make_objective,
    monotonic_penalty,
    OptimizerConfig,
    SegmentationOptimizer,
    segment_size_penalty,
)

scores = np.random.uniform(0, 100, 1000)
labels = np.random.binomial(1, 0.15, 1000)  # target (binary here)
objective = make_objective(
    scores,
    labels,
    metric="r2",
    penalties=[
        monotonic_penalty(weight=0.3),
        segment_size_penalty(min_size=0.05, max_size=0.4, weight=0.2),
    ],
)

config = OptimizerConfig(n_segments=5, pop_size=50, max_iter=100, seed=42)
optimizer = SegmentationOptimizer(config)
optimizer.fit(scores, labels, objective)

print(optimizer.summary())
print(optimizer.get_metrics())
```

### Selecting the number of segments

```python
from pso_segmentation import make_objective, monotonic_penalty, select_n_segments


def objective_factory(scores, labels, n_segments, params):
    return make_objective(
        scores,
        labels,
        metric="r2",
        penalties=[monotonic_penalty(weight=params["monotonic_weight"])],
    )

selection = select_n_segments(
    scores,
    labels,
    segment_range=(3, 7),
    objective_factory=objective_factory,
    param_grid={"monotonic_weight": [0.0, 0.2, 0.5]},
)

print(selection.best_candidate.n_segments)
print(selection.best_candidate.cuts)
```

## Documentation

The full user guide lives in the `docs/` folder. The notebooks are intentionally limited to:

- `notebooks/00_quick_start.ipynb` (generic segmentation + objective contract)
- `notebooks/01_business_use_case.ipynb` (PD segmentation with a custom objective)

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
