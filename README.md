# pso-segmentation

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, professional-grade Python package for segmentation optimization using Particle Swarm Optimization (PSO).

## 🎯 Overview

`pso-segmentation` provides a flexible and extensible framework for optimizing segmentation of continuous variables (scores) using custom objective functions. It features:

- **Custom PSO Implementation**: Pure Python, no heavy external dependencies
- **Flexible Objective Functions**: Users define their own fitness functions
- **Hybrid API**: Object-oriented for advanced users, functional API for quick usage
- **Comprehensive History Tracking**: Monitor optimization convergence across iterations
- **Serialization Support**: Export and import optimized segments
- **Type-Safe**: Full type hints with mypy strict mode

## 📦 Installation

Install from PyPI:

```bash
pip install pso-segmentation
```

Or with visualization support:

```bash
pip install pso-segmentation[viz]
```

Or with full development dependencies:

```bash
pip install pso-segmentation[all]
```

## 🚀 Quick Start

### Simple Usage (Functional API)

```python
import numpy as np
from pso_segmentation import segment_scores, example_fitness_function

# Your data
scores = np.random.uniform(0, 100, 1000)
labels = np.random.binomial(1, 0.15, 1000)

# One-liner segmentation
cuts = segment_scores(scores, labels, example_fitness_function)
print(f"Optimal cuts: {cuts}")
```

### Advanced Usage (OO API)

```python
from pso_segmentation import SegmentationOptimizer, OptimizerConfig, example_fitness_function

# Configure optimizer
config = OptimizerConfig(
    pop_size=50,
    max_iter=100,
    n_segments=5,
    seed=42,
)

# Create and fit optimizer
optimizer = SegmentationOptimizer(
    objective_func=example_fitness_function,
    config=config
)
optimizer.fit(scores, labels)

# Analyze results
print(optimizer.summary())
optimizer.get_history()  # DataFrame with convergence history
optimizer.get_metrics(scores, labels)  # Detailed metrics
```

## 📖 Documentation

Full documentation available at [Read the Docs](https://pso-segmentation.readthedocs.io)

### Key Resources

- [Installation Guide](https://pso-segmentation.readthedocs.io/en/latest/installation.html)
- [Getting Started](https://pso-segmentation.readthedocs.io/en/latest/quickstart.html)
- [Advanced Configuration](https://pso-segmentation.readthedocs.io/en/latest/guides/advanced_config.html)
- [API Reference](https://pso-segmentation.readthedocs.io/en/latest/api/)

## 🛠️ Development

Clone and install in development mode:

```bash
git clone https://github.com/yourusername/pso-segmentation.git
cd pso-segmentation
pip install -e ".[all]"
```

### Running Tests

```bash
pytest
```

### Code Quality Checks

```bash
ruff format .        # Format code
ruff check .         # Lint
mypy src/            # Type checking
pytest               # Tests with coverage
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔮 Roadmap

- [x] Core PSO engine
- [ ] Additional optimization algorithms (v1.1+)
- [ ] Advanced visualization tools (v1.1+)
- [ ] Multi-objective optimization (v2.0+)

## ⚠️ Status

**Version 0.1.0**: Early Alpha - API may change before v1.0 release.
