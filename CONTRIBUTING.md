# Contributing to pso-segmentation

Thank you for your interest in contributing! We welcome contributions from the community.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/pso-segmentation.git
   cd pso-segmentation
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[all]"
   ```

## Making Changes

### Code Style

We use strict code quality standards:

- **Formatting**: `ruff format` (enforced)
- **Linting**: `ruff check` (enforced)
- **Type Checking**: `mypy --strict` (enforced)

Before committing, run:

```bash
ruff format .
ruff check .
mypy src/
pytest
```

### Testing

All new features must include tests. Run tests with:

```bash
pytest
pytest --cov=src/pso_segmentation  # With coverage
```

Aim for coverage > 80% on core modules.

### Commits

Use clear, descriptive commit messages:

```
Add feature: Brief description

Longer explanation if needed.
- Point 1
- Point 2
```

Avoid:
- ❌ `fix bug`
- ❌ `update code`
- ✅ `Fix convergence issue in PSO when pop_size < 2`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run all checks: `ruff format . && ruff check . && mypy src/ && pytest`
4. Commit with clear messages
5. Push to your fork
6. Create a Pull Request with:
   - Clear title and description
   - Reference to any related issues
   - Summary of changes

## Documentation

- Update docstrings for public APIs (Google style)
- Add/update Sphinx docs for new features
- Include examples in docstrings

Example:

```python
def segment_scores(scores: np.ndarray, labels: np.ndarray, objective_func: Callable) -> np.ndarray:
    """
    Optimize segmentation of continuous scores.
    
    Parameters
    ----------
    scores : np.ndarray
        Continuous scores to segment (shape: (n_samples,))
    labels : np.ndarray
        Binary labels (0/1) (shape: (n_samples,))
    objective_func : Callable
        Custom fitness function
    
    Returns
    -------
    np.ndarray
        Optimized segment boundaries
    
    Examples
    --------
    >>> cuts = segment_scores(scores, labels, my_fitness_func)
    """
```

## Reporting Issues

Use GitHub Issues with:

- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Python version and environment
- Minimal code example

## Questions?

- Open a GitHub Discussion
- Check existing documentation and issues first

---

Thank you for contributing! 🎉
