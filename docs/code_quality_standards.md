# Code Quality Standards

## Overview

This document defines code quality standards, style guidelines, and best practices for the skin cancer detection fairness project. All contributors must adhere to these standards to maintain code consistency, readability, and maintainability.

## Table of Contents

1. [Code Style](#code-style)
2. [Type Hints](#type-hints)
3. [Documentation](#documentation)
4. [Testing Requirements](#testing-requirements)
5. [Code Review Checklist](#code-review-checklist)
6. [Linting and Formatting](#linting-and-formatting)
7. [Project Structure](#project-structure)
8. [Security Guidelines](#security-guidelines)

---

## Code Style

### Python Style Guide

We follow **PEP 8** with specific modifications:

**Line Length**: 100 characters (not 79)
```python
# Good
def compute_fairness_metrics(predictions, labels, sensitive_attributes, metric_types):
    pass

# Bad (too long)
def compute_fairness_metrics_for_all_demographic_groups_with_stratification(predictions, labels, sensitive_attributes, metric_types, stratification_method):
    pass
```

**Imports Organization**:
```python
# 1. Standard library imports
import os
import sys
from pathlib import Path

# 2. Third-party imports
import torch
import numpy as np
import pandas as pd

# 3. Local imports
from src.models import ResNet50Classifier
from src.fairness.metrics import compute_eod
```

**Naming Conventions**:
```python
# Variables and functions: snake_case
train_accuracy = 0.85
def compute_loss(predictions, labels):
    pass

# Classes: PascalCase
class FairnessMetricsCalculator:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001

# Private methods/variables: _leading_underscore
def _internal_helper_function():
    pass
```

### Whitespace

**Operators**:
```python
# Good
x = a + b
result = (x * 2) - (y / 3)

# Bad
x=a+b
result = ( x*2 )-( y/3 )
```

**Function Arguments**:
```python
# Good
def train(model, data, epochs=10, lr=0.001):
    pass

train(model, train_data, epochs=20, lr=0.01)

# Bad
def train(model,data,epochs = 10,lr = 0.001):
    pass
```

### Code Organization

**Function Length**: Maximum 50 lines
```python
# If a function exceeds 50 lines, break it into smaller functions
def large_function():
    data = load_data()
    preprocessed = preprocess_data(data)
    model = train_model(preprocessed)
    results = evaluate_model(model)
    return results
```

**Class Length**: Maximum 300 lines
```python
# If a class exceeds 300 lines, consider splitting into multiple classes
# or using composition/inheritance
```

---

## Type Hints

### Required Type Hints

**All function signatures** must include type hints:
```python
from typing import List, Dict, Tuple, Optional
import torch

def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """Compute classification accuracy."""
    correct = (predictions == labels).sum().item()
    total = len(labels)
    return correct / total
```

**Complex Types**:
```python
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, any]] = None
) -> Dict[str, any]:
    """Load configuration from file."""
    pass

def train_model(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    epochs: int = 10
) -> Tuple[torch.nn.Module, List[float]]:
    """Train model and return trained model + loss history."""
    pass
```

**Generic Types**:
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Dataset(Generic[T]):
    def __getitem__(self, index: int) -> T:
        pass
```

### Type Checking

Run type checking with **mypy**:
```bash
mypy src/
```

**mypy configuration** (`mypy.ini`):
```ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
```

---

## Documentation

### Docstring Standards

We use **Google-style docstrings** for all public functions and classes:

**Function Documentation**:
```python
def compute_equalized_odds_difference(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sensitive_attribute: torch.Tensor
) -> float:
    """
    Compute Equalized Odds Difference (EOD) metric.

    EOD measures the maximum difference in TPR or FPR between demographic groups.
    A value of 0 indicates perfect fairness.

    Args:
        predictions: Binary predictions, shape (N,)
        labels: Ground truth labels, shape (N,)
        sensitive_attribute: Demographic group labels, shape (N,)

    Returns:
        EOD value in range [0, 1], where 0 is perfectly fair

    Raises:
        ValueError: If inputs have mismatched shapes
        ValueError: If predictions are not binary

    Examples:
        >>> preds = torch.tensor([1, 0, 1, 0])
        >>> labels = torch.tensor([1, 0, 0, 1])
        >>> fst = torch.tensor([1, 1, 2, 2])
        >>> eod = compute_equalized_odds_difference(preds, labels, fst)
        >>> print(f"EOD: {eod:.3f}")
    """
    pass
```

**Class Documentation**:
```python
class FairnessAwareTrainer:
    """
    Trainer with fairness constraints and bias mitigation.

    This trainer extends standard training by incorporating fairness metrics
    and applying bias mitigation techniques during optimization.

    Attributes:
        model: Neural network model
        optimizer: PyTorch optimizer
        fairness_weight: Weight for fairness loss term (default: 0.1)
        sensitive_groups: List of sensitive attribute groups to monitor

    Examples:
        >>> trainer = FairnessAwareTrainer(model, optimizer)
        >>> trainer.fit(train_loader, val_loader, epochs=10)
        >>> metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        fairness_weight: float = 0.1
    ):
        """
        Initialize fairness-aware trainer.

        Args:
            model: Neural network to train
            optimizer: Optimizer for parameter updates
            fairness_weight: Weight for fairness loss term
        """
        pass
```

**Module Documentation**:
```python
"""
Fairness metrics for demographic parity and equalized odds.

This module implements standard fairness metrics including:
- AUROC per demographic group
- Equalized Odds Difference (EOD)
- Demographic Parity Difference (DPD)
- Expected Calibration Error (ECE)

Usage:
    from src.fairness.metrics import compute_eod, compute_auroc_per_group

    eod = compute_eod(predictions, labels, sensitive_attribute)
    auroc_per_fst = compute_auroc_per_group(probs, labels, fst)
"""
```

### Inline Comments

**When to comment**:
```python
# Good: Explain WHY, not WHAT
# Use exponential moving average for stable gradient updates
momentum = 0.9

# Bad: Redundant with code
# Set momentum to 0.9
momentum = 0.9
```

**Complex Logic**:
```python
# Good: Explain complex algorithm
# Apply focal loss to handle class imbalance
# Formula: FL = -α(1-p)^γ * log(p)
# where α balances classes, γ focuses on hard examples
focal_loss = -alpha * (1 - probs)**gamma * torch.log(probs)
```

---

## Testing Requirements

### Coverage Requirements

**Minimum Coverage**:
- Overall project: **80%**
- Critical modules (fairness, models, data): **90%**
- New features: **100%** (must have tests)

**Test Types Required**:
1. **Unit Tests**: All public functions
2. **Integration Tests**: Critical workflows
3. **Edge Case Tests**: Boundary conditions
4. **Error Handling Tests**: Exception paths

### Test Quality Standards

**Test Independence**:
```python
# Good: Tests are independent
def test_accuracy_calculation():
    predictions = torch.tensor([1, 0, 1])
    labels = torch.tensor([1, 0, 1])
    accuracy = compute_accuracy(predictions, labels)
    assert accuracy == 1.0

# Bad: Tests share state
shared_data = []  # DON'T DO THIS

def test_first():
    shared_data.append(1)

def test_second():
    assert len(shared_data) == 1  # Depends on test_first
```

**Test Naming**:
```python
# Good: Descriptive test names
def test_auroc_returns_one_for_perfect_classifier():
    pass

def test_eod_raises_error_for_mismatched_shapes():
    pass

# Bad: Vague test names
def test_auroc():
    pass

def test_error():
    pass
```

---

## Code Review Checklist

### Before Submitting PR

- [ ] All tests pass (`pytest`)
- [ ] Code coverage meets requirements (>80%)
- [ ] Type hints added to all functions
- [ ] Docstrings added (Google style)
- [ ] Code formatted with `black`
- [ ] Linting passes (`flake8`)
- [ ] Type checking passes (`mypy`)
- [ ] No hardcoded paths or credentials
- [ ] README updated (if adding features)
- [ ] CHANGELOG updated

### Reviewer Checklist

**Functionality**:
- [ ] Code solves the stated problem
- [ ] Edge cases handled correctly
- [ ] Error handling is appropriate
- [ ] No obvious bugs

**Code Quality**:
- [ ] Follows PEP 8 style guide
- [ ] Functions are focused (single responsibility)
- [ ] Variable names are descriptive
- [ ] No code duplication
- [ ] Appropriate use of data structures

**Testing**:
- [ ] Adequate test coverage
- [ ] Tests are meaningful (not just for coverage)
- [ ] Tests are independent
- [ ] Edge cases tested

**Documentation**:
- [ ] Functions have docstrings
- [ ] Complex logic is commented
- [ ] README updated if needed

**Security**:
- [ ] No credentials in code
- [ ] Input validation present
- [ ] No SQL injection vulnerabilities
- [ ] No path traversal vulnerabilities

---

## Linting and Formatting

### Black (Code Formatter)

Format all code with **black**:
```bash
black src/ tests/
```

**Configuration** (`pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py39']
include = '\.pyi?$'
```

### Flake8 (Linter)

Check code style:
```bash
flake8 src/ tests/
```

**Configuration** (`.flake8`):
```ini
[flake8]
max-line-length = 100
exclude = .git,__pycache__,venv,.venv
ignore = E203, W503
```

### isort (Import Sorter)

Organize imports:
```bash
isort src/ tests/
```

**Configuration** (`pyproject.toml`):
```toml
[tool.isort]
profile = "black"
line_length = 100
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

**Configuration** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## Project Structure

### Recommended Structure

```
project/
├── src/                      # Source code
│   ├── __init__.py
│   ├── models/              # Model architectures
│   ├── data/                # Data loading and preprocessing
│   ├── fairness/            # Fairness metrics and mitigation
│   ├── evaluation/          # Evaluation utilities
│   └── utils/               # Helper functions
├── tests/                    # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
├── notebooks/                # Jupyter notebooks
├── docs/                     # Documentation
├── experiments/              # Experiment results
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── pytest.ini               # Pytest configuration
├── .coveragerc              # Coverage configuration
└── README.md                # Project overview
```

### Module Organization

**Single Responsibility**:
```python
# Good: Focused modules
src/fairness/metrics.py       # Fairness metrics only
src/fairness/mitigation.py    # Bias mitigation only

# Bad: Mixed responsibilities
src/fairness/everything.py    # Metrics, mitigation, evaluation
```

---

## Security Guidelines

### Sensitive Data

**Never commit**:
- API keys
- Passwords
- Database credentials
- Private keys
- Personal data

**Use environment variables**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")  # Good
API_KEY = "sk-1234567890abcdef"  # BAD - Never hardcode
```

### Input Validation

**Always validate inputs**:
```python
def process_image(image: torch.Tensor) -> torch.Tensor:
    """Process image tensor."""
    # Validate shape
    if image.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {image.ndim}D")

    # Validate range
    if not torch.all((image >= 0) & (image <= 1)):
        raise ValueError("Image values must be in [0, 1]")

    return image
```

### File Path Safety

**Prevent path traversal**:
```python
from pathlib import Path

def load_model(model_name: str) -> torch.nn.Module:
    """Load model from checkpoints directory."""
    # Good: Validate path
    checkpoint_dir = Path("checkpoints")
    model_path = (checkpoint_dir / model_name).resolve()

    # Ensure path is within checkpoint_dir
    if not model_path.is_relative_to(checkpoint_dir.resolve()):
        raise ValueError("Invalid model path")

    return torch.load(model_path)
```

---

## Performance Guidelines

### Memory Efficiency

**Use generators for large datasets**:
```python
# Good: Generator (memory-efficient)
def load_images(directory):
    for file_path in directory.glob("*.jpg"):
        yield load_image(file_path)

# Bad: Load all in memory
def load_images(directory):
    return [load_image(f) for f in directory.glob("*.jpg")]
```

**Delete large objects**:
```python
# Good: Explicit cleanup
large_tensor = torch.randn(10000, 10000)
result = process(large_tensor)
del large_tensor  # Free memory
torch.cuda.empty_cache()  # If using GPU
```

### Computation Efficiency

**Vectorize operations**:
```python
# Good: Vectorized
accuracies = (predictions == labels).float().mean(dim=0)

# Bad: Loop
accuracies = []
for i in range(predictions.shape[1]):
    acc = (predictions[:, i] == labels).float().mean()
    accuracies.append(acc)
```

---

## Git Workflow

### Commit Messages

**Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build/tooling changes

**Example**:
```
feat(fairness): Add Expected Calibration Error metric

Implement ECE computation for per-group calibration analysis.
Includes binning strategy and visualization utilities.

Closes #42
```

### Branch Naming

```
feature/<feature-name>
bugfix/<bug-description>
hotfix/<critical-fix>
refactor/<refactor-description>
```

---

## Resources

**Style Guides**:
- PEP 8: https://pep8.org/
- Google Python Style: https://google.github.io/styleguide/pyguide.html

**Tools**:
- Black: https://black.readthedocs.io/
- Flake8: https://flake8.pycqa.org/
- mypy: http://mypy-lang.org/

**Last Updated**: 2025-10-13
