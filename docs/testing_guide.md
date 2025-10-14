# Testing Guide

## Overview

This guide provides comprehensive instructions for running tests, interpreting results, and contributing new tests to the skin cancer detection fairness project.

## Table of Contents

1. [Test Infrastructure](#test-infrastructure)
2. [Running Tests](#running-tests)
3. [Test Categories](#test-categories)
4. [Coverage Reports](#coverage-reports)
5. [Writing New Tests](#writing-new-tests)
6. [Continuous Integration](#continuous-integration)
7. [Troubleshooting](#troubleshooting)

---

## Test Infrastructure

### Test Framework

We use **pytest** as our testing framework with the following plugins:
- `pytest-cov`: Code coverage reporting
- `pytest-xdist`: Parallel test execution (optional)

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                   # Shared fixtures
├── unit/                         # Unit tests
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_fairness_metrics.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/                  # Integration tests
│   ├── __init__.py
│   ├── test_training_pipeline.py
│   └── test_evaluation_pipeline.py
└── fixtures/                     # Test data and mocks
    ├── __init__.py
    └── sample_data.py
```

### Configuration Files

- **`pytest.ini`**: Main pytest configuration
- **`.coveragerc`**: Coverage reporting configuration
- **`conftest.py`**: Shared fixtures and pytest hooks

---

## Running Tests

### Basic Test Execution

Run all tests:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run with detailed output and print statements:
```bash
pytest -vv -s
```

### Running Specific Test Categories

Run only unit tests:
```bash
pytest -m unit
```

Run only integration tests:
```bash
pytest -m integration
```

Run only fairness-related tests:
```bash
pytest -m fairness
```

Run only model tests:
```bash
pytest -m models
```

### Running Specific Test Files

Run a specific test file:
```bash
pytest tests/unit/test_fairness_metrics.py
```

Run a specific test class:
```bash
pytest tests/unit/test_fairness_metrics.py::TestAUROCPerFST
```

Run a specific test function:
```bash
pytest tests/unit/test_fairness_metrics.py::TestAUROCPerFST::test_auroc_perfect_classifier
```

### Excluding Slow Tests

Skip slow tests (>5 seconds):
```bash
pytest -m "not slow"
```

### Parallel Execution

Run tests in parallel (requires `pytest-xdist`):
```bash
pytest -n auto
```

Run with 4 parallel workers:
```bash
pytest -n 4
```

### Running Tests Requiring GPU

Run GPU-specific tests (requires CUDA):
```bash
pytest -m requires_gpu
```

Skip GPU tests:
```bash
pytest -m "not requires_gpu"
```

---

## Test Categories

### Unit Tests

**Purpose**: Test individual functions and classes in isolation.

**Characteristics**:
- Fast execution (<1 second per test)
- No external dependencies
- Use mock data
- High code coverage

**Examples**:
- Image normalization functions
- Fairness metric calculations
- Model initialization
- Configuration loading

**Markers**: `@pytest.mark.unit`

### Integration Tests

**Purpose**: Test end-to-end workflows and component interactions.

**Characteristics**:
- Slower execution (1-30 seconds per test)
- Test multiple components together
- Verify complete pipelines
- Use realistic data flows

**Examples**:
- Full training loop
- Evaluation pipeline
- Checkpoint saving/loading
- Multi-model comparison

**Markers**: `@pytest.mark.integration`, `@pytest.mark.pipeline`

### Fairness Tests

**Purpose**: Validate fairness metrics and bias detection.

**Characteristics**:
- Test demographic parity
- Verify equalized odds
- Validate calibration
- FST-stratified analysis

**Examples**:
- AUROC per FST group
- Equalized Odds Difference (EOD)
- Expected Calibration Error (ECE)
- Confusion matrix per demographic

**Markers**: `@pytest.mark.fairness`

### Slow Tests

**Purpose**: Comprehensive tests that take >5 seconds.

**Characteristics**:
- Multi-epoch training
- Large dataset processing
- Extensive model evaluation

**Markers**: `@pytest.mark.slow`

---

## Coverage Reports

### Generating Coverage Reports

Run tests with coverage:
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Coverage Report Types

**Terminal Report**:
```bash
pytest --cov=src --cov-report=term-missing
```

Shows coverage with missing line numbers in terminal.

**HTML Report**:
```bash
pytest --cov=src --cov-report=html
```

Generates detailed HTML report in `htmlcov/index.html`.

**XML Report** (for CI):
```bash
pytest --cov=src --cov-report=xml
```

Generates `coverage.xml` for tools like Codecov.

### Coverage Targets

**Project-wide targets**:
- Overall: **>80%** code coverage
- Critical modules (fairness, models): **>90%**

**Viewing Coverage**:
```bash
# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Interpreting Coverage

**Coverage metrics**:
- **Statements**: Lines executed
- **Branches**: Conditional paths taken
- **Missing**: Lines not covered by tests

**Example output**:
```
Name                            Stmts   Miss  Cover   Missing
-------------------------------------------------------------
src/data/preprocessing.py          45      3    93%   12, 45, 67
src/fairness/metrics.py            78      5    94%   23, 89-92
src/models/resnet.py               56      8    86%   34, 67-73
-------------------------------------------------------------
TOTAL                             179     16    91%
```

---

## Writing New Tests

### Test Naming Conventions

**Files**: `test_<module_name>.py`
```python
test_data_preprocessing.py
test_fairness_metrics.py
```

**Classes**: `Test<Functionality>`
```python
class TestAUROCPerFST:
    pass
```

**Functions**: `test_<what_is_being_tested>`
```python
def test_auroc_perfect_classifier():
    pass
```

### Test Structure (AAA Pattern)

```python
def test_example():
    # Arrange: Set up test data and conditions
    model = create_model()
    data = load_test_data()

    # Act: Execute the functionality being tested
    result = model.predict(data)

    # Assert: Verify expected outcomes
    assert result.shape == (10, 7)
    assert torch.isfinite(result).all()
```

### Using Fixtures

**Accessing shared fixtures** (from `conftest.py`):
```python
def test_with_fixture(mock_dataloader, device):
    # Use pre-configured dataloader and device
    batch = next(iter(mock_dataloader))
    assert batch['image'].device.type == device.type
```

**Creating local fixtures**:
```python
@pytest.fixture
def custom_dataset():
    return MockHAM10000(num_samples=50, seed=42)

def test_custom(custom_dataset):
    assert len(custom_dataset) == 50
```

### Parametrized Tests

Test multiple inputs efficiently:
```python
@pytest.mark.parametrize("batch_size,expected", [
    (8, (8, 3, 224, 224)),
    (16, (16, 3, 224, 224)),
    (32, (32, 3, 224, 224)),
])
def test_batch_shapes(batch_size, expected):
    data = torch.randn(*expected)
    assert data.shape == expected
```

### Marking Tests

Add markers to categorize tests:
```python
@pytest.mark.unit
@pytest.mark.fairness
def test_fairness_metric():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_training():
    pass

@pytest.mark.requires_gpu
def test_cuda_operations():
    pass
```

### Testing Exceptions

Verify error handling:
```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="Expected 4D tensor"):
        process_image(torch.randn(3, 224))  # Wrong dimensions
```

### Skipping Tests

Skip tests conditionally:
```python
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_gpu_training():
    pass
```

### Approximate Comparisons

For floating-point comparisons:
```python
def test_loss_value():
    loss = compute_loss(predictions, labels)
    assert loss == pytest.approx(0.5, abs=0.01)  # Within 0.01
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

### Pre-commit Hooks

Run tests before committing:
```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest -m "not slow" --maxfail=1
```

---

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "ModuleNotFoundError"
```
Solution: Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue**: CUDA out of memory in GPU tests
```
Solution: Reduce batch sizes in tests or skip GPU tests
pytest -m "not requires_gpu"
```

**Issue**: Tests are too slow
```
Solution: Run only fast tests or use parallel execution
pytest -m "not slow" -n auto
```

**Issue**: Fixture not found
```
Solution: Check fixture is defined in conftest.py or test file
Ensure fixture name matches function parameter exactly
```

### Debugging Tests

Run with Python debugger:
```bash
pytest --pdb  # Drop into debugger on failure
```

Run with verbose traceback:
```bash
pytest --tb=long  # Full traceback
pytest --tb=short  # Concise traceback
```

Print output during test:
```bash
pytest -s  # Show print statements
```

### Performance Profiling

Profile test execution time:
```bash
pytest --durations=10  # Show 10 slowest tests
```

---

## Best Practices

### DO:
- Write tests for all new features
- Keep tests independent (no shared state)
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies
- Keep tests fast (<1s for unit tests)
- Aim for >80% code coverage

### DON'T:
- Write tests that depend on external services
- Use hard-coded file paths
- Test implementation details (test behavior, not internals)
- Write overly complex test logic
- Skip writing tests for "simple" code

---

## Resources

**Pytest Documentation**: https://docs.pytest.org/
**Coverage.py Documentation**: https://coverage.readthedocs.io/
**Python Testing Best Practices**: https://realpython.com/pytest-python-testing/

---

## Contact

For questions about testing:
- Check existing tests for examples
- Review this guide
- Consult team documentation

**Last Updated**: 2025-10-13
