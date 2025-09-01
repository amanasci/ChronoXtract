# ChronoXtract Test Suite

This directory contains the complete test suite for ChronoXtract, organized by module and functionality.

## Test Organization

### CARMA Module Tests
- `test_carma_basic.py` - Basic CARMA functionality tests
- `test_carma_comprehensive.py` - Comprehensive CARMA feature tests
- `test_carma_validation.py` - Validation against celerite2 and other implementations
- `test_carma_benchmark.py` - Performance benchmarking for CARMA operations
- `test_carma_performance.py` - Performance comparison with external libraries
- `test_carma_mcmc.py` - MCMC sampling tests for CARMA models

### Core Module Tests
- `test_stats.py` - Statistical function tests
- `test_correlation.py` - Correlation and autocorrelation tests
- `test_fda.py` - Frequency domain analysis tests
- `test_entropy.py` - Entropy measure tests
- `test_misc.py` - Miscellaneous utility tests
- `test_rolling_stats.py` - Rolling statistics tests
- `test_seasonality.py` - Seasonality analysis tests
- `test_shape.py` - Shape feature tests
- `test_higherorder.py` - Higher-order statistical tests

## Running Tests

### Run All CARMA Tests
```bash
python tests/run_carma_tests.py
```

### Run Individual Tests
```bash
python tests/test_carma_validation.py
python tests/test_carma_mcmc.py
# etc.
```

### Run All Tests (using pytest)
```bash
pytest tests/
```

## Test Categories

### Unit Tests
Basic functionality tests for individual functions and modules.

### Integration Tests
Tests that verify the interaction between different modules.

### Performance Tests
Benchmarking and performance comparison tests.

### Validation Tests
Tests that compare results against established implementations (e.g., celerite2).

## Adding New Tests

When adding new CARMA-related tests:

1. Use the naming convention: `test_carma_*.py`
2. Place the file in the `tests/` directory
3. Add the test to `run_carma_tests.py` if it's a CARMA test
4. Include proper docstrings and comments
5. Follow the existing test structure and patterns

## Test Dependencies

Most tests require:
- numpy
- chronoxtract (the package being tested)

Some tests also require:
- celerite2 (for validation tests)
- matplotlib (for plotting tests)
- scipy (for optimization tests)

## Continuous Integration

These tests are designed to run in CI environments and provide comprehensive coverage of the ChronoXtract functionality.
