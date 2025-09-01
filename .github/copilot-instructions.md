# GitHub Copilot Instructions for ChronoXtract

## Project Overview

ChronoXtract is a high-performance Python library for time series feature extraction, built with Rust for optimal speed. It provides comprehensive statistical, temporal, and frequency-domain features from time series data.

### Technology Stack
- **Core Library**: Rust (2021 edition) for performance-critical computations
- **Python Bindings**: PyO3 with ABI3 compatibility (Python 3.8+)
- **Build System**: Maturin for Python package building
- **Dependencies**: NumPy, rustfft, nalgebra, rayon for parallel processing

## Architecture and Code Organization

### Source Structure (`src/`)
- `lib.rs` - Main library entry point and Python module definition
- `stats/` - Basic statistical functions (mean, variance, skewness, etc.)
- `rollingstats/` - Rolling window calculations and moving averages
- `fda/` - Frequency domain analysis (FFT, Lomb-Scargle)
- `correlation/` - Correlation and autocorrelation functions
- `entropy/` - Entropy measures and complexity analysis
- `higherorder/` - Higher-order statistics and Hjorth parameters
- `seasonality/` - Seasonal decomposition and trend analysis
- `shape/` - Shape-based features and pattern analysis
- `peaks/` - Peak detection and prominence analysis
- `carma/` - CARMA (Continuous AutoRegressive Moving Average) models
- `misc/` - Utility functions and helpers

### Key Design Principles
1. **Performance First**: All computationally intensive operations in Rust
2. **Memory Efficient**: Minimize allocations, use iterators where possible
3. **Error Handling**: Comprehensive error propagation with `thiserror`
4. **Python Integration**: Seamless NumPy array integration via PyO3
5. **Parallel Processing**: Leverage `rayon` for parallelizable operations

## Development Workflow

### Building the Project
```bash
# Install maturin if not available
pip install maturin

# Development build (creates Python module)
maturin develop

# Production build
maturin build --release
```

### Testing Strategy
```bash
# Rust unit tests
cargo test

# Integration tests (examples)
python docs/examples/basic_statistics.py
python docs/examples/frequency_analysis.py

# Benchmarks
cargo bench
```

## Coding Standards and Conventions

### Rust Code Guidelines
1. **Documentation**: Every public function must have comprehensive doc comments with:
   - Purpose and mathematical background
   - Parameter descriptions with types and constraints
   - Return value description
   - Example usage
   - Error conditions

2. **Function Signatures**: Follow this pattern for Python-exposed functions:
   ```rust
   /// Comprehensive description with mathematical details
   /// 
   /// # Arguments
   /// * `data` - Input time series data as Vec<f64>
   /// * `param` - Parameter description with valid range
   /// 
   /// # Returns
   /// * Result containing computed value or error
   /// 
   /// # Example
   /// ```python
   /// import chronoxtract as ct
   /// result = ct.function_name([1.0, 2.0, 3.0], param=0.5)
   /// ```
   #[pyfunction]
   pub fn function_name(data: Vec<f64>, param: f64) -> PyResult<f64> {
       // Implementation
   }
   ```

3. **Error Handling**: Use `PyResult<T>` for Python-exposed functions, custom error types for internal functions

4. **Performance**: Always consider:
   - Input validation early
   - Memory allocation patterns
   - Parallelization opportunities with `rayon`
   - SIMD optimizations where applicable

### Python Integration Standards
1. **NumPy Compatibility**: Accept both Python lists and NumPy arrays
2. **Type Hints**: Provide comprehensive type hints in Python stubs
3. **Docstrings**: Mirror Rust documentation in Python format
4. **Error Messages**: Provide clear, actionable error messages

### Naming Conventions
- **Rust Functions**: `snake_case` following Rust conventions
- **Python Functions**: `snake_case` for consistency
- **Parameters**: Descriptive names (avoid single letters except for mathematical contexts)
- **Modules**: Lowercase, descriptive names

## Testing Requirements

### Unit Tests
- Every public function must have unit tests
- Test edge cases: empty arrays, single values, NaN/infinite values
- Test parameter validation and error conditions
- Include performance regression tests for critical functions

### Integration Tests
- Examples in `docs/examples/` serve as integration tests
- Each example must run without errors and produce expected output
- Include real-world datasets when possible

### Benchmarks
- Performance-critical functions must have benchmarks in `benches/`
- Compare against baseline implementations where applicable
- Monitor for performance regressions

## Documentation Standards

### API Documentation
- Update `docs/api_reference.md` for all new public functions
- Include mathematical formulas using LaTeX notation
- Provide comprehensive examples with real data

### Examples
- Add examples to `docs/examples/` for new feature categories
- Include visualization code using matplotlib
- Explain the mathematical intuition and practical applications
- Follow the naming pattern: `feature_category_analysis.py`

### Code Comments
- Explain complex algorithms and mathematical derivations
- Reference academic papers or standard implementations
- Document performance optimizations and trade-offs

## Dependency Management

### Adding Dependencies
- Prefer lightweight, well-maintained crates
- Avoid duplicating functionality already in the standard library
- Consider impact on compile times and binary size
- Document the rationale in commit messages

### Version Constraints
- Use conservative version bounds to ensure stability
- Test with both minimum and latest supported versions
- Update `Cargo.lock` only when necessary

## Performance Considerations

### Optimization Guidelines
1. **Algorithmic Complexity**: Choose optimal algorithms for the expected data sizes
2. **Memory Access Patterns**: Favor cache-friendly sequential access
3. **Parallelization**: Use `rayon` for embarrassingly parallel operations
4. **SIMD**: Consider explicit SIMD for hot loops
5. **Allocation**: Minimize heap allocations in tight loops

### Benchmarking
- Profile before optimizing
- Measure both CPU time and memory usage
- Test with realistic data sizes and patterns
- Document performance characteristics in function documentation

## Commit Message Convention

Follow conventional commits format:
```
type(scope): description

feat(stats): add robust variance calculation with outlier handling
fix(fda): handle empty arrays in FFT computation
docs(examples): add financial time series analysis example
perf(correlation): optimize autocorrelation function using FFT
test(entropy): add comprehensive entropy measure validation
```

### Types
- `feat`: New features or functions
- `fix`: Bug fixes
- `docs`: Documentation changes
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `refactor`: Code refactoring without functional changes
- `style`: Code style/formatting changes
- `build`: Build system or dependency changes

## Pull Request Guidelines

### Before Submitting
1. Run full test suite: `cargo test && maturin develop && python docs/examples/basic_statistics.py`
2. Check formatting: `cargo fmt --check`
3. Run lints: `cargo clippy -- -D warnings`
4. Update documentation for new features
5. Add or update examples as needed

### PR Description Template
```markdown
## Summary
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Breaking change

## Mathematical Background
For new statistical functions, explain the mathematical basis

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Benchmarks added (if applicable)

## Performance Impact
Describe any performance implications

## Breaking Changes
List any breaking changes and migration path
```

## Security Considerations

### Input Validation
- Always validate array sizes and parameter ranges
- Handle edge cases gracefully (empty arrays, NaN values)
- Prevent integer overflow in calculations
- Validate that indices are within bounds

### Memory Safety
- Leverage Rust's ownership system for memory safety
- Be cautious with unsafe code (avoid if possible)
- Use proper error handling instead of panicking

## Feature Development Guidelines

### Adding New Statistical Functions
1. Research mathematical background and existing implementations
2. Choose appropriate algorithms considering numerical stability
3. Implement comprehensive error handling
4. Add thorough documentation with mathematical details
5. Create examples demonstrating practical usage
6. Add unit tests covering edge cases
7. Consider performance implications and optimizations

### CARMA Model Considerations
The CARMA module is particularly complex and requires special attention to:
- Numerical stability in matrix operations
- Proper handling of ill-conditioned systems
- Convergence criteria for iterative algorithms
- Parameter estimation robustness

## Common Patterns and Best Practices

### Error Handling Pattern
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChronoExtractError {
    #[error("Input array is empty")]
    EmptyArray,
    #[error("Invalid parameter value: {value}")]
    InvalidParameter { value: f64 },
    #[error("Computation failed: {reason}")]
    ComputationError { reason: String },
}

pub fn safe_computation(data: &[f64]) -> Result<f64, ChronoExtractError> {
    if data.is_empty() {
        return Err(ChronoExtractError::EmptyArray);
    }
    // Implementation
}
```

### Parallel Processing Pattern
```rust
use rayon::prelude::*;

pub fn parallel_computation(data: &[f64]) -> Vec<f64> {
    data.par_iter()
        .map(|&x| expensive_computation(x))
        .collect()
}
```

### Python Function Wrapper Pattern
```rust
#[pyfunction]
pub fn python_wrapper(py_data: Vec<f64>, param: f64) -> PyResult<f64> {
    let result = internal_computation(&py_data, param)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(result)
}
```

This document should guide all development work on ChronoXtract to ensure consistency, quality, and maintainability.