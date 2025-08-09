# Contributing to ChronoXtract

We welcome contributions to ChronoXtract! This document provides guidelines for contributing to the project.

## 🚀 Ways to Contribute

- 🐛 **Bug Reports**: Report issues or unexpected behavior
- 💡 **Feature Requests**: Suggest new time series analysis features
- 📝 **Documentation**: Improve documentation, examples, or tutorials
- 🔧 **Code Contributions**: Fix bugs or implement new features
- 🧪 **Testing**: Add test cases or improve test coverage
- 📊 **Examples**: Contribute real-world use cases and examples

## 📋 Getting Started

### Prerequisites

- **Rust** (latest stable version)
- **Python** 3.8+
- **Maturin** for Python-Rust bindings
- **Git** for version control

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ChronoXtract.git
   cd ChronoXtract
   ```

2. **Set up Rust development environment**:
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   
   # Install required tools
   cargo install maturin
   ```

3. **Set up Python environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install maturin numpy matplotlib pytest
   ```

4. **Build the project**:
   ```bash
   maturin develop
   ```

5. **Verify installation**:
   ```bash
   python -c "import chronoxtract as ct; print('Setup successful!')"
   ```

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality

3. **Test your changes**:
   ```bash
   # Build and test
   maturin develop
   
   # Run examples to verify
   python docs/examples/basic_statistics.py
   
   # Run any existing tests
   cargo test
   ```

4. **Update documentation**:
   - Update relevant documentation files
   - Add examples if introducing new features
   - Update API reference if needed

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We follow conventional commit format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `style:` Code style changes

Examples:
```
feat: add rolling correlation function
fix: handle empty arrays in variance calculation
docs: add examples for frequency domain analysis
test: add unit tests for statistical functions
```

## 🔧 Code Guidelines

### Rust Code

- **Use `rustfmt` for formatting**:
  ```bash
  cargo fmt
  ```

- **Use `clippy` for linting**:
  ```bash
  cargo clippy
  ```

- **Write documentation comments**:
  ```rust
  /// Calculates the mean of a time series.
  ///
  /// # Arguments
  /// * `time_series` - A vector of f64 values
  ///
  /// # Returns
  /// The arithmetic mean as f64
  #[pyfunction]
  pub fn calculate_mean(time_series: Vec<f64>) -> PyResult<f64> {
      // Implementation
  }
  ```

- **Handle edge cases**:
  ```rust
  // Check for empty input
  if time_series.is_empty() {
      return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
          "Input time series cannot be empty"
      ));
  }
  ```

### Python Code (Examples/Tests)

- **Follow PEP 8 style guidelines**
- **Use type hints where appropriate**:
  ```python
  def analyze_data(data: List[float]) -> Dict[str, float]:
      """Analyze time series data."""
      pass
  ```

- **Write docstrings**:
  ```python
  def example_function(data: List[float]) -> float:
      """
      Calculate example metric from time series data.
      
      Args:
          data: List of time series values
          
      Returns:
          Calculated metric value
          
      Example:
          >>> result = example_function([1, 2, 3, 4, 5])
          >>> print(result)
          3.0
      """
      pass
  ```

## 🧪 Testing

### Adding Tests

1. **For Rust functions**: Add unit tests in the same file
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_calculate_mean() {
           let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
           let result = calculate_mean(data).unwrap();
           assert!((result - 3.0).abs() < f64::EPSILON);
       }
   }
   ```

2. **For Python examples**: Create test scripts
   ```python
   import chronoxtract as ct
   import numpy as np
   
   def test_basic_functionality():
       data = [1, 2, 3, 4, 5]
       result = ct.time_series_summary(data)
       assert abs(result['mean'] - 3.0) < 1e-10
   ```

### Running Tests

```bash
# Rust tests
cargo test

# Python examples (manual verification)
python docs/examples/basic_statistics.py
```

## 📝 Documentation

### API Documentation

- Update `docs/api_reference.md` for new functions
- Include parameter descriptions, return values, and examples
- Follow the existing format and style

### Examples

- Add examples to `docs/examples/` directory
- Include both simple demonstrations and real-world applications
- Provide clear comments and explanations
- Generate visualizations where appropriate

### User Guide

- Update `docs/user_guide.md` for significant new features
- Include best practices and performance considerations
- Provide conceptual explanations, not just function documentation

## 🚀 Feature Development

### Adding New Statistical Functions

1. **Implement in Rust** (`src/stats/mod.rs` or appropriate module):
   ```rust
   #[pyfunction]
   pub fn your_function(data: Vec<f64>) -> PyResult<f64> {
       // Implementation
   }
   ```

2. **Export in lib.rs**:
   ```rust
   m.add_function(wrap_pyfunction!(stats::your_function, m)?)?;
   ```

3. **Add to API documentation**
4. **Create examples**
5. **Add tests**

### Adding New Module Categories

1. **Create new module directory**: `src/your_module/`
2. **Implement functions**: `src/your_module/mod.rs`
3. **Add module to lib.rs**: `mod your_module;`
4. **Export functions**: Add to `chronoxtract` function in lib.rs
5. **Document thoroughly**

## 🐛 Bug Reports

When reporting bugs, please include:

- **Python version**: `python --version`
- **ChronoXtract version**: Check with `pip show chronoxtract`
- **Operating system**: Windows/macOS/Linux
- **Minimal reproduction example**:
  ```python
  import chronoxtract as ct
  
  # Minimal code that reproduces the bug
  data = [1, 2, 3]
  result = ct.some_function(data)  # This fails
  ```
- **Expected behavior**
- **Actual behavior**
- **Error messages** (full traceback)

## 💡 Feature Requests

When suggesting features:

- **Describe the use case**: What problem does this solve?
- **Provide examples**: How would you use this feature?
- **Consider existing alternatives**: Why aren't current functions sufficient?
- **Specify expected behavior**: What should the function return?
- **Mathematical background**: Include relevant formulas or references

## 📋 Pull Request Process

1. **Create descriptive PR title**:
   ```
   feat: add autocorrelation function for lag analysis
   ```

2. **Fill out PR template** (describe changes, testing, etc.)

3. **Ensure all checks pass**:
   - Code compiles without warnings
   - Examples run successfully
   - Documentation is updated

4. **Respond to review feedback**

5. **Squash commits** if requested

## 🔒 Security

If you discover security vulnerabilities:

- **Do not** open a public issue
- **Email** the maintainers directly: kumaramanasci@gmail.com
- **Include** detailed description and reproduction steps

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: kumaramanasci@gmail.com for direct communication

## 📜 License

By contributing to ChronoXtract, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors will be recognized in:
- Project README
- Release notes for significant contributions
- `CONTRIBUTORS.md` file (if we create one)

Thank you for contributing to ChronoXtract! 🚀