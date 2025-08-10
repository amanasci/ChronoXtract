# Changelog

All notable changes to ChronoXtract will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite including API reference, user guide, and examples
- Contributing guidelines for developers
- MIT License
- Examples gallery with practical use cases:
  - Basic statistics examples
  - Rolling statistics demonstrations
  - Frequency analysis examples
  - Variability analysis tutorials
  - Real-world applications (financial, sensor monitoring, astronomical)

### Changed
- Enhanced README.md with complete feature overview and usage examples
- Improved project structure with organized documentation

### Documentation
- Added detailed API reference with function documentation and examples
- Created comprehensive user guide with best practices and tutorials
- Added example scripts for all major feature categories
- Included performance tips and real-world application examples

## [0.0.2] - 2024-01-XX

### Added
- Basic statistical functions:
  - `time_series_summary()`: Comprehensive statistical analysis
  - `time_series_mean_median_mode()`: Central tendency measures
- Rolling statistics:
  - `rolling_mean()`: Sliding window average
  - `rolling_variance()`: Sliding window variance
  - `exponential_moving_average()`: EMA with customizable alpha
  - `sliding_window_entropy()`: Complexity analysis
  - `expanding_sum()`: Cumulative sum calculation
- Frequency domain analysis:
  - `perform_fft_py()`: Fast Fourier Transform
  - `lomb_scargle_py()`: Lomb-Scargle periodogram for irregular data
- Variability analysis:
  - `fractional_variability()`: Measure of relative variation
  - `fractional_variability_error()`: Error estimation
  - `rolling_fractional_variability()`: Time-varying variability
  - `calc_variability_timescale()`: Characteristic timescales

### Technical
- Rust implementation with PyO3 bindings for high performance
- Support for Python 3.8+
- Memory-efficient algorithms for large datasets

## [0.0.1] - Initial Development

### Added
- Project foundation and basic structure
- Core statistical calculations
- Python package infrastructure

---

## Future Roadmap

### Planned Features
- **Peak Analysis** (partially implemented):
  - Peak detection algorithms
  - Peak prominence calculation
  - Peak width analysis
- **Advanced Time Series Analysis**:
  - Seasonality detection
  - Trend analysis
  - Anomaly detection algorithms
  - Time series decomposition
- **Machine Learning Integration**:
  - Feature selection utilities
  - Automated feature extraction pipelines
- **Performance Enhancements**:
  - Parallel processing for large datasets
  - Streaming algorithms for real-time analysis
- **Additional Statistical Methods**:
  - Correlation analysis
  - Spectral density estimation
  - Wavelet transforms

### Documentation Improvements
- Interactive Jupyter notebooks
- Video tutorials
- API documentation with auto-generation
- Benchmarking results and performance comparisons

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Support

- **Issues**: [GitHub Issues](https://github.com/amanasci/ChronoXtract/issues)
- **Email**: kumaramanasci@gmail.com