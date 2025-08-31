# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-08-31

### Added
- **Statistical Functions**: Core statistical measures including mean, median, mode, variance, standard deviation, skewness, kurtosis
- **Higher-order Statistics**: Hjorth parameters (activity, mobility, complexity) and central moments (5th-8th order)
- **Rolling Statistics**: Rolling mean, variance, expanding sum, exponential moving average
- **Frequency Domain Analysis**: FFT analysis and Lomb-Scargle periodogram for irregularly sampled data
- **Entropy and Information Theory**: Sample entropy, approximate entropy, permutation entropy, Lempel-Ziv complexity, multiscale entropy
- **Seasonality and Trend Analysis**: Seasonal and trend strength, STL decomposition, seasonality detection, detrended fluctuation analysis
- **Shape and Peak Features**: Zero crossing rate, slope features, peak detection, peak prominence, variability features
- **Correlation Analysis**: Discrete Correlation Function (DCF), Auto-Correlation Function (ACF), Z-transformed DCF (ZDCF)
- **Variability Analysis**: Fractional variability measures for astronomical and financial time series
- **Comprehensive Documentation**: API reference, user guide, examples gallery, and contributing guidelines
- **Python-Rust Integration**: High-performance implementation using PyO3 bindings

### Features
- 57+ statistical and analytical functions for time series analysis
- Memory-efficient algorithms optimized for large datasets
- Support for both regular and irregularly sampled time series
- Extensive error handling and input validation
- Type safety through Rust implementation
- Python 3.8+ compatibility

### Documentation
- Complete API reference with detailed function descriptions
- Comprehensive user guide with tutorials and best practices
- Examples gallery with practical use cases
- Contributing guidelines for developers
- MIT License and project documentation

## [Unreleased]

### Planned
- Peak analysis algorithms (peak width calculation, burst detection)
- Advanced anomaly detection
- Machine learning integration features
- Performance enhancements with parallel processing
- Interactive Jupyter notebooks
- Video tutorials and benchmarking results

---

## Release Notes

### Version 0.1.0
This is the first stable release of ChronoXtract, providing a comprehensive suite of time series analysis tools with high-performance Rust implementation and seamless Python integration.