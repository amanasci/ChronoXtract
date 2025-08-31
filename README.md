# ChronoXtract

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

ChronoXtract is a high-performance Python library for time series feature extraction, built with Rust for optimal speed. Extract comprehensive statistical, temporal, and frequency-domain features from your time series data with ease.

## 🚀 Key Features

### 📊 **Statistical Analysis**
- **Descriptive Statistics**: Mean, Median, Mode, Variance, Standard Deviation
- **Distribution Analysis**: Skewness, Kurtosis, Quantiles (5%, 25%, 75%, 95%)
- **Range Metrics**: Minimum, Maximum, Range, Sum, Absolute Energy

### 📈 **Rolling Statistics**
- **Rolling Mean & Variance**: Sliding window calculations
- **Exponential Moving Average**: Customizable smoothing factor
- **Expanding Sum**: Cumulative sum calculations
- **Sliding Window Entropy**: Information content analysis

### 🌊 **Frequency Domain Analysis**
- **Fast Fourier Transform (FFT)**: Spectral analysis
- **Lomb-Scargle Periodogram**: For irregularly sampled data

### 📊 **Variability Analysis**
- **Fractional Variability**: Measure of relative variation
- **Rolling Fractional Variability**: Time-varying variability
- **Variability Timescale**: Characteristic time scales

### 📈 **Correlation Analysis**
- **Discrete Correlation Function (DCF)**: For analyzing two time series.
- **Auto-Correlation Function (ACF)**: For analyzing a single time series.
- **Z-transformed Discrete Correlation Function (ZDCF)**: An improved DCF method.

### 🧠 **Higher-order Statistics**
- **Hjorth Parameters**: Activity, mobility, and complexity measures
- **Central Moments**: 5th through 8th order statistical moments
- **Advanced Signal Characterization**: For biomedical and complex signal analysis

### 🔍 **Entropy and Information Theory**
- **Sample Entropy**: Measures signal regularity and predictability
- **Approximate Entropy**: Quantifies pattern complexity
- **Permutation Entropy**: Ordinal pattern-based complexity
- **Multiscale Entropy**: Complexity analysis across time scales
- **Lempel-Ziv Complexity**: Algorithmic complexity measure

### 📅 **Seasonality and Trend Analysis**
- **Seasonal Strength Detection**: Identify and quantify seasonal patterns
- **STL Decomposition**: Separate trend, seasonal, and remainder components
- **Detrended Fluctuation Analysis**: Detect long-range correlations
- **Trend Strength**: Quantify directional movement in data

### 🏔️ **Shape and Peak Features**
- **Peak Detection**: Identify and analyze signal peaks
- **Zero Crossing Rate**: Measure signal oscillation frequency
- **Slope Analysis**: Mean slope, slope variance, and maximum slope
- **Turning Points**: Identify local extrema and direction changes
- **Energy Distribution**: Analyze signal energy characteristics

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install chronoxtract
```

### From Source
```bash
git clone https://github.com/amanasci/ChronoXtract.git
cd ChronoXtract
pip install maturin
maturin develop
```

## 🔥 Quick Start

### Basic Statistical Summary
```python
import chronoxtract as ct
import numpy as np

# Generate sample data
data = np.random.randn(1000)

# Get comprehensive statistical summary
summary = ct.time_series_summary(data)
print(summary)
# Output: {'mean': -0.023, 'median': -0.042, 'std_dev': 0.987, ...}
```

### Rolling Statistics
```python
# Calculate rolling statistics
window_size = 50

rolling_mean = ct.rolling_mean(data, window=window_size)
rolling_var = ct.rolling_variance(data, window=window_size)
exp_moving_avg = ct.exponential_moving_average(data, alpha=0.1)
```

### Frequency Domain Analysis
```python
# Perform FFT analysis
fft_result = ct.perform_fft_py(data)

# For irregularly sampled data
time = np.sort(np.random.uniform(0, 10, 100))
values = np.sin(2 * np.pi * time) + np.random.normal(0, 0.1, 100)
frequencies = np.linspace(0.1, 2.0, 50)

lomb_result = ct.lomb_scargle_py(time.tolist(), values.tolist(), frequencies.tolist())
```

### Variability Analysis
```python
# For astronomical or financial time series
flux = np.random.lognormal(0, 0.5, 1000)
flux_err = np.random.uniform(0.01, 0.1, 1000)

fvar = ct.fractional_variability(flux.tolist(), flux_err.tolist())
fvar_err = ct.fractional_variability_error(flux.tolist(), flux_err.tolist())
```

## 📚 Documentation

- **[API Reference](docs/api_reference.md)** - Detailed function documentation
- **[Examples Gallery](docs/examples/)** - Comprehensive examples and tutorials  
- **[User Guide](docs/user_guide.md)** - In-depth explanations and best practices
- **[Contributing](CONTRIBUTING.md)** - Development and contribution guidelines

## 🎯 Use Cases

- **Financial Analysis**: Stock price volatility, trend analysis
- **Scientific Computing**: Experimental data analysis, signal processing
- **IoT & Sensor Data**: Monitoring systems, anomaly detection
- **Astronomy**: Light curve analysis, periodicity detection
- **Engineering**: Vibration analysis, system monitoring

## 🔧 Performance

ChronoXtract is built with Rust and PyO3, providing:
- **High Performance**: 10-100x faster than pure Python implementations
- **Memory Efficient**: Optimized memory usage for large datasets
- **Type Safety**: Rust's type system prevents runtime errors
- **Comprehensive**: 57+ statistical and analytical functions
- **Versatile**: Supports both regular and irregularly sampled time series

## 🎯 Roadmap

### ✅ Recently Completed
- **Peak Analysis**: Peak detection algorithms and prominence calculation
- **Advanced Time Series Analysis**: Seasonality detection, trend analysis, and STL decomposition
- **Entropy and Information Theory**: Sample entropy, permutation entropy, and multiscale entropy
- **Higher-order Statistics**: Hjorth parameters and central moments
- **Shape Features**: Zero crossing rate, slope analysis, and turning points

### 🚧 In Progress
- **Enhanced Documentation**: Interactive Jupyter notebooks and video tutorials
- **Advanced Anomaly Detection**: Machine learning-based outlier detection
- **Performance Optimizations**: Parallel processing for ultra-large datasets

### 📋 Planned Features
- **Machine Learning Integration**:
  - Feature selection utilities
  - Automated feature extraction pipelines
  - Integration with scikit-learn and other ML libraries
- **Streaming Algorithms**:
  - Real-time analysis for continuous data streams
  - Online statistical computation
- **Advanced Signal Processing**:
  - Wavelet transforms
  - Spectral density estimation
  - Advanced filtering techniques
- **Utility Functions**:
  - Automated report generation from statistical outputs
  - Data quality assessment tools

### Documentation Improvements
- Interactive Jupyter notebooks
- Video tutorials
- Automated API documentation
- Benchmarking results and performance comparisons

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, feedback, or collaborations:
- **Email**: [kumaramanasci@gmail.com](mailto:kumaramanasci@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/amanasci/ChronoXtract/issues)

## 🙏 Acknowledgments

Built with love using:
- [Rust](https://rust-lang.org/) - Systems programming language
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [RustFFT](https://github.com/ejmahler/RustFFT) - Fast Fourier Transform implementation
