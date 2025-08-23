# ChronoXtract User Guide

Welcome to the comprehensive user guide for ChronoXtract! This guide will help you understand the concepts, best practices, and advanced usage patterns for time series feature extraction.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Statistical Analysis](#statistical-analysis)
4. [Rolling Statistics](#rolling-statistics)
5. [Frequency Domain Analysis](#frequency-domain-analysis)
6. [Variability Analysis](#variability-analysis)
7. [Correlation Analysis](#correlation-analysis)
8. [Best Practices](#best-practices)
9. [Performance Tips](#performance-tips)
10. [Real-World Applications](#real-world-applications)

---

## Getting Started

### Installation

ChronoXtract requires Python 3.8 or higher. Install it using pip:

```bash
pip install chronoxtract
```

For development or latest features, install from source:

```bash
git clone https://github.com/amanasci/ChronoXtract.git
cd ChronoXtract
pip install maturin
maturin develop
```

### Basic Usage

```python
import chronoxtract as ct
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
time = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * 0.5 * time) + 0.5 * np.random.randn(1000)

# Basic analysis
summary = ct.time_series_summary(signal.tolist())
print(f"Signal statistics: Mean={summary['mean']:.3f}, Std={summary['standard_deviation']:.3f}")
```

---

## Core Concepts

### Data Types and Input Format

ChronoXtract expects time series data as Python lists of floats:

```python
# Correct format
data = [1.0, 2.5, 3.2, 4.1, 5.8]
result = ct.time_series_summary(data)

# Convert from numpy arrays
numpy_data = np.array([1, 2, 3, 4, 5])
result = ct.time_series_summary(numpy_data.tolist())

# Convert from pandas Series
import pandas as pd
series = pd.Series([1, 2, 3, 4, 5])
result = ct.time_series_summary(series.tolist())
```

### Understanding Return Types

Different functions return different types:

```python
# Dictionary with multiple statistics
summary = ct.time_series_summary(data)  # Returns dict

# Single tuple
mean, median, mode = ct.time_series_mean_median_mode(data)  # Returns tuple

# List of values
rolling_avg = ct.rolling_mean(data, window=5)  # Returns List[float]

# Optional values (may be None)
fvar = ct.fractional_variability(flux, flux_err)  # Returns Optional[float]
```

---

## Statistical Analysis

### Descriptive Statistics

The `time_series_summary()` function provides a comprehensive statistical overview:

```python
import numpy as np

# Create different types of distributions
normal_data = np.random.normal(0, 1, 1000)
skewed_data = np.random.exponential(2, 1000)
bimodal_data = np.concatenate([np.random.normal(-2, 0.5, 500), 
                               np.random.normal(2, 0.5, 500)])

for name, data in [("Normal", normal_data), ("Skewed", skewed_data), ("Bimodal", bimodal_data)]:
    stats = ct.time_series_summary(data.tolist())
    print(f"\n{name} Distribution:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  Skewness: {stats['skewness']:.3f}")
    print(f"  Kurtosis: {stats['kurtosis']:.3f}")
```

### Understanding Skewness and Kurtosis

- **Skewness**: Measures asymmetry
  - Skewness = 0: Symmetric distribution
  - Skewness > 0: Right tail is longer
  - Skewness < 0: Left tail is longer

- **Kurtosis**: Measures tail heaviness
  - Kurtosis = 3: Normal distribution
  - Kurtosis > 3: Heavy tails (leptokurtic)
  - Kurtosis < 3: Light tails (platykurtic)

```python
# Analyze distribution shape
def analyze_distribution(data, name):
    stats = ct.time_series_summary(data)
    skew = stats['skewness']
    kurt = stats['kurtosis']
    
    print(f"{name}:")
    if abs(skew) < 0.5:
        print("  Shape: Approximately symmetric")
    elif skew > 0.5:
        print("  Shape: Right-skewed (positive skew)")
    else:
        print("  Shape: Left-skewed (negative skew)")
    
    if kurt > 3.5:
        print("  Tails: Heavy-tailed")
    elif kurt < 2.5:
        print("  Tails: Light-tailed")
    else:
        print("  Tails: Normal-like")

# Example usage
stock_returns = np.random.laplace(0, 0.02, 1000)  # Typical stock return distribution
analyze_distribution(stock_returns.tolist(), "Stock Returns")
```

---

## Rolling Statistics

### Window Size Selection

Choosing the right window size is crucial for rolling statistics:

```python
# Generate signal with different time scales
t = np.linspace(0, 100, 2000)
trend = 0.01 * t  # Long-term trend
seasonal = 5 * np.sin(2 * np.pi * t / 20)  # 20-unit period
noise = np.random.normal(0, 1, len(t))
signal = trend + seasonal + noise

# Compare different window sizes
windows = [10, 50, 200]
plt.figure(figsize=(12, 8))

plt.subplot(len(windows) + 1, 1, 1)
plt.plot(t, signal, alpha=0.7, label='Original Signal')
plt.legend()
plt.title('Original Signal')

for i, window in enumerate(windows):
    rolling_avg = ct.rolling_mean(signal.tolist(), window=window)
    
    plt.subplot(len(windows) + 1, 1, i + 2)
    plt.plot(t[window-1:], rolling_avg, label=f'Rolling Mean (window={window})')
    plt.plot(t, signal, alpha=0.3, color='gray')
    plt.legend()
    plt.title(f'Window Size: {window}')

plt.tight_layout()
plt.show()
```

### Rolling Variance for Anomaly Detection

```python
# Detect anomalies using rolling variance
def detect_anomalies(data, window=50, threshold=3):
    rolling_var = ct.rolling_variance(data, window=window)
    
    # Calculate z-scores of variance
    mean_var = np.mean(rolling_var)
    std_var = np.std(rolling_var)
    z_scores = [(var - mean_var) / std_var for var in rolling_var]
    
    # Find anomalies
    anomalies = []
    for i, z in enumerate(z_scores):
        if abs(z) > threshold:
            anomalies.append(i + window - 1)  # Adjust for window offset
    
    return anomalies, rolling_var

# Example with artificial anomalies
normal_data = np.random.normal(0, 1, 500)
# Insert anomalies
normal_data[100:110] *= 5  # High variance period
normal_data[300:305] *= 0.1  # Low variance period

anomalies, variance = detect_anomalies(normal_data.tolist())
print(f"Detected {len(anomalies)} anomalous periods")
```

---

## Frequency Domain Analysis

### FFT for Spectral Analysis

```python
def analyze_spectrum(signal, sampling_rate):
    """Comprehensive spectral analysis using FFT"""
    
    # Perform FFT
    fft_result = ct.perform_fft_py(signal)
    
    # Calculate frequencies and power
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/sampling_rate)
    power = [abs(c)**2 for c in fft_result]
    
    # Find dominant frequencies (positive frequencies only)
    positive_freqs = freqs[:n//2]
    positive_power = power[:n//2]
    
    # Find peaks
    peak_indices = []
    for i in range(1, len(positive_power)-1):
        if (positive_power[i] > positive_power[i-1] and 
            positive_power[i] > positive_power[i+1] and
            positive_power[i] > np.max(positive_power) * 0.1):  # At least 10% of max
            peak_indices.append(i)
    
    dominant_freqs = [positive_freqs[i] for i in peak_indices]
    dominant_powers = [positive_power[i] for i in peak_indices]
    
    return {
        'frequencies': positive_freqs,
        'power': positive_power,
        'dominant_frequencies': dominant_freqs,
        'dominant_powers': dominant_powers
    }

# Example: Analyze a complex signal
fs = 1000  # Sampling rate
t = np.arange(0, 2, 1/fs)
signal = (np.sin(2*np.pi*50*t) + 
          0.5*np.sin(2*np.pi*120*t) + 
          0.2*np.sin(2*np.pi*200*t) + 
          0.1*np.random.randn(len(t)))

spectrum = analyze_spectrum(signal.tolist(), fs)
print("Dominant frequencies found:")
for freq, power in zip(spectrum['dominant_frequencies'], spectrum['dominant_powers']):
    print(f"  {freq:.1f} Hz (power: {power:.2f})")
```

### Lomb-Scargle for Irregular Data

```python
def analyze_irregular_data(times, values, freq_range=(0.1, 10), n_freqs=100):
    """Analyze irregularly sampled time series"""
    
    # Create frequency grid
    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    
    # Compute Lomb-Scargle periodogram
    power = ct.lomb_scargle_py(times, values, freqs.tolist())
    
    # Find significant peaks
    power_array = np.array(power)
    threshold = np.percentile(power_array, 95)  # 95th percentile
    
    significant_peaks = []
    for i in range(1, len(power)-1):
        if (power[i] > power[i-1] and 
            power[i] > power[i+1] and 
            power[i] > threshold):
            significant_peaks.append((freqs[i], power[i]))
    
    return {
        'frequencies': freqs,
        'power': power,
        'significant_peaks': significant_peaks,
        'threshold': threshold
    }

# Example: Astronomical observation data
np.random.seed(42)
observation_times = np.sort(np.random.uniform(0, 100, 300))  # Irregular sampling
true_period = 12.5  # True period in the signal
true_signal = 2 * np.sin(2*np.pi*observation_times/true_period)
noise = np.random.normal(0, 0.5, len(observation_times))
observed_values = true_signal + noise

result = analyze_irregular_data(observation_times.tolist(), observed_values.tolist())
print(f"True period: {true_period:.1f}")
print("Detected significant periods:")
for freq, power in result['significant_peaks']:
    period = 1/freq
    print(f"  Period: {period:.1f} (freq: {freq:.3f} Hz, power: {power:.2f})")
```

---

## Variability Analysis

### Fractional Variability

Fractional variability is particularly useful in astronomy and finance:

```python
def variability_analysis(flux, flux_err, window_size=50):
    """Comprehensive variability analysis"""
    
    # Overall fractional variability
    fvar = ct.fractional_variability(flux, flux_err)
    fvar_err = ct.fractional_variability_error(flux, flux_err)
    
    # Rolling fractional variability
    rolling_fvar = ct.rolling_fractional_variability(flux, flux_err, window_size)
    
    # Remove None values
    valid_rolling = [fv for fv in rolling_fvar if fv is not None]
    
    # Statistics on variability
    if valid_rolling:
        max_var = max(valid_rolling)
        min_var = min(valid_rolling)
        avg_var = sum(valid_rolling) / len(valid_rolling)
    else:
        max_var = min_var = avg_var = None
    
    return {
        'overall_fvar': fvar,
        'overall_fvar_error': fvar_err,
        'rolling_fvar': rolling_fvar,
        'max_variability': max_var,
        'min_variability': min_var,
        'average_variability': avg_var
    }

# Example: Stock price volatility analysis
np.random.seed(42)
days = 252  # One trading year
initial_price = 100
returns = np.random.normal(0.001, 0.02, days)  # Daily returns
prices = [initial_price]
for ret in returns:
    prices.append(prices[-1] * (1 + ret))

# Add measurement errors (bid-ask spread, etc.)
price_errors = [p * 0.001 for p in prices]  # 0.1% error

var_analysis = variability_analysis(prices, price_errors)
print(f"Overall fractional variability: {var_analysis['overall_fvar']:.4f} ± {var_analysis['overall_fvar_error']:.4f}")
print(f"Maximum rolling variability: {var_analysis['max_variability']:.4f}")
print(f"Average rolling variability: {var_analysis['average_variability']:.4f}")
```

---

## Correlation Analysis

### Auto-Correlation Function (ACF)

The ACF is used to identify repeating patterns or periods in a single time series.

```python
import numpy as np
import chronoxtract as ct
import matplotlib.pyplot as plt

# Create a time series with a known period
period = 20
t = np.linspace(0, 100, 100)
v = np.sin(2 * np.pi * t / period)
e = np.random.rand(100) * 0.1

# Calculate the ACF
result = ct.acf_py(t.tolist(), v.tolist(), e.tolist(), lag_min=0, lag_max=40, lag_bin_width=0.5)
lags = result['lags']
correlations = result['correlations']

# Plot the ACF
plt.plot(lags, correlations)
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Auto-Correlation Function')
plt.show()
```

### Discrete Correlation Function (DCF)

The DCF is used to find the time lag between two different time series.

```python
import numpy as np
import chronoxtract as ct
import matplotlib.pyplot as plt

# Create two time series with a known lag
t1 = np.linspace(0, 100, 100)
v1 = np.sin(t1)
e1 = np.random.rand(100) * 0.1

lag = 10
t2 = t1 + lag
v2 = np.sin(t1)
e2 = np.random.rand(100) * 0.1

# Calculate the DCF
result = ct.dcf_py(t1.tolist(), v1.tolist(), e1.tolist(), t2.tolist(), v2.tolist(), e2.tolist(), lag_min=-20, lag_max=20, lag_bin_width=0.5)
lags = result['lags']
correlations = result['correlations']

# Plot the DCF
plt.plot(lags, correlations)
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Discrete Correlation Function')
plt.show()
```

### Z-transformed Discrete Correlation Function (ZDCF)

The ZDCF is a more robust version of the DCF, especially for sparsely sampled time series.

```python
import numpy as np
import chronoxtract as ct
import matplotlib.pyplot as plt

# Create two time series with a known lag
t1 = np.linspace(0, 100, 100)
v1 = np.sin(t1)
e1 = np.random.rand(100) * 0.1

lag = 10
t2 = t1 + lag
v2 = np.sin(t1)
e2 = np.random.rand(100) * 0.1

# Calculate the ZDCF
result = ct.zdcf_py(t1.tolist(), v1.tolist(), e1.tolist(), t2.tolist(), v2.tolist(), e2.tolist(), min_points=11, num_mc=100)
lags = result['lags']
correlations = result['correlations']

# Plot the ZDCF
plt.plot(lags, correlations)
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Z-transformed Discrete Correlation Function')
plt.show()
```

---

## Best Practices

### Data Preprocessing

```python
def preprocess_time_series(data, remove_outliers=True, outlier_threshold=3):
    """Basic preprocessing for time series data"""
    
    # Convert to numpy for easier handling
    data_array = np.array(data)
    
    # Remove NaN values
    valid_mask = ~np.isnan(data_array)
    clean_data = data_array[valid_mask]
    
    if remove_outliers:
        # Remove outliers using z-score
        z_scores = np.abs((clean_data - np.mean(clean_data)) / np.std(clean_data))
        outlier_mask = z_scores < outlier_threshold
        clean_data = clean_data[outlier_mask]
    
    return clean_data.tolist()

# Example usage
raw_data = [1, 2, 100, 3, 4, np.nan, 5, 6, -50, 7]  # Contains outliers and NaN
clean_data = preprocess_time_series(raw_data)
print(f"Original length: {len(raw_data)}, Clean length: {len(clean_data)}")
```

### Choosing Appropriate Features

```python
def feature_selection_guide(data_length, sampling_rate=None):
    """Guide for selecting appropriate features based on data characteristics"""
    
    recommendations = []
    
    # Basic statistics (always applicable)
    recommendations.append("Basic statistics (time_series_summary)")
    
    # Rolling statistics
    if data_length > 20:
        max_window = data_length // 4
        recommendations.append(f"Rolling statistics (window size: 5 to {max_window})")
    
    # Frequency analysis
    if data_length > 50:
        if sampling_rate:
            nyquist = sampling_rate / 2
            recommendations.append(f"FFT analysis (up to {nyquist} Hz)")
        else:
            recommendations.append("FFT analysis (consider normalized frequencies)")
    
    # Variability analysis
    if data_length > 100:
        recommendations.append("Fractional variability analysis")
    
    return recommendations

# Example usage
data_length = 1000
features = feature_selection_guide(data_length, sampling_rate=100)
print("Recommended features for your data:")
for feature in features:
    print(f"  - {feature}")
```

---

## Performance Tips

### Memory Efficient Processing

```python
def process_large_dataset(data, chunk_size=10000):
    """Process large datasets in chunks to manage memory"""
    
    if len(data) <= chunk_size:
        return ct.time_series_summary(data)
    
    # Process in chunks and combine results
    n_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size else 0)
    
    chunk_summaries = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        chunk = data[start_idx:end_idx]
        
        chunk_summary = ct.time_series_summary(chunk)
        chunk_summaries.append(chunk_summary)
    
    # Combine results (simplified - for demonstration)
    combined_mean = sum(s['mean'] * len(data[i*chunk_size:(i+1)*chunk_size]) 
                       for i, s in enumerate(chunk_summaries)) / len(data)
    
    print(f"Processed {len(data)} points in {n_chunks} chunks")
    return combined_mean

# Example with large dataset
large_data = np.random.randn(100000).tolist()
result = process_large_dataset(large_data)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import time

def parallel_analysis(datasets):
    """Analyze multiple datasets in parallel"""
    
    def analyze_single(data):
        return ct.time_series_summary(data)
    
    start_time = time.time()
    
    # Sequential processing
    sequential_results = []
    for data in datasets:
        sequential_results.append(analyze_single(data))
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_results = list(executor.map(analyze_single, datasets))
    parallel_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x")
    
    return parallel_results

# Example
datasets = [np.random.randn(10000).tolist() for _ in range(8)]
results = parallel_analysis(datasets)
```

---

## Real-World Applications

### Stock Market Analysis

```python
def stock_analysis_pipeline(prices, volumes):
    """Complete stock analysis using ChronoXtract"""
    
    # Calculate returns
    returns = [(prices[i] - prices[i-1]) / prices[i-1] 
               for i in range(1, len(prices))]
    
    # Basic statistics
    price_stats = ct.time_series_summary(prices)
    return_stats = ct.time_series_summary(returns)
    
    # Volatility analysis
    volatility = ct.rolling_variance(returns, window=20)  # 20-day volatility
    
    # Volume analysis
    volume_stats = ct.time_series_summary(volumes)
    
    # Trend analysis using EMA
    ema_short = ct.exponential_moving_average(prices, alpha=0.1)
    ema_long = ct.exponential_moving_average(prices, alpha=0.05)
    
    # Signal generation (basic)
    signals = []
    for i in range(max(len(ema_short), len(ema_long))):
        if i < len(ema_short) and i < len(ema_long):
            if ema_short[i] > ema_long[i]:
                signals.append("BUY")
            else:
                signals.append("SELL")
    
    return {
        'price_stats': price_stats,
        'return_stats': return_stats,
        'volatility': volatility,
        'volume_stats': volume_stats,
        'signals': signals[-10:]  # Last 10 signals
    }

# Example usage (mock data)
np.random.seed(42)
prices = [100]
volumes = []
for i in range(252):  # One year of trading
    price_change = np.random.normal(0.001, 0.02)
    new_price = prices[-1] * (1 + price_change)
    prices.append(new_price)
    
    # Volume inversely correlated with price stability
    volume = np.random.lognormal(10, abs(price_change) * 50)
    volumes.append(volume)

analysis = stock_analysis_pipeline(prices, volumes)
print(f"Average daily return: {analysis['return_stats']['mean']:.4f}")
print(f"Return volatility: {analysis['return_stats']['standard_deviation']:.4f}")
print(f"Recent signals: {analysis['signals']}")
```

### Sensor Data Monitoring

```python
def sensor_monitoring_system(sensor_data, timestamps):
    """Real-time sensor monitoring and anomaly detection"""
    
    # Basic health check
    stats = ct.time_series_summary(sensor_data)
    
    # Detect anomalies using entropy
    entropy = ct.sliding_window_entropy(sensor_data, window=50, bins=10)
    
    # Rolling variance for stability monitoring
    stability = ct.rolling_variance(sensor_data, window=30)
    
    # Trend detection
    trend = ct.exponential_moving_average(sensor_data, alpha=0.1)
    
    # Calculate health score
    health_score = 100
    
    # Penalize high variance
    if stability:
        avg_var = sum(stability) / len(stability)
        if avg_var > stats['variance'] * 2:
            health_score -= 20
    
    # Penalize low entropy (stuck sensor)
    if entropy:
        avg_entropy = sum(entropy) / len(entropy)
        if avg_entropy < 1.0:  # Low entropy threshold
            health_score -= 30
    
    # Penalize extreme values
    if (stats['maximum'] > stats['mean'] + 3 * stats['standard_deviation'] or
        stats['minimum'] < stats['mean'] - 3 * stats['standard_deviation']):
        health_score -= 25
    
    return {
        'health_score': max(0, health_score),
        'stats': stats,
        'trend': trend[-10:] if trend else [],
        'anomaly_entropy': avg_entropy if entropy else None,
        'stability_variance': avg_var if stability else None
    }

# Example: Temperature sensor monitoring
np.random.seed(42)
normal_temp = 25  # Celsius
temp_readings = []
for i in range(1000):
    if 100 <= i <= 120:  # Simulate sensor malfunction
        temp = normal_temp + np.random.normal(0, 10)  # High variance
    elif 500 <= i <= 520:  # Simulate stuck sensor
        temp = normal_temp  # No variation
    else:
        temp = normal_temp + np.random.normal(0, 1)  # Normal operation
    temp_readings.append(temp)

timestamps = list(range(len(temp_readings)))
health_report = sensor_monitoring_system(temp_readings, timestamps)
print(f"Sensor health score: {health_report['health_score']}/100")
if health_report['health_score'] < 80:
    print("⚠️  Sensor requires attention!")
```

This user guide provides comprehensive coverage of ChronoXtract's capabilities with practical examples. For specific function details, refer to the [API Reference](api_reference.md), and for working code examples, check the [Examples Gallery](examples/).