# API Reference

This document provides detailed information about all functions available in ChronoXtract.

## Table of Contents

- [Statistical Functions](#statistical-functions)
- [Individual Statistical Functions](#individual-statistical-functions)
- [Rolling Statistics](#rolling-statistics)
- [Frequency Domain Analysis](#frequency-domain-analysis)
- [Variability Analysis](#variability-analysis)
- [Correlation Analysis](#correlation-analysis)
- [Higher-order Statistics](#higher-order-statistics)
- [Entropy and Information Theory](#entropy-and-information-theory)
- [Seasonality and Trend Analysis](#seasonality-and-trend-analysis)
- [Shape and Peak Features](#shape-and-peak-features)
- [Peak Detection](#peak-detection)

---

## Statistical Functions

### `time_series_summary(time_series)`

Calculates a comprehensive statistical summary of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing all statistical measures:
  - `mean`: Arithmetic mean
  - `median`: Middle value when sorted
  - `mode`: Most frequently occurring value
  - `variance`: Variance of the data
  - `standard_deviation`: Standard deviation
  - `skewness`: Measure of asymmetry
  - `kurtosis`: Measure of tail heaviness
  - `minimum`: Smallest value
  - `maximum`: Largest value
  - `range`: Difference between max and min
  - `q05`, `q25`, `q75`, `q95`: Quantiles (5%, 25%, 75%, 95%)
  - `sum`: Sum of all values
  - `absolute_energy`: Sum of squared values

**Example:**
```python
import chronoxtract as ct
import numpy as np

data = np.random.randn(1000).tolist()
summary = ct.time_series_summary(data)

print(f"Mean: {summary['mean']:.4f}")
print(f"Std Dev: {summary['standard_deviation']:.4f}")
print(f"Skewness: {summary['skewness']:.4f}")
```

### `time_series_mean_median_mode(time_series)`

Calculates the three central tendencies of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `tuple`: (mean, median, mode)

**Example:**
```python
data = [1, 2, 2, 3, 4, 4, 4, 5]
mean, median, mode = ct.time_series_mean_median_mode(data)
print(f"Mean: {mean}, Median: {median}, Mode: {mode}")
```

---

## Individual Statistical Functions

### `calculate_mean(time_series)`

Calculates the arithmetic mean of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Arithmetic mean of the time series

**Example:**
```python
import chronoxtract as ct

data = [1.0, 2.0, 3.0, 4.0, 5.0]
mean = ct.calculate_mean(data)
print(f"Mean: {mean}")  # Output: Mean: 3.0
```

### `calculate_median(time_series)`

Calculates the median value of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Median value of the time series

### `calculate_mode(time_series)`

Calculates the mode (most frequently occurring value) of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Most frequently occurring value

### `calculate_variance(time_series)`

Calculates the variance of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Variance of the time series

### `calculate_std_dev(time_series)`

Calculates the standard deviation of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Standard deviation of the time series

### `calculate_skewness(time_series)`

Calculates the skewness (measure of asymmetry) of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Skewness value

### `calculate_kurtosis(time_series)`

Calculates the kurtosis (measure of tail heaviness) of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Kurtosis value

### `calculate_min_max_range(time_series)`

Calculates the minimum, maximum, and range of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing 'min', 'max', and 'range' values

### `calculate_quantiles(time_series)`

Calculates various quantiles of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing quantile values (q05, q25, q75, q95)

### `calculate_sum(time_series)`

Calculates the sum of all values in a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Sum of all values

### `calculate_absolute_energy(time_series)`

Calculates the absolute energy (sum of squared values) of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Absolute energy value

---

## Rolling Statistics

### `rolling_mean(series, window)`

Computes the rolling mean over a sliding window.

**Parameters:**
- `series` (List[float]): Input time series
- `window` (int): Size of the sliding window

**Returns:**
- `List[float]`: Rolling mean values

**Example:**
```python
import matplotlib.pyplot as plt

data = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200)
rolling_avg = ct.rolling_mean(data.tolist(), window=20)

plt.plot(data, label='Original', alpha=0.7)
plt.plot(range(19, len(data)), rolling_avg, label='Rolling Mean', linewidth=2)
plt.legend()
plt.show()
```

### `rolling_variance(series, window)`

Computes the rolling variance over a sliding window.

**Parameters:**
- `series` (List[float]): Input time series
- `window` (int): Size of the sliding window

**Returns:**
- `List[float]`: Rolling variance values

**Example:**
```python
# Detect periods of high volatility
prices = np.random.lognormal(0, 0.02, 1000)
volatility = ct.rolling_variance(prices.tolist(), window=30)

high_vol_periods = [i for i, vol in enumerate(volatility) if vol > np.percentile(volatility, 95)]
print(f"High volatility detected at indices: {high_vol_periods[:5]}...")
```

### `expanding_sum(series)`

Computes the cumulative sum (expanding window sum) of the series.

**Parameters:**
- `series` (List[float]): Input time series

**Returns:**
- `List[float]`: Cumulative sum values

**Example:**
```python
daily_sales = [100, 150, 200, 120, 180]
cumulative_sales = ct.expanding_sum(daily_sales)
print(f"Cumulative sales: {cumulative_sales}")
# Output: [100.0, 250.0, 450.0, 570.0, 750.0]
```

### `exponential_moving_average(series, alpha)`

Computes the Exponential Moving Average (EMA) of the series.

**Parameters:**
- `series` (List[float]): Input time series
- `alpha` (float): Smoothing factor (0 < alpha <= 1). Higher values give more weight to recent observations.

**Returns:**
- `List[float]`: EMA values

**Example:**
```python
# Stock price smoothing
prices = [100, 102, 98, 105, 107, 103, 108, 110]
ema_fast = ct.exponential_moving_average(prices, alpha=0.3)  # Fast EMA
ema_slow = ct.exponential_moving_average(prices, alpha=0.1)  # Slow EMA

print("Fast EMA:", [f"{x:.2f}" for x in ema_fast])
print("Slow EMA:", [f"{x:.2f}" for x in ema_slow])
```

### `sliding_window_entropy(series, window, bins)`

Computes the Shannon entropy over sliding windows using histogram binning.

**Parameters:**
- `series` (List[float]): Input time series
- `window` (int): Size of the sliding window
- `bins` (int): Number of histogram bins for entropy calculation

**Returns:**
- `List[float]`: Entropy values for each window

**Example:**
```python
# Detect complexity changes in signal
signal = np.concatenate([
    np.sin(np.linspace(0, 4*np.pi, 100)),  # Simple sine wave
    np.random.randn(100),                   # Random noise
    np.sin(np.linspace(0, 20*np.pi, 100))  # High frequency sine
])

entropy = ct.sliding_window_entropy(signal.tolist(), window=50, bins=10)

# Plot to visualize complexity changes
plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Signal')
plt.subplot(2, 1, 2)
plt.plot(entropy)
plt.title('Sliding Window Entropy')
plt.show()
```

---

## Frequency Domain Analysis

### `perform_fft_py(input)`

Performs Fast Fourier Transform on the input data.

**Parameters:**
- `input` (List[float]): Input time series data

**Returns:**
- `List[complex]`: FFT coefficients as Python complex numbers

**Example:**
```python
# Analyze frequency components
t = np.linspace(0, 1, 500)
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)  # 50Hz + 120Hz

fft_result = ct.perform_fft_py(signal.tolist())

# Calculate power spectrum
power = [abs(c)**2 for c in fft_result]
freqs = np.fft.fftfreq(len(signal), 1/500)

# Plot power spectrum
plt.plot(freqs[:len(freqs)//2], power[:len(power)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.show()
```

### `lomb_scargle_py(t, y, freqs)`

Computes the Lomb-Scargle periodogram for irregularly sampled data.

**Parameters:**
- `t` (List[float]): Time points (can be irregularly spaced)
- `y` (List[float]): Corresponding data values
- `freqs` (List[float]): Frequencies at which to evaluate the periodogram

**Returns:**
- `List[float]`: Power spectral density values at specified frequencies

**Example:**
```python
# Analyze irregularly sampled astronomical data
np.random.seed(42)
t_regular = np.linspace(0, 10, 200)
t_irregular = np.sort(np.random.choice(t_regular, size=100, replace=False))
y_irregular = 2 * np.sin(2*np.pi*0.5*t_irregular) + np.random.normal(0, 0.5, len(t_irregular))

# Define frequency grid
freqs = np.linspace(0.1, 2.0, 100)

# Compute Lomb-Scargle periodogram
power = ct.lomb_scargle_py(t_irregular.tolist(), y_irregular.tolist(), freqs.tolist())

# Find dominant frequency
dominant_freq = freqs[np.argmax(power)]
print(f"Dominant frequency: {dominant_freq:.2f} Hz")

plt.plot(freqs, power)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.axvline(dominant_freq, color='red', linestyle='--', label=f'Peak: {dominant_freq:.2f} Hz')
plt.legend()
plt.show()
```

---

## Variability Analysis

### `fractional_variability(flux, flux_err)`

Calculates the fractional variability of a time series.

**Parameters:**
- `flux` (List[float]): Flux measurements
- `flux_err` (List[float]): Flux measurement errors

**Returns:**
- `Optional[float]`: Fractional variability value, or None if calculation fails

**Example:**
```python
# Astronomical light curve analysis
flux = np.random.lognormal(0, 0.3, 1000)
flux_err = np.random.uniform(0.01, 0.05, 1000) * flux

fvar = ct.fractional_variability(flux.tolist(), flux_err.tolist())
print(f"Fractional Variability: {fvar:.4f}")
```

### `fractional_variability_error(flux, flux_err)`

Calculates the error in fractional variability measurement.

**Parameters:**
- `flux` (List[float]): Flux measurements
- `flux_err` (List[float]): Flux measurement errors

**Returns:**
- `Optional[float]`: Error in fractional variability, or None if calculation fails

**Example:**
```python
flux = np.random.lognormal(0, 0.2, 500)
flux_err = np.random.uniform(0.01, 0.03, 500) * flux

fvar = ct.fractional_variability(flux.tolist(), flux_err.tolist())
fvar_err = ct.fractional_variability_error(flux.tolist(), flux_err.tolist())

print(f"Fractional Variability: {fvar:.4f} ± {fvar_err:.4f}")
```

### `rolling_fractional_variability(flux, flux_err, window_size)`

Computes rolling fractional variability over a sliding window.

**Parameters:**
- `flux` (List[float]): Flux measurements
- `flux_err` (List[float]): Flux measurement errors
- `window_size` (int): Size of the sliding window

**Returns:**
- `List[Optional[float]]`: Rolling fractional variability values

**Example:**
```python
# Monitor changing variability over time
flux = np.concatenate([
    np.random.lognormal(0, 0.1, 200),  # Low variability period
    np.random.lognormal(0, 0.5, 200),  # High variability period
    np.random.lognormal(0, 0.1, 200)   # Low variability again
])
flux_err = np.random.uniform(0.01, 0.03, len(flux)) * flux

rolling_fvar = ct.rolling_fractional_variability(
    flux.tolist(), flux_err.tolist(), window_size=50
)

# Remove None values for plotting
valid_fvar = [fv for fv in rolling_fvar if fv is not None]

plt.subplot(2, 1, 1)
plt.plot(flux)
plt.title('Flux')
plt.subplot(2, 1, 2)
plt.plot(valid_fvar)
plt.title('Rolling Fractional Variability')
plt.xlabel('Time')
plt.show()
```

### `calc_variability_timescale(time, flux, flux_err)`

Calculates the characteristic variability timescale.

**Parameters:**
- `time` (List[float]): Time points
- `flux` (List[float]): Flux measurements
- `flux_err` (List[float]): Flux measurement errors

**Returns:**
- `Optional[float]`: Variability timescale, or None if calculation fails

**Example:**
```python
# Generate time series with known variability timescale
dt = 0.1
time = np.arange(0, 100, dt)
# Create correlated noise with ~10 day timescale
from scipy.stats import norm
kernel = norm.pdf(np.arange(-30, 31), 0, 10)
kernel /= kernel.sum()
noise = np.convolve(np.random.randn(len(time) + 60), kernel, mode='valid')
flux = 1 + 0.3 * noise
flux_err = np.full_like(flux, 0.02)

timescale = ct.calc_variability_timescale(time.tolist(), flux.tolist(), flux_err.tolist())
print(f"Estimated variability timescale: {timescale:.2f} days")
```

---

## Error Handling

All functions return appropriate error types when given invalid input:
- Empty lists will return empty results or None
- Mismatched array lengths will raise appropriate errors
- Invalid window sizes (0 or larger than data) will return empty results

## Performance Notes

- All functions are implemented in Rust for optimal performance
- Large datasets (> 1M points) are handled efficiently
- Memory usage is optimized for streaming calculations where possible

For more examples and tutorials, see the [Examples Gallery](examples/) and [User Guide](user_guide.md).

---

## Correlation Analysis

### `acf_py(t, v, e, lag_min, lag_max, lag_bin_width)`

Calculates the Auto-Correlation Function (ACF) of a time series.

**Parameters:**
- `t` (List[float]): Time points
- `v` (List[float]): Value points
- `e` (List[float]): Error points
- `lag_min` (float): Minimum lag
- `lag_max` (float): Maximum lag
- `lag_bin_width` (float): Width of the lag bins

**Returns:**
- `dict`: Dictionary containing:
  - `lags`: List of lags
  - `correlations`: List of correlation values

**Example:**
```python
import numpy as np
import chronoxtract as ct

period = 20
t = np.linspace(0, 100, 100)
v = np.sin(2 * np.pi * t / period)
e = np.random.rand(100) * 0.1

result = ct.acf_py(t.tolist(), v.tolist(), e.tolist(), lag_min=0, lag_max=40, lag_bin_width=0.5)
```

### `dcf_py(t1, v1, e1, t2, v2, e2, lag_min, lag_max, lag_bin_width)`

Calculates the Discrete Correlation Function (DCF) between two time series.

**Parameters:**
- `t1`, `v1`, `e1` (List[float]): Time, value, and error for the first time series
- `t2`, `v2`, `e2` (List[float]): Time, value, and error for the second time series
- `lag_min` (float): Minimum lag
- `lag_max` (float): Maximum lag
- `lag_bin_width` (float): Width of the lag bins

**Returns:**
- `dict`: Dictionary containing:
  - `lags`: List of lags
  - `correlations`: List of correlation values

**Example:**
```python
import numpy as np
import chronoxtract as ct

t1 = np.linspace(0, 100, 100)
v1 = np.sin(t1)
e1 = np.random.rand(100) * 0.1

lag = 10
t2 = t1 + lag
v2 = np.sin(t1)
e2 = np.random.rand(100) * 0.1

result = ct.dcf_py(t1.tolist(), v1.tolist(), e1.tolist(), t2.tolist(), v2.tolist(), e2.tolist(), lag_min=-20, lag_max=20, lag_bin_width=0.5)
```

### `zdcf_py(t1, v1, e1, t2, v2, e2, min_points, num_mc)`

Calculates the Z-transformed Discrete Correlation Function (ZDCF) between two time series.

**Parameters:**
- `t1`, `v1`, `e1` (List[float]): Time, value, and error for the first time series
- `t2`, `v2`, `e2` (List[float]): Time, value, and error for the second time series
- `min_points` (int): Minimum number of points in a bin
- `num_mc` (int): Number of Monte Carlo simulations

**Returns:**
- `dict`: Dictionary containing:
  - `lags`: List of lags
  - `correlations`: List of correlation values

**Example:**
```python
import numpy as np
import chronoxtract as ct

t1 = np.linspace(0, 100, 100)
v1 = np.sin(t1)
e1 = np.random.rand(100) * 0.1

lag = 10
t2 = t1 + lag
v2 = np.sin(t1)
e2 = np.random.rand(100) * 0.1

result = ct.zdcf_py(t1.tolist(), v1.tolist(), e1.tolist(), t2.tolist(), v2.tolist(), e2.tolist(), min_points=11, num_mc=100)
```

---

## Higher-order Statistics

### `hjorth_parameters(time_series)`

Calculates all three Hjorth parameters (activity, mobility, complexity) of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing 'activity', 'mobility', and 'complexity' values

**Example:**
```python
import chronoxtract as ct
import numpy as np

# Generate a complex signal
t = np.linspace(0, 10, 1000)
signal = np.sin(2*np.pi*t) + 0.5*np.sin(6*np.pi*t) + 0.2*np.random.randn(1000)

hjorth = ct.hjorth_parameters(signal.tolist())
print(f"Activity: {hjorth['activity']:.4f}")
print(f"Mobility: {hjorth['mobility']:.4f}")
print(f"Complexity: {hjorth['complexity']:.4f}")
```

### `hjorth_activity(time_series)`

Calculates the Hjorth activity parameter (variance of the signal).

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Activity parameter value

### `hjorth_mobility(time_series)`

Calculates the Hjorth mobility parameter (measure of mean frequency).

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Mobility parameter value

### `hjorth_complexity(time_series)`

Calculates the Hjorth complexity parameter (measure of frequency distribution).

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Complexity parameter value

### `higher_moments(time_series)`

Calculates higher-order statistical moments of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing higher-order moment values

### `central_moment_5(time_series)` to `central_moment_8(time_series)`

Calculate the 5th through 8th central moments of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Central moment value

---

## Entropy and Information Theory

### `sample_entropy(time_series, m, r)`

Calculates the sample entropy of a time series, measuring signal regularity.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `m` (int): Pattern length
- `r` (float): Tolerance for matching

**Returns:**
- `float`: Sample entropy value

**Example:**
```python
import chronoxtract as ct
import numpy as np

# Regular signal has low entropy
regular = (np.sin(np.linspace(0, 4*np.pi, 100)) * 100).tolist()
entropy_regular = ct.sample_entropy(regular, m=2, r=0.2)

# Random signal has high entropy
random = (np.random.randn(100) * 100).tolist()
entropy_random = ct.sample_entropy(random, m=2, r=0.2)

print(f"Regular signal entropy: {entropy_regular:.4f}")
print(f"Random signal entropy: {entropy_random:.4f}")
```

### `approximate_entropy(time_series, m, r)`

Calculates the approximate entropy of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `m` (int): Pattern length
- `r` (float): Tolerance for matching

**Returns:**
- `float`: Approximate entropy value

### `permutation_entropy(time_series, order)`

Calculates the permutation entropy based on ordinal patterns.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `order` (int): Order of permutation patterns

**Returns:**
- `float`: Permutation entropy value

### `lempel_ziv_complexity(time_series)`

Calculates the Lempel-Ziv complexity of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Lempel-Ziv complexity value

### `multiscale_entropy(time_series, max_scale, m, r)`

Calculates multiscale entropy across different time scales.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `max_scale` (int): Maximum scale factor
- `m` (int): Pattern length
- `r` (float): Tolerance for matching

**Returns:**
- `List[float]`: Entropy values at different scales

---

## Seasonality and Trend Analysis

### `seasonal_trend_strength(time_series, period)`

Calculates both seasonal and trend strength of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `period` (int): Seasonal period

**Returns:**
- `dict`: Dictionary containing 'seasonal_strength' and 'trend_strength' values

**Example:**
```python
import chronoxtract as ct
import numpy as np

# Generate time series with trend and seasonality
t = np.linspace(0, 100, 1000)
trend = 0.02 * t
seasonal = 2 * np.sin(2*np.pi*t/10)  # Period of 10
noise = 0.5 * np.random.randn(1000)
ts = trend + seasonal + noise

strengths = ct.seasonal_trend_strength(ts.tolist(), period=100)
print(f"Seasonal strength: {strengths['seasonal_strength']:.4f}")
print(f"Trend strength: {strengths['trend_strength']:.4f}")
```

### `seasonal_strength(time_series, period)`

Calculates the seasonal strength of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `period` (int): Seasonal period

**Returns:**
- `float`: Seasonal strength value

### `trend_strength(time_series)`

Calculates the trend strength of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Trend strength value

### `simple_stl_decomposition(time_series, period)`

Performs a simple STL (Seasonal and Trend decomposition using Loess) decomposition.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `period` (int): Seasonal period

**Returns:**
- `dict`: Dictionary containing 'trend', 'seasonal', and 'remainder' components

### `detect_seasonality(time_series)`

Detects potential seasonal patterns in a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing seasonality detection results

### `detrended_fluctuation_analysis(time_series)`

Performs detrended fluctuation analysis to detect long-range correlations.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing DFA results including scaling exponent

---

## Shape and Peak Features

### `zero_crossing_rate(time_series)`

Calculates the zero crossing rate of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Zero crossing rate value

**Example:**
```python
import chronoxtract as ct
import numpy as np

# High frequency signal has high zero crossing rate
high_freq = np.sin(20 * np.linspace(0, 2*np.pi, 1000))
zcr_high = ct.zero_crossing_rate(high_freq.tolist())

# Low frequency signal has low zero crossing rate
low_freq = np.sin(2 * np.linspace(0, 2*np.pi, 1000))
zcr_low = ct.zero_crossing_rate(low_freq.tolist())

print(f"High freq ZCR: {zcr_high:.4f}")
print(f"Low freq ZCR: {zcr_low:.4f}")
```

### `slope_features(time_series)`

Calculates comprehensive slope-based features of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing various slope-based features

### `mean_slope(time_series)`

Calculates the mean slope of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Mean slope value

### `slope_variance(time_series)`

Calculates the variance of slopes in a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Slope variance value

### `max_slope(time_series)`

Calculates the maximum absolute slope in a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Maximum slope value

### `enhanced_peak_stats(time_series)`

Calculates enhanced peak statistics including various peak-related measures.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing comprehensive peak statistics

### `peak_to_peak_amplitude(time_series)`

Calculates the peak-to-peak amplitude of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `float`: Peak-to-peak amplitude value

### `variability_features(time_series)`

Calculates comprehensive variability features of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing various variability measures

### `turning_points(time_series)`

Identifies and counts turning points in a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing turning point statistics

### `energy_distribution(time_series)`

Calculates the energy distribution characteristics of a time series.

**Parameters:**
- `time_series` (List[float]): Input time series data

**Returns:**
- `dict`: Dictionary containing energy distribution measures

---

## Peak Detection

### `find_peaks(time_series, height, distance)`

Finds peaks in a time series based on height and distance criteria.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `height` (float): Minimum peak height
- `distance` (int): Minimum distance between peaks

**Returns:**
- `List[int]`: Indices of detected peaks

**Example:**
```python
import chronoxtract as ct
import numpy as np

# Generate signal with clear peaks
t = np.linspace(0, 10, 1000)
signal = np.sin(t) + 0.5*np.sin(3*t) + 0.1*np.random.randn(1000)

peaks = ct.find_peaks(signal.tolist(), height=0.5, distance=20)
print(f"Found {len(peaks)} peaks at indices: {peaks[:5]}...")
```

### `peak_prominence(time_series, peaks)`

Calculates the prominence of detected peaks.

**Parameters:**
- `time_series` (List[float]): Input time series data
- `peaks` (List[int]): Indices of peaks

**Returns:**
- `List[float]`: Prominence values for each peak

### `variability_statistics(flux, flux_err)`

Calculates comprehensive variability statistics for astronomical or financial time series.

**Parameters:**
- `flux` (List[float]): Flux or value measurements
- `flux_err` (List[float]): Error values

**Returns:**
- `dict`: Dictionary containing comprehensive variability statistics

---

## Error Handling

All functions include comprehensive error handling for:
- Empty input arrays
- Invalid parameter values
- Numerical stability issues
- Type conversion errors

When errors occur, functions will raise appropriate Python exceptions with descriptive messages.

---

## Performance Notes

- All functions are implemented in Rust for optimal performance
- Memory usage is optimized for large datasets
- Functions are thread-safe and can be used in parallel processing
- Input validation is performed efficiently without significant overhead
- For very large datasets (>1M points), consider using streaming algorithms where available