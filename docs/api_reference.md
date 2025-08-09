# API Reference

This document provides detailed information about all functions available in ChronoXtract.

## Table of Contents

- [Statistical Functions](#statistical-functions)
- [Rolling Statistics](#rolling-statistics)
- [Frequency Domain Analysis](#frequency-domain-analysis)
- [Variability Analysis](#variability-analysis)

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