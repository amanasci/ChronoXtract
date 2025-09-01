# CARMA Module Documentation

The CARMA (Continuous AutoRegressive Moving Average) module provides state-of-the-art time series modeling capabilities for irregularly sampled data.

## Overview

CARMA models are continuous-time generalizations of ARMA models, particularly suited for:
- Irregularly sampled time series
- Astronomical time series (light curves)
- Financial time series with gaps
- Any time series where traditional ARMA models are insufficient

A CARMA(p,q) model satisfies:
```
L[y(t)] = R[ε(t)]
```
where L is a p-th order differential operator, R is a q-th order polynomial, and ε(t) is white noise.

## Quick Start

```python
import numpy as np
import chronoxtract as ct

# Create a CARMA(2,1) model
model = ct.carma_model(2, 1)
ct.set_carma_parameters(model, [0.3, 0.1], [1.0, 0.4], 1.5)

# Generate irregular time series
times, values = ct.generate_irregular_carma(
    model, duration=20.0, mean_sampling_rate=2.0, 
    sampling_noise=0.3, seed=42
)

# Fit a model to data
result = ct.carma_mle(times, values, 2, 1)
print(f"Fitted model: {result}")
```

## Core Functions

### Model Creation and Configuration

- **`carma_model(p, q)`** - Create a CARMA(p,q) model
- **`set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma)`** - Set model parameters

### Parameter Estimation

- **`carma_mle(times, values, p, q, ...)`** - Maximum likelihood estimation
- **`carma_method_of_moments(times, values, p, q)`** - Method of moments estimation
- **`carma_mcmc(times, values, p, q, n_samples, ...)`** - MCMC sampling (simplified)

### Simulation

- **`simulate_carma(model, times, ...)`** - Simulate CARMA process at given times
- **`generate_irregular_carma(model, duration, mean_rate, ...)`** - Generate irregularly sampled data

### Analysis & Diagnostics

- **`carma_psd(model, frequencies)`** - Power spectral density
- **`carma_covariance(model, time_lags)`** - Covariance function
- **`carma_loglikelihood(model, times, values, ...)`** - Log-likelihood
- **`carma_residuals(model, times, values, ...)`** - Residual analysis

### Prediction & Filtering

- **`carma_predict(model, times, values, prediction_times, ...)`** - Forecasting
- **`carma_kalman_filter(model, times, values, ...)`** - Kalman filtering

### Model Selection

- **`carma_information_criteria(times, values, max_p, max_q, ...)`** - AIC/BIC comparison
- **`carma_cross_validation(times, values, p, q, n_folds, ...)`** - Cross-validation

### Utilities

- **`check_carma_stability(model)`** - Check if model is stable
- **`carma_to_state_space(model)`** - Convert to state-space representation
- **`carma_characteristic_roots(model)`** - Get characteristic polynomial roots

## Example: Complete Workflow

```python
import numpy as np
import chronoxtract as ct

# 1. Generate synthetic data
model_true = ct.carma_model(2, 1)
ct.set_carma_parameters(model_true, [0.3, 0.1], [1.0, 0.4], 1.5)

times, values = ct.generate_irregular_carma(
    model_true, duration=50.0, mean_sampling_rate=1.0, seed=42
)

# 2. Model selection
ic_result = ct.carma_information_criteria(times, values, 3, 2)
best_p, best_q = ic_result.best_aic
print(f"Best model: CARMA({best_p}, {best_q})")

# 3. Fit the selected model
fitted_result = ct.carma_mle(times, values, best_p, best_q)
fitted_model = fitted_result.model

# 4. Validate the model
residuals = ct.carma_residuals(fitted_model, times, values)
print(f"Ljung-Box p-value: {residuals.ljung_box_pvalue:.3f}")

# 5. Make predictions
n_train = int(0.8 * len(times))
train_times, train_values = times[:n_train], values[:n_train]
test_times = times[n_train:]

predictions = ct.carma_predict(fitted_model, train_times, train_values, test_times)
print(f"Prediction RMSE: {np.sqrt(np.mean((predictions.mean - values[n_train:])**2)):.3f}")

# 6. Analyze spectral properties
frequencies = np.logspace(-2, 0, 50)
psd = ct.carma_psd(fitted_model, frequencies)
```

## Performance Notes

The CARMA module is optimized for:
- **Fast simulation**: >1M points/second for typical models
- **Efficient PSD computation**: >2M frequencies/second
- **Scalable fitting**: Handles datasets up to 10K+ points efficiently

## Data Structures

### CarmaModel
Main model object with attributes:
- `p`, `q`: Model orders
- `ar_coeffs`, `ma_coeffs`: Model coefficients
- `sigma`: Noise parameter

### CarmaFitResult
Result from parameter estimation:
- `model`: Fitted CarmaModel
- `loglikelihood`: Log-likelihood value
- `aic`, `bic`: Information criteria
- `convergence_info`: Optimization details

### Other Result Objects
- `CarmaPrediction`: Prediction results with confidence intervals
- `KalmanResult`: Kalman filter output
- `CarmaResiduals`: Residual analysis results
- `InformationCriteriaResult`: Model selection results

## Implementation Details

The module uses:
- **State-space representation** for efficient Kalman filtering
- **Nalgebra** for linear algebra operations
- **Matrix exponential** for irregular time step handling
- **PyO3** for seamless Python integration

## Limitations

Current implementation:
- Simplified optimization (basic method instead of full L-BFGS)
- Limited to small model orders (p ≤ 8 recommended)
- MCMC uses simplified random walk (placeholder)
- Some advanced diagnostics are approximated

## References

1. Kelly, B. C., Bechtold, J., & Siemiginowska, A. (2009). "Are the Variations in Quasar Optical Flux Driven by Thermal Fluctuations?" ApJ, 698, 895.
2. Brockwell, P. J., & Davis, R. A. (2016). "Introduction to Time Series and Forecasting." Springer.
3. Brockwell, P. J. (2001). "Lévy-driven CARMA processes." Annals of the Institute of Statistical Mathematics, 53(1), 113-124.

## Testing

Run the test scripts to verify functionality:

```bash
# Basic functionality test
python test_carma_basic.py

# Comprehensive test
python test_carma_comprehensive.py

# Performance benchmark
python benchmark_carma.py
```