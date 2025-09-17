# CARMA Module Documentation

## Overview

The CARMA (Continuous-time AutoRegressive Moving Average) module provides a complete, high-performance implementation for modeling irregularly sampled time series data. This implementation features modern Rust algorithms with Python bindings for scientific computing.

## Key Features

### 🚀 **High-Performance Implementation**
- **Memory-safe Rust core**: Zero unsafe code with compile-time guarantees
- **Optimized algorithms**: Efficient Kalman filtering and matrix operations
- **Parallel processing**: Multi-core utilization via rayon
- **Scalable**: Handles large datasets with minimal memory overhead

### 📊 **Complete Statistical Framework**
- **Maximum Likelihood Estimation (MLE)**: Multi-start optimization for robust parameter estimation
- **MCMC Sampling**: Full Bayesian inference with parallel tempering
- **Model Selection**: Automated order selection using information criteria
- **Convergence Diagnostics**: R-hat, effective sample size, and acceptance rates

### 🔬 **Scientific Rigor**
- **Irregular Sampling**: Native support for uneven time intervals
- **Numerical Stability**: Robust algorithms for ill-conditioned problems
- **Parameter Validation**: Comprehensive input checking and error handling
- **Reproducible Results**: Proper random seed handling

## Mathematical Background

### CARMA(p,q) Model

A CARMA(p,q) process represents a continuous-time stochastic process where:
- **p**: Autoregressive order (must be > 0)
- **q**: Moving average order (must be < p)

The process is defined by:
- **AR polynomial**: α(s) of order p
- **MA polynomial**: β(s) of order q
- **Power Spectral Density**: P(f) = σ² |β(2πif)|² / |α(2πif)|²

### State-Space Representation

The implementation uses a rotated state-space representation where:
- State transition matrix is diagonal (eigenvalues of AR polynomial)
- Efficient computation of matrix exponentials
- Optimal numerical stability

## API Reference

### Core Classes

#### `CarmaParams`
Standard CARMA model parameterization.

```python
import chronoxtract as ct

# Create CARMA(2,1) model
params = ct.CarmaParams(p=2, q=1)

# Set parameters
params.ar_coeffs = [0.8, 0.3]    # AR coefficients
params.ma_coeffs = [1.0, 0.4]    # MA coefficients (first is always 1.0)
params.sigma = 1.5               # Process noise standard deviation

# Validate parameters
params.validate()

# Compute AR polynomial roots
roots = params.ar_roots()

# Check stationarity
is_stationary = params.is_stationary()
```

#### `McmcParams`
Specialized parameterization for MCMC sampling.

```python
# Create MCMC parameters
mcmc_params = ct.McmcParams(p=2, q=1)

# Set MCMC-specific parameters
mcmc_params.ysigma = 1.2           # Process standard deviation
mcmc_params.measerr_scale = 1.0    # Measurement error scaling
mcmc_params.mu = 0.5               # Mean level
mcmc_params.ar_params = [0.1, 0.2] # AR parameters (not coefficients)
mcmc_params.ma_params = [0.3]      # MA parameters

# Convert to standard parameterization
carma_params = mcmc_params.to_carma_params()
```

### Maximum Likelihood Estimation

#### `carma_mle(times, values, errors, p, q, **kwargs)`

Performs robust maximum likelihood estimation using parallel multi-start optimization.

```python
import numpy as np

# Prepare data
times = np.array([0.0, 1.2, 2.8, 4.1, 5.9])
values = np.array([1.0, 1.5, 0.8, 1.2, 0.9])
errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

# Run MLE estimation
mle_result = ct.carma_mle(
    times, values, errors,
    p=2, q=1,                    # Model order
    n_starts=8,                  # Number of random starting points
    max_iter=1000                # Maximum iterations per start
)

# Access results
print(f"Log-likelihood: {mle_result.loglikelihood}")
print(f"AICc: {mle_result.aicc}")
print(f"BIC: {mle_result.bic}")
print(f"Converged: {mle_result.converged}")
print(f"AR coefficients: {mle_result.params.ar_coeffs}")
print(f"MA coefficients: {mle_result.params.ma_coeffs}")
print(f"Sigma: {mle_result.params.sigma}")
```

**Parameters:**
- `times`: Observation times (irregular spacing supported)
- `values`: Observed values
- `errors`: Measurement error standard deviations
- `p`: Autoregressive order
- `q`: Moving average order (q < p)
- `n_starts`: Number of random starting points (default: 8)
- `max_iter`: Maximum optimization iterations (default: 1000)

**Returns:** `CarmaMLEResult`
- `params`: Fitted CARMA parameters
- `loglikelihood`: Maximum log-likelihood value
- `aic`, `aicc`, `bic`: Information criteria
- `iterations`: Number of iterations used
- `converged`: Whether optimization converged

### MCMC Sampling

#### `carma_mcmc(times, values, errors, p, q, n_samples, **kwargs)`

Performs Bayesian inference using adaptive Metropolis-Hastings MCMC with parallel tempering.

```python
# Run MCMC sampling
mcmc_result = ct.carma_mcmc(
    times, values, errors,
    p=2, q=1,                    # Model order
    n_samples=5000,              # Post burn-in samples
    n_burn=2000,                 # Burn-in samples
    n_chains=6,                  # Parallel tempering chains
    seed=42                      # Random seed for reproducibility
)

# Access results
print(f"Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
print(f"Sample shape: {mcmc_result.samples.shape}")
print(f"R-hat convergence: {np.max(mcmc_result.rhat):.3f}")
print(f"Min ESS: {np.min(mcmc_result.effective_sample_size):.1f}")

# Extract parameter samples
samples = mcmc_result.samples
n_params = samples.shape[1]

# Parameter order: ar_params, ma_params, log(ysigma), log(measerr_scale), mu
p_ar = p
q_ma = q
ar_samples = samples[:, :p_ar]
ma_samples = samples[:, p_ar:p_ar+q_ma]
ysigma_samples = np.exp(samples[:, p_ar+q_ma])
measerr_scale_samples = np.exp(samples[:, p_ar+q_ma+1])
mu_samples = samples[:, -1]
```

**Parameters:**
- `times`, `values`, `errors`: Data arrays
- `p`, `q`: Model orders
- `n_samples`: Number of post burn-in samples
- `n_burn`: Number of burn-in samples (default: n_samples/4)
- `n_chains`: Number of parallel tempering chains (default: 4)
- `seed`: Random seed (default: None)

**Returns:** `CarmaMCMCResult`
- `samples`: Parameter samples from cold chain [n_samples × n_params]
- `loglikelihoods`: Log-likelihood values for each sample
- `acceptance_rate`: Acceptance rate for cold chain
- `rhat`: R-hat convergence diagnostic for each parameter
- `effective_sample_size`: Effective sample size for each parameter
- `n_samples`, `n_burn`: Sample counts
- `p`, `q`: Model orders

### Model Order Selection

#### `carma_choose_order(times, values, errors, max_p, max_q)`

Automatically selects optimal CARMA model order using corrected Akaike Information Criterion (AICc).

```python
# Automatic order selection
order_result = ct.carma_choose_order(
    times, values, errors,
    max_p=3,                     # Maximum AR order to test
    max_q=2                      # Maximum MA order to test
)

# Access results
print(f"Best model: CARMA({order_result.best_p}, {order_result.best_q})")
print(f"Best AICc: {order_result.best_aicc:.4f}")

# Access full results grid
import matplotlib.pyplot as plt
aicc_grid = order_result.aicc_grid
p_values = order_result.p_values  
q_values = order_result.q_values

# Visualize AICc landscape
plt.imshow(aicc_grid, aspect='auto')
plt.xlabel('q')
plt.ylabel('p')
plt.title('AICc Model Selection')
plt.colorbar()
plt.show()
```

**Parameters:**
- `times`, `values`, `errors`: Data arrays
- `max_p`: Maximum autoregressive order to test
- `max_q`: Maximum moving average order to test

**Returns:** `CarmaOrderResult`
- `best_p`, `best_q`: Optimal model orders
- `best_aicc`: AICc value for best model
- `aicc_grid`: Full grid of AICc values [p × q]
- `p_values`, `q_values`: Tested order ranges

### Utility Functions

#### `carma_loglikelihood(params, times, values, errors)`

Computes log-likelihood for given parameters and data.

```python
# Compute log-likelihood
loglik = ct.carma_loglikelihood(params, times, values, errors)
print(f"Log-likelihood: {loglik}")
```

## Usage Examples

### Example 1: Basic CARMA Analysis

```python
import chronoxtract as ct
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic irregular time series
np.random.seed(42)
n_points = 100
times = np.sort(np.random.uniform(0, 50, n_points))
true_values = np.sin(0.1 * times) + 0.3 * np.sin(0.05 * times)
noise = np.random.normal(0, 0.2, n_points)
values = true_values + noise
errors = np.full(n_points, 0.2)

# Plot the data
plt.figure(figsize=(12, 4))
plt.errorbar(times, values, yerr=errors, fmt='o', alpha=0.6)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Irregular Time Series Data')
plt.show()

# Automatic model selection
print("Selecting optimal CARMA order...")
order_result = ct.carma_choose_order(times, values, errors, max_p=3, max_q=2)
p_opt, q_opt = order_result.best_p, order_result.best_q
print(f"Optimal model: CARMA({p_opt}, {q_opt})")

# Maximum likelihood estimation
print("Fitting model with MLE...")
mle_result = ct.carma_mle(times, values, errors, p_opt, q_opt, n_starts=10)
print(f"MLE AICc: {mle_result.aicc:.4f}")
print(f"AR coefficients: {mle_result.params.ar_coeffs}")
print(f"MA coefficients: {mle_result.params.ma_coeffs}")

# Bayesian inference with MCMC
print("Running MCMC for full posterior...")
mcmc_result = ct.carma_mcmc(
    times, values, errors, p_opt, q_opt,
    n_samples=3000, n_burn=1000, n_chains=4, seed=123
)

print(f"MCMC acceptance rate: {mcmc_result.acceptance_rate:.3f}")
print(f"Convergence (max R-hat): {np.max(mcmc_result.rhat):.3f}")
print(f"Min effective sample size: {np.min(mcmc_result.effective_sample_size):.0f}")

# Plot MCMC diagnostics
samples = mcmc_result.samples
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Trace plots
for i in range(min(4, samples.shape[1])):
    row, col = i // 2, i % 2
    axes[row, col].plot(samples[:, i])
    axes[row, col].set_title(f'Parameter {i+1} Trace')
    axes[row, col].set_xlabel('Iteration')

plt.tight_layout()
plt.show()
```

### Example 2: Comparison with Known Parameters

```python
# Generate data from known CARMA process
def generate_carma_data(times, ar_coeffs, ma_coeffs, sigma, seed=None):
    """Generate synthetic CARMA data with known parameters"""
    if seed:
        np.random.seed(seed)
    
    n = len(times)
    values = np.zeros(n)
    
    # Simple AR(1) approximation for demonstration
    if len(ar_coeffs) == 1:
        phi = ar_coeffs[0]
        for i in range(1, n):
            dt = times[i] - times[i-1]
            values[i] = values[i-1] * np.exp(-phi * dt) + np.random.normal(0, sigma * np.sqrt(dt))
    
    return values

# True parameters
true_ar = [0.5]
true_ma = [1.0]
true_sigma = 1.0

# Generate data
times = np.linspace(0, 20, 50)
true_values = generate_carma_data(times, true_ar, true_ma, true_sigma, seed=456)
errors = np.full(len(times), 0.1)

# Add measurement noise
values = true_values + np.random.normal(0, errors)

# Fit with MLE
mle_result = ct.carma_mle(times, values, errors, 1, 0, n_starts=8)

# Compare results
print("Parameter Recovery Test:")
print(f"True AR coefficient: {true_ar[0]:.3f}")
print(f"MLE AR coefficient:  {mle_result.params.ar_coeffs[0]:.3f}")
print(f"True sigma:          {true_sigma:.3f}")
print(f"MLE sigma:           {mle_result.params.sigma:.3f}")

# MCMC for uncertainty quantification
mcmc_result = ct.carma_mcmc(times, values, errors, 1, 0, 
                           n_samples=2000, n_burn=500, seed=789)

# Extract AR parameter samples (first parameter)
ar_samples = mcmc_result.samples[:, 0]
print(f"\nMCMC Results:")
print(f"AR coefficient mean: {np.mean(ar_samples):.3f} ± {np.std(ar_samples):.3f}")
print(f"95% credible interval: [{np.percentile(ar_samples, 2.5):.3f}, {np.percentile(ar_samples, 97.5):.3f}]")
```

### Example 3: Performance Benchmarking

```python
import time

def benchmark_carma_performance():
    """Benchmark CARMA implementation across different data sizes"""
    
    sizes = [50, 100, 200, 500, 1000]
    mle_times = []
    mcmc_times = []
    
    for n in sizes:
        print(f"\nBenchmarking with {n} data points...")
        
        # Generate test data
        np.random.seed(n)
        times = np.sort(np.random.uniform(0, 100, n))
        values = np.random.normal(0, 1, n)
        errors = np.full(n, 0.1)
        
        # Benchmark MLE
        start = time.time()
        try:
            mle_result = ct.carma_mle(times, values, errors, 1, 0, 
                                     n_starts=4, max_iter=100)
            mle_time = time.time() - start
            mle_times.append(mle_time)
            print(f"  MLE: {mle_time:.3f}s")
        except Exception as e:
            print(f"  MLE: FAILED ({e})")
            mle_times.append(None)
        
        # Benchmark MCMC (scaled down for larger datasets)
        n_samples = max(200, 1000 - n)  # Fewer samples for larger datasets
        start = time.time()
        try:
            mcmc_result = ct.carma_mcmc(times, values, errors, 1, 0,
                                       n_samples=n_samples, n_burn=n_samples//4, 
                                       n_chains=2)
            mcmc_time = time.time() - start
            mcmc_times.append(mcmc_time)
            print(f"  MCMC: {mcmc_time:.3f}s ({n_samples} samples)")
        except Exception as e:
            print(f"  MCMC: FAILED ({e})")
            mcmc_times.append(None)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    valid_mle = [(s, t) for s, t in zip(sizes, mle_times) if t is not None]
    valid_mcmc = [(s, t) for s, t in zip(sizes, mcmc_times) if t is not None]
    
    if valid_mle:
        s_mle, t_mle = zip(*valid_mle)
        plt.loglog(s_mle, t_mle, 'o-', label='MLE', linewidth=2)
    
    if valid_mcmc:
        s_mcmc, t_mcmc = zip(*valid_mcmc)
        plt.loglog(s_mcmc, t_mcmc, 's-', label='MCMC', linewidth=2)
    
    plt.xlabel('Number of Data Points')
    plt.ylabel('Computation Time (seconds)')
    plt.title('CARMA Performance Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return sizes, mle_times, mcmc_times

# Run benchmark
benchmark_results = benchmark_carma_performance()
```

## Best Practices

### Data Preparation
1. **Irregular Sampling**: The implementation naturally handles irregular time intervals
2. **Error Estimation**: Provide realistic measurement error estimates
3. **Data Quality**: Remove obvious outliers before fitting
4. **Time Units**: Use consistent time units throughout

### Model Selection
1. **Start Simple**: Begin with low-order models (CARMA(1,0), CARMA(2,1))
2. **Use AICc**: Prefer AICc over AIC for small datasets
3. **Cross-Validation**: Validate selected models on held-out data
4. **Physical Constraints**: Consider physical meaning of parameters

### Parameter Estimation
1. **MLE First**: Use MLE for initial parameter estimates
2. **Multiple Starts**: Use sufficient random starting points (8-16)
3. **Convergence**: Check that optimization converged
4. **MCMC Validation**: Use MCMC to assess parameter uncertainty

### MCMC Sampling
1. **Burn-in**: Use adequate burn-in (≥25% of total samples)
2. **Convergence**: Monitor R-hat < 1.1 for all parameters
3. **Effective Sample Size**: Aim for ESS > 400 per parameter
4. **Parallel Tempering**: Use multiple chains for complex posteriors

### Performance Optimization
1. **Data Size**: Consider computational cost for large datasets (>1000 points)
2. **Model Complexity**: Higher-order models require more computation
3. **Parallel Processing**: Implementation automatically uses multiple cores
4. **Memory Usage**: Monitor memory usage for very large datasets

## Error Handling

The implementation provides comprehensive error checking:

```python
try:
    result = ct.carma_mle(times, values, errors, p, q)
except Exception as e:
    if "InvalidOrder" in str(e):
        print("Invalid model order: ensure p > 0 and q < p")
    elif "InvalidData" in str(e):
        print("Data validation failed: check array lengths and values")
    elif "NonStationaryError" in str(e):
        print("Model is non-stationary: try different parameters")
    elif "OptimizationFailed" in str(e):
        print("Optimization failed: try more starting points or iterations")
    else:
        print(f"Unexpected error: {e}")
```

## Advanced Topics

### State-Space Formulation
The implementation uses an efficient rotated state-space representation:
- Diagonal state transition matrix (AR polynomial eigenvalues)
- Optimal numerical conditioning
- Fast matrix exponential computation

### Parallel Tempering MCMC
The MCMC implementation includes sophisticated features:
- Multiple temperature chains for multimodal exploration
- Adaptive proposal covariance during burn-in
- Automatic temperature swap proposals
- Cold chain extraction for final results

### Numerical Stability
Key numerical considerations:
- Robust eigenvalue computation for AR roots
- Condition number checking for matrices
- Overflow/underflow prevention in log-likelihood
- Proper handling of edge cases

## References

1. Brockwell, P. J., & Davis, R. A. (1991). Time Series: Theory and Methods. Springer.
2. Kelly, B. C., Bechtold, J., & Siemiginowska, A. (2009). Are the Variations in Quasar Optical Flux Driven by Thermal Fluctuations? ApJ, 698, 895.
3. Ambikasaran, S., Foreman-Mackey, D., Greengard, L., Hogg, D. W., & O'Neil, M. (2015). Fast Direct Methods for Gaussian Processes. PAMI, 38, 252.

For more information and updates, visit: https://github.com/amanasci/ChronoXtract