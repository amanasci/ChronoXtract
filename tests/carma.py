# 1. Basic Parameter Creation
import chronoxtract as ct
import numpy as np

# Create CARMA parameters
params = ct.CarmaParams(p=2, q=1)
params.ar_coeffs = [0.8, 0.3]
params.ma_coeffs = [1.0, 0.4] 
params.sigma = 1.5

# MCMC parameters
mcmc_params = ct.McmcParams(p=2, q=1)
mcmc_params.ysigma = 1.2
mcmc_params.measerr_scale = 1.0

# 2. Data Preparation
times = np.array([0.0, 1.2, 2.8, 4.1, 5.9])
values = np.array([1.0, 1.5, 0.8, 1.2, 0.9])
errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

# 3. Maximum Likelihood Estimation
mle_result = ct.carma_mle(
    times, values, errors, p=2, q=1,
    n_starts=8, max_iter=1000
)

print(f"MLE Results:")
print(f"  Log-likelihood: {mle_result.loglikelihood}")
print(f"  AICc: {mle_result.aicc}")
print(f"  AR coeffs: {mle_result.params.ar_coeffs}")
print(f"  MA coeffs: {mle_result.params.ma_coeffs}")

# 4. MCMC Sampling
mcmc_result = ct.carma_mcmc(
    times, values, errors, p=2, q=1,
    n_samples=5000, n_burn=2000, n_chains=6, seed=42
)

print(f"MCMC Results:")
print(f"  Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
print(f"  Sample shape: {mcmc_result.samples.shape}")
print(f"  R-hat: {mcmc_result.rhat}")
print(f"  ESS: {mcmc_result.effective_sample_size}")

# 5. Model Order Selection
order_result = ct.carma_choose_order(
    times, values, errors, max_p=3, max_q=2
)

print(f"Order Selection:")
print(f"  Best model: CARMA({order_result.best_p}, {order_result.best_q})")
print(f"  Best AICc: {order_result.best_aicc}")

# 6. Log-likelihood Computation
loglik = ct.carma_loglikelihood(params, times, values, errors)
print(f"Log-likelihood: {loglik}")
