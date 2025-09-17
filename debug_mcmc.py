#!/usr/bin/env python3
"""
Debug script to understand MCMC adaptive covariance behavior
"""

import numpy as np
import chronoxtract as cx
import time

def debug_mcmc_simple():
    """Debug a simple CARMA(2,1) model with detailed logging"""
    print("🔍 Debugging CARMA MCMC Implementation")
    print("=" * 50)
    
    # Generate simple test data
    np.random.seed(42)
    p, q = 2, 1
    n_points = 200
    n_samples = 100  # Small number for debugging
    
    # Create stable parameters
    ar_coeffs = [0.5, 0.3]
    ma_coeffs = [1.0, 0.2]
    sigma = 0.1
    
    print(f"Testing CARMA({p},{q}) with {n_points} points, {n_samples} samples")
    print(f"True AR coeffs: {ar_coeffs}")
    print(f"True MA coeffs: {ma_coeffs}")
    print(f"True sigma: {sigma}")
    
    # Create and test model
    model = cx.carma_model(p, q)
    cx.set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma)
    
    # Check stability
    is_stable = cx.check_carma_stability(model)
    print(f"Model stable: {is_stable}")
    
    if not is_stable:
        print("❌ Model not stable, stopping")
        return
    
    # Generate data
    t = np.linspace(0, 10, n_points)
    y = cx.simulate_carma(model, t, seed=42)
    print(f"Generated {len(y)} data points")
    
    # Test MLE first
    print("\nTesting MLE...")
    start_time = time.time()
    mle_result = cx.carma_mle(t, y, p, q)
    mle_time = time.time() - start_time
    print(f"MLE time: {mle_time:.3f}s")
    print(f"MLE Log-likelihood: {mle_result.loglikelihood:.2f}")
    
    # Extract fitted parameters
    fitted_model = mle_result.model
    fitted_ar = fitted_model.ar_coeffs
    fitted_ma = fitted_model.ma_coeffs
    fitted_sigma = fitted_model.sigma
    
    print(f"Fitted AR coeffs: {[f'{x:.3f}' for x in fitted_ar]}")
    print(f"Fitted MA coeffs: {[f'{x:.3f}' for x in fitted_ma]}")
    print(f"Fitted sigma: {fitted_sigma:.3f}")
    
    # Test MCMC with short run for debugging
    print("\nTesting MCMC...")
    start_time = time.time()
    mcmc_result = cx.carma_mcmc(t, y, p, q, n_samples, burn_in=50, seed=42)
    mcmc_time = time.time() - start_time
    
    print(f"MCMC time: {mcmc_time:.3f}s")
    print(f"Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
    print(f"Number of samples: {len(mcmc_result.samples)}")
    print(f"R-hat values: {[f'{x:.3f}' for x in mcmc_result.rhat]}")
    print(f"ESS values: {[f'{x:.1f}' for x in mcmc_result.effective_sample_size]}")
    
    # Analyze samples
    if len(mcmc_result.samples) > 0:
        samples = np.array(mcmc_result.samples)
        print(f"Sample shape: {samples.shape}")
        
        # Parameter statistics
        param_means = np.mean(samples, axis=0)
        param_stds = np.std(samples, axis=0)
        
        print(f"Sample means: {[f'{x:.3f}' for x in param_means]}")
        print(f"Sample stds: {[f'{x:.3f}' for x in param_stds]}")
        
        # Check if parameters are moving
        param_ranges = np.max(samples, axis=0) - np.min(samples, axis=0)
        print(f"Parameter ranges: {[f'{x:.3f}' for x in param_ranges]}")
        
        # Check for stuck chains
        if np.any(param_ranges < 1e-6):
            print("⚠️  Some parameters appear to be stuck (very small range)")
    
    # Performance assessment
    acceptance_ok = 0.2 <= mcmc_result.acceptance_rate <= 0.6
    rhat_ok = max(mcmc_result.rhat) < 1.1
    ess_ok = min(mcmc_result.effective_sample_size) > 20  # Lower threshold for debug
    
    print(f"\nPerformance Assessment:")
    print(f"Acceptance rate OK: {acceptance_ok} (target: 0.2-0.6)")
    print(f"R-hat OK: {rhat_ok} (target: <1.1)")
    print(f"ESS OK: {ess_ok} (target: >20 for debug)")
    
    if acceptance_ok and rhat_ok and ess_ok:
        print("✅ MCMC: Good performance")
    else:
        print("❌ MCMC: Needs improvement")

if __name__ == "__main__":
    debug_mcmc_simple()