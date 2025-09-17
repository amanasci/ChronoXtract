#!/usr/bin/env python3
"""
Detailed debug script to understand why MCMC has 0% acceptance rate
"""

import numpy as np
import chronoxtract as cx
import time

def debug_mcmc_detailed():
    """Debug with manual parameter exploration"""
    print("🔍 Detailed MCMC Debugging")
    print("=" * 50)
    
    # Generate simple test data
    np.random.seed(42)
    p, q = 2, 1
    n_points = 50  # Even smaller for debugging
    
    # Create stable parameters
    ar_coeffs = [0.5, 0.2]
    ma_coeffs = [1.0, 0.1]
    sigma = 0.1
    
    print(f"Testing CARMA({p},{q}) with {n_points} points")
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
    t = np.linspace(0, 5, n_points)
    y = cx.simulate_carma(model, t, seed=42)
    print(f"Generated {len(y)} data points")
    print(f"Data range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"Data mean: {np.mean(y):.3f}, std: {np.std(y):.3f}")
    
    # Test MLE first
    print("\nTesting MLE...")
    try:
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
        
        # Check if MLE parameters make sense
        if fitted_sigma > 10 or any(abs(ar) > 1 for ar in fitted_ar):
            print("⚠️  MLE parameters seem unreasonable")
        
    except Exception as e:
        print(f"❌ MLE failed: {e}")
        return
    
    # Test method of moments for comparison
    print("\nTesting Method of Moments...")
    try:
        mom_result = cx.carma_method_of_moments(t, y, p, q)
        print(f"MoM Log-likelihood: {mom_result.loglikelihood:.2f}")
        
        mom_model = mom_result.model
        print(f"MoM AR coeffs: {[f'{x:.3f}' for x in mom_model.ar_coeffs]}")
        print(f"MoM MA coeffs: {[f'{x:.3f}' for x in mom_model.ma_coeffs]}")
        print(f"MoM sigma: {mom_model.sigma:.3f}")
    except Exception as e:
        print(f"❌ Method of Moments failed: {e}")
    
    # Test MCMC with very short run
    print("\nTesting MCMC with minimal settings...")
    try:
        start_time = time.time()
        mcmc_result = cx.carma_mcmc(t, y, p, q, n_samples=20, burn_in=10, seed=42)
        mcmc_time = time.time() - start_time
        
        print(f"MCMC time: {mcmc_time:.3f}s")
        print(f"Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
        print(f"Number of samples: {len(mcmc_result.samples)}")
        
        if len(mcmc_result.samples) > 0:
            samples = np.array(mcmc_result.samples)
            print(f"Sample shape: {samples.shape}")
            
            # Check first few samples
            print("First 5 samples:")
            for i in range(min(5, len(samples))):
                print(f"  Sample {i}: {[f'{x:.3f}' for x in samples[i]]}")
                
            # Check if all samples are identical (stuck chains)
            if len(samples) > 1:
                param_stds = np.std(samples, axis=0)
                print(f"Parameter standard deviations: {[f'{x:.6f}' for x in param_stds]}")
                
                if np.all(param_stds < 1e-8):
                    print("⚠️  All chains appear to be completely stuck!")
        else:
            print("❌ No samples generated!")
            
    except Exception as e:
        print(f"❌ MCMC failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mcmc_detailed()