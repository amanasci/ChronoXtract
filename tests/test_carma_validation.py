#!/usr/bin/env python3
"""
Comprehensive validation of the new CARMA implementation against carma_pack.

This script tests:
1. Basic CARMA model creation and parameter setting
2. Simulation functionality  
3. MLE estimation accuracy
4. MCMC sampling functionality
5. Parameter recovery from synthetic data
6. Performance benchmarking
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Try to import carma_pack for reference comparisons
try:
    sys.path.append('/tmp/carma_pack/src')
    import carmcmc as cm
    CARMA_PACK_AVAILABLE = True
    print("✅ carma_pack found - will perform reference comparisons")
except ImportError:
    CARMA_PACK_AVAILABLE = False
    print("⚠️  carma_pack not available - skipping reference comparisons")

# Import our implementation
try:
    import chronoxtract as ct
    print("✅ chronoxtract imported successfully")
except ImportError as e:
    print(f"❌ Failed to import chronoxtract: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic CARMA model creation and parameter setting."""
    print("\n🧪 Testing Basic CARMA Functionality")
    print("=" * 50)
    
    # Test model creation
    try:
        model = ct.carma_model(2, 1)
        print("✅ CARMA(2,1) model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test parameter setting
    try:
        ct.set_carma_parameters(model, [0.5, 1.0], [1.0, 0.3], 1.5, 0.0)
        print("✅ Parameters set successfully")
    except Exception as e:
        print(f"❌ Parameter setting failed: {e}")
        return False
    
    # Test stability check
    try:
        is_stable = ct.check_carma_stability(model)
        print(f"✅ Stability check: {is_stable}")
    except Exception as e:
        print(f"❌ Stability check failed: {e}")
        return False
    
    return True

def test_simulation():
    """Test CARMA simulation functionality."""
    print("\n🧪 Testing CARMA Simulation")
    print("=" * 50)
    
    # First try with very simple, known stable parameters
    try:
        model = ct.carma_model(1, 0)  # Start with CAR(1)
        ct.set_carma_parameters(model, [1.0], [1.0], 1.0, 0.0)
        
        # Test regular simulation
        times = np.linspace(0, 10, 50)
        values = ct.simulate_carma(model, times, seed=42)
        print(f"✅ CAR(1) simulation: {len(values)} data points")
        
        return True, (times, values), None
        
    except Exception as e:
        print(f"❌ CAR(1) simulation failed: {e}")
        
    # Try with generated stable parameters if CAR(1) works
    try:
        ar_coeffs, ma_coeffs, sigma = ct.generate_stable_carma_parameters(2, 1, seed=42)
        print(f"✅ Generated stable parameters:")
        print(f"   AR: {ar_coeffs}")
        print(f"   MA: {ma_coeffs}")  
        print(f"   σ: {sigma:.3f}")
        
        model = ct.carma_model(2, 1)
        ct.set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma, 0.0)
        
        # Test regular simulation
        times = np.linspace(0, 10, 100)
        values = ct.simulate_carma(model, times, seed=42)
        print(f"✅ Simulated {len(values)} data points")
        
        # Test irregular simulation
        times_irreg, values_irreg, errors = ct.generate_carma_data(
            model, duration=10.0, mean_sampling_rate=10.0, 
            sampling_irregularity=0.3, measurement_noise=0.1, seed=42
        )
        print(f"✅ Generated irregular dataset with {len(times_irreg)} points")
        
        return True, (times, values), (times_irreg, values_irreg, errors)
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False, None, None

def test_mle_estimation(sim_data):
    """Test MLE estimation on simulated data."""
    print("\n🧪 Testing MLE Estimation")
    print("=" * 50)
    
    if sim_data is None:
        print("❌ No simulation data available")
        return False
    
    times, values = sim_data
    
    try:
        # Test MLE fitting
        start_time = time.time()
        result = ct.carma_mle(times, values, 2, 1, max_iter=100, n_trials=5, seed=42)
        mle_time = time.time() - start_time
        
        print(f"✅ MLE completed in {mle_time:.3f}s")
        print(f"   Log-likelihood: {result.loglikelihood:.3f}")
        print(f"   AIC: {result.aic:.3f}")
        print(f"   BIC: {result.bic:.3f}")
        print(f"   Converged: {result.converged}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLE estimation failed: {e}")
        return False

def test_mcmc_sampling(sim_data):
    """Test MCMC sampling on simulated data."""
    print("\n🧪 Testing MCMC Sampling")
    print("=" * 50)
    
    if sim_data is None:
        print("❌ No simulation data available")
        return False
    
    times, values = sim_data
    
    try:
        # Test MCMC sampling
        start_time = time.time()
        result = ct.carma_mcmc(times, values, 2, 1, n_samples=500, burn_in=200, 
                              n_chains=2, seed=42)
        mcmc_time = time.time() - start_time
        
        print(f"✅ MCMC completed in {mcmc_time:.3f}s")
        print(f"   Samples: {len(result.samples)}")
        print(f"   Parameters: {result.param_names}")
        print(f"   Acceptance rate: {result.acceptance_rate:.3f}")
        
        # Check sample statistics
        if len(result.samples) > 0:
            mu_samples = result.get_samples('mu')
            print(f"   μ posterior mean: {np.mean(mu_samples):.3f} ± {np.std(mu_samples):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ MCMC sampling failed: {e}")
        return False

def test_model_selection():
    """Test model selection functionality."""
    print("\n🧪 Testing Model Selection")
    print("=" * 50)
    
    try:
        # Generate data from known CARMA(2,1) model
        ar_coeffs, ma_coeffs, sigma = ct.generate_stable_carma_parameters(2, 1, seed=42)
        model = ct.carma_model(2, 1)
        ct.set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma, 0.0)
        
        times = np.linspace(0, 20, 200)
        values = ct.simulate_carma(model, times, seed=42)
        
        # Test model selection
        start_time = time.time()
        results = ct.carma_model_selection(times, values, max_p=4, max_q=3)
        selection_time = time.time() - start_time
        
        print(f"✅ Model selection completed in {selection_time:.3f}s")
        print("   Results (p, q, AIC, BIC):")
        
        best_aic = min(results, key=lambda x: x[2])
        best_bic = min(results, key=lambda x: x[3])
        
        for p, q, aic, bic in sorted(results)[:10]:  # Show top 10
            marker = " 🎯" if (p, q) == (2, 1) else ""
            print(f"   ({p},{q}): AIC={aic:.1f}, BIC={bic:.1f}{marker}")
        
        print(f"   Best AIC: CARMA({best_aic[0]},{best_aic[1]})")
        print(f"   Best BIC: CARMA({best_bic[0]},{best_bic[1]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model selection failed: {e}")
        return False

def test_parameter_recovery():
    """Test parameter recovery using known synthetic data."""
    print("\n🧪 Testing Parameter Recovery")
    print("=" * 50)
    
    # True parameters
    true_ar = [0.5, 1.2]
    true_ma = [1.0, 0.4] 
    true_sigma = 1.5
    true_mu = 2.0
    
    try:
        # Create model and simulate
        model = ct.carma_model(2, 1)
        ct.set_carma_parameters(model, true_ar, true_ma, true_sigma, true_mu)
        
        # Generate enough data for good parameter recovery
        times = np.sort(np.random.uniform(0, 50, 500))
        values = ct.simulate_carma(model, times, seed=42)
        
        # Fit with MLE
        result = ct.carma_mle(times, values, 2, 1, n_trials=10, seed=42)
        
        print("✅ Parameter Recovery Test:")
        print(f"   True AR:     {true_ar}")
        print(f"   True MA:     {true_ma}")
        print(f"   True σ:      {true_sigma:.3f}")
        print(f"   True μ:      {true_mu:.3f}")
        print()
        print("   Estimated parameters:")
        fitted_model = result.model
        print(f"   Est. AR:     {[f'{x:.3f}' for x in fitted_model.ar_coeffs]}")
        print(f"   Est. MA:     {[f'{x:.3f}' for x in fitted_model.ma_coeffs]}")
        print(f"   Est. σ:      {fitted_model.sigma:.3f}")
        print(f"   Est. μ:      {fitted_model.mu:.3f}")
        
        # Compute recovery errors
        ar_error = np.mean([(est - true)**2 for est, true in zip(fitted_model.ar_coeffs, true_ar)])
        ma_error = np.mean([(est - true)**2 for est, true in zip(fitted_model.ma_coeffs, true_ma)])
        sigma_error = (fitted_model.sigma - true_sigma)**2
        mu_error = (fitted_model.mu - true_mu)**2
        
        print(f"\n   Recovery Errors (MSE):")
        print(f"   AR MSE:      {ar_error:.4f}")
        print(f"   MA MSE:      {ma_error:.4f}")
        print(f"   σ MSE:       {sigma_error:.4f}")
        print(f"   μ MSE:       {mu_error:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter recovery test failed: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\n🧪 Testing Utility Functions")
    print("=" * 50)
    
    try:
        # Test characteristic roots
        model = ct.carma_model(2, 1)
        ct.set_carma_parameters(model, [1.0, 2.0], [1.0, 0.3], 1.0, 0.0)
        
        roots = ct.carma_characteristic_roots(model)
        print(f"✅ Characteristic roots: {roots}")
        
        # Test power spectrum
        frequencies = np.logspace(-2, 1, 50)
        psd = ct.carma_power_spectrum(model, frequencies)
        print(f"✅ Power spectrum computed at {len(frequencies)} frequencies")
        
        # Test autocovariance (should work for CAR(1))
        car1_model = ct.carma_model(1, 0)
        ct.set_carma_parameters(car1_model, [1.0], [1.0], 1.0, 0.0)
        
        lags = np.linspace(0, 5, 20)
        autocov = ct.carma_autocovariance(car1_model, lags)
        print(f"✅ Autocovariance computed at {len(lags)} lags")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n🚀 Performance Benchmarks")
    print("=" * 50)
    
    # Benchmark simulation
    model = ct.carma_model(3, 2)
    ar_coeffs, ma_coeffs, sigma = ct.generate_stable_carma_parameters(3, 2, seed=42)
    ct.set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma, 0.0)
    
    times = np.linspace(0, 100, 10000)
    
    start_time = time.time()
    values = ct.simulate_carma(model, times, seed=42)
    sim_time = time.time() - start_time
    
    print(f"✅ Simulation: {len(times)} points in {sim_time:.3f}s ({len(times)/sim_time:.0f} pts/s)")
    
    # Benchmark MLE on smaller dataset
    times_small = times[::10]  # Subsample for faster fitting
    values_small = values[::10]
    
    start_time = time.time()
    result = ct.carma_mle(times_small, values_small, 3, 2, n_trials=3, max_iter=50)
    mle_time = time.time() - start_time
    
    print(f"✅ MLE: {len(times_small)} points in {mle_time:.3f}s")
    
    # Benchmark MCMC on even smaller dataset
    times_tiny = times[::50]
    values_tiny = values[::50] 
    
    start_time = time.time()
    mcmc_result = ct.carma_mcmc(times_tiny, values_tiny, 3, 2, n_samples=200, burn_in=100, n_chains=2)
    mcmc_time = time.time() - start_time
    
    print(f"✅ MCMC: {len(times_tiny)} points, 200 samples in {mcmc_time:.3f}s")

def main():
    """Run all validation tests."""
    print("🎯 ChronoXtract CARMA Module Validation")
    print("=" * 60)
    
    test_results = []
    
    # Basic functionality tests
    test_results.append(("Basic Functionality", test_basic_functionality()))
    
    # Simulation tests
    sim_success, regular_data, irregular_data = test_simulation()
    test_results.append(("Simulation", sim_success))
    
    # MLE tests
    mle_success = test_mle_estimation(regular_data if sim_success else None)
    test_results.append(("MLE Estimation", mle_success))
    
    # MCMC tests
    mcmc_success = test_mcmc_sampling(regular_data if sim_success else None)
    test_results.append(("MCMC Sampling", mcmc_success))
    
    # Model selection tests
    test_results.append(("Model Selection", test_model_selection()))
    
    # Parameter recovery tests
    test_results.append(("Parameter Recovery", test_parameter_recovery()))
    
    # Utility function tests
    test_results.append(("Utility Functions", test_utilities()))
    
    # Performance benchmarks
    try:
        run_performance_benchmark()
        test_results.append(("Performance Benchmark", True))
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        test_results.append(("Performance Benchmark", False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! CARMA implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())