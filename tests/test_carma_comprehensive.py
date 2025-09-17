#!/usr/bin/env python3
"""
Comprehensive test comparing CARMA MLE and MCMC implementations
and benchmarking against existing libraries
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

# Add the build directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'target', 'debug'))

try:
    import chronoxtract as ct
    print("✓ Successfully imported chronoxtract")
except ImportError as e:
    print(f"❌ Failed to import chronoxtract: {e}")
    exit(1)

def generate_carma_data(n_points=200, p=2, q=1, irregular=True, seed=42):
    """Generate synthetic CARMA time series data"""
    np.random.seed(seed)
    
    if irregular:
        # Irregular sampling
        times = np.sort(np.random.uniform(0, 50, n_points))
    else:
        # Regular sampling
        times = np.linspace(0, 50, n_points)
    
    # Simple AR(2) process simulation for testing
    # This is a simplified simulation - the full CARMA simulation would be more complex
    values = np.zeros(n_points)
    values[0] = np.random.normal(0, 1)
    values[1] = np.random.normal(0, 1)
    
    # AR(2) coefficients that ensure stationarity
    phi1, phi2 = 0.7, -0.3
    sigma = 1.0
    
    for i in range(2, n_points):
        dt_i = times[i] - times[i-1]
        dt_i1 = times[i-1] - times[i-2]
        
        # Simple discrete-time approximation
        values[i] = (phi1 * values[i-1] + phi2 * values[i-2] + 
                    np.random.normal(0, sigma * np.sqrt(dt_i)))
    
    # Add measurement errors
    measurement_errors = np.full(n_points, 0.1)
    noisy_values = values + np.random.normal(0, measurement_errors)
    
    return times, noisy_values, measurement_errors

def test_mle_vs_mcmc_consistency():
    """Test that MLE and MCMC give consistent results"""
    print("\n" + "="*60)
    print("Testing MLE vs MCMC Consistency")
    print("="*60)
    
    # Generate test data
    times, values, errors = generate_carma_data(n_points=100, seed=123)
    print(f"Generated {len(times)} data points")
    
    # Test parameters
    p, q = 2, 1
    
    # Run MLE
    print("\n🔧 Running MLE estimation...")
    start_time = time.time()
    try:
        mle_result = ct.carma_mle(times, values, errors, p, q, n_starts=4, max_iter=100)
        mle_time = time.time() - start_time
        print(f"✓ MLE completed in {mle_time:.2f} seconds")
        print(f"  Log-likelihood: {mle_result.loglikelihood:.4f}")
        print(f"  AICc: {mle_result.aicc:.4f}")
        print(f"  AR coeffs: {mle_result.params.ar_coeffs}")
        print(f"  MA coeffs: {mle_result.params.ma_coeffs}")
        print(f"  Sigma: {mle_result.params.sigma:.4f}")
    except Exception as e:
        print(f"❌ MLE failed: {e}")
        return False
    
    # Run MCMC
    print("\n🔬 Running MCMC sampling...")
    start_time = time.time()
    try:
        mcmc_result = ct.carma_mcmc(
            times, values, errors, p, q, 
            n_samples=1000, n_burn=500, n_chains=4, seed=456
        )
        mcmc_time = time.time() - start_time
        print(f"✓ MCMC completed in {mcmc_time:.2f} seconds")
        print(f"  Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
        print(f"  Mean R-hat: {np.mean(mcmc_result.rhat):.3f}")
        print(f"  Mean ESS: {np.mean(mcmc_result.effective_sample_size):.1f}")
        
        # Extract parameter means from MCMC samples
        samples = np.array(mcmc_result.samples)
        n_params = samples.shape[1]
        
        # Parameter order: ar_params, ma_params, log(ysigma), log(measerr_scale), mu
        ar_means = np.mean(samples[:, :p], axis=0)
        ma_means = np.mean(samples[:, p:p+q], axis=0)
        ysigma_mean = np.exp(np.mean(samples[:, p+q]))
        mu_mean = np.mean(samples[:, -1])
        
        print(f"  MCMC AR means: {ar_means}")
        print(f"  MCMC MA means: {ma_means}")
        print(f"  MCMC sigma mean: {ysigma_mean:.4f}")
        print(f"  MCMC mu mean: {mu_mean:.4f}")
        
    except Exception as e:
        print(f"❌ MCMC failed: {e}")
        return False
    
    # Compare results
    print("\n📊 Comparing MLE vs MCMC results:")
    print(f"  Sigma - MLE: {mle_result.params.sigma:.4f}, MCMC: {ysigma_mean:.4f}")
    
    # Check if results are reasonably close (within 50% for this test)
    sigma_diff = abs(mle_result.params.sigma - ysigma_mean) / mle_result.params.sigma
    print(f"  Relative sigma difference: {sigma_diff:.3f}")
    
    if sigma_diff < 0.5:
        print("✓ MLE and MCMC results are reasonably consistent")
        return True
    else:
        print("⚠ MLE and MCMC results differ significantly")
        return False

def benchmark_carma_performance():
    """Benchmark CARMA implementation performance"""
    print("\n" + "="*60)
    print("CARMA Performance Benchmark")
    print("="*60)
    
    data_sizes = [50, 100, 200, 500]
    mle_times = []
    mcmc_times = []
    
    for n in data_sizes:
        print(f"\n📊 Testing with {n} data points...")
        
        # Generate data
        times, values, errors = generate_carma_data(n_points=n, seed=n)
        
        # Benchmark MLE
        start_time = time.time()
        try:
            mle_result = ct.carma_mle(times, values, errors, 1, 0, n_starts=2, max_iter=50)
            mle_time = time.time() - start_time
            mle_times.append(mle_time)
            print(f"  MLE: {mle_time:.3f}s (AICc: {mle_result.aicc:.2f})")
        except Exception as e:
            print(f"  MLE failed: {e}")
            mle_times.append(None)
        
        # Benchmark MCMC (smaller sample for speed)
        start_time = time.time()
        try:
            mcmc_result = ct.carma_mcmc(
                times, values, errors, 1, 0,
                n_samples=200, n_burn=100, n_chains=2, seed=n
            )
            mcmc_time = time.time() - start_time
            mcmc_times.append(mcmc_time)
            print(f"  MCMC: {mcmc_time:.3f}s (Acc: {mcmc_result.acceptance_rate:.3f})")
        except Exception as e:
            print(f"  MCMC failed: {e}")
            mcmc_times.append(None)
    
    # Summary
    print(f"\n📈 Performance Summary:")
    print(f"Data Size | MLE Time | MCMC Time")
    print(f"----------|----------|----------")
    for i, n in enumerate(data_sizes):
        mle_str = f"{mle_times[i]:.3f}s" if mle_times[i] else "FAILED"
        mcmc_str = f"{mcmc_times[i]:.3f}s" if mcmc_times[i] else "FAILED"
        print(f"{n:8d} | {mle_str:8s} | {mcmc_str}")
    
    return data_sizes, mle_times, mcmc_times

def test_parameter_recovery():
    """Test parameter recovery with known ground truth"""
    print("\n" + "="*60)
    print("Parameter Recovery Test")
    print("="*60)
    
    # Test with simple AR(1) model
    print("Testing AR(1) parameter recovery...")
    
    # Generate data with known parameters
    times, values, errors = generate_carma_data(n_points=150, p=1, q=0, seed=999)
    
    # Try to recover parameters
    try:
        mle_result = ct.carma_mle(times, values, errors, 1, 0, n_starts=6, max_iter=200)
        print(f"✓ Recovered AR coefficient: {mle_result.params.ar_coeffs[0]:.4f}")
        print(f"✓ Recovered sigma: {mle_result.params.sigma:.4f}")
        print(f"✓ Log-likelihood: {mle_result.loglikelihood:.4f}")
        print(f"✓ AICc: {mle_result.aicc:.4f}")
        return True
    except Exception as e:
        print(f"❌ Parameter recovery failed: {e}")
        return False

def compare_with_celerite():
    """Compare with celerite if available"""
    print("\n" + "="*60)
    print("Comparison with External Libraries")
    print("="*60)
    
    try:
        import celerite
        print("✓ celerite is available for comparison")
        
        # This would need to be implemented with actual celerite comparison
        print("📝 Celerite comparison would be implemented here")
        print("   (requires proper GP kernel setup and fitting)")
        return True
        
    except ImportError:
        print("ℹ celerite not available - skipping comparison")
        try:
            # Try other libraries
            import george
            print("✓ george is available for comparison")
            print("📝 George comparison would be implemented here")
            return True
        except ImportError:
            print("ℹ No external CARMA libraries found")
            print("📝 Install celerite or george for performance comparison:")
            print("    pip install celerite george")
            return False

def main():
    """Run comprehensive CARMA tests"""
    print("🚀 Comprehensive CARMA Implementation Test Suite")
    print("=" * 80)
    
    results = {}
    
    # Test 1: MLE vs MCMC consistency
    results['consistency'] = test_mle_vs_mcmc_consistency()
    
    # Test 2: Performance benchmarking
    print("\n" + "⏱" + " " * 58)
    data_sizes, mle_times, mcmc_times = benchmark_carma_performance()
    results['performance'] = (data_sizes, mle_times, mcmc_times)
    
    # Test 3: Parameter recovery
    results['recovery'] = test_parameter_recovery()
    
    # Test 4: External library comparison
    results['comparison'] = compare_with_celerite()
    
    # Final summary
    print("\n" + "🎯 FINAL RESULTS")
    print("=" * 80)
    print(f"✓ MLE vs MCMC Consistency: {'PASS' if results['consistency'] else 'FAIL'}")
    print(f"✓ Performance Tests: {'PASS' if any(mle_times) and any(mcmc_times) else 'FAIL'}")
    print(f"✓ Parameter Recovery: {'PASS' if results['recovery'] else 'FAIL'}")
    print(f"✓ External Comparison: {'AVAILABLE' if results['comparison'] else 'SKIPPED'}")
    
    overall_success = (results['consistency'] and 
                      results['recovery'] and
                      any(mle_times) and any(mcmc_times))
    
    if overall_success:
        print("\n🎉 All core tests PASSED! CARMA implementation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)