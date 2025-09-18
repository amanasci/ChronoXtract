#!/usr/bin/env python3
"""
Simple test script to verify CARMA implementation with known stationary parameters
"""

import numpy as np
import chronoxtract as ct

def test_simple_ar1():
    """Test with a simple AR(1) model that should be stable"""
    print("🔧 Testing simple AR(1) model...")
    
    # Create a simple AR(1) model with known stable parameters
    params = ct.CarmaParams(1, 0)
    params.ar_coeffs = [0.8]  # Stable: |0.8| < 1
    params.ma_coeffs = [1.0]  # Standard normalization
    params.sigma = 1.0
    
    print(f"✓ Created AR(1) parameters: AR={params.ar_coeffs}, MA={params.ma_coeffs}, σ={params.sigma}")
    
    # Test parameter validation
    try:
        params.validate()
        print("✓ Parameter validation passed")
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        return False
    
    # Create simple test data
    np.random.seed(42)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    values = np.array([1.0, 0.8, 0.64, 0.512, 0.41], dtype=np.float64)  # Roughly AR(1) with φ=0.8
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    
    print(f"✓ Created test data: {len(times)} points")
    
    # Test log-likelihood computation
    try:
        loglik = ct.carma_loglikelihood(params, times, values, errors)
        print(f"✓ Log-likelihood computed: {loglik:.4f}")
    except Exception as e:
        print(f"❌ Log-likelihood failed: {e}")
        return False
    
    # Test MLE estimation
    try:
        mle_result = ct.carma_mle(times, values, errors, 1, 0, n_starts=2, max_iter=10)
        print(f"✓ MLE estimation: AICc={mle_result.aicc:.4f}")
        print(f"  AR coeff: {mle_result.params.ar_coeffs[0]:.3f} (true: 0.8)")
    except Exception as e:
        print(f"❌ MLE estimation failed: {e}")
        return False
    
    return True

def test_simple_ar2():
    """Test with a simple AR(2) model"""
    print("\n🔧 Testing simple AR(2) model...")
    
    # Create AR(2) with known stable parameters
    params = ct.CarmaParams(2, 0)
    params.ar_coeffs = [0.6, 0.2]  # Should be stable
    params.ma_coeffs = [1.0]
    params.sigma = 1.0
    
    print(f"✓ Created AR(2) parameters: AR={params.ar_coeffs}, MA={params.ma_coeffs}, σ={params.sigma}")
    
    # Test parameter validation
    try:
        params.validate()
        print("✓ Parameter validation passed")
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        return False
    
    # Create test data
    times = np.linspace(0, 10, 20)
    # Generate simple AR(2) data
    values = np.sin(0.5 * times) + 0.1 * np.random.randn(len(times))
    errors = np.full(len(times), 0.1)
    
    print(f"✓ Created test data: {len(times)} points")
    
    # Test log-likelihood
    try:
        loglik = ct.carma_loglikelihood(params, times, values, errors)
        print(f"✓ Log-likelihood computed: {loglik:.4f}")
    except Exception as e:
        print(f"❌ Log-likelihood failed: {e}")
        return False
    
    return True

def test_carma_11():
    """Test CARMA(1,1) model"""
    print("\n🔧 Testing CARMA(1,1) model...")
    
    params = ct.CarmaParams(1, 1)
    params.ar_coeffs = [0.7]  # Stable
    params.ma_coeffs = [1.0, 0.3]  # MA polynomial
    params.sigma = 1.0
    
    print(f"✓ Created CARMA(1,1) parameters: AR={params.ar_coeffs}, MA={params.ma_coeffs}, σ={params.sigma}")
    
    # Test validation
    try:
        params.validate()
        print("✓ Parameter validation passed")
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        return False
    
    # Simple test data
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Test log-likelihood
    try:
        loglik = ct.carma_loglikelihood(params, times, values, errors)
        print(f"✓ Log-likelihood computed: {loglik:.4f}")
    except Exception as e:
        print(f"❌ Log-likelihood failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Simple CARMA Implementation Test")
    print("=" * 50)
    
    results = []
    results.append(("AR(1)", test_simple_ar1()))
    results.append(("AR(2)", test_simple_ar2()))
    results.append(("CARMA(1,1)", test_carma_11()))
    
    print("\n" + "🎯 Test Results")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:<12}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All basic tests passed!")
    else:
        print("⚠️  Some tests failed - implementation needs work")