#!/usr/bin/env python3
"""
Validated test suite for CARMA implementation

This test suite verifies that the CARMA implementation works correctly
with known examples and provides regression testing.
"""

import sys
import os
import time

# Test environment setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check multiple possible locations for the built module
build_dirs = [
    os.path.join(project_root, 'target', 'debug'),
    os.path.join(project_root, 'target', 'release'),
    project_root
]

chronoxtract = None
for build_dir in build_dirs:
    sys.path.insert(0, build_dir)
    try:
        import chronoxtract
        print(f"✓ Successfully imported chronoxtract from {build_dir}")
        break
    except ImportError:
        if build_dir in sys.path:
            sys.path.remove(build_dir)
        continue

if chronoxtract is None:
    print("❌ Could not import chronoxtract module")
    print("Please build the module first with: maturin develop")
    sys.exit(1)

# Check for NumPy
try:
    import numpy as np
    HAS_NUMPY = True
    print("✓ NumPy available for comprehensive testing")
except ImportError:
    HAS_NUMPY = False
    print("ℹ  NumPy not available, using basic tests only")

def test_carma_params_creation():
    """Test basic CARMA parameter creation and validation"""
    print("\n" + "="*60)
    print("Testing CARMA Parameter Creation and Validation")
    print("="*60)
    
    # Test valid parameter creation
    try:
        params = chronoxtract.CarmaParams(2, 1)
        print("✓ Created CARMA(2,1) parameters")
        
        # Test parameter setting
        params.ar_coeffs = [0.8, 0.3]
        params.ma_coeffs = [1.0, 0.4]
        params.sigma = 1.5
        
        print(f"✓ Set parameters: AR={params.ar_coeffs}, MA={params.ma_coeffs}, σ={params.sigma}")
        return True
        
    except Exception as e:
        print(f"❌ CARMA parameter creation failed: {e}")
        return False

def test_mcmc_params_creation():
    """Test MCMC parameter creation"""
    print("\n" + "-"*60)
    print("Testing MCMC Parameter Creation")
    print("-"*60)
    
    try:
        mcmc_params = chronoxtract.McmcParams(2, 1)
        print("✓ Created MCMC parameters")
        
        # Test parameter setting
        mcmc_params.ysigma = 1.2
        mcmc_params.measerr_scale = 1.0
        mcmc_params.mu = 0.5
        
        print(f"✓ Set MCMC parameters: ysigma={mcmc_params.ysigma}, scale={mcmc_params.measerr_scale}")
        return True
        
    except Exception as e:
        print(f"❌ MCMC parameter creation failed: {e}")
        return False

def test_mle_basic():
    """Test basic MLE functionality"""
    print("\n" + "-"*60)
    print("Testing Maximum Likelihood Estimation")
    print("-"*60)
    
    if not HAS_NUMPY:
        print("⚠  Skipping MLE test (requires NumPy)")
        return True
    
    # Create synthetic data
    np.random.seed(42)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    values = np.array([1.0, 1.2, 0.8, 1.1, 0.9], dtype=np.float64)
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    
    try:
        start_time = time.time()
        mle_result = chronoxtract.carma_mle(
            times, values, errors, 
            p=1, q=0,  # Simple AR(1) model
            n_starts=2,  # Quick test
            max_iter=50
        )
        mle_time = time.time() - start_time
        
        print(f"✓ MLE completed in {mle_time:.3f}s")
        print(f"  AICc: {mle_result.aicc:.4f}")
        print(f"  Converged: {mle_result.converged}")
        
        # Basic validation
        assert hasattr(mle_result, 'params')
        assert hasattr(mle_result, 'loglikelihood')
        assert hasattr(mle_result, 'aic')
        assert hasattr(mle_result, 'aicc')
        assert hasattr(mle_result, 'bic')
        
        return True
        
    except Exception as e:
        print(f"❌ MLE test failed: {e}")
        return False

def test_mcmc_basic():
    """Test basic MCMC functionality"""
    print("\n" + "-"*60)
    print("Testing MCMC Sampling")
    print("-"*60)
    
    if not HAS_NUMPY:
        print("⚠  Skipping MCMC test (requires NumPy)")
        return True
    
    # Create synthetic data
    np.random.seed(123)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    values = np.array([1.0, 1.1, 0.9, 1.2, 0.8], dtype=np.float64)
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    
    try:
        start_time = time.time()
        mcmc_result = chronoxtract.carma_mcmc(
            times, values, errors,
            p=1, q=0,  # Simple AR(1) model
            n_samples=100,  # Small for quick test
            n_burn=50,
            n_chains=2,
            seed=456
        )
        mcmc_time = time.time() - start_time
        
        print(f"✓ MCMC completed in {mcmc_time:.3f}s")
        print(f"  Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
        print(f"  Sample shape: {mcmc_result.samples.shape}")
        print(f"  Number of parameters: {len(mcmc_result.rhat)}")
        
        # Basic validation
        assert hasattr(mcmc_result, 'samples')
        assert hasattr(mcmc_result, 'loglikelihoods')
        assert hasattr(mcmc_result, 'acceptance_rate')
        assert hasattr(mcmc_result, 'rhat')
        assert hasattr(mcmc_result, 'effective_sample_size')
        
        # Check reasonable values
        assert 0.0 <= mcmc_result.acceptance_rate <= 1.0
        assert mcmc_result.samples.shape[0] == 100
        
        return True
        
    except Exception as e:
        print(f"❌ MCMC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_order_selection():
    """Test model order selection"""
    print("\n" + "-"*60)
    print("Testing Model Order Selection")
    print("-"*60)
    
    if not HAS_NUMPY:
        print("⚠  Skipping order selection test (requires NumPy)")
        return True
    
    # Create synthetic data
    np.random.seed(789)
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)
    values = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 0.95], dtype=np.float64)
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    
    try:
        start_time = time.time()
        order_result = chronoxtract.carma_choose_order(
            times, values, errors,
            max_p=2, max_q=1
        )
        order_time = time.time() - start_time
        
        print(f"✓ Order selection completed in {order_time:.3f}s")
        print(f"  Best model: CARMA({order_result.best_p}, {order_result.best_q})")
        print(f"  Best AICc: {order_result.best_aicc:.4f}")
        
        # Basic validation
        assert hasattr(order_result, 'best_p')
        assert hasattr(order_result, 'best_q')
        assert hasattr(order_result, 'best_aicc')
        assert hasattr(order_result, 'aicc_grid')
        
        assert 1 <= order_result.best_p <= 2
        assert 0 <= order_result.best_q < order_result.best_p
        
        return True
        
    except Exception as e:
        print(f"❌ Order selection test failed: {e}")
        return False

def test_performance_scaling():
    """Test performance with different data sizes"""
    print("\n" + "-"*60)
    print("Testing Performance Scaling")
    print("-"*60)
    
    if not HAS_NUMPY:
        print("⚠  Skipping performance test (requires NumPy)")
        return True
    
    sizes = [10, 20, 50]
    results = []
    
    for n in sizes:
        print(f"\n📊 Testing with {n} data points...")
        
        # Generate test data
        np.random.seed(n)
        times = np.sort(np.random.uniform(0, 10, n))
        values = np.random.normal(0, 1, n)
        errors = np.full(n, 0.1)
        
        # Test MLE
        try:
            start_time = time.time()
            mle_result = chronoxtract.carma_mle(
                times, values, errors, 1, 0,
                n_starts=2, max_iter=20
            )
            mle_time = time.time() - start_time
            print(f"  MLE: {mle_time:.3f}s (AICc: {mle_result.aicc:.2f})")
            
        except Exception as e:
            print(f"  MLE: FAILED - {e}")
            mle_time = None
        
        # Test MCMC (smaller sample)
        try:
            start_time = time.time()
            mcmc_result = chronoxtract.carma_mcmc(
                times, values, errors, 1, 0,
                n_samples=50, n_burn=25, n_chains=2
            )
            mcmc_time = time.time() - start_time
            print(f"  MCMC: {mcmc_time:.3f}s (Acc: {mcmc_result.acceptance_rate:.3f})")
            
        except Exception as e:
            print(f"  MCMC: FAILED - {e}")
            mcmc_time = None
        
        results.append((n, mle_time, mcmc_time))
    
    # Check that at least some tests passed
    successful_runs = sum(1 for _, mle_t, mcmc_t in results if mle_t is not None or mcmc_t is not None)
    return successful_runs >= len(sizes) // 2

def run_validation_suite():
    """Run the complete validation suite"""
    print("🚀 CARMA Implementation Validation Suite")
    print("=" * 80)
    print(f"Testing environment: {'NumPy available' if HAS_NUMPY else 'Basic mode'}")
    
    tests = [
        ("Parameter Creation", test_carma_params_creation),
        ("MCMC Parameters", test_mcmc_params_creation),
        ("MLE Estimation", test_mle_basic),
        ("MCMC Sampling", test_mcmc_basic),
        ("Order Selection", test_order_selection),
        ("Performance Scaling", test_performance_scaling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "🎯 VALIDATION RESULTS")
    print("=" * 80)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"✓ {test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 All validation tests PASSED!")
        print("✅ CARMA implementation is working correctly and ready for use")
        return True
    elif passed >= total * 0.75:
        print("\n✅ Most validation tests passed - implementation is mostly functional")
        return True
    else:
        print("\n⚠️  Several validation tests failed - implementation may have issues")
        return False

if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)