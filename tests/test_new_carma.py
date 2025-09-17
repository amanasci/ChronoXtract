#!/usr/bin/env python3
"""
Test script for new CARMA implementation
"""

import numpy as np
import sys
import os

# Add the build directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'target', 'debug'))

try:
    import chronoxtract as ct
    print("✓ Successfully imported chronoxtract")
    
    # Test CARMA parameter creation
    try:
        params = ct.CarmaParams(2, 1)
        print(f"✓ Created CARMA parameters: {params}")
        
        # Set some basic parameters
        params.ar_coeffs = [0.5, 0.3]
        params.ma_coeffs = [1.0, 0.2]
        params.sigma = 1.0
        
        print(f"✓ Set CARMA parameters: AR={params.ar_coeffs}, MA={params.ma_coeffs}, σ={params.sigma}")
        
        # Test parameter validation
        try:
            params.validate()
            print("✓ Parameter validation passed")
        except Exception as e:
            print(f"⚠ Parameter validation issue: {e}")
        
    except Exception as e:
        print(f"✗ Failed to create CARMA parameters: {e}")
    
    # Test basic time series creation
    try:
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        values = np.array([1.0, 1.2, 0.8, 1.1, 0.9], dtype=np.float64)
        errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
        
        print(f"✓ Created test time series: {len(times)} points")
        
        # Test log-likelihood computation
        try:
            loglik = ct.carma_loglikelihood(params, times, values, errors)
            print(f"✓ Computed log-likelihood: {loglik}")
        except Exception as e:
            print(f"⚠ Log-likelihood computation issue: {e}")
        
        # Test MLE estimation
        try:
            mle_result = ct.carma_mle(times, values, errors, 1, 0, n_starts=2, max_iter=10)
            print(f"✓ MLE estimation completed: {mle_result}")
        except Exception as e:
            print(f"⚠ MLE estimation issue: {e}")
        
        # Test model order selection
        try:
            order_result = ct.carma_choose_order(times, values, errors, max_p=2, max_q=1)
            print(f"✓ Model order selection: best p={order_result.best_p}, q={order_result.best_q}")
        except Exception as e:
            print(f"⚠ Model order selection issue: {e}")
        
    except Exception as e:
        print(f"✗ Failed time series tests: {e}")
    
    print("\n🎉 New CARMA implementation basic functionality test completed!")
    
except ImportError as e:
    print(f"✗ Failed to import chronoxtract: {e}")
    print("Please build the module first with: maturin develop")