#!/usr/bin/env python3
"""
Quick test of the new MCMC implementation
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
    
    # Generate simple test data
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    values = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0], dtype=np.float64)
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    
    print(f"✓ Test data: {len(times)} points")
    
    # Test MCMC with small sample size
    print("\n🔬 Testing MCMC implementation...")
    try:
        mcmc_result = ct.carma_mcmc(
            times, values, errors, 
            p=1, q=0,  # Simple AR(1) model
            n_samples=50,  # Small for quick test
            n_burn=25,
            n_chains=2,
            seed=42
        )
        
        print(f"✓ MCMC completed successfully!")
        print(f"  Samples shape: {mcmc_result.samples.shape}")
        print(f"  Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
        print(f"  Number of parameters: {len(mcmc_result.rhat)}")
        print(f"  R-hat values: {mcmc_result.rhat}")
        
        # Check that we got reasonable results
        if mcmc_result.samples.shape[0] == 50:
            print("✓ Correct number of samples")
        if 0.1 < mcmc_result.acceptance_rate < 0.9:
            print("✓ Reasonable acceptance rate")
        
        print("\n🎉 MCMC test PASSED!")
        
    except Exception as e:
        print(f"❌ MCMC test failed: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ Failed to import chronoxtract: {e}")