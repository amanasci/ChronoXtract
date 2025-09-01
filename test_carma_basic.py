#!/usr/bin/env python3
"""
Simple test script for CARMA module functionality
"""

import numpy as np
import sys
import os

# Add the build directory to Python path
sys.path.insert(0, '/home/runner/work/ChronoXtract/ChronoXtract/target/debug')

try:
    import chronoxtract as ct
    print("✓ Successfully imported chronoxtract")
    
    # Test basic CARMA model creation
    try:
        model = ct.carma_model(2, 1)
        print(f"✓ Created CARMA model: {model}")
    except Exception as e:
        print(f"✗ Failed to create CARMA model: {e}")
        
    # Test setting parameters
    try:
        ct.set_carma_parameters(model, [0.5, 0.3], [1.0, 0.2], 1.0)
        print("✓ Set CARMA parameters successfully")
    except Exception as e:
        print(f"✗ Failed to set CARMA parameters: {e}")
        
    # Test simple simulation
    try:
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = ct.simulate_carma(model, times, seed=42)
        print(f"✓ Simulated CARMA: {len(values)} values")
        print(f"  Values: {values}")
    except Exception as e:
        print(f"✗ Failed to simulate CARMA: {e}")
        
    # Test PSD computation
    try:
        frequencies = np.array([0.1, 0.2, 0.5])
        psd = ct.carma_psd(model, frequencies)
        print(f"✓ Computed PSD: {psd}")
    except Exception as e:
        print(f"✗ Failed to compute PSD: {e}")
        
    # Test method of moments
    try:
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
        result = ct.carma_method_of_moments(times, values, 2, 1)
        print(f"✓ Method of moments: {result}")
    except Exception as e:
        print(f"✗ Failed method of moments: {e}")
        
    print("\n🎉 CARMA module basic functionality test completed!")
    
except ImportError as e:
    print(f"✗ Failed to import chronoxtract: {e}")
    print("Please build the module first with: maturin develop")