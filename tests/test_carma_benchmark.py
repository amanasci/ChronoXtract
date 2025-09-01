#!/usr/bin/env python3
"""
Basic benchmarking test for CARMA module
"""

import numpy as np
import time
import chronoxtract as ct

def benchmark_carma():
    print("🔥 CARMA Module Benchmark")
    print("=" * 50)
    
    # Create a CARMA model
    model = ct.carma_model(3, 2)
    ct.set_carma_parameters(model, [0.8, 0.3, 0.1], [1.0, 0.4, 0.2], 1.0)
    
    print(f"Model: {model}")
    
    # Test 1: Simulation performance
    print("\n📊 Simulation Benchmark:")
    sizes = [100, 500, 1000]
    
    for n in sizes:
        times = np.linspace(0, 10, n)
        
        start = time.time()
        values = ct.simulate_carma(model, times, seed=42)
        end = time.time()
        
        print(f"  N={n:4d}: {end-start:.4f}s ({n/(end-start):.0f} pts/sec)")
    
    # Test 2: PSD computation performance  
    print("\n📈 PSD Computation Benchmark:")
    freq_sizes = [50, 100, 200]
    
    for n_freq in freq_sizes:
        frequencies = np.linspace(0.01, 1.0, n_freq)
        
        start = time.time()
        psd = ct.carma_psd(model, frequencies)
        end = time.time()
        
        print(f"  N_freq={n_freq:3d}: {end-start:.4f}s ({n_freq/(end-start):.0f} freq/sec)")
    
    # Test 3: Model fitting performance
    print("\n🎯 Fitting Benchmark:")
    sizes = [50, 100, 200]
    
    for n in sizes:
        # Generate synthetic data
        times = np.sort(np.random.uniform(0, 10, n))
        values = np.random.normal(0, 1, n)
        
        start = time.time()
        result = ct.carma_method_of_moments(times, values, 2, 1)
        end = time.time()
        
        print(f"  N={n:3d}: {end-start:.4f}s (AIC={result.aic:.2f})")
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    benchmark_carma()