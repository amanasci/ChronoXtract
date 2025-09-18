#!/usr/bin/env python3
"""
Comprehensive CARMA benchmark comparing ChronoXtract with reference implementations
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm
import chronoxtract as ct

class ReferenceCarmaAR1:
    """
    Reference AR(1) implementation using simple analytical solutions
    """
    
    def __init__(self, phi, sigma, mu=0.0):
        """
        AR(1) model: x(t) = phi * x(t-1) + sigma * noise + mu
        """
        if abs(phi) >= 1.0:
            raise ValueError(f"AR(1) coefficient must satisfy |phi| < 1, got {phi}")
        
        self.phi = phi
        self.sigma = sigma
        self.mu = mu
    
    def generate_data(self, times):
        """Generate AR(1) data at given times"""
        n = len(times)
        values = np.zeros(n)
        
        # Initialize with stationary mean
        values[0] = self.mu + self.sigma * np.random.randn() / np.sqrt(1 - self.phi**2)
        
        for i in range(1, n):
            dt = times[i] - times[i-1]
            # For irregular sampling, use continuous-time AR(1)
            phi_t = np.exp(-dt / (1.0 / (1.0 - self.phi)))  # Approximate
            values[i] = self.mu + phi_t * (values[i-1] - self.mu) + self.sigma * np.random.randn()
        
        return values
    
    def loglikelihood(self, times, values, errors):
        """Compute log-likelihood for AR(1) model"""
        n = len(values)
        loglik = 0.0
        
        # Initialize with stationary distribution
        state_mean = self.mu
        state_var = self.sigma**2 / (1 - self.phi**2)
        
        for i in range(n):
            # Predict
            pred_mean = state_mean
            pred_var = state_var + errors[i]**2
            
            # Update (simplified)
            innovation = values[i] - pred_mean
            loglik += norm.logpdf(innovation, 0, np.sqrt(pred_var))
            
            # Kalman update (simplified)
            gain = state_var / pred_var
            state_mean = state_mean + gain * innovation
            state_var = (1 - gain) * state_var
            
            # Predict next state (if not last)
            if i < n - 1:
                dt = times[i+1] - times[i]
                phi_t = np.exp(-dt / (1.0 / (1.0 - self.phi)))  # Approximate
                state_mean = self.mu + phi_t * (state_mean - self.mu)
                state_var = self.sigma**2 * (1 - phi_t**2) + phi_t**2 * state_var
        
        return loglik

def benchmark_ar1_comparison(n_points_list=[50, 100, 200, 500]):
    """
    Benchmark AR(1) models comparing ChronoXtract vs Reference
    """
    print("🚀 AR(1) Model Benchmark")
    print("=" * 60)
    
    # Test parameters
    true_phi = 0.7
    true_sigma = 1.0
    measurement_error = 0.1
    
    results = []
    
    for n_points in n_points_list:
        print(f"\n📊 Testing with {n_points} data points...")
        
        # Generate reference data
        ref_model = ReferenceCarmaAR1(true_phi, true_sigma)
        times = np.sort(np.random.uniform(0, 10, n_points))
        times[0] = 0.0  # Ensure we start at t=0
        values = ref_model.generate_data(times)
        errors = np.full(n_points, measurement_error)
        
        print(f"  Generated data: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
        
        # Test reference implementation
        start_time = time.time()
        try:
            ref_loglik = ref_model.loglikelihood(times, values, errors)
            ref_time = time.time() - start_time
            print(f"  ✓ Reference AR(1): loglik={ref_loglik:.4f} ({ref_time*1000:.2f} ms)")
        except Exception as e:
            print(f"  ❌ Reference AR(1): {e}")
            ref_loglik = None
            ref_time = None
        
        # Test ChronoXtract with manual parameters
        start_time = time.time()
        try:
            # Create CARMA(1,0) ≡ AR(1) 
            params = ct.CarmaParams(1, 0)
            # For AR(1): s + α = 0, we want root at -λ where λ > 0 for stability
            # If phi = 0.7, then for continuous time we need α such that the root gives similar behavior
            # Let's try α = 1.0 (which gives root at -1.0, very stable)
            params.ar_coeffs = [1.0]  # This should give stable behavior
            params.ma_coeffs = [1.0]
            params.sigma = true_sigma
            
            ct_loglik = ct.carma_loglikelihood(params, times, values, errors)
            ct_time = time.time() - start_time
            print(f"  ✓ ChronoXtract: loglik={ct_loglik:.4f} ({ct_time*1000:.2f} ms)")
        except Exception as e:
            print(f"  ❌ ChronoXtract: {e}")
            ct_loglik = None
            ct_time = None
        
        # Test ChronoXtract MLE
        start_time = time.time()
        try:
            mle_result = ct.carma_mle(times, values, errors, 1, 0, n_starts=4, max_iter=20)
            mle_time = time.time() - start_time
            print(f"  ✓ ChronoXtract MLE: AICc={mle_result.aicc:.4f} ({mle_time*1000:.2f} ms)")
            print(f"    Fitted AR coeff: {mle_result.params.ar_coeffs[0]:.3f}")
        except Exception as e:
            print(f"  ❌ ChronoXtract MLE: {e}")
            mle_result = None
            mle_time = None
        
        results.append({
            'n_points': n_points,
            'ref_loglik': ref_loglik,
            'ref_time': ref_time,
            'ct_loglik': ct_loglik,
            'ct_time': ct_time,
            'mle_result': mle_result,
            'mle_time': mle_time
        })
    
    return results

def test_parameter_recovery():
    """Test parameter recovery for known AR(1) model"""
    print("\n🔧 Parameter Recovery Test")
    print("=" * 60)
    
    true_params = [
        (0.5, 1.0), (0.8, 0.5), (0.3, 2.0), (0.9, 1.5)
    ]
    
    for true_phi, true_sigma in true_params:
        print(f"\n📋 Testing φ={true_phi}, σ={true_sigma}")
        
        # Generate data
        ref_model = ReferenceCarmaAR1(true_phi, true_sigma)
        n_points = 200
        times = np.linspace(0, 10, n_points)
        values = ref_model.generate_data(times)
        errors = np.full(n_points, 0.1)
        
        # Test if ChronoXtract can recover parameters
        try:
            mle_result = ct.carma_mle(times, values, errors, 1, 0, n_starts=8, max_iter=50)
            print(f"  ✓ MLE succeeded: AICc={mle_result.aicc:.4f}")
            print(f"    AR coeff: {mle_result.params.ar_coeffs[0]:.3f} (true conversion needed)")
            print(f"    Sigma: {mle_result.params.sigma:.3f} (true: {true_sigma:.3f})")
        except Exception as e:
            print(f"  ❌ MLE failed: {e}")

def test_higher_order_models():
    """Test higher order CARMA models"""
    print("\n🔧 Higher Order CARMA Models")
    print("=" * 60)
    
    # Test CARMA(2,0), CARMA(2,1), CARMA(3,1), etc.
    test_orders = [(2, 0), (2, 1), (3, 1), (3, 2)]
    
    for p, q in test_orders:
        print(f"\n📋 Testing CARMA({p},{q})")
        
        # Generate simple synthetic data
        n_points = 100
        times = np.linspace(0, 10, n_points)
        values = np.sin(0.5 * times) + 0.2 * np.sin(2.0 * times) + 0.1 * np.random.randn(n_points)
        errors = np.full(n_points, 0.1)
        
        # Test MLE
        try:
            mle_result = ct.carma_mle(times, values, errors, p, q, n_starts=4, max_iter=20)
            print(f"  ✓ MLE succeeded: AICc={mle_result.aicc:.4f}")
            print(f"    AR coeffs: {[f'{x:.3f}' for x in mle_result.params.ar_coeffs]}")
            print(f"    MA coeffs: {[f'{x:.3f}' for x in mle_result.params.ma_coeffs]}")
        except Exception as e:
            print(f"  ❌ MLE failed: {e}")
        
        # Test MCMC (if MLE worked)
        try:
            mcmc_result = ct.carma_mcmc(times, values, errors, p, q, 
                                      n_samples=100, n_burn=50, n_chains=2)
            print(f"  ✓ MCMC succeeded: acceptance={mcmc_result.acceptance_rate:.3f}")
        except Exception as e:
            print(f"  ❌ MCMC failed: {e}")

if __name__ == "__main__":
    print("🚀 ChronoXtract CARMA Benchmark Suite")
    print("=" * 80)
    
    try:
        # Run AR(1) comparison
        benchmark_results = benchmark_ar1_comparison()
        
        # Run parameter recovery test
        test_parameter_recovery()
        
        # Run higher order tests
        test_higher_order_models()
        
        print(f"\n🎯 Benchmark Summary")
        print("=" * 80)
        print("Note: Current implementation has stationarity validation issues.")
        print("AR root computation needs debugging before reliable benchmarking.")
        print("Once fixed, this will provide comprehensive CARMA vs reference comparison.")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()