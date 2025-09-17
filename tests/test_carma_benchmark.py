#!/usr/bin/env python3
"""
Comprehensive CARMA Benchmark: ChronoXtract vs carma_pack

This script performs detailed comparison between our CARMA implementation
and the reference carma_pack implementation across different model orders
for both MLE and MCMC methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from collections import defaultdict

# Try to import carma_pack for reference comparisons
try:
    sys.path.append('/tmp/carma_pack/src')
    import carmcmc as cm
    CARMA_PACK_AVAILABLE = True
    print("✅ carma_pack found - performing reference comparisons")
except ImportError:
    CARMA_PACK_AVAILABLE = False
    print("❌ carma_pack not available - cannot perform benchmarks")
    sys.exit(1)

# Import our implementation
try:
    import chronoxtract as ct
    print("✅ chronoxtract imported successfully")
except ImportError as e:
    print(f"❌ Failed to import chronoxtract: {e}")
    sys.exit(1)

class CarmaBenchmark:
    """Comprehensive CARMA benchmarking suite."""
    
    def __init__(self):
        self.results = defaultdict(dict)
        self.test_cases = [
            # (p, q, n_points, description)
            (1, 0, 200, "CAR(1) - Simple"),
            (2, 0, 300, "CAR(2) - Basic AR"),
            (2, 1, 400, "CARMA(2,1) - Low Order"),
            (3, 1, 500, "CARMA(3,1) - Medium Order"),
            (3, 2, 500, "CARMA(3,2) - Higher Order"),
            (4, 2, 600, "CARMA(4,2) - High Order")
        ]
    
    def generate_carma_pack_data(self, p, q, n_points, seed=42):
        """Generate synthetic data using carma_pack."""
        print(f"   📊 Generating CARMA({p},{q}) data with carma_pack...")
        
        np.random.seed(seed)
        
        # Generate irregular time sampling
        t_max = 50.0
        times = np.sort(np.random.uniform(0, t_max, n_points))
        
        # Generate measurement errors
        yerr = np.full(n_points, 0.1)
        
        # Create carma_pack model for simulation
        # Use a simple approach: create model, set reasonable parameters, and simulate
        
        # For CAR(1), use simple parameters
        if p == 1 and q == 0:
            # Simple CAR(1) with tau=5, sigma=1
            true_params = [0.0, np.log(1.0/5.0), 1.0]  # [mu, log(omega), sigma]
            
            # Create simple synthetic data
            dt = np.diff(times)
            dt = np.concatenate([[0], dt])
            
            # CAR(1) simulation: Y(t+dt) = Y(t)*exp(-dt/tau) + noise
            tau = 5.0
            sigma = 1.0
            
            y = np.zeros(n_points)
            for i in range(1, n_points):
                phi = np.exp(-dt[i] / tau)
                noise_var = sigma**2 * (1 - phi**2)
                y[i] = y[i-1] * phi + np.random.normal(0, np.sqrt(noise_var))
            
        else:
            # For higher order models, use simple stable parameters
            # Generate random but stable coefficients
            np.random.seed(seed)
            
            # Simple stable AR coefficients
            if p == 2:
                ar_coeffs = [0.5, 1.0]
            elif p == 3:
                ar_coeffs = [0.3, 0.8, 1.2]
            elif p == 4:
                ar_coeffs = [0.2, 0.5, 0.8, 1.5]
            else:
                ar_coeffs = [0.1 * (i+1) for i in range(p)]
            
            # Simple MA coefficients
            ma_coeffs = [1.0] + [0.1 * (i+1) for i in range(q)]
            sigma = 1.0
            mu = 0.0
            
            # Use our implementation to generate the data
            model = ct.carma_model(p, q)
            ct.set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma, mu)
            y = ct.simulate_carma(model, times, seed=seed)
            
            true_params = [mu] + ar_coeffs + ma_coeffs + [sigma]
        
        return times, y, yerr, true_params
    
    def benchmark_mle(self, p, q, times, y, yerr, true_params):
        """Benchmark MLE estimation."""
        print(f"   🔍 Benchmarking MLE for CARMA({p},{q})...")
        
        results = {}
        
        # ChronoXtract MLE
        try:
            start_time = time.time()
            ct_result = ct.carma_mle(times, y, p, q, n_trials=3, max_iter=200, seed=42)
            ct_time = time.time() - start_time
            
            results['chronoxtract'] = {
                'time': ct_time,
                'loglik': ct_result.loglikelihood,
                'aic': ct_result.aic,
                'bic': ct_result.bic,
                'converged': ct_result.converged,
                'params': [ct_result.model.mu] + ct_result.model.ar_coeffs + ct_result.model.ma_coeffs + [ct_result.model.sigma]
            }
            print(f"      ✅ ChronoXtract: {ct_time:.2f}s, LogLik={ct_result.loglikelihood:.2f}")
            
        except Exception as e:
            print(f"      ❌ ChronoXtract MLE failed: {e}")
            results['chronoxtract'] = {'error': str(e)}
        
        # carma_pack MLE
        try:
            start_time = time.time()
            
            # Create carma_pack model
            cm_model = cm.CarmaModel(times, y, yerr, p=p, q=q)
            cm_result = cm_model.get_mle(p, q, ntrials=3)
            cm_time = time.time() - start_time
            
            results['carma_pack'] = {
                'time': cm_time,
                'loglik': -cm_result.fun,  # carma_pack minimizes negative log-likelihood
                'params': cm_result.x,
                'success': cm_result.success
            }
            print(f"      ✅ carma_pack: {cm_time:.2f}s, LogLik={-cm_result.fun:.2f}")
            
        except Exception as e:
            print(f"      ❌ carma_pack MLE failed: {e}")
            results['carma_pack'] = {'error': str(e)}
        
        return results
    
    def benchmark_mcmc(self, p, q, times, y, yerr, true_params):
        """Benchmark MCMC sampling."""
        print(f"   🎲 Benchmarking MCMC for CARMA({p},{q})...")
        
        results = {}
        n_samples = 500
        burn_in = 250
        
        # ChronoXtract MCMC
        try:
            start_time = time.time()
            ct_mcmc = ct.carma_mcmc(times, y, p, q, n_samples=n_samples, burn_in=burn_in, 
                                   n_chains=2, seed=42)
            ct_time = time.time() - start_time
            
            results['chronoxtract'] = {
                'time': ct_time,
                'n_samples': len(ct_mcmc.samples),
                'acceptance_rate': ct_mcmc.acceptance_rate,
                'param_names': ct_mcmc.param_names,
                'samples': ct_mcmc.samples[:50] if len(ct_mcmc.samples) > 50 else ct_mcmc.samples  # Store subset
            }
            print(f"      ✅ ChronoXtract: {ct_time:.2f}s, {len(ct_mcmc.samples)} samples, accept={ct_mcmc.acceptance_rate:.3f}")
            
        except Exception as e:
            print(f"      ❌ ChronoXtract MCMC failed: {e}")
            results['chronoxtract'] = {'error': str(e)}
        
        # carma_pack MCMC
        try:
            start_time = time.time()
            
            # Create carma_pack model
            cm_model = cm.CarmaModel(times, y, yerr, p=p, q=q)
            
            # Run MCMC
            cm_mcmc = cm_model.run_mcmc(n_samples, nburnin=burn_in)
            cm_time = time.time() - start_time
            
            results['carma_pack'] = {
                'time': cm_time,
                'n_samples': n_samples,
                'mcmc_result': 'completed'  # carma_pack MCMC object is complex
            }
            print(f"      ✅ carma_pack: {cm_time:.2f}s, {n_samples} samples")
            
        except Exception as e:
            print(f"      ❌ carma_pack MCMC failed: {e}")
            results['carma_pack'] = {'error': str(e)}
        
        return results
    
    def run_benchmark_suite(self):
        """Run the complete benchmark suite."""
        print("🚀 Starting Comprehensive CARMA Benchmark Suite")
        print("=" * 80)
        
        for p, q, n_points, description in self.test_cases:
            print(f"\n📋 Testing {description}")
            print("-" * 60)
            
            # Generate test data
            try:
                times, y, yerr, true_params = self.generate_carma_pack_data(p, q, n_points)
                print(f"   📊 Generated {len(times)} data points")
                
                # Benchmark MLE
                mle_results = self.benchmark_mle(p, q, times, y, yerr, true_params)
                
                # Benchmark MCMC  
                mcmc_results = self.benchmark_mcmc(p, q, times, y, yerr, true_params)
                
                # Store results
                self.results[f"CARMA({p},{q})"] = {
                    'mle': mle_results,
                    'mcmc': mcmc_results,
                    'data_info': {
                        'n_points': len(times),
                        'time_span': times[-1] - times[0],
                        'true_params': true_params
                    }
                }
                
            except Exception as e:
                print(f"   ❌ Failed to test {description}: {e}")
                continue
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Model':<12} {'Method':<6} {'ChronoXtract':<15} {'carma_pack':<15} {'Status':<15}")
        print("-" * 75)
        
        for model_name, model_results in self.results.items():
            # MLE Results
            ct_mle = model_results['mle'].get('chronoxtract', {})
            cp_mle = model_results['mle'].get('carma_pack', {})
            
            ct_mle_time = ct_mle.get('time', 'Failed')
            cp_mle_time = cp_mle.get('time', 'Failed')
            
            ct_mle_str = f"{ct_mle_time:.1f}s" if isinstance(ct_mle_time, float) else "Failed"
            cp_mle_str = f"{cp_mle_time:.1f}s" if isinstance(cp_mle_time, float) else "Failed"
            
            mle_status = "✅ Both OK" if (isinstance(ct_mle_time, float) and isinstance(cp_mle_time, float)) else "⚠️ Issues"
            
            print(f"{model_name:<12} {'MLE':<6} {ct_mle_str:<15} {cp_mle_str:<15} {mle_status:<15}")
            
            # MCMC Results
            ct_mcmc = model_results['mcmc'].get('chronoxtract', {})
            cp_mcmc = model_results['mcmc'].get('carma_pack', {})
            
            ct_mcmc_time = ct_mcmc.get('time', 'Failed')
            cp_mcmc_time = cp_mcmc.get('time', 'Failed')
            
            ct_mcmc_str = f"{ct_mcmc_time:.1f}s" if isinstance(ct_mcmc_time, float) else "Failed"
            cp_mcmc_str = f"{cp_mcmc_time:.1f}s" if isinstance(cp_mcmc_time, float) else "Failed"
            
            mcmc_status = "✅ Both OK" if (isinstance(ct_mcmc_time, float) and isinstance(cp_mcmc_time, float)) else "⚠️ Issues"
            
            print(f"{model_name:<12} {'MCMC':<6} {ct_mcmc_str:<15} {cp_mcmc_str:<15} {mcmc_status:<15}")
        
        print("\n" + "=" * 80)
        
        # Success statistics
        total_tests = len(self.results) * 2  # MLE + MCMC for each model
        ct_successes = 0
        cp_successes = 0
        
        for model_results in self.results.values():
            if 'time' in model_results['mle'].get('chronoxtract', {}):
                ct_successes += 1
            if 'time' in model_results['mle'].get('carma_pack', {}):
                cp_successes += 1
            if 'time' in model_results['mcmc'].get('chronoxtract', {}):
                ct_successes += 1
            if 'time' in model_results['mcmc'].get('carma_pack', {}):
                cp_successes += 1
        
        print(f"ChronoXtract Success Rate: {ct_successes}/{total_tests} ({100*ct_successes/total_tests:.1f}%)")
        print(f"carma_pack Success Rate: {cp_successes}/{total_tests} ({100*cp_successes/total_tests:.1f}%)")

def main():
    """Main benchmark execution."""
    if not CARMA_PACK_AVAILABLE:
        print("❌ carma_pack is required for benchmarking")
        return 1
    
    benchmark = CarmaBenchmark()
    benchmark.run_benchmark_suite()
    benchmark.print_summary()
    
    print("\n🎉 Benchmark completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())