#!/usr/bin/env python3
"""
CARMA MCMC Fix Validation and Basic Benchmarking

Tests the MCMC fixes and provides basic performance benchmarking
across different CARMA model orders without requiring carma_pack.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from collections import defaultdict

# Import our implementation
try:
    import chronoxtract as ct
    print("✅ chronoxtract imported successfully")
except ImportError as e:
    print(f"❌ Failed to import chronoxtract: {e}")
    sys.exit(1)

class McmcFixValidation:
    """Validation of MCMC fixes and performance benchmarking."""
    
    def __init__(self):
        self.results = {}
        # Test cases: (p, q, n_points, description)
        self.test_cases = [
            (1, 0, 150, "CAR(1) - Simple"),
            (2, 0, 200, "CAR(2) - Basic AR"),
            (2, 1, 250, "CARMA(2,1) - Low Order"),
            (3, 1, 300, "CARMA(3,1) - Medium Order"),
            (3, 2, 350, "CARMA(3,2) - Higher Order"),
            (4, 2, 400, "CARMA(4,2) - High Order")
        ]
    
    def test_mcmc_innovation_fix(self):
        """Test that the innovation variance fix works."""
        print("\n🔧 Testing MCMC Innovation Variance Fix")
        print("=" * 60)
        
        # Use parameters that might cause numerical issues
        try:
            # Create a simple CAR(1) model
            model = ct.carma_model(1, 0)
            ct.set_carma_parameters(model, [1.5], [1.0], 0.5, 0.0)
            
            # Generate data
            times = np.linspace(0, 20, 100)
            values = ct.simulate_carma(model, times, seed=42)
            
            # Test MCMC - this should work now with the fix
            result = ct.carma_mcmc(times, values, 1, 0, n_samples=200, burn_in=100, 
                                  n_chains=2, seed=42)
            
            print(f"✅ MCMC Innovation Fix Test Passed")
            print(f"   Generated {len(result.all_samples)} samples")
            print(f"   Acceptance rate: {result.acceptance_rate:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ MCMC Innovation Fix Test Failed: {e}")
            return False
    
    def generate_stable_test_data(self, p, q, n_points, seed=42):
        """Generate stable test data for given model order."""
        np.random.seed(seed)
        
        # Generate stable parameters
        ar_coeffs, ma_coeffs, sigma = ct.generate_stable_carma_parameters(p, q, seed=seed)
        
        # Create model and simulate
        model = ct.carma_model(p, q)
        ct.set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma, 0.0)
        
        # Generate irregular time points
        duration = 30.0
        times = np.sort(np.random.uniform(0, duration, n_points))
        
        # Simulate
        values = ct.simulate_carma(model, times, seed=seed)
        
        return times, values, model, (ar_coeffs, ma_coeffs, sigma)
    
    def benchmark_model_order(self, p, q, n_points, description):
        """Benchmark a specific model order."""
        print(f"\n📊 Benchmarking {description}")
        print("-" * 50)
        
        try:
            # Generate test data
            times, values, true_model, true_params = self.generate_stable_test_data(p, q, n_points)
            ar_coeffs, ma_coeffs, sigma = true_params
            
            print(f"   Data: {len(times)} points over {times[-1]:.1f} time units")
            print(f"   True AR: {[f'{x:.2f}' for x in ar_coeffs]}")
            print(f"   True MA: {[f'{x:.2f}' for x in ma_coeffs]}")
            print(f"   True σ: {sigma:.2f}")
            
            results = {'model_order': (p, q), 'n_points': n_points}
            
            # Test MLE
            try:
                start_time = time.time()
                mle_result = ct.carma_mle(times, values, p, q, n_trials=3, max_iter=100, seed=42)
                mle_time = time.time() - start_time
                
                results['mle'] = {
                    'time': mle_time,
                    'loglik': mle_result.loglikelihood,
                    'aic': mle_result.aic,
                    'bic': mle_result.bic,
                    'converged': mle_result.converged,
                    'params': {
                        'ar': mle_result.model.ar_coeffs,
                        'ma': mle_result.model.ma_coeffs,
                        'sigma': mle_result.model.sigma,
                        'mu': mle_result.model.mu
                    }
                }
                
                print(f"   ✅ MLE: {mle_time:.1f}s, LogLik={mle_result.loglikelihood:.2f}")
                
            except Exception as e:
                print(f"   ❌ MLE failed: {e}")
                results['mle'] = {'error': str(e)}
            
            # Test MCMC with different configurations
            mcmc_configs = [
                {'n_samples': 300, 'burn_in': 150, 'n_chains': 1, 'name': 'Single Chain'},
                {'n_samples': 400, 'burn_in': 200, 'n_chains': 2, 'name': 'Parallel Tempering'},
            ]
            
            results['mcmc'] = {}
            
            for config in mcmc_configs:
                try:
                    start_time = time.time()
                    mcmc_result = ct.carma_mcmc(
                        times, values, p, q,
                        n_samples=config['n_samples'],
                        burn_in=config['burn_in'],
                        n_chains=config['n_chains'],
                        seed=42
                    )
                    mcmc_time = time.time() - start_time
                    
                    results['mcmc'][config['name']] = {
                        'time': mcmc_time,
                        'n_samples': len(mcmc_result.all_samples),
                        'acceptance_rate': mcmc_result.acceptance_rate,
                        'effective_samples_per_sec': len(mcmc_result.all_samples) / mcmc_time if mcmc_time > 0 else 0
                    }
                    
                    print(f"   ✅ MCMC {config['name']}: {mcmc_time:.1f}s, "
                          f"{len(mcmc_result.all_samples)} samples, accept={mcmc_result.acceptance_rate:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ MCMC {config['name']} failed: {e}")
                    results['mcmc'][config['name']] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
            return {'error': str(e)}
    
    def run_full_benchmark(self):
        """Run comprehensive benchmark suite."""
        print("🚀 CARMA MCMC Fix Validation and Benchmark Suite")
        print("=" * 80)
        
        # Test the innovation variance fix
        fix_success = self.test_mcmc_innovation_fix()
        
        if not fix_success:
            print("\n❌ MCMC fix validation failed - stopping benchmark")
            return
        
        print(f"\n📈 Running Performance Benchmarks")
        print("=" * 80)
        
        # Run benchmarks for each test case
        for p, q, n_points, description in self.test_cases:
            result = self.benchmark_model_order(p, q, n_points, description)
            self.results[f"CARMA({p},{q})"] = result
    
    def print_performance_summary(self):
        """Print performance summary."""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Header
        print(f"{'Model':<12} {'Points':<8} {'MLE Time':<10} {'MCMC (Single)':<15} {'MCMC (Parallel)':<15} {'Status':<10}")
        print("-" * 85)
        
        # Results
        for model_name, results in self.results.items():
            if 'error' in results:
                print(f"{model_name:<12} {'N/A':<8} {'Failed':<10} {'Failed':<15} {'Failed':<15} {'❌':<10}")
                continue
            
            n_points = results.get('n_points', 'N/A')
            
            # MLE time
            mle_time = results.get('mle', {}).get('time', 'Failed')
            mle_str = f"{mle_time:.1f}s" if isinstance(mle_time, float) else "Failed"
            
            # MCMC times
            mcmc_single = results.get('mcmc', {}).get('Single Chain', {})
            mcmc_parallel = results.get('mcmc', {}).get('Parallel Tempering', {})
            
            single_time = mcmc_single.get('time', 'Failed')
            parallel_time = mcmc_parallel.get('time', 'Failed')
            
            single_str = f"{single_time:.1f}s" if isinstance(single_time, float) else "Failed"
            parallel_str = f"{parallel_time:.1f}s" if isinstance(parallel_time, float) else "Failed"
            
            # Status
            all_success = (isinstance(mle_time, float) and 
                          isinstance(single_time, float) and 
                          isinstance(parallel_time, float))
            status = "✅" if all_success else "⚠️"
            
            print(f"{model_name:<12} {n_points:<8} {mle_str:<10} {single_str:<15} {parallel_str:<15} {status:<10}")
        
        # Success rate
        print("\n" + "-" * 85)
        total_models = len(self.results)
        successful_models = sum(1 for r in self.results.values() if 'error' not in r)
        
        if total_models > 0:
            print(f"Overall Success Rate: {successful_models}/{total_models} ({100*successful_models/total_models:.1f}%)")
        else:
            print("No models tested")
        
        # Performance insights
        if successful_models > 0:
            print(f"\n📈 Performance Insights:")
            
            # Average MLE time
            mle_times = [r['mle']['time'] for r in self.results.values() 
                        if 'mle' in r and 'time' in r['mle'] and isinstance(r['mle']['time'], float)]
            
            if mle_times:
                print(f"   Average MLE time: {np.mean(mle_times):.1f}s (range: {min(mle_times):.1f}s - {max(mle_times):.1f}s)")
            
            # Average MCMC acceptance rates
            accept_rates = []
            for r in self.results.values():
                for mcmc_type in ['Single Chain', 'Parallel Tempering']:
                    mcmc_result = r.get('mcmc', {}).get(mcmc_type, {})
                    if 'acceptance_rate' in mcmc_result:
                        accept_rates.append(mcmc_result['acceptance_rate'])
            
            if accept_rates:
                print(f"   Average MCMC acceptance rate: {np.mean(accept_rates):.3f}")
                
            # Efficiency
            efficiencies = []
            for r in self.results.values():
                for mcmc_type in ['Single Chain', 'Parallel Tempering']:
                    mcmc_result = r.get('mcmc', {}).get(mcmc_type, {})
                    if 'effective_samples_per_sec' in mcmc_result:
                        efficiencies.append(mcmc_result['effective_samples_per_sec'])
            
            if efficiencies:
                print(f"   Average MCMC efficiency: {np.mean(efficiencies):.1f} samples/second")

def main():
    """Main validation and benchmark execution."""
    print("🔧 CARMA MCMC Fix Validation and Benchmarking")
    print("=" * 60)
    
    validator = McmcFixValidation()
    validator.run_full_benchmark()
    validator.print_performance_summary()
    
    print("\n🎉 Validation and benchmark completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())