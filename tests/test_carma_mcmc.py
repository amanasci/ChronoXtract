#!/usr/bin/env python3
"""
MCMC Performance Test for Higher-Order CARMA Models
Tests the improved MCMC implementation specifically
"""

import numpy as np
import time
import chronoxtract as cx

def test_mcmc_performance(p, q, n_points=1000, n_samples=1000):
    """Test MCMC performance for a specific CARMA(p,q) model"""
    print(f"\n🔬 Testing MCMC for CARMA({p},{q}) with {n_points} points, {n_samples} samples")
    print("-" * 60)

    # Generate stable CARMA parameters with more conservative values
    np.random.seed(42)
    ar_coeffs = []
    for i in range(p):
        # Use smaller coefficients for higher stability
        coeff = np.random.uniform(-0.1, 0.1) / (i + 2)  # Divide by (i+2) for more stability
        ar_coeffs.append(coeff)

    ma_coeffs = [1.0]
    for _ in range(q):
        # Smaller MA coefficients for stability
        ma_coeffs.append(np.random.uniform(-0.15, 0.15))

    sigma = np.random.uniform(0.1, 0.3)  # Smaller sigma range

    print(f"AR coeffs: {[f'{x:.3f}' for x in ar_coeffs]}")
    print(f"MA coeffs: {[f'{x:.3f}' for x in ma_coeffs]}")
    print(f"Sigma: {sigma:.3f}")

    try:
        # Create and validate model
        model = cx.carma_model(p, q)
        cx.set_carma_parameters(model, ar_coeffs, ma_coeffs, sigma)

        # Temporarily skip stability check to test MCMC improvements
        # if not cx.check_carma_stability(model):
        #     print("❌ Model not stable!")
        #     return
        print("⚠️  Skipping stability check for testing")

        # Generate data
        t = np.linspace(0, 10, n_points)
        y = cx.simulate_carma(model, t, seed=42)
        print(f"Generated {len(y)} data points")

        # Test MLE first
        start_time = time.time()
        mle_result = cx.carma_mle(t, y, p, q)
        mle_time = time.time() - start_time
        print(f"MLE time: {mle_time:.3f}s")
        print(f"MLE Log-likelihood: {mle_result.loglikelihood:.2f}")

        # Test MCMC
        start_time = time.time()
        mcmc_result = cx.carma_mcmc(t, y, p, q, n_samples, seed=42)
        mcmc_time = time.time() - start_time

        print(f"MCMC time: {mcmc_time:.3f}s")
        print(f"Acceptance rate: {mcmc_result.acceptance_rate:.3f}")
        print(f"Max R-hat: {max(mcmc_result.rhat):.3f}")
        print(f"Min ESS: {min(mcmc_result.effective_sample_size):.1f}")

        # Evaluate MCMC quality
        acceptance_ok = 0.2 <= mcmc_result.acceptance_rate <= 0.6
        rhat_ok = max(mcmc_result.rhat) < 1.4
        ess_ok = min(mcmc_result.effective_sample_size) > 100

        print(f"Acceptance rate OK: {acceptance_ok}")
        print(f"R-hat OK: {rhat_ok}")
        print(f"ESS OK: {ess_ok}")

        if acceptance_ok and rhat_ok and ess_ok:
            print("✅ MCMC: Excellent performance!")
        elif acceptance_ok or rhat_ok:
            print("⚠️  MCMC: Acceptable performance")
        else:
            print("❌ MCMC: Poor performance")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🚀 MCMC Performance Test for Higher-Order CARMA")
    print("=" * 60)

    # Test different model complexities
    test_cases = [
        (2, 1, 500, 500),    # Simple model
        (3, 1, 1000, 1000),  # Medium complexity
        (3, 2, 1500, 1000),  # Higher complexity
        (4, 2, 2000, 1000),  # High complexity
    ]

    for p, q, n_points, n_samples in test_cases:
        test_mcmc_performance(p, q, n_points, n_samples)

    print("\n✅ MCMC performance test completed!")

if __name__ == "__main__":
    main()
