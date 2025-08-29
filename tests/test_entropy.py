import pytest
import numpy as np
import chronoxtract as ct

def test_sample_entropy():
    """Test Sample Entropy calculation"""
    # Test with a regular signal (low entropy)
    data = np.sin(np.linspace(0, 4*np.pi, 100))
    entropy = ct.sample_entropy(data, m=2, r=0.2)
    
    assert entropy >= 0.0
    assert np.isfinite(entropy)

def test_sample_entropy_constant():
    """Test Sample Entropy for constant signal"""
    data = np.array([1.0] * 50)
    entropy = ct.sample_entropy(data, m=2, r=0.1)
    
    # Constant signal should have infinite entropy (no variation)
    assert entropy == float('inf') or entropy < 0.1

def test_sample_entropy_random():
    """Test Sample Entropy for random signal"""
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    entropy = ct.sample_entropy(data, m=2, r=0.2)
    
    assert entropy > 0.0
    assert np.isfinite(entropy)

def test_approximate_entropy():
    """Test Approximate Entropy calculation"""
    data = np.sin(np.linspace(0, 4*np.pi, 100))
    entropy = ct.approximate_entropy(data, m=2, r=0.2)
    
    assert entropy >= 0.0
    assert np.isfinite(entropy)

def test_approximate_entropy_vs_sample_entropy():
    """Test that ApEn and SampEn are both finite and reasonable"""
    np.random.seed(123)
    data = np.random.normal(0, 1, 100)
    
    apen = ct.approximate_entropy(data, m=2, r=0.2)
    sampen = ct.sample_entropy(data, m=2, r=0.2)
    
    assert np.isfinite(apen)
    assert np.isfinite(sampen)
    # Both should be positive for random data
    assert apen > 0.0
    assert sampen > 0.0
    # Both should be reasonable values (not too extreme)
    assert apen < 10.0
    assert sampen < 10.0

def test_permutation_entropy():
    """Test Permutation Entropy calculation"""
    data = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0])
    entropy = ct.permutation_entropy(data, m=3, delay=1)
    
    assert entropy >= 0.0
    assert np.isfinite(entropy)

def test_permutation_entropy_sine():
    """Test Permutation Entropy for sine wave"""
    t = np.linspace(0, 4*np.pi, 100)
    data = np.sin(t)
    entropy = ct.permutation_entropy(data, m=3, delay=1)
    
    assert entropy >= 0.0
    assert np.isfinite(entropy)

def test_permutation_entropy_random():
    """Test Permutation Entropy for random signal"""
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    entropy = ct.permutation_entropy(data, m=3, delay=1)
    
    assert entropy >= 0.0
    assert np.isfinite(entropy)

def test_lempel_ziv_complexity():
    """Test Lempel-Ziv Complexity calculation"""
    # Simple periodic pattern
    data = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    complexity = ct.lempel_ziv_complexity(data, threshold=None)
    
    assert complexity > 0.0
    assert np.isfinite(complexity)

def test_lempel_ziv_complexity_with_threshold():
    """Test Lempel-Ziv Complexity with specified threshold"""
    data = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
    complexity = ct.lempel_ziv_complexity(data, threshold=2.0)
    
    assert complexity > 0.0
    assert np.isfinite(complexity)

def test_lempel_ziv_complexity_random():
    """Test that random data has higher complexity than periodic"""
    np.random.seed(42)
    random_data = np.random.normal(0, 1, 50)
    periodic_data = np.tile([1.0, 2.0], 25)
    
    complexity_random = ct.lempel_ziv_complexity(random_data, threshold=None)
    complexity_periodic = ct.lempel_ziv_complexity(periodic_data, threshold=None)
    
    # Random data should generally have higher complexity
    assert complexity_random >= complexity_periodic

def test_multiscale_entropy():
    """Test Multiscale Entropy calculation"""
    t = np.linspace(0, 4*np.pi, 200)
    data = np.sin(t) + 0.1 * np.random.normal(0, 1, 200)
    
    entropies = ct.multiscale_entropy(data, m=2, r=0.15, max_scale=5)
    
    assert len(entropies) == 5
    assert all(np.isfinite(e) or np.isnan(e) for e in entropies)
    # At least the first few scales should have finite entropy
    assert np.isfinite(entropies[0])

def test_multiscale_entropy_consistency():
    """Test that first scale of MSE matches regular sample entropy"""
    np.random.seed(123)
    data = np.random.normal(0, 1, 100)
    
    mse_entropies = ct.multiscale_entropy(data, m=2, r=0.2, max_scale=3)
    single_entropy = ct.sample_entropy(data, m=2, r=0.2)
    
    # First scale should be very close to single sample entropy
    assert abs(mse_entropies[0] - single_entropy) < 1e-10

def test_entropy_parameter_validation():
    """Test parameter validation for entropy functions"""
    data = np.array([1.0, 2.0, 3.0])
    
    # Test insufficient data length
    with pytest.raises(ValueError):
        ct.sample_entropy(data, m=5, r=0.1)  # m too large
    
    with pytest.raises(ValueError):
        ct.approximate_entropy(data, m=5, r=0.1)  # m too large
    
    with pytest.raises(ValueError):
        ct.permutation_entropy(data, m=5, delay=1)  # m too large
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        ct.sample_entropy(data, m=2, r=-0.1)  # negative r
    
    with pytest.raises(ValueError):
        ct.permutation_entropy(data, m=1, delay=1)  # m too small
    
    with pytest.raises(ValueError):
        ct.permutation_entropy(data, m=2, delay=0)  # delay too small

def test_entropy_empty_input():
    """Test error handling for empty input"""
    with pytest.raises(ValueError):
        ct.lempel_ziv_complexity(np.array([]), threshold=None)

def test_entropy_stability_large_data():
    """Test entropy calculations on larger datasets"""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    
    # These should complete without errors
    sampen = ct.sample_entropy(data, m=2, r=0.2)
    apen = ct.approximate_entropy(data, m=2, r=0.2)
    permen = ct.permutation_entropy(data, m=3, delay=1)
    lzc = ct.lempel_ziv_complexity(data, threshold=None)
    
    assert all(np.isfinite(x) for x in [sampen, apen, permen, lzc])

def test_deterministic_vs_stochastic():
    """Test that deterministic signals have different entropy than stochastic"""
    # Deterministic sine wave
    t = np.linspace(0, 10*np.pi, 500)
    deterministic = np.sin(t)
    
    # Stochastic noise
    np.random.seed(42)
    stochastic = np.random.normal(0, 1, 500)
    
    det_sampen = ct.sample_entropy(deterministic, m=2, r=0.15)
    sto_sampen = ct.sample_entropy(stochastic, m=2, r=0.15)
    
    det_permen = ct.permutation_entropy(deterministic, m=3, delay=1)
    sto_permen = ct.permutation_entropy(stochastic, m=3, delay=1)
    
    # Stochastic signals should generally have higher entropy
    assert sto_sampen > det_sampen
    assert sto_permen > det_permen

def test_entropy_different_scales():
    """Test multiscale entropy behavior across scales"""
    # Create a signal with structure at different scales
    t = np.linspace(0, 10, 1000)
    data = np.sin(2*np.pi*t) + 0.5*np.sin(2*np.pi*5*t) + 0.1*np.random.normal(0, 1, 1000)
    
    entropies = ct.multiscale_entropy(data, m=2, r=0.2, max_scale=10)
    
    # Should have at least some finite entropies
    finite_entropies = [e for e in entropies if np.isfinite(e)]
    assert len(finite_entropies) >= 5

def test_permutation_entropy_ordinal_patterns():
    """Test permutation entropy with known ordinal patterns"""
    # Monotonic increasing sequence
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    entropy = ct.permutation_entropy(data, m=3, delay=1)
    
    # Should have low entropy (only one ordinal pattern: [0,1,2])
    assert entropy >= 0.0
    assert entropy < 1.0  # Should be quite low

def test_lempel_ziv_binary_conversion():
    """Test LZC with different binary conversion thresholds"""
    data = np.array([1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.5, 4.5])
    
    lzc_low = ct.lempel_ziv_complexity(data, threshold=2.0)
    lzc_high = ct.lempel_ziv_complexity(data, threshold=3.0)
    
    # Different thresholds should potentially give different complexities
    assert lzc_low > 0.0
    assert lzc_high > 0.0
    assert np.isfinite(lzc_low)
    assert np.isfinite(lzc_high)