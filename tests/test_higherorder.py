import pytest
import numpy as np
import chronoxtract as ct

def test_hjorth_parameters():
    """Test Hjorth parameters calculation"""
    # Generate a simple sine wave
    t = np.linspace(0, 4*np.pi, 100)
    data = np.sin(t)
    
    activity, mobility, complexity = ct.hjorth_parameters(data)
    
    assert activity > 0.0
    assert mobility > 0.0
    assert complexity > 0.0
    
    # For a pure sine wave, complexity should be close to 1
    assert abs(complexity - 1.0) < 0.5

def test_hjorth_activity():
    """Test Hjorth activity (variance)"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    activity = ct.hjorth_activity(data)
    
    # Should match variance: mean=3, variance=2
    assert abs(activity - 2.0) < 1e-10

def test_hjorth_mobility():
    """Test Hjorth mobility"""
    # Use a quadratic signal to ensure non-zero mobility
    data = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # squares: 1^2, 2^2, 3^2, 4^2, 5^2
    mobility = ct.hjorth_mobility(data)
    
    assert mobility > 0.0
    assert np.isfinite(mobility)

def test_hjorth_complexity():
    """Test Hjorth complexity"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    complexity = ct.hjorth_complexity(data)
    
    assert complexity > 0.0
    assert np.isfinite(complexity)

def test_hjorth_constant_signal():
    """Test Hjorth parameters for constant signal"""
    data = np.array([5.0] * 10)
    
    activity, mobility, complexity = ct.hjorth_parameters(data)
    
    assert abs(activity - 0.0) < 1e-10  # No variance
    assert abs(mobility - 0.0) < 1e-10  # No mobility
    assert abs(complexity - 1.0) < 1e-10  # Default complexity

def test_higher_moments():
    """Test higher-order central moments (5th-8th)"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    m5, m6, m7, m8 = ct.higher_moments(data)
    
    # All moments should be finite
    assert np.isfinite(m5)
    assert np.isfinite(m6)
    assert np.isfinite(m7)
    assert np.isfinite(m8)

def test_individual_central_moments():
    """Test individual central moment functions"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    m5 = ct.central_moment_5(data)
    m6 = ct.central_moment_6(data)
    m7 = ct.central_moment_7(data)
    m8 = ct.central_moment_8(data)
    
    # Check that individual functions match combined function
    combined_m5, combined_m6, combined_m7, combined_m8 = ct.higher_moments(data)
    
    assert abs(m5 - combined_m5) < 1e-10
    assert abs(m6 - combined_m6) < 1e-10
    assert abs(m7 - combined_m7) < 1e-10
    assert abs(m8 - combined_m8) < 1e-10

def test_symmetric_signal_moments():
    """Test that symmetric signals have appropriate moment properties"""
    # Generate symmetric gaussian noise
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    
    m5, m6, m7, m8 = ct.higher_moments(data)
    
    # For symmetric distributions, odd moments should be smaller than even moments
    # (though with finite samples, they won't be exactly 0)
    assert abs(m5) < abs(m6)  # 5th should be smaller than 6th
    assert abs(m7) < abs(m8)  # 7th should be smaller than 8th
    
    # Even moments should be positive for a normal distribution
    assert m6 > 0
    assert m8 > 0
    
    # Check that values are reasonable for normal distribution
    assert 10 < m6 < 20  # 6th moment should be around 15 for N(0,1)
    assert 80 < m8 < 130  # 8th moment should be around 105 for N(0,1)

def test_hjorth_empty_input():
    """Test error handling for empty input"""
    with pytest.raises(ValueError):
        ct.hjorth_activity(np.array([]))
    
    with pytest.raises(ValueError):
        ct.hjorth_parameters(np.array([]))

def test_hjorth_insufficient_data():
    """Test error handling for insufficient data"""
    # Need at least 2 points for mobility
    with pytest.raises(ValueError):
        ct.hjorth_mobility(np.array([1.0]))
    
    # Need at least 3 points for complexity and full parameters
    with pytest.raises(ValueError):
        ct.hjorth_complexity(np.array([1.0, 2.0]))
    
    with pytest.raises(ValueError):
        ct.hjorth_parameters(np.array([1.0, 2.0]))

def test_moments_empty_input():
    """Test error handling for empty input in moments"""
    with pytest.raises(ValueError):
        ct.higher_moments(np.array([]))
    
    with pytest.raises(ValueError):
        ct.central_moment_5(np.array([]))

def test_chaotic_signal_hjorth():
    """Test Hjorth parameters on a chaotic-like signal"""
    # Create a more complex signal (sum of multiple frequencies)
    t = np.linspace(0, 10, 1000)
    data = np.sin(t) + 0.5 * np.sin(3*t) + 0.25 * np.sin(7*t)
    
    activity, mobility, complexity = ct.hjorth_parameters(data)
    
    assert activity > 0.0
    assert mobility > 0.0
    assert complexity > 1.0  # Should be more complex than pure sine wave

def test_gaussian_noise_properties():
    """Test properties of Gaussian white noise"""
    np.random.seed(123)
    data = np.random.normal(0, 1, 1000)
    
    activity, mobility, complexity = ct.hjorth_parameters(data)
    
    # Activity should be close to variance (1.0)
    assert abs(activity - 1.0) < 0.2
    
    # Mobility and complexity should be reasonable
    assert mobility > 0.0
    assert complexity > 0.0

def test_large_signal():
    """Test performance on larger signals"""
    # Test with 10k points
    t = np.linspace(0, 100, 10000)
    data = np.sin(0.1 * t) + 0.1 * np.random.normal(0, 1, 10000)
    
    activity, mobility, complexity = ct.hjorth_parameters(data)
    
    assert activity > 0.0
    assert mobility > 0.0
    assert complexity > 0.0
    assert np.isfinite(activity)
    assert np.isfinite(mobility)
    assert np.isfinite(complexity)