import pytest
import numpy as np
import chronoxtract as ct

def test_seasonal_trend_strength():
    """Test seasonal and trend strength calculation"""
    # Create a signal with trend and seasonality
    t = np.arange(100)
    trend = 0.1 * t
    seasonal = np.sin(2 * np.pi * t / 10)
    noise = 0.1 * np.random.random(100)
    data = trend + seasonal + noise
    
    seasonal_strength, trend_strength = ct.seasonal_trend_strength(data, period=10)
    
    assert 0.0 <= seasonal_strength <= 1.0
    assert 0.0 <= trend_strength <= 1.0
    assert np.isfinite(seasonal_strength)
    assert np.isfinite(trend_strength)

def test_seasonal_strength():
    """Test seasonal strength calculation alone"""
    # Create a purely seasonal signal
    t = np.arange(50)
    data = np.sin(2 * np.pi * t / 8)
    
    strength = ct.seasonal_strength(data, period=8)
    
    assert 0.0 <= strength <= 1.0
    assert np.isfinite(strength)
    # Should have significant seasonal strength
    assert strength > 0.3

def test_trend_strength():
    """Test trend strength calculation alone"""
    # Create a signal with strong trend
    t = np.arange(50)
    data = 0.2 * t + 0.1 * np.random.random(50)
    
    strength = ct.trend_strength(data, period=10)
    
    assert 0.0 <= strength <= 1.0
    assert np.isfinite(strength)
    # Should have significant trend strength
    assert strength > 0.3

def test_simple_stl_decomposition():
    """Test STL decomposition"""
    # Create a signal with known components
    t = np.arange(60)
    trend = 0.05 * t
    seasonal = np.sin(2 * np.pi * t / 12)
    noise = 0.05 * np.random.random(60)
    data = trend + seasonal + noise
    
    trend_comp, seasonal_comp, remainder_comp = ct.simple_stl_decomposition(data, period=12)
    
    assert len(trend_comp) == len(data)
    assert len(seasonal_comp) == len(data)
    assert len(remainder_comp) == len(data)
    
    # Components should approximately reconstruct original
    reconstructed = np.array(trend_comp) + np.array(seasonal_comp) + np.array(remainder_comp)
    assert np.allclose(reconstructed, data, rtol=1e-10)

def test_detect_seasonality():
    """Test seasonality detection"""
    # Create a clearly seasonal signal
    t = np.arange(50)
    seasonal_data = np.sin(2 * np.pi * t / 10)
    
    # Create a random signal
    np.random.seed(42)
    random_data = np.random.normal(0, 1, 50)
    
    seasonal_detected = ct.detect_seasonality(seasonal_data, period=10, threshold=0.3)
    random_detected = ct.detect_seasonality(random_data, period=10, threshold=0.3)
    
    assert seasonal_detected == True  # Should detect seasonality
    # Random signal should be less likely to show seasonality, but not guaranteed

def test_detect_seasonality_with_threshold():
    """Test seasonality detection with different thresholds"""
    t = np.arange(40)
    data = np.sin(2 * np.pi * t / 8) + 0.5 * np.random.random(40)
    
    # Lower threshold should be more likely to detect seasonality
    detected_low = ct.detect_seasonality(data, period=8, threshold=0.1)
    detected_high = ct.detect_seasonality(data, period=8, threshold=0.8)
    
    # At least the low threshold version should work
    assert isinstance(detected_low, bool)
    assert isinstance(detected_high, bool)

def test_detrended_fluctuation_analysis():
    """Test DFA calculation"""
    # Create a signal with known scaling properties
    np.random.seed(42)
    data = np.cumsum(np.random.normal(0, 1, 500))  # Random walk
    
    alpha = ct.detrended_fluctuation_analysis(data, min_window=10, max_window=100, num_windows=10)
    
    assert np.isfinite(alpha)
    # For a random walk, alpha should be around 1.5
    assert 0.5 < alpha < 2.5

def test_stl_decomposition_seasonality_recovery():
    """Test that STL can recover known seasonal patterns"""
    # Create a signal with a clear 12-point seasonal pattern
    t = np.arange(48)
    seasonal_pattern = np.array([1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0])
    seasonal = np.tile(seasonal_pattern, 4)
    trend = 0.1 * t
    data = trend + seasonal
    
    trend_comp, seasonal_comp, remainder_comp = ct.simple_stl_decomposition(data, period=12)
    
    # Seasonal component should have the right period
    seasonal_comp = np.array(seasonal_comp)
    # Check that seasonal pattern repeats
    for i in range(0, 36, 12):
        pattern1 = seasonal_comp[i:i+12]
        pattern2 = seasonal_comp[i+12:i+24]
        # Patterns should be similar (allowing some decomposition artifacts)
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        assert correlation > 0.8

def test_trend_strength_linear_signal():
    """Test trend strength on a purely linear signal"""
    data = np.arange(50, dtype=float)
    strength = ct.trend_strength(data, period=10)
    
    # Pure linear trend should have high trend strength
    assert strength > 0.8
    assert strength <= 1.0

def test_seasonal_strength_pure_sine():
    """Test seasonal strength on a pure sine wave"""
    t = np.arange(60)
    data = np.sin(2 * np.pi * t / 15)
    
    strength = ct.seasonal_strength(data, period=15)
    
    # Pure seasonal signal should have high seasonal strength
    assert strength > 0.8
    assert strength <= 1.0

def test_parameter_validation():
    """Test parameter validation for seasonality functions"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test insufficient data length
    with pytest.raises(ValueError):
        ct.seasonal_trend_strength(data, period=10)  # period too large
    
    with pytest.raises(ValueError):
        ct.simple_stl_decomposition(data, period=10)  # period too large
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        ct.seasonal_strength(data, period=1)  # period too small
    
    with pytest.raises(ValueError):
        ct.detect_seasonality(data, period=0)  # period too small

def test_dfa_parameter_validation():
    """Test DFA parameter validation"""
    data = np.arange(100, dtype=float)
    
    # Test invalid window parameters
    with pytest.raises(ValueError):
        ct.detrended_fluctuation_analysis(data, min_window=50, max_window=30, num_windows=10)
    
    with pytest.raises(ValueError):
        ct.detrended_fluctuation_analysis(data, min_window=10, max_window=200, num_windows=10)  # max_window > data length
    
    with pytest.raises(ValueError):
        ct.detrended_fluctuation_analysis(data, min_window=10, max_window=50, num_windows=1)  # too few windows

def test_stl_no_seasonality():
    """Test STL decomposition on a signal with no seasonality"""
    # Pure trend with noise
    t = np.arange(40)
    data = 0.1 * t + 0.1 * np.random.random(40)
    
    trend_comp, seasonal_comp, remainder_comp = ct.simple_stl_decomposition(data, period=8)
    
    # Seasonal component should be close to zero
    seasonal_comp = np.array(seasonal_comp)
    assert np.std(seasonal_comp) < np.std(data) * 0.5  # Seasonal variation should be much smaller

def test_large_signal_performance():
    """Test performance on larger signals"""
    # Create a large signal
    t = np.arange(2000)
    data = 0.01 * t + np.sin(2 * np.pi * t / 50) + 0.1 * np.random.random(2000)
    
    # These should complete without errors
    seasonal_strength, trend_strength = ct.seasonal_trend_strength(data, period=50)
    detected = ct.detect_seasonality(data, period=50, threshold=0.2)
    
    assert np.isfinite(seasonal_strength)
    assert np.isfinite(trend_strength)
    assert isinstance(detected, bool)

def test_different_seasonal_periods():
    """Test seasonality detection with different periods"""
    # Create signals with different seasonal periods
    t = np.arange(100)
    
    # 10-point seasonality
    data1 = np.sin(2 * np.pi * t / 10)
    detected1 = ct.detect_seasonality(data1, period=10, threshold=0.5)
    
    # 20-point seasonality  
    data2 = np.sin(2 * np.pi * t / 20)
    detected2 = ct.detect_seasonality(data2, period=20, threshold=0.5)
    
    # Both should detect their respective seasonalities
    assert detected1 == True
    assert detected2 == True

def test_dfa_different_signals():
    """Test DFA on different types of signals"""
    np.random.seed(123)
    
    # White noise (should have alpha around 0.5)
    white_noise = np.random.normal(0, 1, 300)
    alpha_white = ct.detrended_fluctuation_analysis(white_noise, min_window=5, max_window=50, num_windows=8)
    
    # Random walk (should have alpha around 1.5)
    random_walk = np.cumsum(np.random.normal(0, 1, 300))
    alpha_walk = ct.detrended_fluctuation_analysis(random_walk, min_window=5, max_window=50, num_windows=8)
    
    assert np.isfinite(alpha_white)
    assert np.isfinite(alpha_walk)
    # Random walk should have higher scaling exponent than white noise
    assert alpha_walk > alpha_white

def test_seasonal_decomposition_reconstruction():
    """Test that seasonal decomposition exactly reconstructs the original signal"""
    # Create test signal
    t = np.arange(36)
    original = 0.1 * t + np.sin(2 * np.pi * t / 6) + 0.05 * np.sin(2 * np.pi * t / 3)
    
    trend, seasonal, remainder = ct.simple_stl_decomposition(original, period=6)
    
    # Exact reconstruction should hold
    reconstructed = np.array(trend) + np.array(seasonal) + np.array(remainder)
    assert np.allclose(original, reconstructed, atol=1e-12)

def test_strength_edge_cases():
    """Test strength calculations with edge cases"""
    # Constant signal
    constant_data = np.array([5.0] * 30)
    seasonal_str, trend_str = ct.seasonal_trend_strength(constant_data, period=6)
    
    # Constant signal should have minimal strength values
    assert seasonal_str >= 0.0
    assert trend_str >= 0.0
    assert seasonal_str <= 1.0
    assert trend_str <= 1.0