import pytest
import numpy as np
import chronoxtract as ct

def test_zero_crossing_rate():
    """Test zero-crossing rate calculation"""
    # Alternating signal should have high zero-crossing rate
    data = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    zcr = ct.zero_crossing_rate(data)
    
    # Should be close to 1.0 (crossing at every sample)
    assert 0.8 <= zcr <= 1.0
    assert np.isfinite(zcr)

def test_zero_crossing_rate_constant():
    """Test zero-crossing rate for constant signal"""
    data = np.array([1.0] * 10)
    zcr = ct.zero_crossing_rate(data)
    
    # Constant signal should have zero crossings
    assert zcr == 0.0

def test_zero_crossing_rate_sine():
    """Test zero-crossing rate for sine wave"""
    t = np.linspace(0, 4*np.pi, 100)
    data = np.sin(t)
    zcr = ct.zero_crossing_rate(data)
    
    # Sine wave should have multiple zero crossings
    assert zcr > 0.02  # Relaxed threshold
    assert zcr < 0.1

def test_slope_features():
    """Test slope features calculation"""
    # Create a signal with varying slopes
    data = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
    mean_slope, slope_var, max_slope = ct.slope_features(data)
    
    assert np.isfinite(mean_slope)
    assert slope_var >= 0.0
    assert max_slope >= 0.0

def test_individual_slope_functions():
    """Test individual slope functions"""
    data = np.array([1.0, 4.0, 2.0, 6.0, 3.0])
    
    mean_s = ct.mean_slope(data)
    var_s = ct.slope_variance(data)
    max_s = ct.max_slope(data)
    
    # Check consistency with combined function
    combined = ct.slope_features(data)
    assert abs(mean_s - combined[0]) < 1e-10
    assert abs(var_s - combined[1]) < 1e-10
    assert abs(max_s - combined[2]) < 1e-10

def test_slope_features_linear():
    """Test slope features for linear signal"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean_slope, slope_var, max_slope = ct.slope_features(data)
    
    # Linear signal should have constant slope
    assert abs(mean_slope - 1.0) < 1e-10
    assert abs(slope_var - 0.0) < 1e-10
    assert abs(max_slope - 1.0) < 1e-10

def test_enhanced_peak_stats():
    """Test enhanced peak statistics"""
    # Create a signal with clear peaks
    t = np.linspace(0, 4*np.pi, 200)
    data = np.sin(t) + 0.5 * np.sin(3*t)
    
    stats = ct.enhanced_peak_stats(data, min_prominence=0.1, min_distance=5)
    
    assert len(stats) == 6  # Returns tuple of 6 values
    num_peaks, mean_prom, mean_spacing, mean_width, max_p2p, peak_density = stats
    
    assert num_peaks > 0
    assert mean_prom >= 0.0
    assert mean_spacing >= 0.0
    assert mean_width >= 0.0
    assert max_p2p >= 0.0
    assert 0.0 <= peak_density <= 1.0

def test_enhanced_peak_stats_no_peaks():
    """Test enhanced peak stats when no peaks are found"""
    # Flat signal should have no peaks
    data = np.array([1.0] * 20)
    stats = ct.enhanced_peak_stats(data, min_prominence=0.1, min_distance=1)
    
    num_peaks, mean_prom, mean_spacing, mean_width, max_p2p, peak_density = stats
    
    # Should find no peaks
    assert num_peaks == 0
    assert mean_prom == 0.0
    assert mean_spacing == 0.0
    assert mean_width == 0.0
    assert max_p2p == 0.0
    assert peak_density == 0.0

def test_peak_to_peak_amplitude():
    """Test peak-to-peak amplitude calculation"""
    # Create signal with known amplitudes
    t = np.linspace(0, 2*np.pi, 100)
    data = np.sin(t)  # Amplitude 1, should have p2p of ~2
    
    max_p2p, mean_p2p, std_p2p = ct.peak_to_peak_amplitude(data)
    
    assert max_p2p >= 0.0
    assert mean_p2p >= 0.0
    assert std_p2p >= 0.0
    assert np.isfinite(max_p2p)
    assert np.isfinite(mean_p2p)
    assert np.isfinite(std_p2p)

def test_variability_features():
    """Test variability features calculation"""
    # Use a signal with known statistical properties
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)  # mean=10, std=2
    
    cv, qcd, mad, iqr = ct.variability_features(data)
    
    assert cv >= 0.0  # Coefficient of variation
    assert qcd >= 0.0  # Quartile coefficient of dispersion
    assert mad >= 0.0  # Median absolute deviation
    assert iqr >= 0.0  # Interquartile range
    
    # CV should be around 2/10 = 0.2 for this distribution
    assert 0.1 < cv < 0.3

def test_variability_features_constant():
    """Test variability features for constant signal"""
    data = np.array([5.0] * 50)
    cv, qcd, mad, iqr = ct.variability_features(data)
    
    # Constant signal should have zero variability
    assert cv == 0.0
    assert qcd == 0.0
    assert mad == 0.0
    assert iqr == 0.0

def test_turning_points():
    """Test turning points calculation"""
    # Create a signal with known turning points
    data = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0])
    num_tp, tp_rate = ct.turning_points(data)
    
    assert num_tp > 0
    assert 0.0 <= tp_rate <= 1.0
    assert np.isfinite(tp_rate)

def test_turning_points_monotonic():
    """Test turning points for monotonic signal"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    num_tp, tp_rate = ct.turning_points(data)
    
    # Monotonic signal should have no turning points
    assert num_tp == 0
    assert tp_rate == 0.0

def test_energy_distribution():
    """Test energy distribution features"""
    # Create a signal with varying energy distribution
    data = np.array([1.0, 0.1, 0.1, 5.0, 0.1, 0.1, 2.0])
    
    energy_entropy, norm_energy, energy_conc = ct.energy_distribution(data)
    
    assert energy_entropy >= 0.0
    assert norm_energy >= 0.0
    assert 0.0 <= energy_conc <= 1.0
    assert np.isfinite(energy_entropy)
    assert np.isfinite(norm_energy)

def test_energy_distribution_uniform():
    """Test energy distribution for uniform signal"""
    data = np.array([1.0] * 10)
    energy_entropy, norm_energy, energy_conc = ct.energy_distribution(data)
    
    # Uniform signal should have high entropy and low concentration
    assert energy_entropy > 2.0  # High entropy
    assert energy_conc <= 0.2  # Low concentration (energy spread out)

def test_parameter_validation():
    """Test parameter validation for shape functions"""
    # Test insufficient data
    short_data = np.array([1.0])
    
    with pytest.raises(ValueError):
        ct.zero_crossing_rate(short_data)
    
    with pytest.raises(ValueError):
        ct.slope_features(short_data)
    
    with pytest.raises(ValueError):
        ct.enhanced_peak_stats(short_data, min_prominence=0.1)
    
    with pytest.raises(ValueError):
        ct.turning_points(short_data)

def test_empty_input_handling():
    """Test handling of empty inputs"""
    with pytest.raises(ValueError):
        ct.variability_features(np.array([]))
    
    with pytest.raises(ValueError):
        ct.energy_distribution(np.array([]))

def test_sine_wave_characteristics():
    """Test various features on a sine wave"""
    t = np.linspace(0, 2*np.pi, 100)
    data = np.sin(t)
    
    # Zero-crossing rate
    zcr = ct.zero_crossing_rate(data)
    assert zcr > 0.005  # Should have some zero crossings
    assert zcr < 0.1
    
    # Turning points (peaks and valleys)
    num_tp, tp_rate = ct.turning_points(data)
    assert num_tp >= 2  # At least one peak and one valley
    
    # Peak stats
    stats = ct.enhanced_peak_stats(data, min_prominence=0.5, min_distance=10)
    num_peaks = stats[0]
    assert num_peaks > 0  # Should find at least one peak

def test_random_walk_properties():
    """Test features on a random walk"""
    np.random.seed(42)
    steps = np.random.choice([-1, 1], size=200)
    data = np.cumsum(steps).astype(float)
    
    # Should have reasonable variability
    cv, qcd, mad, iqr = ct.variability_features(data)
    assert cv > 0.1  # Should have some variability
    
    # Should have many turning points
    num_tp, tp_rate = ct.turning_points(data)
    assert tp_rate > 0.3  # Random walk should have many turning points

def test_feature_stability_large_data():
    """Test feature stability on larger datasets"""
    np.random.seed(123)
    t = np.linspace(0, 20*np.pi, 2000)
    data = np.sin(t) + 0.1 * np.random.normal(0, 1, 2000)
    
    # All features should complete without errors
    zcr = ct.zero_crossing_rate(data)
    slope_feats = ct.slope_features(data)
    peak_stats = ct.enhanced_peak_stats(data, min_prominence=0.1, min_distance=5)
    p2p_amps = ct.peak_to_peak_amplitude(data)
    var_feats = ct.variability_features(data)
    turn_pts = ct.turning_points(data)
    energy_dist = ct.energy_distribution(data)
    
    # All results should be finite and reasonable
    assert np.isfinite(zcr)
    assert all(np.isfinite(x) for x in slope_feats)
    assert all(np.isfinite(x) for x in peak_stats)
    assert all(np.isfinite(x) for x in p2p_amps)
    assert all(np.isfinite(x) for x in var_feats)
    assert all(np.isfinite(x) for x in turn_pts)
    assert all(np.isfinite(x) for x in energy_dist)

def test_periodic_vs_aperiodic():
    """Test that features can distinguish periodic from aperiodic signals"""
    # Periodic signal
    t = np.linspace(0, 10*np.pi, 300)
    periodic = np.sin(t)
    
    # Aperiodic signal (random)
    np.random.seed(42)
    aperiodic = np.random.normal(0, 1, 300)
    
    # Get features for both
    zcr_per = ct.zero_crossing_rate(periodic)
    zcr_aper = ct.zero_crossing_rate(aperiodic)
    
    _, tp_rate_per = ct.turning_points(periodic)
    _, tp_rate_aper = ct.turning_points(aperiodic)
    
    # Periodic signal should have more regular zero crossings
    # Random signal should have more turning points
    assert tp_rate_aper > tp_rate_per

def test_impulse_response():
    """Test features on an impulse signal"""
    data = np.zeros(50)
    data[25] = 10.0  # Single impulse
    
    # Should detect the impulse as a peak
    stats = ct.enhanced_peak_stats(data, min_prominence=1.0, min_distance=1)
    num_peaks = stats[0]
    assert num_peaks == 1
    
    # Should have high energy concentration
    _, _, energy_conc = ct.energy_distribution(data)
    assert energy_conc > 0.8  # Most energy in one sample

def test_step_function():
    """Test features on a step function"""
    # Create a step function that actually crosses zero
    data = np.concatenate([-np.ones(25), np.ones(25)])
    
    # Should have one zero crossing
    zcr = ct.zero_crossing_rate(data)
    assert 0.01 < zcr < 0.05  # One crossing out of 49 transitions
    
    # Should have large slope at the step
    _, _, max_slope = ct.slope_features(data)
    assert max_slope == 2.0  # Step from -1 to 1 should be the maximum slope