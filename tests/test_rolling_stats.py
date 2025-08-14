import pytest
import numpy as np
import chronoxtract as ct

def test_rolling_mean():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3
    result = ct.rolling_mean(data, window)
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(result, expected)

def test_rolling_variance():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3
    result = ct.rolling_variance(data, window)
    expected = np.array([0.66666667, 0.66666667, 0.66666667])
    assert np.allclose(result, expected, atol=1e-5)

def test_expanding_sum():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ct.expanding_sum(data)
    expected = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    assert np.allclose(result, expected)

def test_exponential_moving_average():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.5
    result = ct.exponential_moving_average(data, alpha)
    expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
    assert np.allclose(result, expected)

def test_sliding_window_entropy():
    data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    window_size = 3
    num_bins = 2
    result = ct.sliding_window_entropy(data, window_size, num_bins)
    expected = np.array([0.0, 0.91829583, 0.91829583, 0.0])
    assert np.allclose(result, expected, atol=1e-5)
