import pytest
import numpy as np
import chronoxtract as ct

def test_perform_fft():
    data = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    fft_result = ct.perform_fft_py(data)

    # The function returns a list of complex numbers
    # Compare with numpy's fft
    np_fft = np.fft.fft(data)

    assert np.allclose(fft_result, np_fft)

def test_perform_fft_odd_length():
    data = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    fft_result = ct.perform_fft_py(data)
    np_fft = np.fft.fft(data)
    assert np.allclose(fft_result, np_fft)

def test_perform_fft_prime_length():
    data = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    fft_result = ct.perform_fft_py(data)
    np_fft = np.fft.fft(data)
    assert np.allclose(fft_result, np_fft)
