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

def test_lomb_scargle():
    time = np.linspace(0, 1, 100)
    values = np.sin(2 * np.pi * 5 * time)
    frequencies = np.linspace(0.1, 10, 100)

    power = ct.lomb_scargle_py(time.tolist(), values.tolist(), frequencies.tolist())

    # Check that the peak is at the right frequency
    assert frequencies[np.argmax(power)] == pytest.approx(5, abs=0.1)
