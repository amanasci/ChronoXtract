import pytest
import numpy as np
import chronoxtract as ct

def test_fractional_variability():
    flux = np.array([100.0, 120.0, 110.0, 105.0, 95.0])
    flux_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    fvar = ct.fractional_variability(flux, flux_err)

    expected_fvar = 0.090241183
    assert np.isclose(fvar, expected_fvar, atol=1e-5)

def test_fractional_variability_error():
    flux = np.array([100.0, 120.0, 110.0, 105.0, 95.0])
    flux_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    fvar_err = ct.fractional_variability_error(flux, flux_err)

    expected_fvar_err = 0.0042163702
    assert np.isclose(fvar_err, expected_fvar_err, atol=1e-4)

def test_find_peaks():
    data = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float64)
    peaks = ct.find_peaks(data, height=1.5)
    assert np.array_equal(peaks, [3, 5])

def test_peak_prominence():
    data = np.array([0, 1, 0, 3, 0, 2, 0], dtype=np.float64)
    peaks = np.array([1, 3, 5], dtype=np.uint64)
    prominences = ct.peak_prominence(data, peaks)
    assert np.allclose(prominences, [1.0, 3.0, 2.0])
