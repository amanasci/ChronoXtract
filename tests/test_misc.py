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

def test_fractional_variability_zero_flux():
    flux = np.array([0.0, 0.0, 0.0])
    flux_err = np.array([1.0, 1.0, 1.0])
    fvar = ct.fractional_variability(flux, flux_err)
    assert np.isnan(fvar)

def test_fractional_variability_negative_flux():
    flux = np.array([-10.0, -20.0, -15.0])
    flux_err = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        ct.fractional_variability(flux, flux_err)

def test_find_peaks_plateau():
    data = np.array([0, 1, 1, 1, 0], dtype=np.float64)
    peaks = ct.find_peaks(data, height=0.5)
    assert np.array_equal(peaks, [2])

def test_find_peaks_multiple_close():
    data = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    peaks = ct.find_peaks(data, height=0.5)
    assert np.array_equal(peaks, [1, 3, 5])

def test_peak_prominence_boundary():
    data = np.array([2, 1, 0, 3, 0, 2, 0], dtype=np.float64)
    peaks = np.array([0, 3, 5], dtype=np.uint64)
    prominences = ct.peak_prominence(data, peaks)
    assert np.allclose(prominences, [2.0, 3.0, 2.0])
