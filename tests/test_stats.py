import pytest
import numpy as np
import chronoxtract as ct

def test_time_series_summary():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    summary = ct.time_series_summary(data)

    assert np.isclose(summary['mean'], 3.0)
    assert np.isclose(summary['median'], 3.0)
    assert np.isclose(summary['mode'], 1.0)  # Mode of a uniform distribution is the first element
    assert np.isclose(summary['variance'], 2.0)
    assert np.isclose(summary['standard_deviation'], np.sqrt(2.0))
    assert np.isclose(summary['skewness'], 0.0)
    assert np.isclose(summary['kurtosis'], -1.3)
    assert np.isclose(summary['minimum'], 1.0)
    assert np.isclose(summary['maximum'], 5.0)
    assert np.isclose(summary['range'], 4.0)
    assert np.isclose(summary['sum'], 15.0)
    assert np.isclose(summary['absolute_energy'], 55.0)

def test_time_series_summary_empty():
    data = []
    with pytest.raises(ValueError):
        ct.time_series_summary(data)

def test_time_series_mean_median_mode():
    data = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    mean, median, mode = ct.time_series_mean_median_mode(data)

    assert np.isclose(mean, 2.6666666666666665)
    assert np.isclose(median, 2.5)
    assert np.isclose(mode, 1.0)

def test_time_series_mean_median_mode_empty():
    data = []
    with pytest.raises(ValueError):
        ct.time_series_mean_median_mode(data)
