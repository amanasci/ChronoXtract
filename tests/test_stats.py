import pytest
import numpy as np
import chronoxtract as ct

def test_time_series_summary():
    data = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    summary = ct.time_series_summary(data)

    assert np.isclose(summary['mean'], 2.8333333333333335)
    assert np.isclose(summary['median'], 2.5)
    assert np.isclose(summary['mode'], 2.0)
    assert np.isclose(summary['variance'], 1.8055555555555556)
    assert np.isclose(summary['standard_deviation'], 1.343709624584882)
    assert np.isclose(summary['skewness'], 0.3053162697580514)
    assert np.isclose(summary['kurtosis'], -1.151715976331361)
    assert np.isclose(summary['minimum'], 1.0)
    assert np.isclose(summary['maximum'], 5.0)
    assert np.isclose(summary['range'], 4.0)
    assert np.isclose(summary['sum'], 17.0)
    assert np.isclose(summary['absolute_energy'], 59.0)

def test_time_series_summary_empty():
    data = np.array([])
    with pytest.raises(ValueError):
        ct.time_series_summary(data)

def test_time_series_mean_median_mode():
    data = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    mean, median, mode = ct.time_series_mean_median_mode(data)

    assert np.isclose(mean, 2.6666666666666665)
    assert np.isclose(median, 2.5)
    assert np.isclose(mode, 1.0)

def test_time_series_mean_median_mode_empty():
    data = np.array([])
    with pytest.raises(ValueError):
        ct.time_series_mean_median_mode(data)
