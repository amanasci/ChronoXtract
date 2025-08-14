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

def test_time_series_summary_int():
    data = np.array([1, 2, 2, 3, 4, 5], dtype=np.float64)
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

def test_time_series_summary_single_element():
    data = np.array([5.0])
    summary = ct.time_series_summary(data)

    assert np.isclose(summary['mean'], 5.0)
    assert np.isclose(summary['median'], 5.0)
    assert np.isclose(summary['mode'], 5.0)
    assert np.isclose(summary['variance'], 0.0)
    assert np.isclose(summary['standard_deviation'], 0.0)
    assert 'skewness' not in summary
    assert 'kurtosis' not in summary
    assert np.isclose(summary['minimum'], 5.0)
    assert np.isclose(summary['maximum'], 5.0)
    assert np.isclose(summary['range'], 0.0)
    assert np.isclose(summary['sum'], 5.0)
    assert np.isclose(summary['absolute_energy'], 25.0)

def test_time_series_summary_all_same():
    data = np.array([3.0, 3.0, 3.0, 3.0])
    summary = ct.time_series_summary(data)

    assert np.isclose(summary['mean'], 3.0)
    assert np.isclose(summary['median'], 3.0)
    assert np.isclose(summary['mode'], 3.0)
    assert np.isclose(summary['variance'], 0.0)
    assert np.isclose(summary['standard_deviation'], 0.0)
    assert 'skewness' not in summary
    assert 'kurtosis' not in summary
    assert np.isclose(summary['minimum'], 3.0)
    assert np.isclose(summary['maximum'], 3.0)
    assert np.isclose(summary['range'], 0.0)
    assert np.isclose(summary['sum'], 12.0)
    assert np.isclose(summary['absolute_energy'], 36.0)

def test_time_series_summary_with_nan():
    data = np.array([1.0, 2.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        ct.time_series_summary(data)

def test_time_series_mean_median_mode_int():
    data = np.array([1, 1, 2, 3, 4, 5], dtype=np.float64)
    mean, median, mode = ct.time_series_mean_median_mode(data)

    assert np.isclose(mean, 2.6666666666666665)
    assert np.isclose(median, 2.5)
    assert np.isclose(mode, 1.0)

def test_time_series_mean_median_mode_single_element():
    data = np.array([5.0])
    mean, median, mode = ct.time_series_mean_median_mode(data)

    assert np.isclose(mean, 5.0)
    assert np.isclose(median, 5.0)
    assert np.isclose(mode, 5.0)

def test_time_series_mean_median_mode_all_same():
    data = np.array([3.0, 3.0, 3.0, 3.0])
    mean, median, mode = ct.time_series_mean_median_mode(data)

    assert np.isclose(mean, 3.0)
    assert np.isclose(median, 3.0)
    assert np.isclose(mode, 3.0)
