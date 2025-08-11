use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyReadonlyArray1;

mod rollingstats;
mod stats;
mod fda;
mod peaks;
mod misc;


#[pyfunction]
fn time_series_summary<'py>(py: Python<'py>, time_series: PyReadonlyArray1<'py, f64>) -> PyResult<Py<PyDict>> {
    let summary = PyDict::new(py);
    let ts_view = time_series.as_array();

    if ts_view.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input time series cannot be empty"));
    }

    // Basic statistics
    let mean = stats::_calculate_mean(ts_view);
    let median = stats::_calculate_median(ts_view);
    let mode = stats::_calculate_mode(ts_view);
    let variance = stats::_calculate_variance(ts_view);
    let std_dev = stats::_calculate_std_dev(ts_view);
    let skewness = if std_dev > 0.0 { Some(stats::_calculate_skewness(ts_view, mean, std_dev)) } else { None };
    let kurtosis = if std_dev > 0.0 { Some(stats::_calculate_kurtosis(ts_view, mean, std_dev)) } else { None };
    
    // Range statistics
    let (min_val, max_val, range) = stats::_calculate_min_max_range(ts_view);
    
    // Quantiles
    let quantiles_vec = stats::_calculate_quantiles(ts_view);
    
    // Sum and energy
    let sum_val: f64 = ts_view.sum();
    let energy: f64 = ts_view.mapv(|x| x.powi(2)).sum();

    // Add all values to the dictionary
    summary.set_item("mean", mean)?;
    summary.set_item("median", median)?;
    summary.set_item("mode", mode)?;
    summary.set_item("variance", variance)?;
    summary.set_item("standard_deviation", std_dev)?;
    if let Some(s) = skewness { summary.set_item("skewness", s)?; }
    if let Some(k) = kurtosis { summary.set_item("kurtosis", k)?; }
    summary.set_item("minimum", min_val)?;
    summary.set_item("maximum", max_val)?;
    summary.set_item("range", range)?;
    summary.set_item("q05", quantiles_vec[0])?;
    summary.set_item("q25", quantiles_vec[1])?;
    summary.set_item("q75", quantiles_vec[2])?;
    summary.set_item("q95", quantiles_vec[3])?;
    summary.set_item("sum", sum_val)?;
    summary.set_item("absolute_energy", energy)?;

    Ok(summary.into())
}

#[pyfunction]
fn time_series_mean_median_mode(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input time series cannot be empty"));
    }
    let mean = stats::_calculate_mean(ts_view);
    let median = stats::_calculate_median(ts_view);
    let mode = stats::_calculate_mode(ts_view);
    Ok((mean, median, mode))
}

/// A Python module implemented in Rust.
#[pymodule]
fn chronoxtract(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(time_series_mean_median_mode, m)?)?;
    m.add_function(wrap_pyfunction!(time_series_summary, m)?)?;
    m.add_function(wrap_pyfunction!(fda::perform_fft_py, m)?)?;
    m.add_function(wrap_pyfunction!(fda::lomb_scargle_py, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::rolling_variance, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::expanding_sum, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::exponential_moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::sliding_window_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(peaks::find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(peaks::peak_prominence, m)?)?;
    m.add_function(wrap_pyfunction!(misc::fractional_variability, m)?)?;
    m.add_function(wrap_pyfunction!(misc::fractional_variability_error, m)?)?;
    m.add_function(wrap_pyfunction!(misc::rolling_fractional_variability, m)?)?;
    m.add_function(wrap_pyfunction!(misc::calc_variability_timescale, m)?)?;
    Ok(())
}