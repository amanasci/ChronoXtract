use pyo3::prelude::*;
use pyo3::types::PyDict;
mod rollingstats;
mod stats;
mod fda;
mod peaks;
mod misc;

#[pyfunction]
fn time_series_summary<'py>(py: Python<'py>, time_series: Vec<f64>) -> PyResult<Py<PyDict>> {
    let summary = PyDict::new(py);
    
    // Basic statistics
    let mean = stats::calculate_mean(time_series.clone())?;
    let median = stats::calculate_median(time_series.clone())?;
    let mode = stats::calculate_mode(time_series.clone())?;
    let variance = stats::calculate_variance(time_series.clone())?;
    let std_dev = stats::calculate_std_dev(time_series.clone())?;
    let skewness = stats::calculate_skewness(time_series.clone())?;
    let kurtosis = stats::calculate_kurtosis(time_series.clone())?;
    
    // Range statistics
    let (min_val, max_val, range) = stats::calculate_min_max_range(time_series.clone())?;
    
    // Quantiles
    let quantiles = stats::calculate_quantiles(time_series.clone())?;
    
    // Sum and energy
    let sum_val = stats::calculate_sum(time_series.clone())?;
    let energy = stats::calculate_absolute_energy(time_series)?;

    // Add all values to the dictionary
    summary.set_item("mean", mean)?;
    summary.set_item("median", median)?;
    summary.set_item("mode", mode)?;
    summary.set_item("variance", variance)?;
    summary.set_item("standard_deviation", std_dev)?;
    summary.set_item("skewness", skewness)?;
    summary.set_item("kurtosis", kurtosis)?;
    summary.set_item("minimum", min_val)?;
    summary.set_item("maximum", max_val)?;
    summary.set_item("range", range)?;
    summary.set_item("q05", quantiles[0])?;
    summary.set_item("q25", quantiles[1])?;
    summary.set_item("q75", quantiles[2])?;
    summary.set_item("q95", quantiles[3])?;
    summary.set_item("sum", sum_val)?;
    summary.set_item("absolute_energy", energy)?;

    Ok(summary.into())
}







#[pyfunction]
fn time_series_mean_median_mode(time_series: Vec<f64>) -> PyResult<(f64, f64, f64)> {
    let mean = stats::calculate_mean(time_series.clone())?;
    let median = stats::calculate_median(time_series.clone())?;
    let mode = stats::calculate_mode(time_series)?;
    Ok((mean, median, mode))
}

/// A Python module implemented in Rust.
#[pymodule]
fn chronoxtract(m: &Bound<'_, PyModule>) -> PyResult<()> {
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