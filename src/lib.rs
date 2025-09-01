use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyReadonlyArray1;

mod rollingstats;
mod stats;
mod fda;
mod peaks;
mod misc;
mod correlation;
mod higherorder;
mod entropy;
mod seasonality;
mod shape;
mod carma;


/// Calculate a comprehensive statistical summary of a time series.
/// 
/// This function computes multiple statistical measures in a single pass for efficiency,
/// including descriptive statistics, distribution measures, and energy metrics.
///
/// # Arguments
/// * `time_series` - Input time series data as a numpy array
///
/// # Returns
/// A dictionary containing:
/// - `mean`: Arithmetic mean
/// - `median`: Middle value when sorted
/// - `mode`: Most frequently occurring value
/// - `variance`: Sample variance
/// - `standard_deviation`: Square root of variance
/// - `skewness`: Measure of asymmetry (None if std_dev too small)
/// - `kurtosis`: Measure of tail heaviness (None if std_dev too small)
/// - `minimum`: Smallest value
/// - `maximum`: Largest value
/// - `range`: Difference between max and min
/// - `q05`, `q25`, `q75`, `q95`: Quantiles at 5%, 25%, 75%, 95%
/// - `sum`: Sum of all values
/// - `absolute_energy`: Sum of squared values
///
/// # Errors
/// Returns `PyValueError` if:
/// - Input time series is empty
/// - Input contains NaN values
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// summary = ct.time_series_summary(data)
/// print(f"Mean: {summary['mean']}")
/// print(f"Standard deviation: {summary['standard_deviation']}")
/// ```
#[pyfunction]
fn time_series_summary<'py>(py: Python<'py>, time_series: PyReadonlyArray1<'py, f64>) -> PyResult<Py<PyDict>> {
    let summary = PyDict::new(py);
    let ts_view = time_series.as_array();

    if ts_view.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input time series cannot be empty"));
    }

    if ts_view.iter().any(|&x| x.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err("Input contains NaN values"));
    }

    // Single-pass statistics
    let stats_summary = stats::_calculate_summary_statistics(ts_view);

    // Median and Quantiles (one sort)
    let (median, quantiles_vec) = stats::_calculate_median_and_quantiles(ts_view);

    // Mode
    let mode = stats::_calculate_mode(ts_view);

    // Add all values to the dictionary
    summary.set_item("mean", stats_summary.mean)?;
    summary.set_item("median", median)?;
    summary.set_item("mode", mode)?;
    summary.set_item("variance", stats_summary.variance)?;
    summary.set_item("standard_deviation", stats_summary.std_dev)?;
    if let Some(s) = stats_summary.skewness { summary.set_item("skewness", s)?; }
    if let Some(k) = stats_summary.kurtosis { summary.set_item("kurtosis", k)?; }
    summary.set_item("minimum", stats_summary.min)?;
    summary.set_item("maximum", stats_summary.max)?;
    summary.set_item("range", stats_summary.range)?;
    summary.set_item("q05", quantiles_vec[0])?;
    summary.set_item("q25", quantiles_vec[1])?;
    summary.set_item("q75", quantiles_vec[2])?;
    summary.set_item("q95", quantiles_vec[3])?;
    summary.set_item("sum", stats_summary.sum)?;
    summary.set_item("absolute_energy", stats_summary.energy)?;

    Ok(summary.into())
}

/// Calculate the mean, median, and mode of a time series.
/// 
/// This is a convenience function that returns the three most common measures 
/// of central tendency in a single function call.
///
/// # Arguments
/// * `time_series` - Input time series data as a numpy array
///
/// # Returns
/// A tuple containing (mean, median, mode)
///
/// # Errors
/// Returns `PyValueError` if:
/// - Input time series is empty
/// - Input contains NaN values
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// mean, median, mode = ct.time_series_mean_median_mode(data)
/// print(f"Mean: {mean}, Median: {median}, Mode: {mode}")
/// ```
#[pyfunction]
fn time_series_mean_median_mode(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Input time series cannot be empty"));
    }
    if ts_view.iter().any(|&x| x.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err("Input contains NaN values"));
    }
    let stats_summary = stats::_calculate_summary_statistics(ts_view);
    let (median, _) = stats::_calculate_median_and_quantiles(ts_view);
    let mode = stats::_calculate_mode(ts_view);
    Ok((stats_summary.mean, median, mode))
}

/// A Python module implemented in Rust.
#[pymodule]
fn chronoxtract(_py: Python, m: &PyModule) -> PyResult<()> {
    // Main functions
    m.add_function(wrap_pyfunction!(time_series_summary, m)?)?;
    m.add_function(wrap_pyfunction!(time_series_mean_median_mode, m)?)?;

    // Individual stat functions
    m.add_function(wrap_pyfunction!(stats::calculate_mean, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_median, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_mode, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_variance, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_skewness, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_min_max_range, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_sum, m)?)?;
    m.add_function(wrap_pyfunction!(stats::calculate_absolute_energy, m)?)?;

    // FDA functions
    m.add_function(wrap_pyfunction!(fda::perform_fft_py, m)?)?;
    m.add_function(wrap_pyfunction!(fda::lomb_scargle_py, m)?)?;

    // Rolling stats functions
    m.add_function(wrap_pyfunction!(rollingstats::rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::rolling_variance, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::expanding_sum, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::exponential_moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(rollingstats::sliding_window_entropy, m)?)?;

    // Peak functions
    m.add_function(wrap_pyfunction!(peaks::find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(peaks::peak_prominence, m)?)?;

    // Misc functions
    m.add_function(wrap_pyfunction!(misc::fractional_variability, m)?)?;
    m.add_function(wrap_pyfunction!(misc::fractional_variability_error, m)?)?;
    m.add_function(wrap_pyfunction!(misc::rolling_fractional_variability, m)?)?;
    m.add_function(wrap_pyfunction!(misc::calc_variability_timescale, m)?)?;
    m.add_function(wrap_pyfunction!(misc::variability_statistics, m)?)?;

    // Correlation functions
    m.add_function(wrap_pyfunction!(correlation::dcf_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::acf_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::zdcf_py, m)?)?;

    // Higher-order statistics
    m.add_function(wrap_pyfunction!(higherorder::hjorth_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::hjorth_activity, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::hjorth_mobility, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::hjorth_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::higher_moments, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::central_moment_5, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::central_moment_6, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::central_moment_7, m)?)?;
    m.add_function(wrap_pyfunction!(higherorder::central_moment_8, m)?)?;

    // Entropy and information-theoretic measures
    m.add_function(wrap_pyfunction!(entropy::sample_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::approximate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::lempel_ziv_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::multiscale_entropy, m)?)?;

    // Seasonality and trend analysis
    m.add_function(wrap_pyfunction!(seasonality::seasonal_trend_strength, m)?)?;
    m.add_function(wrap_pyfunction!(seasonality::seasonal_strength, m)?)?;
    m.add_function(wrap_pyfunction!(seasonality::trend_strength, m)?)?;
    m.add_function(wrap_pyfunction!(seasonality::simple_stl_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(seasonality::detect_seasonality, m)?)?;
    m.add_function(wrap_pyfunction!(seasonality::detrended_fluctuation_analysis, m)?)?;

    // Shape and peak features
    m.add_function(wrap_pyfunction!(shape::zero_crossing_rate, m)?)?;
    m.add_function(wrap_pyfunction!(shape::slope_features, m)?)?;
    m.add_function(wrap_pyfunction!(shape::mean_slope, m)?)?;
    m.add_function(wrap_pyfunction!(shape::slope_variance, m)?)?;
    m.add_function(wrap_pyfunction!(shape::max_slope, m)?)?;
    m.add_function(wrap_pyfunction!(shape::enhanced_peak_stats, m)?)?;
    m.add_function(wrap_pyfunction!(shape::peak_to_peak_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(shape::variability_features, m)?)?;
    m.add_function(wrap_pyfunction!(shape::turning_points, m)?)?;
    m.add_function(wrap_pyfunction!(shape::energy_distribution, m)?)?;

    // CARMA functions
    m.add_function(wrap_pyfunction!(carma::carma_model, m)?)?;
    m.add_function(wrap_pyfunction!(carma::set_carma_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_mle, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_method_of_moments, m)?)?;
    m.add_function(wrap_pyfunction!(carma::simulate_carma, m)?)?;
    m.add_function(wrap_pyfunction!(carma::generate_irregular_carma, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_psd, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_loglikelihood, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_predict, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_information_criteria, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_cross_validation, m)?)?;
    m.add_function(wrap_pyfunction!(carma::check_carma_stability, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_to_state_space, m)?)?;
    m.add_function(wrap_pyfunction!(carma::carma_characteristic_roots, m)?)?;

    Ok(())
}