use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;

/// Calculate Hjorth parameters: activity, mobility, and complexity
/// 
/// Activity: variance of the signal
/// Mobility: variance of the first derivative divided by variance of the signal
/// Complexity: mobility of the first derivative divided by mobility of the signal
///
/// # Arguments
/// * `time_series` - Input time series data
///
/// # Returns
/// Tuple of (activity, mobility, complexity)
#[pyfunction]
pub fn hjorth_parameters(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 3 {
        return Err(PyValueError::new_err("Time series must have at least 3 points for Hjorth parameters"));
    }

    let activity = _calculate_hjorth_activity(ts_view);
    let mobility = _calculate_hjorth_mobility(ts_view)?;
    let complexity = _calculate_hjorth_complexity(ts_view)?;

    Ok((activity, mobility, complexity))
}

/// Calculate Hjorth activity (variance of the signal)
#[pyfunction]
pub fn hjorth_activity(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_hjorth_activity(ts_view))
}

/// Calculate Hjorth mobility
#[pyfunction]
pub fn hjorth_mobility(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Time series must have at least 2 points for mobility"));
    }
    _calculate_hjorth_mobility(ts_view)
}

/// Calculate Hjorth complexity
#[pyfunction]
pub fn hjorth_complexity(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 3 {
        return Err(PyValueError::new_err("Time series must have at least 3 points for complexity"));
    }
    _calculate_hjorth_complexity(ts_view)
}

/// Calculate central moments of order 5-8
#[pyfunction]
pub fn higher_moments(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }

    let (moment5, moment6, moment7, moment8) = _calculate_higher_moments(ts_view);
    Ok((moment5, moment6, moment7, moment8))
}

/// Calculate the 5th central moment
#[pyfunction]
pub fn central_moment_5(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_central_moment(ts_view, 5))
}

/// Calculate the 6th central moment
#[pyfunction]
pub fn central_moment_6(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_central_moment(ts_view, 6))
}

/// Calculate the 7th central moment
#[pyfunction]
pub fn central_moment_7(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_central_moment(ts_view, 7))
}

/// Calculate the 8th central moment
#[pyfunction]
pub fn central_moment_8(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_central_moment(ts_view, 8))
}

// Internal implementation functions

fn _calculate_hjorth_activity(data: ArrayView1<f64>) -> f64 {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}

fn _calculate_hjorth_mobility(data: ArrayView1<f64>) -> PyResult<f64> {
    if data.len() < 2 {
        return Err(PyValueError::new_err("Need at least 2 points for derivative"));
    }

    // Calculate first derivative (discrete differences)
    let mut first_derivative = Vec::with_capacity(data.len() - 1);
    for i in 1..data.len() {
        first_derivative.push(data[i] - data[i - 1]);
    }

    let activity = _calculate_hjorth_activity(data);
    let derivative_variance = _calculate_variance(&first_derivative);

    if activity <= 0.0 {
        return Ok(0.0);
    }

    Ok((derivative_variance / activity).sqrt())
}

fn _calculate_hjorth_complexity(data: ArrayView1<f64>) -> PyResult<f64> {
    if data.len() < 3 {
        return Err(PyValueError::new_err("Need at least 3 points for second derivative"));
    }

    // Calculate first derivative
    let mut first_derivative = Vec::with_capacity(data.len() - 1);
    for i in 1..data.len() {
        first_derivative.push(data[i] - data[i - 1]);
    }

    // Calculate second derivative
    let mut second_derivative = Vec::with_capacity(first_derivative.len() - 1);
    for i in 1..first_derivative.len() {
        second_derivative.push(first_derivative[i] - first_derivative[i - 1]);
    }

    let derivative_variance = _calculate_variance(&first_derivative);
    let second_derivative_variance = _calculate_variance(&second_derivative);

    if derivative_variance <= 0.0 {
        return Ok(1.0); // Default complexity for constant signal
    }

    let mobility_original = _calculate_hjorth_mobility(data)?;
    let mobility_derivative = (second_derivative_variance / derivative_variance).sqrt();

    if mobility_original <= 0.0 {
        return Ok(1.0);
    }

    Ok(mobility_derivative / mobility_original)
}

fn _calculate_variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}

fn _calculate_higher_moments(data: ArrayView1<f64>) -> (f64, f64, f64, f64) {
    let moment5 = _calculate_central_moment(data, 5);
    let moment6 = _calculate_central_moment(data, 6);
    let moment7 = _calculate_central_moment(data, 7);
    let moment8 = _calculate_central_moment(data, 8);
    (moment5, moment6, moment7, moment8)
}

fn _calculate_central_moment(data: ArrayView1<f64>, order: u32) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let n = data.len() as f64;

    data.iter()
        .map(|&x| (x - mean).powi(order as i32))
        .sum::<f64>() / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    #[test]
    fn test_hjorth_activity_internal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let activity = _calculate_hjorth_activity(Array1::from(data).view());
        assert!(activity > 0.0);
        // For data [1,2,3,4,5], mean = 3, variance = 2
        assert!((activity - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hjorth_constant_signal_internal() {
        let data = vec![5.0; 10];
        let activity = _calculate_hjorth_activity(Array1::from(data).view());
        assert!((activity - 0.0).abs() < 1e-10); // No variance
    }

    #[test]
    fn test_higher_moments_internal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (m5, m6, m7, m8) = _calculate_higher_moments(Array1::from(data).view());
        
        // All moments should be finite
        assert!(m5.is_finite());
        assert!(m6.is_finite());
        assert!(m7.is_finite());
        assert!(m8.is_finite());
    }

    #[test]
    fn test_sine_wave_hjorth_internal() {
        // Generate a sine wave - should have reasonable Hjorth parameters
        let mut data = Vec::new();
        for i in 0..100 {
            data.push((2.0 * std::f64::consts::PI * i as f64 / 10.0).sin());
        }
        
        let data_array = Array1::from(data);
        let data_view = data_array.view();
        let activity = _calculate_hjorth_activity(data_view);
        assert!(activity > 0.0);
        
        // Test mobility calculation (this will have an error result, so we just check it doesn't crash)
        let mobility_result = _calculate_hjorth_mobility(data_view);
        assert!(mobility_result.is_ok());
        let mobility = mobility_result.unwrap();
        assert!(mobility > 0.0);
    }

    #[test]
    fn test_variance_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = _calculate_variance(&data);
        assert!((variance - 2.0).abs() < 1e-10);
    }
}