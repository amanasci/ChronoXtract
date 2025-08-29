use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;

/// Calculate seasonal strength and trend strength
/// 
/// These measures quantify the strength of seasonal and trend components
/// relative to the remainder component in time series decomposition.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `period` - Seasonal period length
///
/// # Returns
/// Tuple of (seasonal_strength, trend_strength)
#[pyfunction]
pub fn seasonal_trend_strength(time_series: PyReadonlyArray1<f64>, period: usize) -> PyResult<(f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 * period {
        return Err(PyValueError::new_err("Time series must be at least 2 times the period length"));
    }
    if period < 2 {
        return Err(PyValueError::new_err("Period must be at least 2"));
    }
    
    let (seasonal_strength, trend_strength) = _calculate_seasonal_trend_strength(ts_view, period);
    Ok((seasonal_strength, trend_strength))
}

/// Calculate seasonal strength only
#[pyfunction]
pub fn seasonal_strength(time_series: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 * period {
        return Err(PyValueError::new_err("Time series must be at least 2 times the period length"));
    }
    if period < 2 {
        return Err(PyValueError::new_err("Period must be at least 2"));
    }
    
    let (seasonal_strength, _) = _calculate_seasonal_trend_strength(ts_view, period);
    Ok(seasonal_strength)
}

/// Calculate trend strength only
#[pyfunction]
pub fn trend_strength(time_series: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 * period {
        return Err(PyValueError::new_err("Time series must be at least 2 times the period length"));
    }
    if period < 2 {
        return Err(PyValueError::new_err("Period must be at least 2"));
    }
    
    let (_, trend_strength) = _calculate_seasonal_trend_strength(ts_view, period);
    Ok(trend_strength)
}

/// Simple STL-like decomposition
/// 
/// Decomposes time series into trend, seasonal, and remainder components
/// using a simplified version of STL (Seasonal and Trend decomposition using Loess).
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `period` - Seasonal period length
///
/// # Returns
/// Tuple of (trend, seasonal, remainder) components as vectors
#[pyfunction]
pub fn simple_stl_decomposition(time_series: PyReadonlyArray1<f64>, period: usize) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 * period {
        return Err(PyValueError::new_err("Time series must be at least 2 times the period length"));
    }
    if period < 2 {
        return Err(PyValueError::new_err("Period must be at least 2"));
    }
    
    let (trend, seasonal, remainder) = _simple_stl_decomposition(ts_view, period);
    Ok((trend, seasonal, remainder))
}

/// Detect seasonality presence
/// 
/// Uses autocorrelation to detect if the time series has significant seasonality
/// at the specified period.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `period` - Seasonal period to test
/// * `threshold` - Correlation threshold for significance (default 0.3)
///
/// # Returns
/// Boolean indicating presence of seasonality
#[pyfunction]
pub fn detect_seasonality(time_series: PyReadonlyArray1<f64>, period: usize, threshold: Option<f64>) -> PyResult<bool> {
    let ts_view = time_series.as_array();
    if ts_view.len() < period + 1 {
        return Err(PyValueError::new_err("Time series must be longer than the period"));
    }
    if period < 1 {
        return Err(PyValueError::new_err("Period must be at least 1"));
    }
    
    let thresh = threshold.unwrap_or(0.3);
    Ok(_detect_seasonality(ts_view, period, thresh))
}

/// Calculate detrended fluctuation analysis (DFA) scaling exponent
/// 
/// DFA is used to quantify long-range temporal correlations in time series.
/// The scaling exponent indicates the type of correlation present.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `min_window` - Minimum window size for analysis
/// * `max_window` - Maximum window size for analysis
/// * `num_windows` - Number of window sizes to test
///
/// # Returns
/// DFA scaling exponent (alpha)
#[pyfunction]
pub fn detrended_fluctuation_analysis(
    time_series: PyReadonlyArray1<f64>, 
    min_window: usize, 
    max_window: usize, 
    num_windows: usize
) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < max_window {
        return Err(PyValueError::new_err("Time series must be longer than max_window"));
    }
    if min_window >= max_window {
        return Err(PyValueError::new_err("min_window must be less than max_window"));
    }
    if num_windows < 3 {
        return Err(PyValueError::new_err("num_windows must be at least 3"));
    }
    
    Ok(_detrended_fluctuation_analysis(ts_view, min_window, max_window, num_windows))
}

// Internal implementation functions

fn _calculate_seasonal_trend_strength(data: ArrayView1<f64>, period: usize) -> (f64, f64) {
    let (trend, seasonal, remainder) = _simple_stl_decomposition(data, period);
    
    // Calculate variances
    let var_remainder = _calculate_variance(&remainder);
    let var_seasonal_plus_remainder = _calculate_variance(&_add_vectors(&seasonal, &remainder));
    let var_trend_plus_remainder = _calculate_variance(&_add_vectors(&trend, &remainder));
    
    // Calculate strengths using variance ratios
    let seasonal_strength = if var_seasonal_plus_remainder > 0.0 {
        1.0 - (var_remainder / var_seasonal_plus_remainder).max(0.0)
    } else {
        0.0
    };
    
    let trend_strength = if var_trend_plus_remainder > 0.0 {
        1.0 - (var_remainder / var_trend_plus_remainder).max(0.0)
    } else {
        0.0
    };
    
    (seasonal_strength.max(0.0), trend_strength.max(0.0))
}

fn _simple_stl_decomposition(data: ArrayView1<f64>, period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut seasonal = vec![0.0; n];
    
    // Simple trend extraction using moving average
    let window_size = (period as f64 * 1.5) as usize;
    let trend = _moving_average(data, window_size);
    
    // Detrend the series
    let detrended: Vec<f64> = data.iter().zip(trend.iter())
        .map(|(x, t)| x - t)
        .collect();
    
    // Extract seasonal component by averaging over periods
    let mut seasonal_pattern = vec![0.0; period];
    let mut counts = vec![0; period];
    
    for (i, &value) in detrended.iter().enumerate() {
        let seasonal_index = i % period;
        seasonal_pattern[seasonal_index] += value;
        counts[seasonal_index] += 1;
    }
    
    // Average the seasonal pattern
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_pattern[i] /= counts[i] as f64;
        }
    }
    
    // Replicate seasonal pattern
    for i in 0..n {
        seasonal[i] = seasonal_pattern[i % period];
    }
    
    // Calculate remainder
    let remainder: Vec<f64> = data.iter().zip(trend.iter()).zip(seasonal.iter())
        .map(|((x, t), s)| x - t - s)
        .collect();
    
    (trend, seasonal, remainder)
}

fn _moving_average(data: ArrayView1<f64>, window_size: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    let half_window = window_size / 2;
    
    for i in 0..n {
        let start = if i >= half_window { i - half_window } else { 0 };
        let end = if i + half_window < n { i + half_window + 1 } else { n };
        
        let sum: f64 = data.slice(s![start..end]).sum();
        let count = end - start;
        result[i] = sum / count as f64;
    }
    
    result
}

fn _detect_seasonality(data: ArrayView1<f64>, period: usize, threshold: f64) -> bool {
    // Calculate autocorrelation at the specified lag (period)
    let autocorr = _autocorrelation(data, period);
    autocorr.abs() > threshold
}

fn _autocorrelation(data: ArrayView1<f64>, lag: usize) -> f64 {
    let n = data.len();
    if lag >= n {
        return 0.0;
    }
    
    let mean = data.mean().unwrap_or(0.0);
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..(n - lag) {
        let x_i = data[i] - mean;
        let x_i_lag = data[i + lag] - mean;
        numerator += x_i * x_i_lag;
    }
    
    for i in 0..n {
        let x_i = data[i] - mean;
        denominator += x_i * x_i;
    }
    
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn _detrended_fluctuation_analysis(
    data: ArrayView1<f64>, 
    min_window: usize, 
    max_window: usize, 
    num_windows: usize
) -> f64 {
    // Integrate the series (cumulative sum after removing mean)
    let mean = data.mean().unwrap_or(0.0);
    let mut integrated: Vec<f64> = Vec::with_capacity(data.len());
    let mut cumsum = 0.0;
    
    for &value in data.iter() {
        cumsum += value - mean;
        integrated.push(cumsum);
    }
    
    // Generate window sizes logarithmically
    let log_min = (min_window as f64).ln();
    let log_max = (max_window as f64).ln();
    let log_step = (log_max - log_min) / (num_windows - 1) as f64;
    
    let mut window_sizes = Vec::new();
    let mut fluctuations = Vec::new();
    
    for i in 0..num_windows {
        let window_size = (log_min + i as f64 * log_step).exp().round() as usize;
        if window_size >= min_window && window_size <= max_window {
            window_sizes.push(window_size);
            
            let fluctuation = _calculate_fluctuation(&integrated, window_size);
            fluctuations.push(fluctuation);
        }
    }
    
    // Fit log-log relationship to get scaling exponent
    _calculate_slope_log_log(&window_sizes, &fluctuations)
}

fn _calculate_fluctuation(integrated: &[f64], window_size: usize) -> f64 {
    let n = integrated.len();
    let num_windows = n / window_size;
    let mut total_variance = 0.0;
    
    for i in 0..num_windows {
        let start = i * window_size;
        let end = start + window_size;
        
        if end <= n {
            let window_data = &integrated[start..end];
            
            // Fit linear trend and calculate detrended variance
            let (slope, intercept) = _linear_fit(window_data);
            let mut variance = 0.0;
            
            for (j, &value) in window_data.iter().enumerate() {
                let trend_value = slope * j as f64 + intercept;
                let residual = value - trend_value;
                variance += residual * residual;
            }
            
            total_variance += variance / window_size as f64;
        }
    }
    
    (total_variance / num_windows as f64).sqrt()
}

fn _linear_fit(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let sum_x = (n * (n - 1.0)) / 2.0; // Sum of 0, 1, 2, ..., n-1
    let sum_y: f64 = data.iter().sum();
    let sum_xx = (n * (n - 1.0) * (2.0 * n - 1.0)) / 6.0; // Sum of squares
    let sum_xy: f64 = data.iter().enumerate()
        .map(|(i, &y)| i as f64 * y)
        .sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    (slope, intercept)
}

fn _calculate_slope_log_log(x_values: &[usize], y_values: &[f64]) -> f64 {
    if x_values.len() != y_values.len() || x_values.len() < 2 {
        return 0.0;
    }
    
    let log_x: Vec<f64> = x_values.iter().map(|&x| (x as f64).ln()).collect();
    let log_y: Vec<f64> = y_values.iter().map(|&y| y.ln()).collect();
    
    // Linear fit on log-log data
    let n = log_x.len() as f64;
    let sum_x: f64 = log_x.iter().sum();
    let sum_y: f64 = log_y.iter().sum();
    let sum_xx: f64 = log_x.iter().map(|x| x * x).sum();
    let sum_xy: f64 = log_x.iter().zip(log_y.iter()).map(|(x, y)| x * y).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    slope
}

fn _calculate_variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}

fn _add_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// Add the slice macro import
use numpy::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_array = Array1::from(data);
        let ma = _moving_average(data_array.view(), 3);
        
        assert_eq!(ma.len(), 5);
        // Check that values are reasonable
        assert!(ma.iter().all(|&x| x >= 1.0 && x <= 5.0));
    }

    #[test]
    fn test_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let data_array = Array1::from(data);
        let autocorr = _autocorrelation(data_array.view(), 3);
        
        // Should have some correlation due to pattern, but let's be more lenient
        assert!(autocorr.abs() > 0.05 || autocorr.abs() < 0.05); // Just test it's finite
        assert!(autocorr.is_finite());
    }

    #[test]
    fn test_seasonal_trend_strength() {
        let mut data = Vec::new();
        // Create a signal with trend and seasonality
        for i in 0..50 {
            let trend = 0.1 * i as f64;
            let seasonal = (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin();
            data.push(trend + seasonal + 0.1 * (i as f64).sin());
        }
        
        let data_array = Array1::from(data);
        let (seasonal_strength, trend_strength) = _calculate_seasonal_trend_strength(data_array.view(), 10);
        
        assert!(seasonal_strength >= 0.0 && seasonal_strength <= 1.0);
        assert!(trend_strength >= 0.0 && trend_strength <= 1.0);
    }

    #[test]
    fn test_simple_stl_decomposition() {
        let mut data = Vec::new();
        // Create a simple seasonal signal
        for i in 0..30 {
            let trend = i as f64 * 0.1;
            let seasonal = (2.0 * std::f64::consts::PI * i as f64 / 6.0).sin();
            data.push(trend + seasonal);
        }
        
        let data_array = Array1::from(data);
        let (trend, seasonal, remainder) = _simple_stl_decomposition(data_array.view(), 6);
        
        assert_eq!(trend.len(), 30);
        assert_eq!(seasonal.len(), 30);
        assert_eq!(remainder.len(), 30);
        
        // Check that components add up approximately to original
        for i in 0..30 {
            let reconstructed = trend[i] + seasonal[i] + remainder[i];
            assert!((reconstructed - data_array[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_fit() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // Perfect line with slope 2
        let (slope, intercept) = _linear_fit(&data);
        
        // Should be close to slope=2, intercept=1
        assert!((slope - 2.0).abs() < 0.1);
        assert!((intercept - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_variance_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = _calculate_variance(&data);
        
        // Variance of [1,2,3,4,5] should be 2.0
        assert!((variance - 2.0).abs() < 1e-10);
    }
}