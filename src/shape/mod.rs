use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;

/// Calculate zero-crossing rate
/// 
/// Zero-crossing rate is the rate at which the signal changes sign.
/// It's a measure of the signal's frequency content and noisiness.
///
/// # Arguments
/// * `time_series` - Input time series data
///
/// # Returns
/// Zero-crossing rate (number of sign changes per sample)
#[pyfunction]
pub fn zero_crossing_rate(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Time series must have at least 2 points"));
    }
    
    Ok(_calculate_zero_crossing_rate(ts_view))
}

/// Calculate slope-based features
/// 
/// Computes various slope-based features including mean slope, 
/// slope variance, and maximum absolute slope.
///
/// # Arguments
/// * `time_series` - Input time series data
///
/// # Returns
/// Tuple of (mean_slope, slope_variance, max_slope)
#[pyfunction]
pub fn slope_features(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Time series must have at least 2 points"));
    }
    
    let (mean_slope, slope_variance, max_slope) = _calculate_slope_features(ts_view);
    Ok((mean_slope, slope_variance, max_slope))
}

/// Calculate mean slope
#[pyfunction]
pub fn mean_slope(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Time series must have at least 2 points"));
    }
    
    let (mean_slope, _, _) = _calculate_slope_features(ts_view);
    Ok(mean_slope)
}

/// Calculate slope variance
#[pyfunction]
pub fn slope_variance(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Time series must have at least 2 points"));
    }
    
    let (_, slope_variance, _) = _calculate_slope_features(ts_view);
    Ok(slope_variance)
}

/// Calculate maximum absolute slope
#[pyfunction]
pub fn max_slope(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Time series must have at least 2 points"));
    }
    
    let (_, _, max_slope) = _calculate_slope_features(ts_view);
    Ok(max_slope)
}

/// Calculate enhanced peak statistics
/// 
/// Provides comprehensive peak analysis including count, prominence,
/// width, spacing, and amplitude features.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `min_prominence` - Minimum prominence for peak detection
/// * `min_distance` - Minimum distance between peaks
///
/// # Returns
/// Dictionary with peak statistics
#[pyfunction]
pub fn enhanced_peak_stats(
    time_series: PyReadonlyArray1<f64>, 
    min_prominence: Option<f64>, 
    min_distance: Option<usize>
) -> PyResult<(usize, f64, f64, f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 3 {
        return Err(PyValueError::new_err("Time series must have at least 3 points"));
    }
    
    let prominence_threshold = min_prominence.unwrap_or(0.1);
    let distance_threshold = min_distance.unwrap_or(1);
    
    let stats = _calculate_enhanced_peak_stats(ts_view, prominence_threshold, distance_threshold);
    Ok(stats)
}

/// Calculate peak-to-peak amplitude statistics
#[pyfunction]
pub fn peak_to_peak_amplitude(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 3 {
        return Err(PyValueError::new_err("Time series must have at least 3 points"));
    }
    
    let (max_p2p, mean_p2p, std_p2p) = _calculate_peak_to_peak_amplitude(ts_view);
    Ok((max_p2p, mean_p2p, std_p2p))
}

/// Calculate signal variability features
/// 
/// Computes various measures of signal variability and shape complexity.
///
/// # Arguments
/// * `time_series` - Input time series data
///
/// # Returns
/// Tuple of (coefficient_variation, quartile_coefficient_dispersion, 
///           median_absolute_deviation, interquartile_range)
#[pyfunction]
pub fn variability_features(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    
    let (cv, qcd, mad, iqr) = _calculate_variability_features(ts_view);
    Ok((cv, qcd, mad, iqr))
}

/// Calculate turning points
/// 
/// Identifies local maxima and minima (turning points) in the time series.
///
/// # Arguments
/// * `time_series` - Input time series data
///
/// # Returns
/// Tuple of (num_turning_points, turning_point_rate)
#[pyfunction]
pub fn turning_points(time_series: PyReadonlyArray1<f64>) -> PyResult<(usize, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 3 {
        return Err(PyValueError::new_err("Time series must have at least 3 points"));
    }
    
    let (num_turning_points, turning_point_rate) = _calculate_turning_points(ts_view);
    Ok((num_turning_points, turning_point_rate))
}

/// Calculate signal energy distribution features
#[pyfunction]
pub fn energy_distribution(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    
    let (energy_entropy, normalized_energy, energy_concentration) = _calculate_energy_distribution(ts_view);
    Ok((energy_entropy, normalized_energy, energy_concentration))
}

// Internal implementation functions

fn _calculate_zero_crossing_rate(data: ArrayView1<f64>) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    
    let mut crossings = 0;
    for i in 1..n {
        if (data[i] >= 0.0 && data[i-1] < 0.0) || (data[i] < 0.0 && data[i-1] >= 0.0) {
            crossings += 1;
        }
    }
    
    crossings as f64 / (n - 1) as f64
}

fn _calculate_slope_features(data: ArrayView1<f64>) -> (f64, f64, f64) {
    let n = data.len();
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    
    // Calculate slopes (first differences)
    let mut slopes = Vec::with_capacity(n - 1);
    for i in 1..n {
        slopes.push(data[i] - data[i-1]);
    }
    
    // Mean slope
    let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
    
    // Slope variance
    let slope_variance = slopes.iter()
        .map(|&s| (s - mean_slope).powi(2))
        .sum::<f64>() / slopes.len() as f64;
    
    // Maximum absolute slope
    let max_slope = slopes.iter()
        .map(|&s| s.abs())
        .fold(0.0, f64::max);
    
    (mean_slope, slope_variance, max_slope)
}

fn _calculate_enhanced_peak_stats(
    data: ArrayView1<f64>, 
    min_prominence: f64, 
    min_distance: usize
) -> (usize, f64, f64, f64, f64, f64) {
    let peaks = _find_peaks_with_prominence(data, min_prominence, min_distance);
    
    if peaks.is_empty() {
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    
    let num_peaks = peaks.len();
    
    // Calculate peak prominences
    let prominences: Vec<f64> = peaks.iter()
        .map(|&peak_idx| _calculate_prominence(data, peak_idx))
        .collect();
    
    let mean_prominence = prominences.iter().sum::<f64>() / prominences.len() as f64;
    
    // Calculate peak spacing
    let mut spacings = Vec::new();
    for i in 1..peaks.len() {
        spacings.push((peaks[i] - peaks[i-1]) as f64);
    }
    let mean_spacing = if spacings.is_empty() {
        0.0
    } else {
        spacings.iter().sum::<f64>() / spacings.len() as f64
    };
    
    // Calculate peak widths (simplified)
    let mean_width = _calculate_mean_peak_width(data, &peaks);
    
    // Calculate peak-to-peak amplitude
    let (max_p2p_amplitude, _) = _calculate_peak_amplitudes(data, &peaks);
    
    // Peak density (peaks per unit length)
    let peak_density = num_peaks as f64 / data.len() as f64;
    
    (num_peaks, mean_prominence, mean_spacing, mean_width, max_p2p_amplitude, peak_density)
}

fn _find_peaks_with_prominence(data: ArrayView1<f64>, min_prominence: f64, min_distance: usize) -> Vec<usize> {
    let n = data.len();
    if n < 3 {
        return Vec::new();
    }
    
    let mut peaks = Vec::new();
    
    // Find local maxima
    for i in 1..(n-1) {
        if data[i] > data[i-1] && data[i] > data[i+1] {
            let prominence = _calculate_prominence(data, i);
            if prominence >= min_prominence {
                peaks.push(i);
            }
        }
    }
    
    // Apply minimum distance constraint
    _filter_peaks_by_distance(peaks, min_distance)
}

fn _calculate_prominence(data: ArrayView1<f64>, peak_idx: usize) -> f64 {
    let peak_value = data[peak_idx];
    
    // Find minimum to the left
    let left_min = (0..peak_idx)
        .map(|i| data[i])
        .fold(peak_value, f64::min);
    
    // Find minimum to the right
    let right_min = ((peak_idx + 1)..data.len())
        .map(|i| data[i])
        .fold(peak_value, f64::min);
    
    // Prominence is the minimum of the drops on both sides
    peak_value - left_min.max(right_min)
}

fn _filter_peaks_by_distance(mut peaks: Vec<usize>, min_distance: usize) -> Vec<usize> {
    if peaks.len() <= 1 {
        return peaks;
    }
    
    peaks.sort_unstable();
    let mut filtered = vec![peaks[0]];
    
    for &peak in peaks.iter().skip(1) {
        if peak - filtered.last().unwrap() >= min_distance {
            filtered.push(peak);
        }
    }
    
    filtered
}

fn _calculate_mean_peak_width(data: ArrayView1<f64>, peaks: &[usize]) -> f64 {
    if peaks.is_empty() {
        return 0.0;
    }
    
    let mut total_width = 0.0;
    
    for &peak_idx in peaks {
        // Simple width calculation: find half-prominence points
        let peak_value = data[peak_idx];
        let prominence = _calculate_prominence(data, peak_idx);
        let half_prominence_level = peak_value - prominence / 2.0;
        
        // Find left boundary
        let mut left_idx = peak_idx;
        while left_idx > 0 && data[left_idx] > half_prominence_level {
            left_idx -= 1;
        }
        
        // Find right boundary
        let mut right_idx = peak_idx;
        while right_idx < data.len() - 1 && data[right_idx] > half_prominence_level {
            right_idx += 1;
        }
        
        total_width += (right_idx - left_idx) as f64;
    }
    
    total_width / peaks.len() as f64
}

fn _calculate_peak_to_peak_amplitude(data: ArrayView1<f64>) -> (f64, f64, f64) {
    let peaks = _find_peaks_with_prominence(data, 0.01, 1);
    let (max_amplitude, amplitudes) = _calculate_peak_amplitudes(data, &peaks);
    
    if amplitudes.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let mean_amplitude = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
    let variance = amplitudes.iter()
        .map(|&a| (a - mean_amplitude).powi(2))
        .sum::<f64>() / amplitudes.len() as f64;
    let std_amplitude = variance.sqrt();
    
    (max_amplitude, mean_amplitude, std_amplitude)
}

fn _calculate_peak_amplitudes(data: ArrayView1<f64>, peaks: &[usize]) -> (f64, Vec<f64>) {
    let mut amplitudes = Vec::new();
    
    for i in 1..peaks.len() {
        let peak1_value = data[peaks[i-1]];
        let peak2_value = data[peaks[i]];
        
        // Find minimum between peaks
        let min_between = (peaks[i-1]..=peaks[i])
            .map(|idx| data[idx])
            .fold(f64::INFINITY, f64::min);
        
        let amplitude1 = peak1_value - min_between;
        let amplitude2 = peak2_value - min_between;
        amplitudes.push(amplitude1.max(amplitude2));
    }
    
    let max_amplitude = amplitudes.iter().copied().fold(0.0, f64::max);
    (max_amplitude, amplitudes)
}

fn _calculate_variability_features(data: ArrayView1<f64>) -> (f64, f64, f64, f64) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    // Calculate basic statistics
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    
    // Coefficient of variation
    let cv = if mean != 0.0 { std_dev / mean.abs() } else { 0.0 };
    
    // Calculate quartiles
    let mut sorted_data: Vec<f64> = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let q1_idx = n / 4;
    let q2_idx = n / 2;
    let q3_idx = 3 * n / 4;
    
    let q1 = sorted_data[q1_idx];
    let q2 = sorted_data[q2_idx]; // median
    let q3 = sorted_data[q3_idx];
    
    // Quartile coefficient of dispersion
    let qcd = if q1 + q3 != 0.0 { (q3 - q1) / (q3 + q1) } else { 0.0 };
    
    // Median absolute deviation
    let mad = {
        let deviations: Vec<f64> = data.iter()
            .map(|&x| (x - q2).abs())
            .collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_deviations[sorted_deviations.len() / 2]
    };
    
    // Interquartile range
    let iqr = q3 - q1;
    
    (cv, qcd, mad, iqr)
}

fn _calculate_turning_points(data: ArrayView1<f64>) -> (usize, f64) {
    let n = data.len();
    if n < 3 {
        return (0, 0.0);
    }
    
    let mut turning_points = 0;
    
    for i in 1..(n-1) {
        let is_local_max = data[i] > data[i-1] && data[i] > data[i+1];
        let is_local_min = data[i] < data[i-1] && data[i] < data[i+1];
        
        if is_local_max || is_local_min {
            turning_points += 1;
        }
    }
    
    let turning_point_rate = turning_points as f64 / (n - 2) as f64;
    
    (turning_points, turning_point_rate)
}

fn _calculate_energy_distribution(data: ArrayView1<f64>) -> (f64, f64, f64) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    
    // Calculate squared values (energy)
    let energies: Vec<f64> = data.iter().map(|&x| x * x).collect();
    let total_energy: f64 = energies.iter().sum();
    
    if total_energy == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    
    // Normalized energy
    let normalized_energy = total_energy / n as f64;
    
    // Energy entropy
    let energy_entropy = {
        let mut entropy = 0.0;
        for &energy in &energies {
            if energy > 0.0 {
                let prob = energy / total_energy;
                entropy -= prob * prob.ln();
            }
        }
        entropy
    };
    
    // Energy concentration (percentage of energy in top 10% of samples)
    let mut sorted_energies = energies.clone();
    sorted_energies.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let top_10_percent = (n as f64 * 0.1).ceil() as usize;
    let concentrated_energy: f64 = sorted_energies.iter().take(top_10_percent).sum();
    let energy_concentration = concentrated_energy / total_energy;
    
    (energy_entropy, normalized_energy, energy_concentration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    #[test]
    fn test_zero_crossing_rate() {
        // Alternating signal should have high zero-crossing rate
        let data = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let data_array = Array1::from(data);
        let zcr = _calculate_zero_crossing_rate(data_array.view());
        
        assert!(zcr > 0.8); // Should be 1.0 or close to it
    }

    #[test]
    fn test_slope_features() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 1.0]; // varying slopes
        let data_array = Array1::from(data);
        let (mean_slope, slope_var, max_slope) = _calculate_slope_features(data_array.view());
        
        assert!(mean_slope.is_finite());
        assert!(slope_var >= 0.0);
        assert!(max_slope >= 0.0);
    }

    #[test]
    fn test_peak_detection() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0]; // has peaks at indices 1, 3, 5
        let data_array = Array1::from(data);
        let peaks = _find_peaks_with_prominence(data_array.view(), 0.5, 1);
        
        assert!(!peaks.is_empty());
        // Should find peaks at positions where local maxima exist
        assert!(peaks.contains(&1) || peaks.contains(&3) || peaks.contains(&5));
    }

    #[test]
    fn test_variability_features() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_array = Array1::from(data);
        let (cv, qcd, mad, iqr) = _calculate_variability_features(data_array.view());
        
        assert!(cv > 0.0);
        assert!(qcd >= 0.0);
        assert!(mad >= 0.0);
        assert!(iqr >= 0.0);
    }

    #[test]
    fn test_turning_points() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 1.0]; // has turning points
        let data_array = Array1::from(data);
        let (num_tp, tp_rate) = _calculate_turning_points(data_array.view());
        
        assert!(num_tp > 0);
        assert!(tp_rate > 0.0);
        assert!(tp_rate <= 1.0);
    }

    #[test]
    fn test_energy_distribution() {
        let data = vec![1.0, 2.0, 0.5, 3.0, 0.1];
        let data_array = Array1::from(data);
        let (entropy, norm_energy, concentration) = _calculate_energy_distribution(data_array.view());
        
        assert!(entropy >= 0.0);
        assert!(norm_energy >= 0.0);
        assert!(concentration >= 0.0 && concentration <= 1.0);
    }
}