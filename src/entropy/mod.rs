use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

/// Calculate Sample Entropy (SampEn)
/// 
/// Sample Entropy is a measure of regularity and complexity in time series.
/// It measures the probability that patterns that are similar remain similar
/// on the next increment.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `m` - Pattern length (template length)
/// * `r` - Tolerance for matching
///
/// # Returns
/// Sample entropy value
#[pyfunction]
pub fn sample_entropy(time_series: PyReadonlyArray1<f64>, m: usize, r: f64) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < m + 1 {
        return Err(PyValueError::new_err("Time series too short for given pattern length"));
    }
    if r <= 0.0 {
        return Err(PyValueError::new_err("Tolerance r must be positive"));
    }
    
    Ok(_calculate_sample_entropy(ts_view, m, r))
}

/// Calculate Approximate Entropy (ApEn)
/// 
/// Approximate Entropy quantifies the regularity and complexity of time series data.
/// It measures the likelihood that patterns of observations that are similar
/// will be followed by similar patterns.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `m` - Pattern length
/// * `r` - Tolerance for matching
///
/// # Returns
/// Approximate entropy value
#[pyfunction]
pub fn approximate_entropy(time_series: PyReadonlyArray1<f64>, m: usize, r: f64) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < m + 1 {
        return Err(PyValueError::new_err("Time series too short for given pattern length"));
    }
    if r <= 0.0 {
        return Err(PyValueError::new_err("Tolerance r must be positive"));
    }
    
    Ok(_calculate_approximate_entropy(ts_view, m, r))
}

/// Calculate Permutation Entropy (PE)
/// 
/// Permutation Entropy is a complexity measure for time series based on 
/// comparing neighboring values and determining their relative order.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `m` - Embedding dimension (order)
/// * `delay` - Time delay (tau)
///
/// # Returns
/// Permutation entropy value
#[pyfunction]
pub fn permutation_entropy(time_series: PyReadonlyArray1<f64>, m: usize, delay: usize) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < m * delay {
        return Err(PyValueError::new_err("Time series too short for given embedding dimension and delay"));
    }
    if m < 2 {
        return Err(PyValueError::new_err("Embedding dimension must be at least 2"));
    }
    if delay < 1 {
        return Err(PyValueError::new_err("Delay must be at least 1"));
    }
    
    Ok(_calculate_permutation_entropy(ts_view, m, delay))
}

/// Calculate Lempel-Ziv Complexity (LZC)
/// 
/// Lempel-Ziv Complexity measures the complexity of a finite binary sequence.
/// The time series is first converted to a binary sequence.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `threshold` - Optional threshold for binarization (uses median if None)
///
/// # Returns
/// Lempel-Ziv complexity value
#[pyfunction]
pub fn lempel_ziv_complexity(time_series: PyReadonlyArray1<f64>, threshold: Option<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    
    Ok(_calculate_lempel_ziv_complexity(ts_view, threshold))
}

/// Calculate Multiscale Entropy (MSE)
/// 
/// Multiscale Entropy calculates Sample Entropy at multiple time scales
/// by coarse-graining the time series.
///
/// # Arguments
/// * `time_series` - Input time series data
/// * `m` - Pattern length for sample entropy
/// * `r` - Tolerance for matching
/// * `max_scale` - Maximum scale factor
///
/// # Returns
/// Vector of entropy values at different scales
#[pyfunction]
pub fn multiscale_entropy(time_series: PyReadonlyArray1<f64>, m: usize, r: f64, max_scale: usize) -> PyResult<Vec<f64>> {
    let ts_view = time_series.as_array();
    if ts_view.len() < (m + 1) * max_scale {
        return Err(PyValueError::new_err("Time series too short for given parameters"));
    }
    if r <= 0.0 {
        return Err(PyValueError::new_err("Tolerance r must be positive"));
    }
    if max_scale < 1 {
        return Err(PyValueError::new_err("Max scale must be at least 1"));
    }
    
    Ok(_calculate_multiscale_entropy(ts_view, m, r, max_scale))
}

// Internal implementation functions

fn _calculate_sample_entropy(data: ArrayView1<f64>, m: usize, r: f64) -> f64 {
    let n = data.len();
    
    // Count matches for patterns of length m and m+1
    let a = _count_matches(data, m + 1, r);
    let b = _count_matches(data, m, r);
    
    if b == 0 {
        return f64::INFINITY; // No matches of length m
    }
    
    let phi_m = (b as f64) / ((n - m) as f64);
    let phi_m_plus_1 = (a as f64) / ((n - m - 1) as f64);
    
    if phi_m_plus_1 == 0.0 {
        return f64::INFINITY;
    }
    
    -((phi_m_plus_1) / phi_m).ln()
}

fn _calculate_approximate_entropy(data: ArrayView1<f64>, m: usize, r: f64) -> f64 {
    let phi_m = _phi(data, m, r);
    let phi_m_plus_1 = _phi(data, m + 1, r);
    
    phi_m - phi_m_plus_1
}

fn _phi(data: ArrayView1<f64>, m: usize, r: f64) -> f64 {
    let n = data.len();
    let mut patterns = Vec::new();
    
    // Extract all possible patterns of length m
    for i in 0..=(n - m) {
        let pattern: Vec<f64> = data.slice(s![i..i + m]).to_vec();
        patterns.push(pattern);
    }
    
    let mut phi_sum = 0.0;
    
    for i in 0..patterns.len() {
        let mut matches = 0;
        
        for j in 0..patterns.len() {
            if _patterns_match(&patterns[i], &patterns[j], r) {
                matches += 1;
            }
        }
        
        if matches > 0 {
            phi_sum += (matches as f64 / patterns.len() as f64).ln();
        }
    }
    
    phi_sum / patterns.len() as f64
}

fn _patterns_match(pattern1: &[f64], pattern2: &[f64], r: f64) -> bool {
    pattern1.iter().zip(pattern2.iter()).all(|(a, b)| (a - b).abs() <= r)
}

fn _count_matches(data: ArrayView1<f64>, m: usize, r: f64) -> usize {
    let n = data.len();
    let mut count = 0;
    
    for i in 0..=(n - m) {
        for j in (i + 1)..=(n - m) {
            let mut match_found = true;
            
            for k in 0..m {
                if (data[i + k] - data[j + k]).abs() > r {
                    match_found = false;
                    break;
                }
            }
            
            if match_found {
                count += 1;
            }
        }
    }
    
    count
}

fn _calculate_permutation_entropy(data: ArrayView1<f64>, m: usize, delay: usize) -> f64 {
    let n = data.len();
    let mut ordinal_patterns = HashMap::new();
    let mut total_patterns = 0;
    
    // Generate all possible ordinal patterns
    for i in 0..=(n - (m - 1) * delay - 1) {
        let mut pattern_values = Vec::new();
        
        // Extract values for the current pattern
        for j in 0..m {
            pattern_values.push((data[i + j * delay], j));
        }
        
        // Sort by value to get ordinal pattern
        pattern_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Extract the ordinal pattern (ranks)
        let ordinal_pattern: Vec<usize> = pattern_values.iter().map(|(_, idx)| *idx).collect();
        
        *ordinal_patterns.entry(ordinal_pattern).or_insert(0) += 1;
        total_patterns += 1;
    }
    
    // Calculate entropy
    let mut entropy = 0.0;
    for &count in ordinal_patterns.values() {
        let probability = count as f64 / total_patterns as f64;
        entropy -= probability * probability.ln();
    }
    
    entropy
}

fn _calculate_lempel_ziv_complexity(data: ArrayView1<f64>, threshold: Option<f64>) -> f64 {
    // Determine threshold for binarization
    let thresh = threshold.unwrap_or_else(|| {
        let mut sorted_data: Vec<f64> = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_data[sorted_data.len() / 2] // median
    });
    
    // Convert to binary string
    let binary_string: Vec<u8> = data.iter()
        .map(|&x| if x >= thresh { 1 } else { 0 })
        .collect();
    
    // Calculate LZ complexity
    let mut complexity = 0;
    let mut i = 0;
    let n = binary_string.len();
    
    while i < n {
        let mut j = 1;
        
        // Find the longest prefix that appears before
        while i + j <= n {
            let current_substring = &binary_string[i..i + j];
            let mut found = false;
            
            // Check if this substring appears earlier
            for k in 0..i {
                if k + j <= i && &binary_string[k..k + j] == current_substring {
                    found = true;
                    break;
                }
            }
            
            if !found {
                break;
            }
            j += 1;
        }
        
        complexity += 1;
        i += j.max(1);
    }
    
    complexity as f64
}

fn _calculate_multiscale_entropy(data: ArrayView1<f64>, m: usize, r: f64, max_scale: usize) -> Vec<f64> {
    let mut entropies = Vec::with_capacity(max_scale);
    
    for scale in 1..=max_scale {
        let coarse_grained = _coarse_grain(data, scale);
        
        if coarse_grained.len() >= m + 1 {
            let entropy = _calculate_sample_entropy(
                numpy::ndarray::Array1::from(coarse_grained).view(),
                m,
                r
            );
            entropies.push(entropy);
        } else {
            entropies.push(f64::NAN);
        }
    }
    
    entropies
}

fn _coarse_grain(data: ArrayView1<f64>, scale: usize) -> Vec<f64> {
    let n = data.len();
    let new_length = n / scale;
    let mut coarse_grained = Vec::with_capacity(new_length);
    
    for i in 0..new_length {
        let start = i * scale;
        let end = (start + scale).min(n);
        let sum: f64 = data.slice(s![start..end]).sum();
        coarse_grained.push(sum / (end - start) as f64);
    }
    
    coarse_grained
}

// Add the slice macro import
use numpy::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    #[test]
    fn test_sample_entropy_constant() {
        let data = vec![1.0; 100];
        let data_array = Array1::from(data);
        let data_view = data_array.view();
        let entropy = _calculate_sample_entropy(data_view, 2, 0.1);
        // Constant signal should have very low entropy (could be infinity due to no matches)
        assert!(entropy == f64::INFINITY || entropy < 0.1);
    }

    #[test]
    fn test_permutation_entropy_calculation() {
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let data_array = Array1::from(data);
        let data_view = data_array.view();
        let entropy = _calculate_permutation_entropy(data_view, 3, 1);
        assert!(entropy.is_finite());
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_lempel_ziv_complexity() {
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Periodic pattern
        let data_array = Array1::from(data);
        let data_view = data_array.view();
        let complexity = _calculate_lempel_ziv_complexity(data_view, Some(1.5));
        assert!(complexity > 0.0);
        assert!(complexity.is_finite());
    }

    #[test]
    fn test_coarse_graining() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_array = Array1::from(data);
        let data_view = data_array.view();
        let coarse_grained = _coarse_grain(data_view, 2);
        assert_eq!(coarse_grained.len(), 3);
        assert!((coarse_grained[0] - 1.5).abs() < 1e-10); // (1+2)/2
        assert!((coarse_grained[1] - 3.5).abs() < 1e-10); // (3+4)/2
        assert!((coarse_grained[2] - 5.5).abs() < 1e-10); // (5+6)/2
    }

    #[test]
    fn test_patterns_match() {
        let pattern1 = vec![1.0, 2.0, 3.0];
        let pattern2 = vec![1.1, 1.9, 3.1];
        assert!(_patterns_match(&pattern1, &pattern2, 0.2));
        assert!(!_patterns_match(&pattern1, &pattern2, 0.05));
    }
}