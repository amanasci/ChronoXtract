use pyo3::prelude::*;

/// Computes the rolling mean over a sliding window.
///
/// # Arguments
/// * `series` - A vector of f64 values representing the time series.
/// * `window` - The size of the window.
///
/// # Returns
/// A vector with the mean for each sliding window.
#[pyfunction]
pub fn rolling_mean(series: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    let n = series.len();
    let mut means = Vec::new();
    if window == 0 || window > n {
        return Ok(means);
    }
    let mut sum: f64 = series.iter().take(window).sum();
    means.push(sum / window as f64);
    for i in window..n {
        sum += series[i] - series[i - window];
        means.push(sum / window as f64);
    }
    Ok(means)
}

/// Computes the rolling variance over a sliding window.
///
/// # Arguments
/// * `series` - A vector of f64 values.
/// * `window` - The size of the sliding window.
///
/// # Returns
/// A vector containing the variance for each window.
#[pyfunction]
pub fn rolling_variance(series: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    let n = series.len();
    let mut variances = Vec::new();
    if window == 0 || window > n {
        return Ok(variances);
    }
    for i in 0..=(n - window) {
        let window_slice = &series[i..i + window];
        let mean = window_slice.iter().sum::<f64>() / window as f64;
        let var = window_slice.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window as f64;
        variances.push(var);
    }
    Ok(variances)
}

/// Computes the expanding window sum (cumulative sum) of the series.
///
/// # Arguments
/// * `series` - A vector of f64 values.
///
/// # Returns
/// A vector where each entry is the sum of all previous values up to that index.
#[pyfunction]
pub fn expanding_sum(series: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut sums = Vec::with_capacity(series.len());
    let mut total = 0.0;
    for x in series {
        total += x;
        sums.push(total);
    }
    Ok(sums)
}

/// Computes the Exponential Moving Average (EMA) of the series.
///
/// # Arguments
/// * `series` - A vector of f64 values.
/// * `alpha` - The smoothing factor (0 < alpha <= 1) that determines the weight for recent data.
///
/// # Returns
/// A vector with the EMA computed over the series.
#[pyfunction]
pub fn exponential_moving_average(series: Vec<f64>, alpha: f64) -> PyResult<Vec<f64>> {
    let mut ema = Vec::with_capacity(series.len());
    if series.is_empty() {
        return Ok(ema);
    }
    ema.push(series[0]);
    for i in 1..series.len() {
        let prev = ema[i - 1];
        let current = alpha * series[i] + (1.0 - alpha) * prev;
        ema.push(current);
    }
    Ok(ema)
}

/// Computes the entropy over sliding windows of the series using a histogram approach.
///
/// # Arguments
/// * `series` - A vector of f64 values.
/// * `window` - The sliding window size.
/// * `bins` - The number of bins to use for the histogram.
///
/// # Returns
/// A vector containing the Shannon entropy for each window.
#[pyfunction]
pub fn sliding_window_entropy(series: Vec<f64>, window: usize, bins: usize) -> PyResult<Vec<f64>> {
    let n = series.len();
    let mut entropies = Vec::new();
    if window == 0 || window > n || bins == 0 {
        return Ok(entropies);
    }

    for i in 0..=(n - window) {
        let window_slice = &series[i..i + window];
        let min_value = window_slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_value = window_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        if range == 0.0 {
            entropies.push(0.0);
            continue;
        }
        
        let mut counts = vec![0; bins];
        for &value in window_slice {
            let mut bin = ((value - min_value) / range * bins as f64).floor() as usize;
            if bin >= bins {
                bin = bins - 1;
            }
            counts[bin] += 1;
        }
        
        let total = window as f64;
        let mut entropy = 0.0;
        for &count in &counts {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.ln();
            }
        }
        entropies.push(entropy);
    }
    Ok(entropies)
}
