use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, ndarray::s};

/// Computes the rolling mean over a sliding window.
#[pyfunction]
pub fn rolling_mean(py: Python, series: PyReadonlyArray1<f64>, window: usize) -> PyResult<Py<PyArray1<f64>>> {
    let series = series.as_array();
    let n = series.len();
    let mut means = Vec::new();
    if window > 0 && window <= n {
        let mut sum: f64 = series.slice(s![..window]).sum();
        means.push(sum / window as f64);
        for i in window..n {
            sum += series[i] - series[i - window];
            means.push(sum / window as f64);
        }
    }
    Ok(PyArray1::from_vec(py, means).to_owned())
}

/// Computes the rolling variance over a sliding window.
#[pyfunction]
pub fn rolling_variance(py: Python, series: PyReadonlyArray1<f64>, window: usize) -> PyResult<Py<PyArray1<f64>>> {
    let series = series.as_array();
    let n = series.len();
    let mut variances = Vec::new();
    if window > 0 && window <= n {
        for i in 0..=(n - window) {
            let window_slice = series.slice(s![i..i + window]);
            let var = window_slice.var(0.0);
            variances.push(var);
        }
    }
    Ok(PyArray1::from_vec(py, variances).to_owned())
}

/// Computes the expanding window sum (cumulative sum) of the series.
#[pyfunction]
pub fn expanding_sum(py: Python, series: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let series = series.as_array();
    let mut sums = Vec::with_capacity(series.len());
    let mut total = 0.0;
    for &x in series {
        total += x;
        sums.push(total);
    }
    Ok(PyArray1::from_vec(py, sums).to_owned())
}

/// Computes the Exponential Moving Average (EMA) of the series.
#[pyfunction]
pub fn exponential_moving_average(py: Python, series: PyReadonlyArray1<f64>, alpha: f64) -> PyResult<Py<PyArray1<f64>>> {
    let series = series.as_array();
    let mut ema = Vec::with_capacity(series.len());
    if !series.is_empty() {
        ema.push(series[0]);
        for i in 1..series.len() {
            let prev = ema[i - 1];
            let current = alpha * series[i] + (1.0 - alpha) * prev;
            ema.push(current);
        }
    }
    Ok(PyArray1::from_vec(py, ema).to_owned())
}

/// Computes the entropy over sliding windows of the series using a histogram approach.
#[pyfunction]
pub fn sliding_window_entropy(py: Python, series: PyReadonlyArray1<f64>, window: usize, bins: usize) -> PyResult<Py<PyArray1<f64>>> {
    let series = series.as_array();
    let n = series.len();
    let mut entropies = Vec::new();
    if window > 0 && window <= n && bins > 0 {
        for i in 0..=(n - window) {
            let window_slice = series.slice(s![i..i + window]);
            let min_value = window_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_value = window_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max_value - min_value;
            if range == 0.0 {
                entropies.push(0.0);
                continue;
            }

            let mut counts = vec![0; bins];
            for &value in &window_slice {
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
                    entropy -= p * p.log2();
                }
            }
            entropies.push(entropy);
        }
    }
    Ok(PyArray1::from_vec(py, entropies).to_owned())
}
