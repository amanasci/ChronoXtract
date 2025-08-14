use std::collections::HashMap;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;

// Internal logic functions that operate on slices
pub(crate) fn _calculate_mean(time_series: ArrayView1<f64>) -> f64 {
    time_series.mean().unwrap_or(0.0)
}

pub(crate) fn _calculate_median(time_series: ArrayView1<f64>) -> f64 {
    let mut sorted = time_series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

pub(crate) fn _calculate_mode(time_series: ArrayView1<f64>) -> f64 {
    let mut counts = HashMap::new();
    for &value in time_series {
        *counts.entry(value.to_bits()).or_insert(0) += 1;
    }
    let mut max_count = 0;
    let mut mode_value = 0u64;
    for (&bits, &count) in &counts {
        if count > max_count {
            max_count = count;
            mode_value = bits;
        }
    }
    f64::from_bits(mode_value)
}

pub(crate) fn _calculate_variance(time_series: ArrayView1<f64>) -> f64 {
    time_series.var(0.0)
}

pub(crate) fn _calculate_std_dev(time_series: ArrayView1<f64>) -> f64 {
    time_series.std(0.0)
}

pub(crate) fn _calculate_skewness(time_series: ArrayView1<f64>, mean: f64, std_dev: f64) -> f64 {
    let n = time_series.len() as f64;
    time_series.iter()
        .map(|x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / n
}

pub(crate) fn _calculate_kurtosis(time_series: ArrayView1<f64>, mean: f64, std_dev: f64) -> f64 {
    let n = time_series.len() as f64;
    let kurtosis = time_series.iter()
        .map(|x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>() / n;
    kurtosis - 3.0
}

pub(crate) fn _calculate_min_max_range(time_series: ArrayView1<f64>) -> (f64, f64, f64) {
    let min = *time_series.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = *time_series.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let range = max - min;
    (min, max, range)
}

pub(crate) fn _calculate_quantiles(time_series: ArrayView1<f64>) -> Vec<f64> {
    let mut sorted = time_series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    vec![0.05, 0.25, 0.75, 0.95]
        .into_iter()
        .map(|q| {
            let pos: f64 = q * (n - 1) as f64;
            let floor = pos.floor() as usize;
            let ceil = pos.ceil() as usize;
            if floor == ceil {
                sorted[floor]
            } else {
                let frac = pos - floor as f64;
                sorted[floor] * (1.0 - frac) + sorted[ceil] * frac
            }
        })
        .collect()
}

// PyO3 wrappers
#[pyfunction]
pub fn calculate_mean(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_mean(ts_view))
}

#[pyfunction]
pub fn calculate_median(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_median(ts_view))
}

#[pyfunction]
pub fn calculate_mode(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_mode(ts_view))
}

#[pyfunction]
pub fn calculate_variance(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_variance(ts_view))
}

#[pyfunction]
pub fn calculate_std_dev(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_std_dev(ts_view))
}

#[pyfunction]
pub fn calculate_skewness(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Skewness requires at least 2 data points"));
    }
    let mean = _calculate_mean(ts_view);
    let std_dev = _calculate_std_dev(ts_view);
    if std_dev == 0.0 {
        return Ok(0.0);
    }
    Ok(_calculate_skewness(ts_view, mean, std_dev))
}

#[pyfunction]
pub fn calculate_kurtosis(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.len() < 2 {
        return Err(PyValueError::new_err("Kurtosis requires at least 2 data points"));
    }
    let mean = _calculate_mean(ts_view);
    let std_dev = _calculate_std_dev(ts_view);
    if std_dev == 0.0 {
        return Ok(0.0);
    }
    Ok(_calculate_kurtosis(ts_view, mean, std_dev))
}

#[pyfunction]
pub fn calculate_min_max_range(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_min_max_range(ts_view))
}

#[pyfunction]
pub fn calculate_quantiles(py: Python, time_series: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    let quantiles_vec = _calculate_quantiles(ts_view);
    Ok(PyArray1::from_vec(py, quantiles_vec).to_owned())
}

#[pyfunction]
pub fn calculate_sum(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    Ok(ts_view.sum())
}

#[pyfunction]
pub fn calculate_absolute_energy(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    Ok(ts_view.mapv(|x| x.powi(2)).sum())
}
