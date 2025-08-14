use std::collections::HashMap;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;

// Internal logic functions that operate on slices
pub(crate) struct SummaryStatistics {
    pub(crate) mean: f64,
    pub(crate) variance: f64,
    pub(crate) std_dev: f64,
    pub(crate) skewness: Option<f64>,
    pub(crate) kurtosis: Option<f64>,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) range: f64,
    pub(crate) sum: f64,
    pub(crate) energy: f64,
}

pub(crate) fn _calculate_summary_statistics(time_series: ArrayView1<f64>) -> SummaryStatistics {
    let n = time_series.len() as f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;

    for &x in time_series {
        let x2 = x * x;
        s1 += x;
        s2 += x2;
        s3 += x2 * x;
        s4 += x2 * x2;
        min = min.min(x);
        max = max.max(x);
    }

    let m1 = s1 / n;
    let m2 = s2 / n;
    let m3 = s3 / n;
    let m4 = s4 / n;

    let mean = m1;
    let variance = m2 - m1 * m1;
    let std_dev = variance.sqrt();

    let (skewness, kurtosis) = if std_dev > 1e-9 { // Use a small epsilon to avoid division by zero
        let m1_pow2 = m1 * m1;
        let m1_pow3 = m1_pow2 * m1;
        let m1_pow4 = m1_pow3 * m1;

        let mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1_pow3;
        let mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1_pow2 * m2 - 3.0 * m1_pow4;

        let var_pow1_5 = variance.powf(1.5);
        let var_pow2 = variance * variance;

        let skew = mu3 / var_pow1_5;
        let kurt = mu4 / var_pow2 - 3.0; // excess kurtosis
        (Some(skew), Some(kurt))
    } else {
        (None, None)
    };

    SummaryStatistics {
        mean,
        variance,
        std_dev,
        skewness,
        kurtosis,
        min,
        max,
        range: max - min,
        sum: s1,
        energy: s2,
    }
}

pub(crate) fn _calculate_median_and_quantiles(time_series: ArrayView1<f64>) -> (f64, Vec<f64>) {
    let n = time_series.len();
    if n == 0 {
        return (f64::NAN, vec![f64::NAN; 4]);
    }

    let mut sorted_data = time_series.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate median from sorted data
    let median = if n % 2 == 0 {
        (sorted_data[(n / 2) - 1] + sorted_data[n / 2]) / 2.0
    } else {
        sorted_data[n / 2]
    };

    // Calculate quantiles from sorted data
    let quantiles_vec = vec![0.05, 0.25, 0.75, 0.95]
        .into_iter()
        .map(|q| {
            let pos = q * (n - 1) as f64;
            let floor_idx = pos.floor() as usize;
            let ceil_idx = pos.ceil() as usize;
            if floor_idx == ceil_idx {
                sorted_data[floor_idx]
            } else {
                let frac = pos - floor_idx as f64;
                sorted_data[floor_idx] * (1.0 - frac) + sorted_data[ceil_idx] * frac
            }
        })
        .collect();

    (median, quantiles_vec)
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

// PyO3 wrappers

#[pyfunction]
pub fn calculate_mode(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_mode(ts_view))
}

#[pyfunction]
pub fn calculate_mean(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_summary_statistics(ts_view).mean)
}

#[pyfunction]
pub fn calculate_median(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_median_and_quantiles(ts_view).0)
}

#[pyfunction]
pub fn calculate_variance(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_summary_statistics(ts_view).variance)
}

#[pyfunction]
pub fn calculate_std_dev(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_summary_statistics(ts_view).std_dev)
}

#[pyfunction]
pub fn calculate_skewness(time_series: PyReadonlyArray1<f64>) -> PyResult<Option<f64>> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_summary_statistics(ts_view).skewness)
}

#[pyfunction]
pub fn calculate_kurtosis(time_series: PyReadonlyArray1<f64>) -> PyResult<Option<f64>> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_summary_statistics(ts_view).kurtosis)
}

#[pyfunction]
pub fn calculate_min_max_range(time_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    let stats = _calculate_summary_statistics(ts_view);
    Ok((stats.min, stats.max, stats.range))
}

#[pyfunction]
pub fn calculate_quantiles(py: Python, time_series: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    let quantiles_vec = _calculate_median_and_quantiles(ts_view).1;
    Ok(PyArray1::from_vec(py, quantiles_vec).to_owned())
}

#[pyfunction]
pub fn calculate_sum(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_summary_statistics(ts_view).sum)
}

#[pyfunction]
pub fn calculate_absolute_energy(time_series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let ts_view = time_series.as_array();
    if ts_view.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    Ok(_calculate_summary_statistics(ts_view).energy)
}
