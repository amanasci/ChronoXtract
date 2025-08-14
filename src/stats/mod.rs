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
    let mut sorted = time_series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();

    // Calculate median
    let mid = n / 2;
    let median = if n % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };

    // Calculate quantiles
    let quantiles_to_calc = vec![0.05, 0.25, 0.75, 0.95];
    let quantiles_vec = quantiles_to_calc
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


