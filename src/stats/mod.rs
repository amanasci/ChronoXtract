use std::collections::HashMap;
use pyo3::prelude::*;

#[pyfunction]
pub fn calculate_mean(time_series: Vec<f64>) -> PyResult<f64> {
    let mean = time_series.iter().sum::<f64>() / time_series.len() as f64;
    Ok(mean)
}

#[pyfunction]
pub fn calculate_median(time_series: Vec<f64>) -> PyResult<f64> {
    let mut sorted = time_series.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    Ok(median)
}

#[pyfunction]
pub fn calculate_mode(time_series: Vec<f64>) -> PyResult<f64> {
    let mut counts = HashMap::new();
    for &value in &time_series {
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
    Ok(f64::from_bits(mode_value))
}

#[pyfunction]
pub fn calculate_variance(time_series: Vec<f64>) -> PyResult<f64> {
    let mean = calculate_mean(time_series.clone())?;
    let variance = time_series.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / time_series.len() as f64;
    Ok(variance)
}

#[pyfunction]
pub fn calculate_std_dev(time_series: Vec<f64>) -> PyResult<f64> {
    let variance = calculate_variance(time_series)?;
    Ok(variance.sqrt())
}

#[pyfunction]
pub fn calculate_skewness(time_series: Vec<f64>) -> PyResult<f64> {
    let mean = calculate_mean(time_series.clone())?;
    let std_dev = calculate_std_dev(time_series.clone())?;
    let n = time_series.len() as f64;
    
    let skewness = time_series.iter()
        .map(|x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / n;
    Ok(skewness)
}

#[pyfunction]
pub fn calculate_kurtosis(time_series: Vec<f64>) -> PyResult<f64> {
    let mean = calculate_mean(time_series.clone())?;
    let std_dev = calculate_std_dev(time_series.clone())?;
    let n = time_series.len() as f64;
    
    let kurtosis = time_series.iter()
        .map(|x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>() / n - 3.0; // Excess Kurtosis
    Ok(kurtosis)
}

#[pyfunction]
pub fn calculate_min_max_range(time_series: Vec<f64>) -> PyResult<(f64, f64, f64)> {
    let min = *time_series.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = *time_series.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let range = max - min;
    Ok((min, max, range))
}

#[pyfunction]
pub fn calculate_quantiles(time_series: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut sorted = time_series.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    
    let quantiles = vec![0.05, 0.25, 0.75, 0.95]
        .into_iter()
        .map(|q| {
            let pos = q * (n - 1) as f64;
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
    
    Ok(quantiles)
}

#[pyfunction]
pub fn calculate_sum(time_series: Vec<f64>) -> PyResult<f64> {
    Ok(time_series.iter().sum())
}

#[pyfunction]
pub fn calculate_absolute_energy(time_series: Vec<f64>) -> PyResult<f64> {
    Ok(time_series.iter().map(|x| x.powi(2)).sum())
}
