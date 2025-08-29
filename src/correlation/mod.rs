// src/correlation/mod.rs

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use pyo3::types::PyDict;

pub mod dcf;
pub mod acf;
pub mod zdcf;

/// Compute Discrete Correlation Function (DCF) between two time series.
/// 
/// The DCF measures the correlation between two time series as a function of lag.
/// It's particularly useful for unevenly sampled data and can handle measurement
/// uncertainties.
///
/// # Arguments
/// * `t1` - Time points for first time series
/// * `v1` - Values for first time series
/// * `e1` - Errors for first time series
/// * `t2` - Time points for second time series
/// * `v2` - Values for second time series
/// * `e2` - Errors for second time series
/// * `lag_min` - Minimum lag to compute
/// * `lag_max` - Maximum lag to compute
/// * `lag_bin_width` - Width of lag bins
///
/// # Returns
/// A dictionary containing:
/// - `lags`: Array of lag values
/// - `correlations`: Array of correlation coefficients
/// - `errors`: Array of correlation errors
///
/// # Errors
/// Returns `PyValueError` if time series have fewer than 2 points
#[pyfunction]
pub fn dcf_py<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray1<f64>,
    v1: PyReadonlyArray1<f64>,
    e1: PyReadonlyArray1<f64>,
    t2: PyReadonlyArray1<f64>,
    v2: PyReadonlyArray1<f64>,
    e2: PyReadonlyArray1<f64>,
    lag_min: f64,
    lag_max: f64,
    lag_bin_width: f64,
) -> PyResult<Py<PyDict>> {
    let t1 = t1.as_slice()?;
    let v1 = v1.as_slice()?;
    let e1 = e1.as_slice()?;
    let t2 = t2.as_slice()?;
    let v2 = v2.as_slice()?;
    let e2 = e2.as_slice()?;

    if t1.len() < 2 || t2.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Time series must have at least 2 points",
        ));
    }

    let series1: Vec<dcf::TimeSeriesPoint> = t1
        .iter()
        .zip(v1.iter())
        .zip(e1.iter())
        .map(|((t, v), e)| dcf::TimeSeriesPoint {
            time: *t,
            value: *v,
            error: *e,
        })
        .collect();

    let series2: Vec<dcf::TimeSeriesPoint> = t2
        .iter()
        .zip(v2.iter())
        .zip(e2.iter())
        .map(|((t, v), e)| dcf::TimeSeriesPoint {
            time: *t,
            value: *v,
            error: *e,
        })
        .collect();

    let correlation_points = dcf::dcf(&series1, &series2, lag_min, lag_max, lag_bin_width);

    let result = PyDict::new(py);
    let lags: Vec<f64> = correlation_points.iter().map(|p| p.lag).collect();
    let correlations: Vec<f64> = correlation_points.iter().map(|p| p.correlation).collect();
    let errors: Vec<f64> = correlation_points.iter().map(|p| p.error).collect();

    result.set_item("lags", PyArray1::from_vec(py, lags))?;
    result.set_item("correlations", PyArray1::from_vec(py, correlations))?;
    result.set_item("errors", PyArray1::from_vec(py, errors))?;

    Ok(result.into())
}

/// Compute Auto-Correlation Function (ACF) of a time series.
/// 
/// The ACF measures how a time series correlates with a lagged version of itself.
/// It's useful for detecting repeating patterns and periodicities in the data.
///
/// # Arguments
/// * `t` - Time points as a numpy array
/// * `v` - Values as a numpy array
/// * `e` - Errors as a numpy array
/// * `lag_min` - Minimum lag to compute
/// * `lag_max` - Maximum lag to compute
/// * `lag_bin_width` - Width of lag bins
///
/// # Returns
/// A dictionary containing:
/// - `lags`: Array of lag values
/// - `correlations`: Array of autocorrelation coefficients
/// - `errors`: Array of correlation errors
///
/// # Errors
/// Returns `PyValueError` if time series has fewer than 2 points
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// t = np.linspace(0, 10, 100)
/// v = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(100)
/// e = np.full_like(v, 0.1)
/// acf_result = ct.acf_py(t, v, e, 0, 5, 0.1)
/// print(f"Peak correlation at lag: {acf_result['lags'][np.argmax(acf_result['correlations'])]}")
/// ```
#[pyfunction]
pub fn acf_py<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    e: PyReadonlyArray1<f64>,
    lag_min: f64,
    lag_max: f64,
    lag_bin_width: f64,
) -> PyResult<Py<PyDict>> {
    let t = t.as_slice()?;
    let v = v.as_slice()?;
    let e = e.as_slice()?;

    if t.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Time series must have at least 2 points",
        ));
    }

    let series: Vec<dcf::TimeSeriesPoint> = t
        .iter()
        .zip(v.iter())
        .zip(e.iter())
        .map(|((t, v), e)| dcf::TimeSeriesPoint {
            time: *t,
            value: *v,
            error: *e,
        })
        .collect();

    let correlation_points = acf::acf(&series, lag_min, lag_max, lag_bin_width);

    let result = PyDict::new(py);
    let lags: Vec<f64> = correlation_points.iter().map(|p| p.lag).collect();
    let correlations: Vec<f64> = correlation_points.iter().map(|p| p.correlation).collect();
    let errors: Vec<f64> = correlation_points.iter().map(|p| p.error).collect();

    result.set_item("lags", PyArray1::from_vec(py, lags))?;
    result.set_item("correlations", PyArray1::from_vec(py, correlations))?;
    result.set_item("errors", PyArray1::from_vec(py, errors))?;

    Ok(result.into())
}

/// Compute z-transformed Discrete Correlation Function (ZDCF).
/// 
/// The ZDCF is an improved version of the DCF that uses z-transformation
/// and Monte Carlo methods to better estimate correlation uncertainties,
/// especially for sparse or irregular time series.
///
/// # Arguments
/// * `t1` - Time points for first time series
/// * `v1` - Values for first time series
/// * `e1` - Errors for first time series
/// * `t2` - Time points for second time series
/// * `v2` - Values for second time series
/// * `e2` - Errors for second time series
/// * `min_points` - Minimum number of points required in each lag bin
/// * `num_mc` - Number of Monte Carlo iterations for error estimation
///
/// # Returns
/// A dictionary containing:
/// - `lags`: Array of lag values
/// - `correlations`: Array of correlation coefficients
/// - `errors`: Array of correlation errors
///
/// # Errors
/// Returns `PyValueError` if time series have fewer than 2 points
#[pyfunction]
pub fn zdcf_py<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray1<f64>,
    v1: PyReadonlyArray1<f64>,
    e1: PyReadonlyArray1<f64>,
    t2: PyReadonlyArray1<f64>,
    v2: PyReadonlyArray1<f64>,
    e2: PyReadonlyArray1<f64>,
    min_points: usize,
    num_mc: usize,
) -> PyResult<Py<PyDict>> {
    let t1 = t1.as_slice()?;
    let v1 = v1.as_slice()?;
    let e1 = e1.as_slice()?;
    let t2 = t2.as_slice()?;
    let v2 = v2.as_slice()?;
    let e2 = e2.as_slice()?;

    if t1.len() < 2 || t2.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Time series must have at least 2 points",
        ));
    }

    let series1: Vec<dcf::TimeSeriesPoint> = t1
        .iter()
        .zip(v1.iter())
        .zip(e1.iter())
        .map(|((t, v), e)| dcf::TimeSeriesPoint {
            time: *t,
            value: *v,
            error: *e,
        })
        .collect();

    let series2: Vec<dcf::TimeSeriesPoint> = t2
        .iter()
        .zip(v2.iter())
        .zip(e2.iter())
        .map(|((t, v), e)| dcf::TimeSeriesPoint {
            time: *t,
            value: *v,
            error: *e,
        })
        .collect();

    let correlation_points = zdcf::zdcf(&series1, &series2, min_points, num_mc);

    let result = PyDict::new(py);
    let lags: Vec<f64> = correlation_points.iter().map(|p| p.lag).collect();
    let correlations: Vec<f64> = correlation_points.iter().map(|p| p.correlation).collect();
    let errors: Vec<f64> = correlation_points.iter().map(|p| p.error).collect();

    result.set_item("lags", PyArray1::from_vec(py, lags))?;
    result.set_item("correlations", PyArray1::from_vec(py, correlations))?;
    result.set_item("errors", PyArray1::from_vec(py, errors))?;

    Ok(result.into())
}
