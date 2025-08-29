mod frac;
mod var;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use pyo3::types::PyDict;

use pyo3::exceptions::PyValueError;

/// Calculate fractional variability of a flux time series.
/// 
/// Fractional variability is a measure of the intrinsic variability of an astronomical
/// source, corrected for measurement uncertainties. It quantifies the excess variance
/// above what would be expected from measurement errors alone.
///
/// # Arguments
/// * `flux` - Flux measurements as a numpy array
/// * `flux_err` - Flux measurement uncertainties as a numpy array
///
/// # Returns
/// The fractional variability value
///
/// # Errors
/// Returns `PyValueError` if arrays have different lengths or other computation errors
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// flux = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
/// flux_err = np.array([0.05, 0.06, 0.04, 0.05, 0.05])
/// fvar = ct.fractional_variability(flux, flux_err)
/// print(f"Fractional variability: {fvar}")
/// ```
#[pyfunction]
#[pyo3(signature = (flux, flux_err))]
pub fn fractional_variability(flux: PyReadonlyArray1<f64>, flux_err: PyReadonlyArray1<f64>) -> PyResult<f64> {
    match frac::fractional_variability(flux.as_slice()?, flux_err.as_slice()?) {
        Ok(val) => Ok(val),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Calculate the error on fractional variability.
/// 
/// Computes the uncertainty in the fractional variability measurement,
/// accounting for finite sample size and measurement errors.
///
/// # Arguments
/// * `flux` - Flux measurements as a numpy array
/// * `flux_err` - Flux measurement uncertainties as a numpy array
///
/// # Returns
/// The error on the fractional variability
///
/// # Errors
/// Returns `PyValueError` if arrays have different lengths or other computation errors
#[pyfunction]
#[pyo3(signature = (flux, flux_err))]
pub fn fractional_variability_error(flux: PyReadonlyArray1<f64>, flux_err: PyReadonlyArray1<f64>) -> PyResult<f64> {
    match frac::fractional_variability_error(flux.as_slice()?, flux_err.as_slice()?) {
        Ok(val) => Ok(val),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Calculate rolling fractional variability over a sliding window.
/// 
/// Computes fractional variability and its error in a sliding window,
/// providing insight into how variability changes over time.
///
/// # Arguments
/// * `flux` - Flux measurements as a numpy array
/// * `flux_err` - Flux measurement uncertainties as a numpy array
/// * `window_size` - Size of the sliding window
///
/// # Returns
/// A dictionary containing:
/// - `fvar`: Array of fractional variability values
/// - `fvar_err`: Array of fractional variability errors
///
/// # Errors
/// Returns `PyValueError` if arrays have different lengths or other computation errors
#[pyfunction]
#[pyo3(signature = (flux, flux_err, window_size))]
pub fn rolling_fractional_variability<'py>(
    py: Python<'py>,
    flux: PyReadonlyArray1<f64>,
    flux_err: PyReadonlyArray1<f64>,
    window_size: usize,
) -> PyResult<PyObject> {
    match frac::rolling_fractional_variability(flux.as_slice()?, flux_err.as_slice()?, window_size) {
        Ok((variability, error)) => {
            let py_variability = PyArray1::from_vec(py, variability).to_owned();
            let py_error = PyArray1::from_vec(py, error).to_owned();
            let dict = PyDict::new(py);
            dict.set_item("fvar", py_variability)?;
            dict.set_item("fvar_err", py_error)?;
            Ok(dict.into())
        },
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Calculate characteristic variability timescale.
/// 
/// Estimates the timescale over which significant variability occurs in the
/// time series, useful for understanding the physical processes driving
/// the observed variability.
///
/// # Arguments
/// * `time` - Time points as a numpy array
/// * `flux` - Flux measurements as a numpy array
/// * `flux_err` - Flux measurement uncertainties as a numpy array
///
/// # Returns
/// The variability timescale as Option<f64> (None if calculation fails)
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// time = np.linspace(0, 100, 200)
/// flux = np.sin(2 * np.pi * time / 10) + 0.1 * np.random.randn(200)
/// flux_err = np.full_like(flux, 0.05)
/// timescale = ct.calc_variability_timescale(time, flux, flux_err)
/// print(f"Variability timescale: {timescale}")
/// ```
#[pyfunction]
#[pyo3(signature = (time,flux,flux_err))]
pub fn calc_variability_timescale(
    time: PyReadonlyArray1<f64>,
    flux: PyReadonlyArray1<f64>,
    flux_err: PyReadonlyArray1<f64>,
) -> PyResult<Option<f64>> {
    Ok(var::calc_variability_timescale(time.as_slice()?, flux.as_slice()?, flux_err.as_slice()?))
}

/// Calculate comprehensive variability statistics.
/// 
/// Computes a suite of statistical measures for analyzing variability
/// in time series data, including basic descriptive statistics.
///
/// # Arguments
/// * `time` - Time points as a numpy array
/// * `flux` - Flux measurements as a numpy array
/// * `flux_err` - Flux measurement uncertainties as a numpy array
///
/// # Returns
/// A dictionary containing variability statistics:
/// - `min`: Minimum flux value
/// - `max`: Maximum flux value
/// - `mean`: Mean flux value
/// - `median`: Median flux value
/// - `std_dev`: Standard deviation of flux
/// - `count`: Number of data points
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// time = np.linspace(0, 10, 100)
/// flux = np.random.randn(100) + 5.0
/// flux_err = np.full_like(flux, 0.1)
/// stats = ct.variability_statistics(time, flux, flux_err)
/// print(f"Mean flux: {stats['mean']}")
/// ```
#[pyfunction]
#[pyo3(signature = (time, flux, flux_err))]
pub fn variability_statistics<'py>(
    py: Python<'py>,
    time: PyReadonlyArray1<f64>,
    flux: PyReadonlyArray1<f64>,
    flux_err: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let stats = var::variability_statistics(time.as_slice()?, flux.as_slice()?, flux_err.as_slice()?);
    let dict = PyDict::new(py);
    dict.set_item("min", stats.min)?;
    dict.set_item("max", stats.max)?;
    dict.set_item("mean", stats.mean)?;
    dict.set_item("median", stats.median)?;
    dict.set_item("std_dev", stats.std_dev)?;
    dict.set_item("count", stats.count)?;
    Ok(dict.into())
}