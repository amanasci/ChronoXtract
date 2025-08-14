mod frac;
mod var;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (flux, flux_err))]
pub fn fractional_variability(flux: PyReadonlyArray1<f64>, flux_err: PyReadonlyArray1<f64>) -> PyResult<Option<f64>> {
    Ok(frac::fractional_variability(flux.as_slice()?, flux_err.as_slice()?))
}

#[pyfunction]
#[pyo3(signature = (flux, flux_err))]
pub fn fractional_variability_error(flux: PyReadonlyArray1<f64>, flux_err: PyReadonlyArray1<f64>) -> PyResult<Option<f64>> {
    Ok(frac::fractional_variability_error(flux.as_slice()?, flux_err.as_slice()?))
}

#[pyfunction]
#[pyo3(signature = (flux, flux_err, window_size))]
pub fn rolling_fractional_variability<'py>(
    py: Python<'py>,
    flux: PyReadonlyArray1<f64>,
    flux_err: PyReadonlyArray1<f64>,
    window_size: usize,
) -> PyResult<PyObject> {
    match frac::rolling_fractional_variability(flux.as_slice()?, flux_err.as_slice()?, window_size) {
        Some((variability, error)) => {
            let py_variability = PyArray1::from_vec(py, variability).to_owned();
            let py_error = PyArray1::from_vec(py, error).to_owned();
            let dict = PyDict::new(py);
            dict.set_item("fvar", py_variability)?;
            dict.set_item("fvar_err", py_error)?;
            Ok(dict.into())
        },
        None => Ok(py.None())
    }
}

#[pyfunction]
#[pyo3(signature = (time,flux,flux_err))]
pub fn calc_variability_timescale(
    time: PyReadonlyArray1<f64>,
    flux: PyReadonlyArray1<f64>,
    flux_err: PyReadonlyArray1<f64>,
) -> PyResult<Option<f64>> {
    Ok(var::calc_variability_timescale(time.as_slice()?, flux.as_slice()?, flux_err.as_slice()?))
}