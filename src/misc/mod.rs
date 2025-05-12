mod frac;
mod var;

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (flux, flux_err))]
pub fn fractional_variability(flux: Vec<f64>, flux_err: Vec<f64>) -> PyResult<Option<f64>> {
    Ok(frac::fractional_variability(&flux, &flux_err))
}

#[pyfunction]
#[pyo3(signature = (flux, flux_err))]
pub fn fractional_variability_error(flux: Vec<f64>, flux_err: Vec<f64>) -> PyResult<Option<f64>> {
    Ok(frac::fractional_variability_error(&flux, &flux_err))
}

#[pyfunction]
#[pyo3(signature = (flux, flux_err,window_size))]
pub fn rolling_fractional_variability(
    flux: Vec<f64>,
    flux_err: Vec<f64>,
    window_size: usize,
) -> PyResult<Vec<Option<f64>>> {
    match frac::rolling_fractional_variability(&flux, &flux_err, window_size) {
        Some((variability, _error)) => Ok(variability.into_iter().map(Some).collect()),
        None => Ok(Vec::new())
    }
}

#[pyfunction]
#[pyo3(signature = (time,flux,flux_err))]
pub fn calc_variability_timescale(
    time: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
) -> PyResult<Option<f64>> {
    Ok(var::calc_variability_timescale(&time, &flux, &flux_err))
}