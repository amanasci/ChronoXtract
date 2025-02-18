mod frac;

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