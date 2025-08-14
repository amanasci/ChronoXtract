mod peak_func;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};

#[pyfunction]
#[pyo3(signature = (data, height=None, distance=None))]
pub fn find_peaks(
    py: Python,
    data: PyReadonlyArray1<f64>,
    height: Option<f64>, 
    distance: Option<usize>
) -> PyResult<Py<PyArray1<usize>>> {
    let data_array = data.as_array();
    let peaks = peak_func::find_peaks(data_array.as_slice().unwrap(), height, distance);
    Ok(PyArray1::from_vec(py, peaks).to_owned())
}

#[pyfunction]
#[pyo3(signature = (data, peaks))]
pub fn peak_prominence(py: Python, data: PyReadonlyArray1<f64>, peaks: PyReadonlyArray1<usize>) -> PyResult<Py<PyArray1<f64>>> {
    let data_array = data.as_array();
    let peaks_array = peaks.as_array();
    let prominences = peak_func::peak_prominence(data_array.as_slice().unwrap(), peaks_array.as_slice().unwrap());
    Ok(PyArray1::from_vec(py, prominences).to_owned())
}