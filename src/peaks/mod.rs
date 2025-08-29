mod peak_func;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};

/// Find peaks in a time series signal.
/// 
/// Identifies local maxima in the data that satisfy optional height and distance criteria.
/// A peak is a point that is higher than its immediate neighbors.
///
/// # Arguments
/// * `data` - Input time series data as a numpy array
/// * `height` - Optional minimum height threshold for peaks
/// * `distance` - Optional minimum distance between peaks (in samples)
///
/// # Returns
/// A numpy array of indices where peaks are located
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// # Create a signal with peaks
/// data = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0])
/// peaks = ct.find_peaks(data, height=1.5)
/// print(f"Peaks at indices: {peaks}")
/// ```
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

/// Calculate the prominence of peaks in a signal.
/// 
/// Prominence measures how much a peak stands out from the surrounding baseline.
/// It is calculated as the minimum vertical distance the signal must descend
/// from the peak before either reaching a higher peak or the edge of the data.
///
/// # Arguments
/// * `data` - Input time series data as a numpy array
/// * `peaks` - Array of peak indices (typically from find_peaks)
///
/// # Returns
/// A numpy array of prominence values corresponding to each peak
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// data = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0])
/// peaks = ct.find_peaks(data)
/// prominences = ct.peak_prominence(data, peaks)
/// print(f"Peak prominences: {prominences}")
/// ```
#[pyfunction]
#[pyo3(signature = (data, peaks))]
pub fn peak_prominence(py: Python, data: PyReadonlyArray1<f64>, peaks: PyReadonlyArray1<usize>) -> PyResult<Py<PyArray1<f64>>> {
    let data_array = data.as_array();
    let peaks_array = peaks.as_array();
    let prominences = peak_func::peak_prominence(data_array.as_slice().unwrap(), peaks_array.as_slice().unwrap());
    Ok(PyArray1::from_vec(py, prominences).to_owned())
}