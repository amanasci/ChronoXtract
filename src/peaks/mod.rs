mod peak_func;

use pyo3::prelude::*;

// #[pyfunction]
// #[pyo3(signature = (data, height=None, distance=None))]
// pub fn find_peaks(
//     data: Vec<f64>, 
//     height: Option<f64>, 
//     distance: Option<usize>
// ) -> PyResult<Vec<usize>> {
//     // Assuming detect_peaks takes a slice of f64 and returns a Vec<usize>
//     Ok(peak_func::find_peaks(data, height, distance))
// }

// #[pyfunction]
// pub fn peak_prominence(data: Vec<f64>, peaks: Vec<usize>) -> PyResult<Vec<f64>> {
//     // Assuming peak_prominence takes a slice of f64 and a slice of usize and returns a Vec<f64>
//     Ok(peak_func::peak_prominence(data, peaks))
// }