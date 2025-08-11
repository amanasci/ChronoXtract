use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use rustfft::num_complex::Complex;
use numpy::Complex64;

mod fft;
mod lombscargle;

#[pyfunction]
pub fn perform_fft_py(py: Python, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<Complex64>>> {
    let input_array = input.as_array();

    // Convert the input into Complex numbers.
    let input_complex: Vec<Complex<f32>> = input_array
        .iter()
        .map(|&x| Complex::new(x as f32, 0.0))
        .collect();

    // Call the FFT function from fft.rs.
    let output = fft::perform_fft(&input_complex);

    // Convert the result to Complex64 for numpy
    let output_complex64: Vec<Complex64> = output.iter().map(|c| Complex64::new(c.re as f64, c.im as f64)).collect();

    Ok(PyArray1::from_vec(py, output_complex64).to_owned())
}

#[pyfunction]
pub fn lomb_scargle_py(py: Python, t: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>, freqs: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let t_array = t.as_array();
    let y_array = y.as_array();
    let freqs_array = freqs.as_array();

    // Call the Lomb-Scargle function from lombscargle.rs.
    let power = lombscargle::lomb_scargle(t_array.to_slice().unwrap(), y_array.to_slice().unwrap(), freqs_array.to_slice().unwrap());

    Ok(PyArray1::from_vec(py, power).to_owned())
}
