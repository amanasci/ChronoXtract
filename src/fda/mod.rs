use pyo3::prelude::*;
use pyo3::types::PyList;
use rustfft::num_complex::Complex;
mod fft;

#[pyfunction]
pub fn perform_fft_py(py: Python, input: Vec<f64>) -> PyResult<Py<PyList>> {
    // Convert the input into Complex numbers.
    let input_complex: Vec<Complex<f32>> = input
        .into_iter()
        .map(|x| Complex::new(x as f32, 0.0))
        .collect();

    // Call the FFT function from fft.rs.
    let output = fft::perform_fft(&input_complex);

    // Convert the FFT output into a vector of tuples (real, imaginary)
    let output_tuples: Vec<(f32, f32)> = output.iter()
        .map(|c| (c.re, c.im))
        .collect();

    // Convert the tuple vector into a Python list and return it.
    Ok(PyList::new(py, output_tuples)?.into())
}