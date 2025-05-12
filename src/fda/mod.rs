use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList};
use rustfft::num_complex::Complex;
mod fft;
mod lombscargle;

#[pyfunction]
pub fn perform_fft_py(py: Python, input: Vec<f64>) -> PyResult<Py<PyList>> {
    // Convert the input into Complex numbers.
    let input_complex: Vec<Complex<f32>> = input
        .into_iter()
        .map(|x| Complex::new(x as f32, 0.0))
        .collect();

    // Call the FFT function from fft.rs.
    // let output = fft::perform_fft(&input_complex);
    let output = fft::perform_fft(&input_complex);

    // Create an empty Python list.
    let py_list = PyList::empty(py);

    // Directly convert each FFT output into a Python complex number.
    for c in output.iter() {
        // Replace PyComplex::new with Python's built-in complex constructor.
        let complex_cstr = std::ffi::CStr::from_bytes_with_nul(b"complex\0").expect("CStr creation failed");
        let py_complex = py.eval(complex_cstr, None, None)?.call1((c.re as f64, c.im as f64))?;
        py_list.append(py_complex)?;
    }

    Ok(py_list.into())
}

#[pyfunction]
pub fn lomb_scargle_py(py: Python, t: Vec<f64>, y: Vec<f64>, freqs: Vec<f64>) -> PyResult<Py<PyList>> {
    // Call the Lomb-Scargle function from lombscargle.rs.
    let power = lombscargle::lomb_scargle(&t, &y, &freqs);

    // Create an empty Python list.
    let py_list = PyList::empty(py);

    // Directly convert each power value into a Python float.
    for p in power.iter() {
        let py_float = PyFloat::new(py, *p);
        py_list.append(py_float)?;
    }

    Ok(py_list.into())
}
