use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use rustfft::num_complex::Complex;
use numpy::Complex64;

mod fft;
mod lombscargle;

/// Perform Fast Fourier Transform (FFT) on a real-valued time series.
/// 
/// Computes the discrete Fourier transform of the input signal, transforming
/// it from the time domain to the frequency domain.
///
/// # Arguments
/// * `input` - Input time series data as a numpy array
///
/// # Returns
/// A numpy array of complex numbers representing the frequency domain signal
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// # Create a simple sinusoidal signal
/// t = np.linspace(0, 1, 100)
/// signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
/// fft_result = ct.perform_fft_py(signal)
/// print(f"FFT shape: {fft_result.shape}")
/// ```
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

/// Compute Lomb-Scargle periodogram for unevenly sampled data.
/// 
/// The Lomb-Scargle periodogram is used to find periodicities in unevenly sampled
/// time series data. It estimates the power spectral density at specified frequencies.
///
/// # Arguments
/// * `t` - Time points as a numpy array
/// * `y` - Observed values as a numpy array
/// * `freqs` - Frequencies at which to evaluate the periodogram
///
/// # Returns
/// A numpy array of power values at the specified frequencies
///
/// # Example
/// ```python
/// import chronoxtract as ct
/// import numpy as np
/// 
/// # Create irregularly sampled data
/// t = np.sort(np.random.uniform(0, 10, 50))
/// y = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(50)
/// freqs = np.linspace(0.1, 2, 100)
/// power = ct.lomb_scargle_py(t, y, freqs)
/// print(f"Peak frequency index: {np.argmax(power)}")
/// ```
#[pyfunction]
pub fn lomb_scargle_py(py: Python, t: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>, freqs: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let t_array = t.as_array();
    let y_array = y.as_array();
    let freqs_array = freqs.as_array();

    // Call the Lomb-Scargle function from lombscargle.rs.
    let power = lombscargle::lomb_scargle(t_array.to_slice().unwrap(), y_array.to_slice().unwrap(), freqs_array.to_slice().unwrap());

    Ok(PyArray1::from_vec(py, power).to_owned())
}
