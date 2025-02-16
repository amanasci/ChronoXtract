use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Performs FFT on the provided input signal using rustfft library.
///
/// # Arguments
///
/// * `input` - A slice of complex numbers representing the time domain signal.
///
/// # Returns
///
/// Returns a vector of complex numbers representing the frequency domain.
pub fn perform_fft(input: &[Complex<f32>]) -> Vec<Complex<f32>> {
    // Create a plan for FFT computation
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(input.len());

    // Copy input data to mutable buffer for in-place FFT computation
    let mut buffer = input.to_vec();

    // Execute the FFT
    fft.process(&mut buffer);

    buffer
}