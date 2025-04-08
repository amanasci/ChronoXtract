use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

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

pub fn fft_from_scratch(input: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let n = input.len();
    
    // Base case: if length is 1, return the input
    if n == 1 {
        return input.to_vec();
    }
    
    // Ensure input length is a power of 2
    assert!(n.is_power_of_two(), "Input length must be a power of 2");
    
    // Divide the input into even and odd indices
    let mut even = Vec::with_capacity(n/2);
    let mut odd = Vec::with_capacity(n/2);
    
    for i in 0..n/2 {
        even.push(input[2*i]);
        odd.push(input[2*i + 1]);
    }
    
    // Recursively compute FFT for even and odd parts
    let even_fft = fft_from_scratch(&even);
    let odd_fft = fft_from_scratch(&odd);
    
    // Combine the results using the butterfly operation
    let mut result = vec![Complex::new(0.0, 0.0); n];
    
    for k in 0..n/2 {
        let twiddle = Complex::new(0.0, -2.0 * PI * (k as f32) / (n as f32)).exp();
        let p = even_fft[k];
        let q = twiddle * odd_fft[k];
        
        result[k] = p + q;
        result[k + n/2] = p - q;
    }
    
    result
}
