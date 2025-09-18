//! Mathematical utilities and algorithms for CARMA models
//!
//! This module contains the core mathematical functions needed for CARMA model
//! operations, including polynomial root finding, matrix operations, and
//! specialized numerical algorithms.

use crate::carma::types::{CarmaError, CarmaParams};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Compute the roots of a polynomial using eigenvalue decomposition
/// 
/// For an AR polynomial α(s) = s^p + α₁s^(p-1) + ... + αₚ,
/// constructs the companion matrix and finds its eigenvalues.
/// 
/// # Arguments
/// * `ar_coeffs` - AR coefficients [α₁, α₂, ..., αₚ]
/// 
/// # Returns
/// Complex roots of the polynomial, which should have negative real parts for stationarity
pub fn compute_ar_roots(ar_coeffs: &[f64]) -> Result<Vec<Complex64>, CarmaError> {
    let p = ar_coeffs.len();
    if p == 0 {
        return Err(CarmaError::InvalidParameters("Empty AR coefficients".to_string()));
    }
    
    if p == 1 {
        // Simple case: single root
        let root = Complex64::new(-ar_coeffs[0], 0.0);
        return Ok(vec![root]);
    }
    
    // Construct companion matrix for the AR polynomial
    // The polynomial is s^p + a₁s^(p-1) + ... + aₚ = 0
    // Companion matrix has the form:
    // [0  1  0  ... 0 ]
    // [0  0  1  ... 0 ]
    // [⋮  ⋮  ⋮  ⋱  ⋮ ]
    // [0  0  0  ... 1 ]
    // [-aₚ -aₚ₋₁ ... -a₁]
    
    let mut companion = DMatrix::zeros(p, p);
    
    // Fill the super-diagonal with 1s
    for i in 0..p-1 {
        companion[(i, i+1)] = 1.0;
    }
    
    // Fill the last row with negative AR coefficients (reversed order)
    // For polynomial s^p + α₁s^(p-1) + ... + αₚ = 0
    // Companion matrix last row should be [-αₚ, -αₚ₋₁, ..., -α₁]
    for j in 0..p {
        companion[(p-1, j)] = -ar_coeffs[p-1-j];
    }
    
    // Compute eigenvalues using nalgebra
    let eigenvalues = companion.complex_eigenvalues();
    
    // Convert to Complex64
    let mut roots: Vec<Complex64> = eigenvalues.iter()
        .map(|&eig| Complex64::new(eig.re, eig.im))
        .collect();
    
    // Sort roots to ensure complex conjugates are adjacent
    // This is important for the matrix exponential computation
    roots.sort_by(|a, b| {
        // First sort by real part (ascending)
        match a.re.partial_cmp(&b.re) {
            Some(std::cmp::Ordering::Equal) => {
                // If real parts are equal, sort by imaginary part
                // Positive imaginary part first
                b.im.partial_cmp(&a.im).unwrap_or(std::cmp::Ordering::Equal)
            }
            Some(ordering) => ordering,
            None => std::cmp::Ordering::Equal,
        }
    });
    
    Ok(roots)
}

/// Compute the observation vector for state-space representation
/// 
/// In the rotated state-space representation, the observation vector
/// relates the MA polynomial coefficients to the state variables.
/// 
/// # Arguments
/// * `ma_coeffs` - MA coefficients [β₀, β₁, ..., βₑ]
/// * `lambda` - AR roots (eigenvalues)
/// 
/// # Returns
/// Observation vector that transforms state to observable output
pub fn compute_observation_vector(ma_coeffs: &[f64], lambda: &[Complex64]) -> Result<DVector<f64>, CarmaError> {
    let p = lambda.len();
    let q = ma_coeffs.len().saturating_sub(1);
    
    if ma_coeffs.is_empty() {
        return Err(CarmaError::InvalidParameters("Empty MA coefficients".to_string()));
    }
    
    // For the rotated state-space representation, we need to solve for
    // the observation vector C such that the MA polynomial is satisfied
    // This involves evaluating the MA polynomial at the AR roots
    
    let mut observation = DVector::zeros(p);
    
    for i in 0..p {
        let mut ma_value = Complex64::new(0.0, 0.0);
        let root = lambda[i];
        
        // Evaluate MA polynomial: β₀ + β₁s + β₂s² + ... + βₑsᵃ
        for (k, &coeff) in ma_coeffs.iter().enumerate() {
            if k <= q {
                let power = root.powf(k as f64);
                ma_value += Complex64::new(coeff, 0.0) * power;
            }
        }
        
        // For real-valued observation, take the real part
        // This assumes the implementation handles complex conjugate pairs properly
        observation[i] = ma_value.re;
    }
    
    Ok(observation)
}

/// Compute process noise covariance matrix
/// 
/// In the rotated coordinate system, this matrix describes how
/// white noise drives the state variables.
/// 
/// # Arguments
/// * `lambda` - AR roots (diagonal elements of transition matrix)
/// * `sigma` - Process noise standard deviation
/// 
/// # Returns
/// Process noise covariance matrix in rotated coordinates
pub fn compute_process_noise_covariance(lambda: &[Complex64], sigma: f64) -> Result<DMatrix<f64>, CarmaError> {
    let p = lambda.len();
    
    if sigma <= 0.0 {
        return Err(CarmaError::InvalidParameters("sigma must be positive".to_string()));
    }
    
    // In the rotated representation, the process noise covariance
    // is related to how the driving white noise affects each mode
    let mut cov = DMatrix::zeros(p, p);
    
    // Simplified implementation - assumes diagonal structure
    // In the full implementation, this would involve more complex
    // relationships between the modes
    let variance = sigma * sigma;
    
    for i in 0..p {
        // Each mode gets noise based on its eigenvalue
        // More negative eigenvalues (faster decay) get more noise
        let lambda_real = lambda[i].re.abs();
        
        // Add numerical stability check
        if lambda_real < 1e-8 {
            return Err(CarmaError::NumericalError(
                format!("Eigenvalue {} has nearly zero real part: {:.2e}", i, lambda_real)
            ));
        }
        
        cov[(i, i)] = variance / (2.0 * lambda_real);
        
        // Sanity check on the resulting covariance
        if !cov[(i, i)].is_finite() || cov[(i, i)] <= 0.0 {
            return Err(CarmaError::NumericalError(
                format!("Invalid process noise covariance: {:.2e}", cov[(i, i)])
            ));
        }
    }
    
    Ok(cov)
}

/// Compute stationary covariance matrix
/// 
/// Solves the continuous-time Lyapunov equation to find the
/// steady-state covariance of the state vector.
/// 
/// # Arguments
/// * `lambda` - AR roots (diagonal transition matrix eigenvalues)
/// * `process_noise_cov` - Process noise covariance matrix
/// 
/// # Returns
/// Stationary state covariance matrix
pub fn compute_stationary_covariance(
    lambda: &[Complex64], 
    process_noise_cov: &DMatrix<f64>
) -> Result<DMatrix<f64>, CarmaError> {
    let p = lambda.len();
    
    // Verify all eigenvalues have negative real parts (stability condition)
    for (i, &eigenval) in lambda.iter().enumerate() {
        if eigenval.re >= -1e-8 {  // Allow small numerical errors
            return Err(CarmaError::NumericalError(
                format!("Eigenvalue {} has non-negative real part: {:.2e} + {:.2e}i", 
                       i, eigenval.re, eigenval.im)
            ));
        }
    }
    
    // For a diagonal system with eigenvalues λᵢ, the Lyapunov equation
    // A*X + X*A' + Q = 0 has a simple solution when A is diagonal
    let mut stationary_cov = DMatrix::zeros(p, p);
    
    for i in 0..p {
        for j in 0..p {
            let lambda_i = lambda[i];
            let lambda_j = lambda[j];
            
            // Lyapunov equation solution: X[i,j] = -Q[i,j] / (λᵢ + λⱼ*)
            let denominator = lambda_i + lambda_j.conj();
            
            if denominator.norm() < 1e-12 {
                return Err(CarmaError::NumericalError(
                    "Singular denominator in Lyapunov equation".to_string()
                ));
            }
            
            // For stable systems, λᵢ + λⱼ* should have negative real part
            // The stationary covariance should be positive definite
            let complex_result = -Complex64::new(process_noise_cov[(i, j)], 0.0) / denominator;
            stationary_cov[(i, j)] = complex_result.re;
            
            // Sanity check - diagonal elements should be positive
            if i == j && stationary_cov[(i, j)] <= 0.0 {
                return Err(CarmaError::NumericalError(
                    format!("Non-positive diagonal element in stationary covariance: [{},{}] = {:.2e}", 
                           i, j, stationary_cov[(i, j)])
                ));
            }
        }
    }
    
    Ok(stationary_cov)
}

/// Compute matrix exponential for state transition
/// 
/// For a diagonal matrix with eigenvalues λ, the matrix exponential
/// is simply diag(exp(λᵢ * dt)).
/// 
/// # Arguments
/// * `lambda` - Diagonal matrix eigenvalues
/// * `dt` - Time step
/// 
/// # Returns
/// Matrix exponential exp(Λ * dt) where Λ = diag(λ)
pub fn matrix_exponential_diagonal(lambda: &[Complex64], dt: f64) -> Result<DMatrix<f64>, CarmaError> {
    let p = lambda.len();
    let mut exp_matrix = DMatrix::zeros(p, p);
    
    let mut i = 0;
    while i < p {
        let eigenval = lambda[i];
        
        if eigenval.im.abs() < 1e-12 {
            // Real eigenvalue - simple case
            let exp_val = (eigenval.re * dt).exp();
            exp_matrix[(i, i)] = exp_val;
            i += 1;
        } else {
            // Complex eigenvalue - should have conjugate pair
            if i + 1 >= p {
                return Err(CarmaError::NumericalError(
                    "Complex eigenvalue without conjugate pair".to_string()
                ));
            }
            
            let eigenval_conj = lambda[i + 1];
            
            // Verify they are conjugates (approximately)
            if (eigenval.re - eigenval_conj.re).abs() > 1e-10 || 
               (eigenval.im + eigenval_conj.im).abs() > 1e-10 {
                return Err(CarmaError::NumericalError(
                    "Complex eigenvalues are not conjugate pairs".to_string()
                ));
            }
            
            // For conjugate pair α ± βi, the 2x2 block exponential is:
            // exp(α*dt) * [cos(β*dt)  -sin(β*dt)]
            //             [sin(β*dt)   cos(β*dt)]
            let alpha = eigenval.re;
            let beta = eigenval.im;
            let exp_alpha = (alpha * dt).exp();
            let cos_beta = (beta * dt).cos();
            let sin_beta = (beta * dt).sin();
            
            exp_matrix[(i, i)] = exp_alpha * cos_beta;
            exp_matrix[(i, i + 1)] = -exp_alpha * sin_beta;
            exp_matrix[(i + 1, i)] = exp_alpha * sin_beta;
            exp_matrix[(i + 1, i + 1)] = exp_alpha * cos_beta;
            
            i += 2; // Skip both eigenvalues in the conjugate pair
        }
    }
    
    Ok(exp_matrix)
}

/// Compute Power Spectral Density (PSD) of CARMA model
/// 
/// The PSD is given by: P(f) = σ² |β(2πif)|² / |α(2πif)|²
/// where α and β are the AR and MA polynomials respectively.
/// 
/// # Arguments
/// * `params` - CARMA model parameters
/// * `frequencies` - Frequencies at which to evaluate PSD
/// 
/// # Returns
/// PSD values at the specified frequencies
pub fn compute_psd(params: &CarmaParams, frequencies: &[f64]) -> Result<Vec<f64>, CarmaError> {
    let mut psd_values = Vec::with_capacity(frequencies.len());
    
    for &freq in frequencies {
        let omega = Complex64::new(0.0, 2.0 * PI * freq);
        
        // Evaluate AR polynomial α(iω)
        let mut ar_poly = Complex64::new(1.0, 0.0); // Start with s^p term (coefficient 1)
        for (k, &coeff) in params.ar_coeffs.iter().enumerate() {
            let power = omega.powf((params.p - 1 - k) as f64);
            ar_poly += Complex64::new(coeff, 0.0) * power;
        }
        
        // Evaluate MA polynomial β(iω)
        let mut ma_poly = Complex64::new(0.0, 0.0);
        for (k, &coeff) in params.ma_coeffs.iter().enumerate() {
            let power = omega.powf(k as f64);
            ma_poly += Complex64::new(coeff, 0.0) * power;
        }
        
        // Compute PSD: σ² |β(iω)|² / |α(iω)|²
        let ar_mag_sq = ar_poly.norm_sqr();
        let ma_mag_sq = ma_poly.norm_sqr();
        
        if ar_mag_sq < 1e-15 {
            return Err(CarmaError::NumericalError(
                "AR polynomial evaluates to zero".to_string()
            ));
        }
        
        let psd = params.sigma * params.sigma * ma_mag_sq / ar_mag_sq;
        psd_values.push(psd);
    }
    
    Ok(psd_values)
}

/// Validate time series data for CARMA modeling
/// 
/// Checks for common issues that would prevent successful CARMA fitting.
/// 
/// # Arguments
/// * `times` - Observation times
/// * `values` - Observed values
/// * `errors` - Optional measurement errors
/// 
/// # Returns
/// Unit result or error describing the validation failure
pub fn validate_time_series(
    times: &[f64], 
    values: &[f64], 
    errors: Option<&[f64]>
) -> Result<(), CarmaError> {
    if times.is_empty() || values.is_empty() {
        return Err(CarmaError::InvalidData("Empty time series".to_string()));
    }
    
    if times.len() != values.len() {
        return Err(CarmaError::InvalidData(
            format!("Time and value arrays have different lengths: {} vs {}", 
                   times.len(), values.len())
        ));
    }
    
    if let Some(errs) = errors {
        if errs.len() != times.len() {
            return Err(CarmaError::InvalidData(
                format!("Error array length {} doesn't match time series length {}", 
                       errs.len(), times.len())
            ));
        }
        
        if errs.iter().any(|&e| e <= 0.0 || !e.is_finite()) {
            return Err(CarmaError::InvalidData(
                "All measurement errors must be positive and finite".to_string()
            ));
        }
    }
    
    // Check for NaN or infinite values
    if times.iter().any(|&t| !t.is_finite()) {
        return Err(CarmaError::InvalidData("Times contain non-finite values".to_string()));
    }
    
    if values.iter().any(|&v| !v.is_finite()) {
        return Err(CarmaError::InvalidData("Values contain non-finite values".to_string()));
    }
    
    // Check that times are in ascending order
    if !times.windows(2).all(|w| w[0] < w[1]) {
        return Err(CarmaError::InvalidData("Times must be in strictly ascending order".to_string()));
    }
    
    // Check minimum number of data points
    if times.len() < 3 {
        return Err(CarmaError::InvalidData(
            "Need at least 3 data points for CARMA fitting".to_string()
        ));
    }
    
    Ok(())
}

/// Compute AIC, AICc, and BIC information criteria
/// 
/// # Arguments
/// * `loglikelihood` - Maximum log-likelihood value
/// * `n_params` - Number of model parameters
/// * `n_data` - Number of data points
/// 
/// # Returns
/// Tuple of (AIC, AICc, BIC)
pub fn compute_information_criteria(
    loglikelihood: f64, 
    n_params: usize, 
    n_data: usize
) -> (f64, f64, f64) {
    let k = n_params as f64;
    let n = n_data as f64;
    
    // AIC = 2k - 2*ln(L)
    let aic = 2.0 * k - 2.0 * loglikelihood;
    
    // AICc = AIC + 2k(k+1)/(n-k-1) for small sample correction
    let aicc = if n > k + 1.0 {
        aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    } else {
        f64::INFINITY // Model too complex for data size
    };
    
    // BIC = k*ln(n) - 2*ln(L)
    let bic = k * n.ln() - 2.0 * loglikelihood;
    
    (aic, aicc, bic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_ar_roots_simple() {
        // Test AR(1): s + a = 0 => root = -a
        let ar_coeffs = vec![2.0];
        let roots = compute_ar_roots(&ar_coeffs).unwrap();
        assert_eq!(roots.len(), 1);
        assert_relative_eq!(roots[0].re, -2.0, epsilon = 1e-10);
        assert_relative_eq!(roots[0].im, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_validate_time_series_valid() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![1.0, 2.0, 1.5, 2.5];
        let errors = vec![0.1, 0.1, 0.1, 0.1];
        
        assert!(validate_time_series(&times, &values, Some(&errors)).is_ok());
    }
    
    #[test]
    fn test_validate_time_series_empty() {
        let times: Vec<f64> = vec![];
        let values: Vec<f64> = vec![];
        
        assert!(validate_time_series(&times, &values, None).is_err());
    }
    
    #[test]
    fn test_validate_time_series_mismatched_lengths() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![1.0, 2.0];
        
        assert!(validate_time_series(&times, &values, None).is_err());
    }
    
    #[test]
    fn test_information_criteria() {
        let loglik = -100.0;
        let n_params = 3;
        let n_data = 50;
        
        let (aic, aicc, bic) = compute_information_criteria(loglik, n_params, n_data);
        
        assert_relative_eq!(aic, 206.0, epsilon = 1e-10); // 2*3 - 2*(-100)
        assert!(aicc > aic); // AICc should be larger than AIC
        assert_relative_eq!(bic, 3.0 * (50.0_f64).ln() + 200.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_matrix_exponential_diagonal() {
        let lambda = vec![Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0)];
        let dt = 1.0;
        
        let exp_matrix = matrix_exponential_diagonal(&lambda, dt).unwrap();
        
        assert_relative_eq!(exp_matrix[(0, 0)], (-1.0_f64).exp(), epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 1)], (-2.0_f64).exp(), epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 0)], 0.0, epsilon = 1e-10);
    }
}