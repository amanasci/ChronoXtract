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
    for j in 0..p {
        companion[(p-1, j)] = -ar_coeffs[p-1-j];
    }
    
    // Compute eigenvalues using nalgebra
    let eigenvalues = companion.complex_eigenvalues();
    
    // Convert to Complex64
    let roots: Vec<Complex64> = eigenvalues.iter()
        .map(|&eig| Complex64::new(eig.re, eig.im))
        .collect();
    
    Ok(roots)
}

/// Compute the input vector for the rotated state-space representation
/// 
/// The input vector J satisfies the equation: EigenMat * J = R
/// where EigenMat is the Vandermonde matrix of AR roots and R = [0, 0, ..., 0, 1]
/// 
/// # Arguments
/// * `lambda` - AR roots (eigenvalues)
/// 
/// # Returns
/// Input vector that maps white noise to state variables in rotated basis
pub fn compute_input_vector(lambda: &[Complex64]) -> Result<DVector<Complex64>, CarmaError> {
    let p = lambda.len();
    
    if p == 0 {
        return Err(CarmaError::InvalidParameters("Empty lambda vector".to_string()));
    }
    
    // Construct Vandermonde matrix: V[i,j] = lambda[i]^j
    let mut vander_matrix = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            vander_matrix[(i, j)] = lambda[i].powf(j as f64);
        }
    }
    
    // Right-hand side vector: [0, 0, ..., 0, 1]
    let mut rhs = DVector::zeros(p);
    rhs[p - 1] = Complex64::new(1.0, 0.0);
    
    // Solve Vandermonde * J = rhs
    let decomp = vander_matrix.lu();
    let input_vector = decomp.solve(&rhs).ok_or_else(|| {
        CarmaError::LinearAlgebraError("Failed to solve for input vector".to_string())
    })?;
    
    Ok(input_vector)
}

/// Compute the observation vector for state-space representation
/// 
/// In the rotated state-space representation, the observation vector
/// C = β * EigenMat where β is the MA coefficient vector and EigenMat
/// is the Vandermonde matrix of AR roots.
/// 
/// # Arguments
/// * `ma_coeffs` - MA coefficients [β₀, β₁, ..., βₑ]
/// * `lambda` - AR roots (eigenvalues)
/// 
/// # Returns
/// Observation vector that transforms state to observable output in rotated basis
pub fn compute_observation_vector(ma_coeffs: &[f64], lambda: &[Complex64]) -> Result<DVector<Complex64>, CarmaError> {
    let p = lambda.len();
    let q = ma_coeffs.len().saturating_sub(1);
    
    if ma_coeffs.is_empty() {
        return Err(CarmaError::InvalidParameters("Empty MA coefficients".to_string()));
    }
    
    if p == 0 {
        return Err(CarmaError::InvalidParameters("Empty lambda vector".to_string()));
    }
    
    // Construct Vandermonde matrix: V[i,j] = lambda[i]^j
    let mut vander_matrix = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            vander_matrix[(i, j)] = lambda[i].powf(j as f64);
        }
    }
    
    // MA coefficient vector, extended with zeros if necessary
    let mut beta_vec = DVector::zeros(p);
    for i in 0..=(q.min(p - 1)) {
        beta_vec[i] = Complex64::new(ma_coeffs[i], 0.0);
    }
    
    // Observation vector: C = β * EigenMat
    let observation = beta_vec.transpose() * vander_matrix;
    
    Ok(observation.transpose())
}

/// Compute matrix exponential for state transition
/// 
/// For a diagonal matrix with eigenvalues λ, the matrix exponential
/// is diag(exp(λᵢ * dt)). This properly handles complex eigenvalues
/// by computing the full complex exponential.
/// 
/// # Arguments
/// * `lambda` - Diagonal matrix eigenvalues
/// * `dt` - Time step
/// 
/// # Returns
/// Matrix exponential exp(Λ * dt) where Λ = diag(λ)
pub fn matrix_exponential_diagonal(lambda: &[Complex64], dt: f64) -> Result<DMatrix<Complex64>, CarmaError> {
    let p = lambda.len();
    let mut exp_matrix = DMatrix::zeros(p, p);
    
    for i in 0..p {
        let exp_val = (lambda[i] * dt).exp();
        exp_matrix[(i, i)] = exp_val;
    }
    
    Ok(exp_matrix)
}

/// Compute process noise covariance matrix integrated over time step dt
/// 
/// For the rotated state-space representation, the integrated process
/// noise covariance has elements:
/// Q[i,j] = -σ² * J_i * J_j* * (exp((λᵢ + λⱼ*) * dt) - 1) / (λᵢ + λⱼ*)
/// 
/// # Arguments
/// * `lambda` - AR roots (diagonal elements of transition matrix)
/// * `input_vector` - Input vector J for white noise driving
/// * `sigma` - Process noise standard deviation
/// * `dt` - Time step
/// 
/// # Returns
/// Process noise covariance matrix integrated over time step dt
pub fn compute_process_noise_covariance_dt(
    lambda: &[Complex64], 
    input_vector: &DVector<Complex64>,
    sigma: f64, 
    dt: f64
) -> Result<DMatrix<Complex64>, CarmaError> {
    let p = lambda.len();
    
    if sigma <= 0.0 {
        return Err(CarmaError::InvalidParameters("sigma must be positive".to_string()));
    }
    
    if input_vector.len() != p {
        return Err(CarmaError::InvalidParameters("Input vector size mismatch".to_string()));
    }
    
    let mut cov = DMatrix::zeros(p, p);
    let sigma_squared = sigma * sigma;
    
    for i in 0..p {
        for j in 0..p {
            let lambda_i = lambda[i];
            let lambda_j = lambda[j];
            let j_i = input_vector[i];
            let j_j_conj = input_vector[j].conj();
            
            // Q[i,j] = -σ² * J_i * J_j* * (exp((λᵢ + λⱼ*) * dt) - 1) / (λᵢ + λⱼ*)
            let sum_lambda = lambda_i + lambda_j.conj();
            
            if sum_lambda.norm() < 1e-12 {
                // For nearly zero eigenvalue sum, use limiting case: Q[i,j] = σ² * J_i * J_j* * dt
                cov[(i, j)] = sigma_squared * j_i * j_j_conj * dt;
            } else {
                let exp_term = (sum_lambda * dt).exp() - Complex64::new(1.0, 0.0);
                cov[(i, j)] = -sigma_squared * j_i * j_j_conj * exp_term / sum_lambda;
            }
        }
    }
    
    Ok(cov)
}

/// Compute stationary covariance matrix
/// 
/// Solves the continuous-time Lyapunov equation to find the
/// steady-state covariance of the state vector using the correct formula:
/// P_inf[i,j] = -σ² * J_i * J_j_conj / (λ_i + λ_j_conj)
/// 
/// # Arguments
/// * `lambda` - AR roots (diagonal transition matrix eigenvalues)
/// * `input_vector` - Input vector J for white noise driving
/// * `sigma` - Process noise standard deviation
/// 
/// # Returns
/// Stationary state covariance matrix in rotated coordinates
pub fn compute_stationary_covariance(
    lambda: &[Complex64], 
    input_vector: &DVector<Complex64>,
    sigma: f64
) -> Result<DMatrix<Complex64>, CarmaError> {
    let p = lambda.len();
    
    if sigma <= 0.0 {
        return Err(CarmaError::InvalidParameters("sigma must be positive".to_string()));
    }
    
    if input_vector.len() != p {
        return Err(CarmaError::InvalidParameters("Input vector size mismatch".to_string()));
    }
    
    let mut stationary_cov = DMatrix::zeros(p, p);
    let sigma_squared = sigma * sigma;
    
    for i in 0..p {
        for j in 0..p {
            let lambda_i = lambda[i];
            let lambda_j = lambda[j];
            let j_i = input_vector[i];
            let j_j_conj = input_vector[j].conj();
            
            // Lyapunov equation solution: P[i,j] = -σ² * J_i * J_j* / (λᵢ + λⱼ*)
            let denominator = lambda_i + lambda_j.conj();
            
            if denominator.norm() < 1e-12 {
                return Err(CarmaError::NumericalError(
                    "Singular denominator in Lyapunov equation".to_string()
                ));
            }
            
            stationary_cov[(i, j)] = -sigma_squared * j_i * j_j_conj / denominator;
        }
    }
    
    Ok(stationary_cov)
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
        
        assert_relative_eq!(exp_matrix[(0, 0)].re, (-1.0_f64).exp(), epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 1)].re, (-2.0_f64).exp(), epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(0, 1)].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 0)].norm(), 0.0, epsilon = 1e-10);
        
        // Check that imaginary parts are zero for real eigenvalues
        assert_relative_eq!(exp_matrix[(0, 0)].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 1)].im, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_matrix_exponential_complex_eigenvalues() {
        // Test with complex conjugate pair (oscillatory behavior)
        let lambda = vec![
            Complex64::new(-0.5, 2.0),  // -0.5 + 2i
            Complex64::new(-0.5, -2.0), // -0.5 - 2i (conjugate)
        ];
        let dt = 0.1;
        
        let exp_matrix = matrix_exponential_diagonal(&lambda, dt).unwrap();
        
        // For λ = -0.5 + 2i, exp(λ*dt) = exp((-0.5 + 2i)*0.1) = exp(-0.05 + 0.2i)
        let expected_1 = (lambda[0] * dt).exp();
        let expected_2 = (lambda[1] * dt).exp();
        
        assert_relative_eq!(exp_matrix[(0, 0)].re, expected_1.re, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(0, 0)].im, expected_1.im, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 1)].re, expected_2.re, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 1)].im, expected_2.im, epsilon = 1e-10);
        
        // Off-diagonal elements should be zero
        assert_relative_eq!(exp_matrix[(0, 1)].norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(1, 0)].norm(), 0.0, epsilon = 1e-10);
        
        // Verify that the conjugate pair property is preserved
        assert_relative_eq!(exp_matrix[(0, 0)].re, exp_matrix[(1, 1)].re, epsilon = 1e-10);
        assert_relative_eq!(exp_matrix[(0, 0)].im, -exp_matrix[(1, 1)].im, epsilon = 1e-10);
    }
    
    #[test]
    fn test_state_space_model_complex_roots() {
        use crate::carma::types::{CarmaParams, StateSpaceModel};
        
        // Test CARMA(2,1) model that produces complex roots
        let mut params = CarmaParams::new(2, 1).unwrap();
        
        // AR coefficients that produce complex conjugate roots
        // For polynomial s^2 + 1.0*s + 2.0 = 0
        // Roots are (-1 ± i√7)/2 ≈ -0.5 ± 1.32i
        params.ar_coeffs = vec![1.0, 2.0];
        params.ma_coeffs = vec![1.0, 0.5];
        params.sigma = 1.0;
        
        // This should not panic or return an error with complex roots
        let state_space = StateSpaceModel::new(&params).unwrap();
        
        // Verify we have complex roots
        assert_eq!(state_space.lambda.len(), 2);
        
        // Check that the roots have negative real parts (stationarity)
        for &root in &state_space.lambda {
            assert!(root.re < 0.0, "Root has non-negative real part: {}", root);
        }
        
        // Verify the state space components have correct dimensions
        assert_eq!(state_space.observation.len(), 2);
        assert_eq!(state_space.input_vector.len(), 2);
        assert_eq!(state_space.stationary_cov.nrows(), 2);
        assert_eq!(state_space.stationary_cov.ncols(), 2);
        
        println!("Complex AR roots: {:?}", state_space.lambda);
        println!("Input vector: {:?}", state_space.input_vector);
        println!("Observation vector: {:?}", state_space.observation);
    }
}