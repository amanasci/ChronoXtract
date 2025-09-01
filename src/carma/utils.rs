use pyo3::prelude::*;
use nalgebra::{DMatrix, DVector, Complex};
use num_complex::Complex64;
use crate::carma::carma_model::{CarmaModel, StateSpaceModel, CarmaError};

/// Check if CARMA model is stable (all roots in left half-plane)
#[pyfunction]
pub fn check_carma_stability(model: &CarmaModel) -> PyResult<bool> {
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    let roots = compute_characteristic_roots(&model.ar_coeffs)?;
    Ok(roots.iter().all(|root| root.re < 0.0))
}

/// Get characteristic roots of the CARMA model
#[pyfunction]
pub fn carma_characteristic_roots(model: &CarmaModel) -> PyResult<Vec<(f64, f64)>> {
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    let roots = compute_characteristic_roots(&model.ar_coeffs)?;
    Ok(roots.iter().map(|c| (c.re, c.im)).collect())
}

/// Convert CARMA model to state-space representation
#[pyfunction]
pub fn carma_to_state_space(model: &CarmaModel) -> PyResult<StateSpaceModel> {
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    let p = model.p;
    
    // Construct companion form transition matrix
    let mut transition = vec![vec![0.0; p]; p];
    
    // Identity in upper p-1 x p-1 block (shifted down by one row)
    for i in 0..p-1 {
        transition[i][i+1] = 1.0;
    }
    
    // AR coefficients in the last row (negated for companion form)
    for j in 0..p {
        transition[p-1][j] = -model.ar_coeffs[p-1-j];
    }
    
    // Observation vector (selects first state variable)
    let mut observation = vec![0.0; p];
    observation[0] = 1.0;
    
    // Process noise matrix (only affects last state)
    let mut process_noise = vec![vec![0.0; p]; p];
    process_noise[p-1][p-1] = model.sigma * model.sigma;
    
    Ok(StateSpaceModel {
        transition_matrix: transition,
        observation_vector: observation,
        process_noise_matrix: process_noise,
        observation_noise: 0.0, // Observational noise is typically separate
    })
}

/// Compute matrix exponential for state transition
pub fn matrix_exponential(matrix: &DMatrix<f64>, dt: f64) -> Result<DMatrix<f64>, CarmaError> {
    let scaled_matrix = matrix * dt;
    
    // Use scaling and squaring for matrix exponential
    // For small matrices, we can use eigen-decomposition or series expansion
    if matrix.nrows() <= 10 {
        matrix_exp_small(&scaled_matrix)
    } else {
        Err(CarmaError::NumericalError("Matrix too large for exponential".to_string()))
    }
}

/// Matrix exponential for small matrices using series expansion
fn matrix_exp_small(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    let n = matrix.nrows();
    let mut result = DMatrix::identity(n, n);
    let mut term = DMatrix::identity(n, n);
    
    // Series expansion: exp(A) = I + A + A²/2! + A³/3! + ...
    for k in 1..=20 { // Limit to 20 terms for numerical stability
        term = &term * matrix / k as f64;
        result += &term;
        
        // Check convergence
        if term.norm() < 1e-12 {
            break;
        }
    }
    
    Ok(result)
}

/// Solve continuous-time Lyapunov equation: AX + XA' + Q = 0
pub fn solve_lyapunov(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    let n = a.nrows();
    
    // For small systems, use direct method
    if n <= 5 {
        solve_lyapunov_direct(a, q)
    } else {
        Err(CarmaError::NumericalError("Matrix too large for Lyapunov solver".to_string()))
    }
}

/// Direct solution of Lyapunov equation for small matrices
fn solve_lyapunov_direct(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    let n = a.nrows();
    
    // Vectorize the equation: (I ⊗ A + A' ⊗ I) vec(X) = -vec(Q)
    let mut kronecker = DMatrix::zeros(n * n, n * n);
    
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let row = i * n + j;
                    let col = k * n + l;
                    
                    if j == l {
                        kronecker[(row, col)] += a[(i, k)];
                    }
                    if i == k {
                        kronecker[(row, col)] += a[(j, l)];
                    }
                }
            }
        }
    }
    
    // Solve the linear system
    let q_vec = vectorize_matrix(q);
    let neg_q_vec = -q_vec;
    
    match kronecker.lu().solve(&neg_q_vec) {
        Some(x_vec) => Ok(unvectorize_matrix(&x_vec, n)),
        None => Err(CarmaError::NumericalError("Failed to solve Lyapunov equation".to_string())),
    }
}

/// Convert matrix to vector (column-major order)
fn vectorize_matrix(matrix: &DMatrix<f64>) -> DVector<f64> {
    let n = matrix.nrows();
    let m = matrix.ncols();
    let mut vec = DVector::zeros(n * m);
    
    for j in 0..m {
        for i in 0..n {
            vec[j * n + i] = matrix[(i, j)];
        }
    }
    
    vec
}

/// Convert vector back to matrix (column-major order)
fn unvectorize_matrix(vec: &DVector<f64>, n: usize) -> DMatrix<f64> {
    let m = vec.len() / n;
    let mut matrix = DMatrix::zeros(n, m);
    
    for j in 0..m {
        for i in 0..n {
            matrix[(i, j)] = vec[j * n + i];
        }
    }
    
    matrix
}

/// Compute roots of characteristic polynomial
fn compute_characteristic_roots(ar_coeffs: &[f64]) -> Result<Vec<Complex64>, CarmaError> {
    let n = ar_coeffs.len();
    
    if n == 1 {
        // Simple case: single root
        Ok(vec![Complex64::new(-ar_coeffs[0], 0.0)])
    } else if n == 2 {
        // Quadratic formula
        let a = 1.0;
        let b = ar_coeffs[1];
        let c = ar_coeffs[0];
        
        let discriminant = b * b - 4.0 * a * c;
        if discriminant >= 0.0 {
            let sqrt_d = discriminant.sqrt();
            Ok(vec![
                Complex64::new((-b + sqrt_d) / (2.0 * a), 0.0),
                Complex64::new((-b - sqrt_d) / (2.0 * a), 0.0),
            ])
        } else {
            let sqrt_d = (-discriminant).sqrt();
            Ok(vec![
                Complex64::new(-b / (2.0 * a), sqrt_d / (2.0 * a)),
                Complex64::new(-b / (2.0 * a), -sqrt_d / (2.0 * a)),
            ])
        }
    } else {
        // For higher order, we would need a root-finding algorithm
        // For now, return an error as this requires more sophisticated methods
        Err(CarmaError::NumericalError(format!("Root finding for degree {} not implemented", n)))
    }
}

/// Validate time series data
pub fn validate_time_series(times: &[f64], values: &[f64], errors: Option<&[f64]>) -> Result<(), CarmaError> {
    if times.is_empty() || values.is_empty() {
        return Err(CarmaError::InvalidData("Empty time series".to_string()));
    }
    
    if times.len() != values.len() {
        return Err(CarmaError::InvalidData("Time and value arrays have different lengths".to_string()));
    }
    
    if let Some(errs) = errors {
        if errs.len() != times.len() {
            return Err(CarmaError::InvalidData("Error array has different length".to_string()));
        }
        
        if errs.iter().any(|&e| e <= 0.0 || !e.is_finite()) {
            return Err(CarmaError::InvalidData("Errors must be positive and finite".to_string()));
        }
    }
    
    if times.iter().any(|&t| !t.is_finite()) {
        return Err(CarmaError::InvalidData("Times must be finite".to_string()));
    }
    
    if values.iter().any(|&v| !v.is_finite()) {
        return Err(CarmaError::InvalidData("Values must be finite".to_string()));
    }
    
    // Check if times are sorted
    for i in 1..times.len() {
        if times[i] <= times[i-1] {
            return Err(CarmaError::InvalidData("Times must be strictly increasing".to_string()));
        }
    }
    
    Ok(())
}

/// Compute autocorrelation function for model validation
pub fn autocorrelation_function(_ar_coeffs: &[f64], _ma_coeffs: &[f64], lags: &[f64]) -> Vec<f64> {
    // Placeholder implementation - would need proper autocorrelation calculation
    lags.iter().map(|&lag| (-lag.abs()).exp()).collect()
}

/// Compute power spectral density
pub fn power_spectral_density(ar_coeffs: &[f64], ma_coeffs: &[f64], sigma: f64, frequencies: &[f64]) -> Vec<f64> {
    frequencies.iter().map(|&freq| {
        let omega = 2.0 * std::f64::consts::PI * freq;
        let i = Complex64::new(0.0, 1.0);
        
        // Evaluate MA polynomial at i*omega
        let mut ma_val = Complex64::new(0.0, 0.0);
        for (k, &coeff) in ma_coeffs.iter().enumerate() {
            ma_val += coeff * (i * omega).powf(k as f64);
        }
        
        // Evaluate AR polynomial at i*omega  
        let mut ar_val = Complex64::new(1.0, 0.0);
        for (k, &coeff) in ar_coeffs.iter().enumerate() {
            ar_val += coeff * (i * omega).powf((k + 1) as f64);
        }
        
        // PSD = sigma^2 * |MA(iω)|² / |AR(iω)|²
        let ma_mag_sq = ma_val.norm_sqr();
        let ar_mag_sq = ar_val.norm_sqr();
        
        if ar_mag_sq > 0.0 {
            sigma * sigma * ma_mag_sq / ar_mag_sq
        } else {
            0.0
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carma::carma_model::CarmaModel;
    
    #[test]
    fn test_carma_stability() {
        let mut model = CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![1.5, -0.5]; // Stable: roots have negative real parts
        model.ma_coeffs = vec![1.0, 0.3];
        model.sigma = 1.0;
        
        assert!(check_carma_stability(&model).unwrap());
    }
    
    #[test]
    fn test_state_space_conversion() {
        let mut model = CarmaModel::new(3, 1).unwrap();
        model.ar_coeffs = vec![2.0, -1.5, 0.5];
        model.ma_coeffs = vec![1.0, 0.3];
        model.sigma = 1.0;
        
        let ss = carma_to_state_space(&model).unwrap();
        assert_eq!(ss.transition_matrix.len(), 3);
        assert_eq!(ss.observation_vector.len(), 3);
        assert_eq!(ss.process_noise_matrix.len(), 3);
    }
    
    #[test]
    fn test_time_series_validation() {
        let times = vec![1.0, 2.0, 3.0];
        let values = vec![1.1, 2.2, 3.3];
        let errors = vec![0.1, 0.2, 0.15];
        
        assert!(validate_time_series(&times, &values, Some(&errors)).is_ok());
        
        // Test various error conditions
        assert!(validate_time_series(&[], &[], None).is_err());
        assert!(validate_time_series(&times, &[], None).is_err());
        assert!(validate_time_series(&times, &values, Some(&[0.1])).is_err());
        assert!(validate_time_series(&times, &values, Some(&[0.1, -0.2, 0.15])).is_err());
        
        let unsorted_times = vec![2.0, 1.0, 3.0];
        assert!(validate_time_series(&unsorted_times, &values, None).is_err());
    }
    
    #[test]
    fn test_matrix_exponential() {
        let matrix = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, -2.0]);
        let result = matrix_exponential(&matrix, 0.1).unwrap();
        
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert!(result[(0,0)].is_finite());
    }
    
    #[test] 
    fn test_characteristic_roots() {
        // Test simple cases
        let roots1 = compute_characteristic_roots(&[1.0]).unwrap();
        assert_eq!(roots1.len(), 1);
        assert_eq!(roots1[0].re, -1.0);
        
        let roots2 = compute_characteristic_roots(&[1.0, 2.0]).unwrap();
        assert_eq!(roots2.len(), 2);
    }
    
    #[test]
    fn test_power_spectral_density() {
        let ar_coeffs = vec![1.5, -0.5];
        let ma_coeffs = vec![1.0, 0.3];
        let sigma = 1.0;
        let frequencies = vec![0.1, 0.2, 0.5];
        
        let psd = power_spectral_density(&ar_coeffs, &ma_coeffs, sigma, &frequencies);
        assert_eq!(psd.len(), 3);
        assert!(psd.iter().all(|&p| p >= 0.0 && p.is_finite()));
    }
}