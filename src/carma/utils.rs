use crate::carma::model::{CarmaModel, CarmaError};
use pyo3::prelude::*;
use num_complex::Complex;

/// Check if a CARMA model is stable (all characteristic roots have negative real parts)
#[pyfunction]
pub fn check_carma_stability(model: &CarmaModel) -> PyResult<bool> {
    Ok(model.is_stable())
}

/// Get characteristic polynomial roots of a CARMA model
#[pyfunction] 
pub fn carma_characteristic_roots(model: &CarmaModel) -> PyResult<Vec<(f64, f64)>> {
    let roots = compute_characteristic_roots(model)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(roots)
}

/// Compute model order selection criteria for a range of (p,q) values
#[pyfunction]
pub fn carma_model_selection(
    times: numpy::PyReadonlyArray1<f64>,
    values: numpy::PyReadonlyArray1<f64>,
    max_p: usize,
    max_q: Option<usize>,
    errors: Option<numpy::PyReadonlyArray1<f64>>,
) -> PyResult<Vec<(usize, usize, f64, f64)>> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let errors_slice = errors.as_ref().map(|e| e.as_slice()).transpose()?;
    
    let max_q_val = max_q.unwrap_or(max_p - 1);
    
    if max_p == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_p must be greater than 0"
        ));
    }
    
    if max_q_val >= max_p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_q must be less than max_p"
        ));
    }
    
    let mut results = Vec::new();
    
    // Try all (p,q) combinations
    for p in 1..=max_p {
        for q in 0..=max_q_val.min(p-1) {
            // Quick MLE fit to get likelihood
            match quick_mle_fit(times_slice, values_slice, errors_slice, p, q) {
                Ok((loglik, n_params)) => {
                    let n_data = times_slice.len() as f64;
                    let aic = -2.0 * loglik + 2.0 * n_params as f64;
                    let bic = -2.0 * loglik + n_params as f64 * n_data.ln();
                    results.push((p, q, aic, bic));
                }
                Err(_) => {
                    // Skip if fit failed
                    continue;
                }
            }
        }
    }
    
    Ok(results)
}

/// Convert CARMA model to power spectral density function
#[pyfunction]
pub fn carma_power_spectrum(
    model: &CarmaModel,
    frequencies: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let freq_slice = frequencies.as_slice()?;
    
    let psd = compute_power_spectral_density(model, freq_slice)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(psd)
}

/// Compute autocovariance function of CARMA model
#[pyfunction]
pub fn carma_autocovariance(
    model: &CarmaModel,
    lags: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let lags_slice = lags.as_slice()?;
    
    let autocov = compute_autocovariance_function(model, lags_slice)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(autocov)
}

/// Internal functions

/// Compute characteristic polynomial roots
fn compute_characteristic_roots(model: &CarmaModel) -> Result<Vec<(f64, f64)>, CarmaError> {
    let coeffs = model.get_characteristic_polynomial();
    
    // For simple cases, use analytical solutions
    match model.p {
        1 => {
            // s + α_0 = 0 => s = -α_0
            let root = -model.ar_coeffs[0];
            Ok(vec![(root, 0.0)])
        }
        2 => {
            // s^2 + α_1 s + α_0 = 0
            let a = 1.0;
            let b = model.ar_coeffs[1];
            let c = model.ar_coeffs[0];
            
            let discriminant = b * b - 4.0 * a * c;
            
            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let root1 = (-b + sqrt_disc) / (2.0 * a);
                let root2 = (-b - sqrt_disc) / (2.0 * a);
                Ok(vec![(root1, 0.0), (root2, 0.0)])
            } else {
                let real_part = -b / (2.0 * a);
                let imag_part = (-discriminant).sqrt() / (2.0 * a);
                Ok(vec![(real_part, imag_part), (real_part, -imag_part)])
            }
        }
        _ => {
            // For higher-order, use numerical root finding (simplified)
            // This is a placeholder - in practice, we'd use a proper root finder
            Err(CarmaError::NumericalError {
                message: format!("Characteristic root computation not implemented for p > 2, got p = {}", model.p)
            })
        }
    }
}

/// Quick MLE fit for model selection (simplified)
fn quick_mle_fit(
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    p: usize,
    q: usize,
) -> Result<(f64, usize), CarmaError> {
    use crate::carma::mle::{fit_carma_mle, MleConfig};
    
    // Use simplified configuration for speed
    let config = MleConfig {
        max_iter: 100,
        tolerance: 1e-4,
        n_trials: 3,
        n_jobs: 1,
        seed: Some(42),
    };
    
    let result = fit_carma_mle(times, values, errors, p, q, &config)?;
    let n_params = result.model.num_params();
    
    Ok((result.loglikelihood, n_params))
}

/// Compute power spectral density at given frequencies
fn compute_power_spectral_density(model: &CarmaModel, frequencies: &[f64]) -> Result<Vec<f64>, CarmaError> {
    let mut psd = Vec::with_capacity(frequencies.len());
    
    for &f in frequencies {
        let omega = 2.0 * std::f64::consts::PI * f;
        let s = Complex::new(0.0, omega);
        
        let psd_val = compute_psd_at_frequency(model, s)?;
        psd.push(psd_val);
    }
    
    Ok(psd)
}

/// Compute PSD at a single complex frequency
fn compute_psd_at_frequency(model: &CarmaModel, s: Complex<f64>) -> Result<f64, CarmaError> {
    // PSD(ω) = σ² |Q(iω)|² / |P(iω)|²
    // where P(s) is the AR polynomial and Q(s) is the MA polynomial
    
    // Compute AR polynomial P(s) = s^p + α_{p-1} s^{p-1} + ... + α_0
    let mut ar_poly = Complex::new(0.0, 0.0);
    let mut s_power = Complex::new(1.0, 0.0);
    
    for i in 0..model.p {
        ar_poly += Complex::new(model.ar_coeffs[i], 0.0) * s_power;
        s_power *= s;
    }
    ar_poly += s_power; // Add s^p term
    
    // Compute MA polynomial Q(s) = β_q s^q + ... + β_0
    let mut ma_poly = Complex::new(0.0, 0.0);
    s_power = Complex::new(1.0, 0.0);
    
    for i in 0..=model.q {
        ma_poly += Complex::new(model.ma_coeffs[i], 0.0) * s_power;
        s_power *= s;
    }
    
    // Compute PSD
    let ar_mag_sq = ar_poly.norm_sqr();
    let ma_mag_sq = ma_poly.norm_sqr();
    
    if ar_mag_sq < 1e-14 {
        return Err(CarmaError::NumericalError {
            message: "AR polynomial is too close to zero".to_string()
        });
    }
    
    let psd = model.sigma * model.sigma * ma_mag_sq / ar_mag_sq;
    
    Ok(psd)
}

/// Compute autocovariance function at given lags
fn compute_autocovariance_function(model: &CarmaModel, lags: &[f64]) -> Result<Vec<f64>, CarmaError> {
    let mut autocov = Vec::with_capacity(lags.len());
    
    // For CARMA models, the autocovariance has a specific analytical form
    // This is a simplified implementation for low-order models
    
    for &lag in lags {
        let cov_val = compute_autocovariance_at_lag(model, lag.abs())?;
        autocov.push(cov_val);
    }
    
    Ok(autocov)
}

/// Compute autocovariance at a single lag
fn compute_autocovariance_at_lag(model: &CarmaModel, lag: f64) -> Result<f64, CarmaError> {
    // For CAR(1): C(τ) = σ²/(2α) * exp(-α*τ)
    if model.p == 1 && model.q == 0 {
        let alpha = model.ar_coeffs[0];
        if alpha <= 0.0 {
            return Err(CarmaError::InvalidParameter {
                message: "AR coefficient must be positive for CAR(1)".to_string()
            });
        }
        
        let variance = model.sigma * model.sigma / (2.0 * alpha);
        let autocov = variance * (-alpha * lag).exp();
        return Ok(autocov);
    }
    
    // For higher-order models, this requires more complex calculations
    // involving the characteristic roots and residues
    // For now, return a placeholder
    Err(CarmaError::NumericalError {
        message: format!("Autocovariance computation not implemented for CARMA({}, {})", model.p, model.q)
    })
}

/// Validate time series data for CARMA fitting
pub fn validate_carma_data(
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
) -> Result<(), CarmaError> {
    if times.is_empty() || values.is_empty() {
        return Err(CarmaError::DataValidationError {
            message: "Time series data cannot be empty".to_string()
        });
    }
    
    if times.len() != values.len() {
        return Err(CarmaError::DataValidationError {
            message: format!("Times and values must have same length: {} vs {}", times.len(), values.len())
        });
    }
    
    if let Some(errs) = errors {
        if errs.len() != times.len() {
            return Err(CarmaError::DataValidationError {
                message: format!("Errors must have same length as times: {} vs {}", errs.len(), times.len())
            });
        }
        
        // Check that errors are positive
        for (i, &err) in errs.iter().enumerate() {
            if err <= 0.0 || !err.is_finite() {
                return Err(CarmaError::DataValidationError {
                    message: format!("Error value at index {} is not positive and finite: {}", i, err)
                });
            }
        }
    }
    
    // Check that times are finite and in ascending order
    for (i, &t) in times.iter().enumerate() {
        if !t.is_finite() {
            return Err(CarmaError::DataValidationError {
                message: format!("Time value at index {} is not finite: {}", i, t)
            });
        }
        
        if i > 0 && t <= times[i-1] {
            return Err(CarmaError::DataValidationError {
                message: format!("Times must be in strictly ascending order: times[{}]={} <= times[{}]={}", 
                    i, t, i-1, times[i-1])
            });
        }
    }
    
    // Check that values are finite
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(CarmaError::DataValidationError {
                message: format!("Value at index {} is not finite: {}", i, v)
            });
        }
    }
    
    // Check minimum number of data points
    if times.len() < 10 {
        return Err(CarmaError::DataValidationError {
            message: format!("Need at least 10 data points for reliable fitting, got {}", times.len())
        });
    }
    
    Ok(())
}

/// Compute theoretical variance of CARMA process
pub fn carma_theoretical_variance(model: &CarmaModel) -> Result<f64, CarmaError> {
    // For CAR(1): Var = σ²/(2α)
    if model.p == 1 && model.q == 0 {
        let alpha = model.ar_coeffs[0];
        if alpha <= 0.0 {
            return Err(CarmaError::InvalidParameter {
                message: "AR coefficient must be positive".to_string()
            });
        }
        return Ok(model.sigma * model.sigma / (2.0 * alpha));
    }
    
    // For higher-order models, this involves solving the Lyapunov equation
    // which we implement in the likelihood module
    use crate::carma::likelihood::StateSpaceModel;
    
    let state_space = StateSpaceModel::from_carma_model(model)?;
    let variance = state_space.steady_state_cov[(0, 0)];
    
    Ok(variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_characteristic_roots_car1() {
        let mut model = CarmaModel::new(1, 0).unwrap();
        model.ar_coeffs = vec![2.0];
        
        let roots = compute_characteristic_roots(&model).unwrap();
        assert_eq!(roots.len(), 1);
        assert!((roots[0].0 + 2.0).abs() < 1e-10);
        assert!(roots[0].1.abs() < 1e-10);
    }
    
    #[test]
    fn test_characteristic_roots_carma21() {
        let mut model = CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![1.0, 3.0]; // s^2 + 3s + 1 = 0
        
        let roots = compute_characteristic_roots(&model).unwrap();
        assert_eq!(roots.len(), 2);
        
        // Verify roots satisfy the equation
        for &(re, im) in &roots {
            let s = Complex::new(re, im);
            let poly_val = s * s + Complex::new(3.0, 0.0) * s + Complex::new(1.0, 0.0);
            assert!(poly_val.norm() < 1e-10);
        }
    }
    
    #[test]
    fn test_data_validation() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![1.0, 2.0, 3.0];
        let errors = vec![0.1, 0.1, 0.1];
        
        assert!(validate_carma_data(&times, &values, Some(&errors)).is_ok());
        
        // Test mismatched lengths
        let bad_values = vec![1.0, 2.0];
        assert!(validate_carma_data(&times, &bad_values, None).is_err());
        
        // Test non-ascending times
        let bad_times = vec![0.0, 2.0, 1.0];
        assert!(validate_carma_data(&bad_times, &values, None).is_err());
        
        // Test negative errors
        let bad_errors = vec![0.1, -0.1, 0.1];
        assert!(validate_carma_data(&times, &values, Some(&bad_errors)).is_err());
    }
    
    #[test]
    fn test_car1_variance() {
        let mut model = CarmaModel::new(1, 0).unwrap();
        model.ar_coeffs = vec![2.0];
        model.sigma = 1.0;
        
        let variance = carma_theoretical_variance(&model).unwrap();
        let expected = 1.0 / (2.0 * 2.0); // σ²/(2α)
        assert!((variance - expected).abs() < 1e-10);
    }
}