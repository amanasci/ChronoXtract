use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use num_complex::Complex64;
use crate::carma::carma_model::{CarmaModel, CarmaResiduals, CarmaError};
use crate::carma::utils::{power_spectral_density, autocorrelation_function, validate_time_series};
use crate::carma::kalman::run_kalman_filter;
use crate::carma::utils::carma_to_state_space;
use nalgebra::{DMatrix, DVector};

/// Compute power spectral density of CARMA model
#[pyfunction]
pub fn carma_psd(
    py: Python,
    model: &CarmaModel,
    frequencies: PyReadonlyArray1<f64>
) -> PyResult<Py<PyArray1<f64>>> {
    let freq_slice = frequencies.as_slice()?;
    
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    if freq_slice.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Frequencies cannot be empty"));
    }
    
    if freq_slice.iter().any(|&f| f < 0.0 || !f.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err("Frequencies must be non-negative and finite"));
    }
    
    // Compute PSD
    let psd_values = power_spectral_density(&model.ar_coeffs, &model.ma_coeffs, model.sigma, freq_slice);
    
    Ok(PyArray1::from_vec(py, psd_values).into())
}

/// Compute covariance function of CARMA model
#[pyfunction]
pub fn carma_covariance(
    py: Python,
    model: &CarmaModel,
    time_lags: PyReadonlyArray1<f64>
) -> PyResult<Py<PyArray1<f64>>> {
    let lags_slice = time_lags.as_slice()?;
    
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    if lags_slice.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Time lags cannot be empty"));
    }
    
    if lags_slice.iter().any(|&lag| !lag.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err("Time lags must be finite"));
    }
    
    // Compute covariance function
    let cov_values = compute_covariance_function(model, lags_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(PyArray1::from_vec(py, cov_values).into())
}

/// Compute log-likelihood of CARMA model given data
#[pyfunction]
pub fn carma_loglikelihood(
    model: &CarmaModel,
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    errors: Option<PyReadonlyArray1<f64>>
) -> PyResult<f64> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let errors_slice = errors.as_ref().map(|e| e.as_slice()).transpose()?;
    
    // Validate inputs
    validate_time_series(times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    // Compute log-likelihood using Kalman filter
    let loglik = compute_log_likelihood_internal(model, times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(loglik)
}

/// Compute residuals and diagnostic statistics
#[pyfunction]
pub fn carma_residuals(
    model: &CarmaModel,
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    errors: Option<PyReadonlyArray1<f64>>
) -> PyResult<CarmaResiduals> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let errors_slice = errors.as_ref().map(|e| e.as_slice()).transpose()?;
    
    // Validate inputs
    validate_time_series(times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    // Compute residuals
    let residuals_result = compute_residuals_internal(model, times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(residuals_result)
}

/// Internal covariance function computation
fn compute_covariance_function(model: &CarmaModel, lags: &[f64]) -> Result<Vec<f64>, CarmaError> {
    // For CARMA models, the covariance function can be computed from the state-space representation
    // This is a simplified implementation
    
    // Use autocorrelation function as approximation
    let acf_values = autocorrelation_function(&model.ar_coeffs, &model.ma_coeffs, lags);
    
    // Scale by variance (sigma^2)
    let variance = model.sigma * model.sigma;
    let cov_values: Vec<f64> = acf_values.iter().map(|&acf| variance * acf).collect();
    
    Ok(cov_values)
}

/// Internal log-likelihood computation
fn compute_log_likelihood_internal(
    model: &CarmaModel,
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
) -> Result<f64, CarmaError> {
    // Convert to state space
    let ss = carma_to_state_space(model)?;
    let p = model.p;
    let transition = DMatrix::from_vec(p, p, ss.transition_matrix.into_iter().flatten().collect());
    let observation = DVector::from_vec(ss.observation_vector);
    let process_noise = DMatrix::from_vec(p, p, ss.process_noise_matrix.into_iter().flatten().collect());
    
    // Run Kalman filter
    let kalman_result = run_kalman_filter(&transition, &observation, &process_noise, times, values, errors)?;
    
    Ok(kalman_result.loglikelihood)
}

/// Internal residuals computation
fn compute_residuals_internal(
    model: &CarmaModel,
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
) -> Result<CarmaResiduals, CarmaError> {
    // Convert to state space
    let ss = carma_to_state_space(model)?;
    let p = model.p;
    let transition = DMatrix::from_vec(p, p, ss.transition_matrix.into_iter().flatten().collect());
    let observation = DVector::from_vec(ss.observation_vector);
    let process_noise = DMatrix::from_vec(p, p, ss.process_noise_matrix.into_iter().flatten().collect());
    
    // Run Kalman filter to get predictions
    let kalman_result = run_kalman_filter(&transition, &observation, &process_noise, times, values, errors)?;
    
    // Compute residuals
    let n = values.len();
    let mut residuals = Vec::with_capacity(n);
    let mut standardized_residuals = Vec::with_capacity(n);
    
    for i in 0..n {
        let residual = values[i] - kalman_result.predicted_mean[i];
        residuals.push(residual);
        
        // Standardize by predicted variance
        let std_residual = if kalman_result.predicted_variance[i] > 0.0 {
            residual / kalman_result.predicted_variance[i].sqrt()
        } else {
            residual
        };
        standardized_residuals.push(std_residual);
    }
    
    // Compute Ljung-Box test statistic
    let (ljung_box_stat, ljung_box_pvalue) = ljung_box_test(&standardized_residuals, 10)?;
    
    Ok(CarmaResiduals {
        residuals,
        standardized_residuals,
        ljung_box_statistic: ljung_box_stat,
        ljung_box_pvalue,
    })
}

/// Ljung-Box test for residual autocorrelation
fn ljung_box_test(residuals: &[f64], max_lag: usize) -> Result<(f64, f64), CarmaError> {
    let n = residuals.len();
    
    if n <= max_lag {
        return Ok((0.0, 1.0)); // Not enough data
    }
    
    // Compute sample autocorrelations
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let variance = residuals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    
    if variance <= 0.0 {
        return Ok((0.0, 1.0));
    }
    
    let mut autocorrelations = Vec::with_capacity(max_lag);
    
    for lag in 1..=max_lag {
        let mut sum = 0.0;
        for i in lag..n {
            sum += (residuals[i] - mean) * (residuals[i - lag] - mean);
        }
        let autocorr = sum / ((n - lag) as f64 * variance);
        autocorrelations.push(autocorr);
    }
    
    // Compute Ljung-Box statistic
    let mut lb_statistic = 0.0;
    for (lag, &autocorr) in autocorrelations.iter().enumerate() {
        let h = lag + 1;
        lb_statistic += autocorr * autocorr / (n - h) as f64;
    }
    lb_statistic *= n as f64 * (n + 2) as f64;
    
    // Approximate p-value using chi-squared distribution
    let degrees_of_freedom = max_lag as f64;
    let p_value = 1.0 - chi_squared_cdf(lb_statistic, degrees_of_freedom);
    
    Ok((lb_statistic, p_value))
}

/// Approximate chi-squared CDF (simplified implementation)
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    
    // Very rough approximation for demonstration
    // A real implementation would use proper gamma functions
    let normalized = x / df;
    if normalized < 0.5 {
        normalized / 2.0
    } else if normalized < 2.0 {
        0.25 + (normalized - 0.5) * 0.5
    } else {
        0.95 // Saturate for large values
    }
}

/// Compute spectral density at specific frequencies using analytic formula
pub fn compute_analytic_psd(
    ar_coeffs: &[f64],
    ma_coeffs: &[f64],
    sigma: f64,
    frequencies: &[f64],
) -> Vec<f64> {
    frequencies.iter().map(|&freq| {
        let omega = 2.0 * std::f64::consts::PI * freq;
        let i = Complex64::new(0.0, 1.0);
        
        // Evaluate polynomials at i*omega
        let mut ar_poly = Complex64::new(1.0, 0.0);
        for (k, &coeff) in ar_coeffs.iter().enumerate() {
            ar_poly += coeff * (i * omega).powi(k as i32 + 1);
        }
        
        let mut ma_poly = Complex64::new(0.0, 0.0);
        for (k, &coeff) in ma_coeffs.iter().enumerate() {
            ma_poly += coeff * (i * omega).powi(k as i32);
        }
        
        // PSD = sigma^2 * |MA(iω)|² / |AR(iω)|²
        let ma_mag_sq = ma_poly.norm_sqr();
        let ar_mag_sq = ar_poly.norm_sqr();
        
        if ar_mag_sq > 1e-12 {
            sigma * sigma * ma_mag_sq / ar_mag_sq
        } else {
            0.0
        }
    }).collect()
}

/// Compute model goodness-of-fit statistics
pub fn compute_goodness_of_fit(
    model: &CarmaModel,
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
) -> Result<(f64, f64, f64), CarmaError> {
    // Get residuals
    let residuals_result = compute_residuals_internal(model, times, values, errors)?;
    
    // Compute statistics
    let n = values.len() as f64;
    let residuals = &residuals_result.residuals;
    
    // Mean squared error
    let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / n;
    
    // Root mean squared error
    let rmse = mse.sqrt();
    
    // Mean absolute error
    let mae = residuals.iter().map(|&r| r.abs()).sum::<f64>() / n;
    
    Ok((mse, rmse, mae))
}

/// Validate spectral analysis inputs
pub fn validate_spectral_inputs(
    model: &CarmaModel,
    frequencies: &[f64],
) -> Result<(), CarmaError> {
    if !model.is_valid() {
        return Err(CarmaError::InvalidParameters("Invalid model".to_string()));
    }
    
    if frequencies.is_empty() {
        return Err(CarmaError::InvalidData("Frequencies cannot be empty".to_string()));
    }
    
    if frequencies.iter().any(|&f| f < 0.0 || !f.is_finite()) {
        return Err(CarmaError::InvalidData("Frequencies must be non-negative and finite".to_string()));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carma::carma_model::CarmaModel;
    use numpy::PyArray1;
    use pyo3::Python;
    
    #[test]
    fn test_analytic_psd() {
        let ar_coeffs = vec![1.5, -0.5];
        let ma_coeffs = vec![1.0, 0.3];
        let sigma = 1.0;
        let frequencies = vec![0.1, 0.2, 0.5];
        
        let psd = compute_analytic_psd(&ar_coeffs, &ma_coeffs, sigma, &frequencies);
        
        assert_eq!(psd.len(), 3);
        assert!(psd.iter().all(|&p| p >= 0.0 && p.is_finite()));
    }
    
    #[test]
    fn test_ljung_box_test() {
        let residuals = vec![0.1, -0.2, 0.15, -0.1, 0.05, -0.08, 0.12, -0.15, 0.09, -0.03];
        let (stat, pvalue) = ljung_box_test(&residuals, 3).unwrap();
        
        assert!(stat >= 0.0 && stat.is_finite());
        assert!(pvalue >= 0.0 && pvalue <= 1.0);
    }
    
    #[test]
    fn test_goodness_of_fit() {
        let mut model = CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![1.5, -0.5];
        model.ma_coeffs = vec![1.0, 0.3];
        model.sigma = 1.0;
        
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![1.0, 1.1, 0.9, 1.05];
        
        let (mse, rmse, mae) = compute_goodness_of_fit(&model, &times, &values, None).unwrap();
        
        assert!(mse >= 0.0 && mse.is_finite());
        assert!(rmse >= 0.0 && rmse.is_finite());
        assert!(mae >= 0.0 && mae.is_finite());
        assert!((rmse * rmse - mse).abs() < 1e-10);
    }
    
    #[test]
    fn test_carma_psd_setup() {
        Python::with_gil(|py| {
            let mut model = CarmaModel::new(2, 1).unwrap();
            model.ar_coeffs = vec![1.5, -0.5];
            model.ma_coeffs = vec![1.0, 0.3];
            model.sigma = 1.0;
            
            let frequencies = PyArray1::from_vec(py, vec![0.1, 0.2, 0.5, 1.0]);
            let result = carma_psd(py, &model, frequencies.readonly());
            
            assert!(result.is_ok());
            let psd_array = result.unwrap();
            let psd_vec: Vec<f64> = psd_array.as_ref(py).to_vec().unwrap();
            
            assert_eq!(psd_vec.len(), 4);
            assert!(psd_vec.iter().all(|&p| p >= 0.0 && p.is_finite()));
        });
    }
    
    #[test]
    fn test_carma_covariance_setup() {
        Python::with_gil(|py| {
            let mut model = CarmaModel::new(2, 1).unwrap();
            model.ar_coeffs = vec![1.5, -0.5];
            model.ma_coeffs = vec![1.0, 0.3];
            model.sigma = 1.0;
            
            let lags = PyArray1::from_vec(py, vec![0.0, 1.0, 2.0, 5.0]);
            let result = carma_covariance(py, &model, lags.readonly());
            
            assert!(result.is_ok());
            let cov_array = result.unwrap();
            let cov_vec: Vec<f64> = cov_array.as_ref(py).to_vec().unwrap();
            
            assert_eq!(cov_vec.len(), 4);
            assert!(cov_vec.iter().all(|&c| c.is_finite()));
            
            // Covariance at lag 0 should be maximum (variance)
            assert!(cov_vec[0] >= cov_vec[1].abs());
        });
    }
    
    #[test]
    fn test_spectral_validation() {
        let mut model = CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![1.5, -0.5];
        model.ma_coeffs = vec![1.0, 0.3];
        model.sigma = 1.0;
        
        let valid_freqs = vec![0.0, 0.1, 0.5, 1.0];
        assert!(validate_spectral_inputs(&model, &valid_freqs).is_ok());
        
        let invalid_freqs = vec![-0.1, 0.5, 1.0];
        assert!(validate_spectral_inputs(&model, &invalid_freqs).is_err());
        
        let empty_freqs = vec![];
        assert!(validate_spectral_inputs(&model, &empty_freqs).is_err());
    }
}