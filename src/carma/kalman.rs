use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use nalgebra::{DMatrix, DVector};
use crate::carma::carma_model::{CarmaModel, KalmanResult, CarmaPrediction, CarmaError};
use crate::carma::utils::{matrix_exponential, solve_lyapunov, validate_time_series, carma_to_state_space};

/// Kalman filter for irregular time sampling
#[pyfunction]
pub fn carma_kalman_filter(
    model: &CarmaModel,
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    errors: Option<PyReadonlyArray1<f64>>
) -> PyResult<KalmanResult> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let errors_slice = errors.as_ref().map(|e| e.as_slice()).transpose()?;
    
    // Validate inputs
    validate_time_series(times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    // Convert to state space
    let ss = carma_to_state_space(model)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    // Convert to nalgebra matrices
    let p = model.p;
    let transition = DMatrix::from_vec(p, p, ss.transition_matrix.into_iter().flatten().collect());
    let observation = DVector::from_vec(ss.observation_vector);
    let process_noise = DMatrix::from_vec(p, p, ss.process_noise_matrix.into_iter().flatten().collect());
    
    // Run Kalman filter
    run_kalman_filter(&transition, &observation, &process_noise, times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Prediction using Kalman filter
#[pyfunction]
pub fn carma_predict(
    model: &CarmaModel,
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    prediction_times: PyReadonlyArray1<f64>,
    errors: Option<PyReadonlyArray1<f64>>,
    confidence_level: Option<f64>
) -> PyResult<CarmaPrediction> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let pred_times_slice = prediction_times.as_slice()?;
    let errors_slice = errors.as_ref().map(|e| e.as_slice()).transpose()?;
    let conf_level = confidence_level.unwrap_or(0.95);
    
    // Validate inputs
    validate_time_series(times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Confidence level must be between 0 and 1"));
    }
    
    // Convert to state space
    let ss = carma_to_state_space(model)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    let p = model.p;
    let transition = DMatrix::from_vec(p, p, ss.transition_matrix.into_iter().flatten().collect());
    let observation = DVector::from_vec(ss.observation_vector);
    let process_noise = DMatrix::from_vec(p, p, ss.process_noise_matrix.into_iter().flatten().collect());
    
    // Run prediction
    run_prediction(&transition, &observation, &process_noise, 
                   times_slice, values_slice, pred_times_slice, errors_slice, conf_level)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Internal Kalman filter implementation
pub fn run_kalman_filter(
    transition: &DMatrix<f64>,
    observation: &DVector<f64>,
    process_noise: &DMatrix<f64>,
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
) -> Result<KalmanResult, CarmaError> {
    let n = times.len();
    let p = transition.nrows();
    
    // Initialize state and covariance
    let mut state = DVector::zeros(p);
    let mut covariance = DMatrix::identity(p, p) * 1000.0; // Large initial uncertainty
    
    let mut filtered_means = Vec::with_capacity(n);
    let mut filtered_variances = Vec::with_capacity(n);
    let mut predicted_means = Vec::with_capacity(n);
    let mut predicted_variances = Vec::with_capacity(n);
    let mut log_likelihood = 0.0;
    
    for i in 0..n {
        // Time step
        let dt = if i == 0 { 1.0 } else { times[i] - times[i-1] };
        
        // Predict step
        let transition_matrix = matrix_exponential(transition, dt)?;
        let predicted_state = &transition_matrix * &state;
        
        // Compute process noise covariance for this time step
        let process_cov = compute_process_covariance(transition, process_noise, dt)?;
        let predicted_covariance = &transition_matrix * &covariance * transition_matrix.transpose() + process_cov;
        
        // Predicted observation
        let predicted_obs = observation.dot(&predicted_state);
        let predicted_obs_var = observation.dot(&(&predicted_covariance * observation));
        
        predicted_means.push(predicted_obs);
        predicted_variances.push(predicted_obs_var);
        
        // Update step
        let obs_noise_var = if let Some(errs) = errors { 
            errs[i] * errs[i] 
        } else { 
            0.01 // Small default observation noise
        };
        
        let innovation_var = predicted_obs_var + obs_noise_var;
        
        if innovation_var > 1e-12 {
            let kalman_gain = (&predicted_covariance * observation) / innovation_var;
            let innovation = values[i] - predicted_obs;
            
            state = predicted_state + &kalman_gain * innovation;
            covariance = &predicted_covariance - &kalman_gain * observation.transpose() * &predicted_covariance;
            
            // Update log-likelihood
            log_likelihood += -0.5 * (innovation * innovation / innovation_var + innovation_var.ln() + (2.0 * std::f64::consts::PI).ln());
        } else {
            // Singular case - use predicted values
            state = predicted_state;
            covariance = predicted_covariance;
        }
        
        // Store filtered results
        let filtered_obs = observation.dot(&state);
        let filtered_obs_var = observation.dot(&(&covariance * observation));
        
        filtered_means.push(filtered_obs);
        filtered_variances.push(filtered_obs_var);
    }
    
    Ok(KalmanResult {
        filtered_mean: filtered_means,
        filtered_variance: filtered_variances,
        predicted_mean: predicted_means,
        predicted_variance: predicted_variances,
        loglikelihood: log_likelihood,
    })
}

/// Internal prediction implementation
fn run_prediction(
    transition: &DMatrix<f64>,
    observation: &DVector<f64>,
    process_noise: &DMatrix<f64>,
    times: &[f64],
    values: &[f64],
    prediction_times: &[f64],
    errors: Option<&[f64]>,
    confidence_level: f64,
) -> Result<CarmaPrediction, CarmaError> {
    // First run Kalman filter on historical data
    let kalman_result = run_kalman_filter(transition, observation, process_noise, times, values, errors)?;
    
    // Get final state from Kalman filter
    let p = transition.nrows();
    let n_hist = times.len();
    
    // Initialize from last filtered state (simplified - would need proper state reconstruction)
    let mut state = DVector::zeros(p);
    state[0] = kalman_result.filtered_mean[n_hist - 1]; // Approximate
    let mut covariance = DMatrix::identity(p, p) * kalman_result.filtered_variance[n_hist - 1];
    
    let mut pred_means = Vec::with_capacity(prediction_times.len());
    let mut pred_variances = Vec::with_capacity(prediction_times.len());
    
    let last_time = times[n_hist - 1];
    
    for &pred_time in prediction_times {
        let dt = pred_time - last_time;
        
        if dt <= 0.0 {
            return Err(CarmaError::InvalidData("Prediction times must be after observation times".to_string()));
        }
        
        // Propagate state forward
        let transition_matrix = matrix_exponential(transition, dt)?;
        state = &transition_matrix * &state;
        
        // Propagate covariance
        let process_cov = compute_process_covariance(transition, process_noise, dt)?;
        covariance = &transition_matrix * &covariance * transition_matrix.transpose() + process_cov;
        
        // Predict observation
        let pred_mean = observation.dot(&state);
        let pred_var = observation.dot(&(&covariance * observation));
        
        pred_means.push(pred_mean);
        pred_variances.push(pred_var);
    }
    
    // Compute confidence bounds
    let z_score = normal_quantile((1.0 + confidence_level) / 2.0);
    let lower_bounds: Vec<f64> = pred_means.iter().zip(pred_variances.iter())
        .map(|(&mean, &var)| mean - z_score * var.sqrt())
        .collect();
    let upper_bounds: Vec<f64> = pred_means.iter().zip(pred_variances.iter())
        .map(|(&mean, &var)| mean + z_score * var.sqrt())
        .collect();
    
    Ok(CarmaPrediction {
        mean: pred_means,
        variance: pred_variances,
        lower_bound: lower_bounds,
        upper_bound: upper_bounds,
    })
}

/// Compute process noise covariance for a given time step
fn compute_process_covariance(
    transition: &DMatrix<f64>,
    process_noise: &DMatrix<f64>,
    dt: f64,
) -> Result<DMatrix<f64>, CarmaError> {
    // For continuous-time systems, we need to compute:
    // ∫₀^dt exp(A*s) * Q * exp(A'*s) ds
    
    // Simplified approximation for small dt
    if dt < 0.1 {
        Ok(process_noise * dt)
    } else {
        // Use Lyapunov equation approach
        // More sophisticated implementation would solve the integral properly
        solve_lyapunov(transition, process_noise)
            .map(|cov| cov * dt)
    }
}

/// Approximate normal distribution quantile (for confidence intervals)
fn normal_quantile(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm approximation for normal quantile
    if p <= 0.0 { return -f64::INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.5 { return 0.0; }
    
    let q = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * q.ln()).sqrt();
    
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    
    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let z = t - numerator / denominator;
    
    if p > 0.5 { z } else { -z }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carma::carma_model::CarmaModel;
    use numpy::PyArray1;
    use pyo3::Python;
    
    #[test]
    fn test_matrix_operations() {
        let transition = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, -1.0]);
        let process_noise = DMatrix::identity(2, 2) * 0.1;
        
        let cov = compute_process_covariance(&transition, &process_noise, 0.1).unwrap();
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
    }
    
    #[test]
    fn test_normal_quantile() {
        assert!((normal_quantile(0.5) - 0.0).abs() < 1e-6);
        assert!(normal_quantile(0.975) > 1.9);
        assert!(normal_quantile(0.975) < 2.0);
        assert!(normal_quantile(0.025) < -1.9);
        assert!(normal_quantile(0.025) > -2.0);
    }
    
    #[test]
    fn test_kalman_filter_setup() {
        Python::with_gil(|py| {
            let mut model = CarmaModel::new(2, 1).unwrap();
            model.ar_coeffs = vec![1.5, -0.5];
            model.ma_coeffs = vec![1.0, 0.3];
            model.sigma = 1.0;
            
            let times = PyArray1::from_vec(py, vec![1.0, 2.0, 3.0, 4.0]);
            let values = PyArray1::from_vec(py, vec![1.1, 2.2, 3.3, 4.4]);
            let errors = PyArray1::from_vec(py, vec![0.1, 0.1, 0.1, 0.1]);
            
            let result = carma_kalman_filter(
                &model,
                times.readonly(),
                values.readonly(),
                Some(errors.readonly()),
            );
            
            assert!(result.is_ok());
            let kalman_result = result.unwrap();
            assert_eq!(kalman_result.filtered_mean.len(), 4);
            assert_eq!(kalman_result.filtered_variance.len(), 4);
        });
    }
    
    #[test]
    fn test_prediction_setup() {
        Python::with_gil(|py| {
            let mut model = CarmaModel::new(2, 1).unwrap();
            model.ar_coeffs = vec![1.5, -0.5];
            model.ma_coeffs = vec![1.0, 0.3];
            model.sigma = 1.0;
            
            let times = PyArray1::from_vec(py, vec![1.0, 2.0, 3.0]);
            let values = PyArray1::from_vec(py, vec![1.1, 2.2, 3.3]);
            let pred_times = PyArray1::from_vec(py, vec![4.0, 5.0]);
            
            let result = carma_predict(
                &model,
                times.readonly(),
                values.readonly(),
                pred_times.readonly(),
                None,
                Some(0.95),
            );
            
            assert!(result.is_ok());
            let pred_result = result.unwrap();
            assert_eq!(pred_result.mean.len(), 2);
            assert_eq!(pred_result.variance.len(), 2);
            assert_eq!(pred_result.lower_bound.len(), 2);
            assert_eq!(pred_result.upper_bound.len(), 2);
        });
    }
}