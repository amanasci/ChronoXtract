use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use std::collections::HashMap;
use crate::carma::carma_model::{InformationCriteriaResult, CrossValidationResult, CarmaError, CarmaModel};
use crate::carma::estimation::{perform_mle_optimization, perform_method_of_moments};
use crate::carma::utils::validate_time_series;
use rand::prelude::*;

/// Compute AIC/BIC for different model orders
#[pyfunction]
pub fn carma_information_criteria(
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    max_p: usize,
    max_q: usize,
    errors: Option<PyReadonlyArray1<f64>>,
) -> PyResult<InformationCriteriaResult> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let errors_slice = errors.as_ref().map(|e| e.as_slice()).transpose()?;
    
    // Validate inputs
    validate_time_series(times_slice, values_slice, errors_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    if max_p == 0 || max_q >= max_p {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid max_p, max_q: must have max_p > 0 and max_q < max_p"));
    }
    
    // Limit search space for computational efficiency
    let max_p_limited = max_p.min(8);
    let max_q_limited = max_q.min(max_p_limited - 1);
    
    // Compute information criteria
    let result = compute_information_criteria_grid(
        times_slice, values_slice, errors_slice, max_p_limited, max_q_limited)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(result)
}

/// Cross-validation for model assessment
#[pyfunction]
pub fn carma_cross_validation(
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    p: usize,
    q: usize,
    n_folds: usize,
    seed: Option<u64>,
) -> PyResult<CrossValidationResult> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    
    // Validate inputs
    validate_time_series(times_slice, values_slice, None)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    if p == 0 || q >= p {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid p, q: must have p > 0 and q < p"));
    }
    
    if n_folds < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("n_folds must be at least 2"));
    }
    
    if times_slice.len() < n_folds * 5 {
        return Err(pyo3::exceptions::PyValueError::new_err("Not enough data for cross-validation"));
    }
    
    // Perform cross-validation
    let result = perform_cross_validation(times_slice, values_slice, p, q, n_folds, seed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(result)
}

/// Internal implementation of information criteria grid search
fn compute_information_criteria_grid(
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    max_p: usize,
    max_q: usize,
) -> Result<InformationCriteriaResult, CarmaError> {
    let mut results = HashMap::new();
    let mut best_aic = (1, 0);
    let mut best_bic = (1, 0);
    let mut min_aic = f64::INFINITY;
    let mut min_bic = f64::INFINITY;
    
    // Grid search over (p, q) combinations
    for p in 1..=max_p {
        for q in 0..p.min(max_q + 1) {
            let mut model_results = HashMap::new();
            
            // Try to fit model - use method of moments for robustness
            match fit_model_robust(times, values, errors, p, q) {
                Ok((loglik, aic, bic, _converged)) => {
                    model_results.insert("loglikelihood".to_string(), loglik);
                    model_results.insert("aic".to_string(), aic);
                    model_results.insert("bic".to_string(), bic);
                    model_results.insert("converged".to_string(), if _converged { 1.0 } else { 0.0 });
                    
                    // Update best models
                    if aic < min_aic {
                        min_aic = aic;
                        best_aic = (p, q);
                    }
                    if bic < min_bic {
                        min_bic = bic;
                        best_bic = (p, q);
                    }
                }
                Err(_) => {
                    // If fitting fails, record with poor criteria
                    model_results.insert("loglikelihood".to_string(), f64::NEG_INFINITY);
                    model_results.insert("aic".to_string(), f64::INFINITY);
                    model_results.insert("bic".to_string(), f64::INFINITY);
                    model_results.insert("converged".to_string(), 0.0);
                }
            }
            
            let key = format!("CARMA({},{})", p, q);
            results.insert(key, model_results);
        }
    }
    
    Ok(InformationCriteriaResult {
        results,
        best_aic,
        best_bic,
    })
}

/// Robust model fitting that tries multiple methods
fn fit_model_robust(
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    p: usize,
    q: usize,
) -> Result<(f64, f64, f64, bool), CarmaError> {
    // First try method of moments (more robust)
    if let Ok(mom_result) = perform_method_of_moments(times, values, p, q) {
        let _n = times.len() as f64;
        let _k = mom_result.model.parameter_count() as f64;
        return Ok((mom_result.loglikelihood, mom_result.aic, mom_result.bic, true));
    }
    
    // If that fails, try MLE with reduced tolerance
    if let Ok(mle_result) = perform_mle_optimization(times, values, errors, p, q, 100, 1e-4) {
        return Ok((mle_result.loglikelihood, mle_result.aic, mle_result.bic, 
                  mle_result.convergence_info.get("converged").copied().unwrap_or(0.0) > 0.5));
    }
    
    Err(CarmaError::ConvergenceError("All fitting methods failed".to_string()))
}

/// Internal cross-validation implementation
fn perform_cross_validation(
    times: &[f64],
    values: &[f64],
    p: usize,
    q: usize,
    n_folds: usize,
    seed: Option<u64>,
) -> Result<CrossValidationResult, CarmaError> {
    let n = times.len();
    
    // Create fold indices
    let mut indices: Vec<usize> = (0..n).collect();
    
    if let Some(s) = seed {
        let mut rng = StdRng::seed_from_u64(s);
        indices.shuffle(&mut rng);
    } else {
        let mut rng = StdRng::from_entropy();
        indices.shuffle(&mut rng);
    }
    
    let fold_size = n / n_folds;
    let mut fold_scores = Vec::with_capacity(n_folds);
    
    for fold in 0..n_folds {
        let test_start = fold * fold_size;
        let test_end = if fold == n_folds - 1 { n } else { (fold + 1) * fold_size };
        
        // Split data into train and test
        let (train_times, train_values, test_times, test_values) = 
            split_data_for_fold(times, values, &indices, test_start, test_end);
        
        if train_times.len() < p + q + 2 {
            continue; // Skip fold if insufficient training data
        }
        
        // Fit model on training data
        let score = match fit_and_evaluate_fold(&train_times, &train_values, &test_times, &test_values, p, q) {
            Ok(s) => s,
            Err(_) => f64::INFINITY, // Penalize failed fits
        };
        
        fold_scores.push(score);
    }
    
    if fold_scores.is_empty() {
        return Err(CarmaError::InvalidData("No valid folds for cross-validation".to_string()));
    }
    
    // Compute statistics
    let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
    let variance = fold_scores.iter()
        .map(|&score| (score - mean_score).powi(2))
        .sum::<f64>() / fold_scores.len() as f64;
    let std_score = variance.sqrt();
    
    Ok(CrossValidationResult {
        mean_score,
        std_score,
        fold_scores,
    })
}

/// Split data for a single fold
fn split_data_for_fold(
    times: &[f64],
    values: &[f64],
    indices: &[usize],
    test_start: usize,
    test_end: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut train_times = Vec::new();
    let mut train_values = Vec::new();
    let mut test_times = Vec::new();
    let mut test_values = Vec::new();
    
    for (i, &idx) in indices.iter().enumerate() {
        if i >= test_start && i < test_end {
            test_times.push(times[idx]);
            test_values.push(values[idx]);
        } else {
            train_times.push(times[idx]);
            train_values.push(values[idx]);
        }
    }
    
    // Sort by time (needed for CARMA fitting)
    let mut train_pairs: Vec<(f64, f64)> = train_times.into_iter().zip(train_values.into_iter()).collect();
    train_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let mut test_pairs: Vec<(f64, f64)> = test_times.into_iter().zip(test_values.into_iter()).collect();
    test_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let (train_times, train_values): (Vec<f64>, Vec<f64>) = train_pairs.into_iter().unzip();
    let (test_times, test_values): (Vec<f64>, Vec<f64>) = test_pairs.into_iter().unzip();
    
    (train_times, train_values, test_times, test_values)
}

/// Fit model on training data and evaluate on test data
fn fit_and_evaluate_fold(
    train_times: &[f64],
    train_values: &[f64],
    test_times: &[f64],
    test_values: &[f64],
    p: usize,
    q: usize,
) -> Result<f64, CarmaError> {
    // Fit model on training data
    let _fit_result = fit_model_robust(train_times, train_values, None, p, q)?;
    let _model = CarmaModel::new(p, q).unwrap(); // This will be improved with actual fitted parameters
    
    // For now, use simple prediction error as score
    // A full implementation would use proper forecasting
    let mut total_error = 0.0;
    let mut count = 0;
    
    // Simple linear interpolation for evaluation (placeholder)
    for (_i, (&test_time, &test_value)) in test_times.iter().zip(test_values.iter()).enumerate() {
        // Find nearest training points for simple prediction
        let prediction = if let Some(pred) = simple_predict(train_times, train_values, test_time) {
            pred
        } else {
            train_values.iter().sum::<f64>() / train_values.len() as f64 // Mean as fallback
        };
        
        let error = (test_value - prediction).abs();
        total_error += error;
        count += 1;
    }
    
    if count > 0 {
        Ok(total_error / count as f64)
    } else {
        Ok(f64::INFINITY)
    }
}

/// Simple prediction for cross-validation (placeholder)
fn simple_predict(times: &[f64], values: &[f64], target_time: f64) -> Option<f64> {
    if times.is_empty() {
        return None;
    }
    
    // Find closest time points
    let mut closest_idx = 0;
    let mut min_dist = (times[0] - target_time).abs();
    
    for (i, &time) in times.iter().enumerate() {
        let dist = (time - target_time).abs();
        if dist < min_dist {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    // Simple interpolation if we have neighbors
    if closest_idx > 0 && closest_idx < times.len() - 1 {
        let t0 = times[closest_idx - 1];
        let t1 = times[closest_idx + 1];
        let v0 = values[closest_idx - 1];
        let v1 = values[closest_idx + 1];
        
        if (t1 - t0).abs() > 1e-12 {
            let alpha = (target_time - t0) / (t1 - t0);
            Some(v0 + alpha * (v1 - v0))
        } else {
            Some(values[closest_idx])
        }
    } else {
        Some(values[closest_idx])
    }
}

/// Model selection based on multiple criteria
pub fn select_best_model(
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    max_p: usize,
    max_q: usize,
    criterion: &str,
) -> Result<(usize, usize, f64), CarmaError> {
    let ic_result = compute_information_criteria_grid(times, values, errors, max_p, max_q)?;
    
    match criterion.to_lowercase().as_str() {
        "aic" => Ok((ic_result.best_aic.0, ic_result.best_aic.1, 
                    get_criterion_value(&ic_result.results, ic_result.best_aic, "aic"))),
        "bic" => Ok((ic_result.best_bic.0, ic_result.best_bic.1,
                    get_criterion_value(&ic_result.results, ic_result.best_bic, "bic"))),
        _ => Err(CarmaError::InvalidParameters(format!("Unknown criterion: {}", criterion))),
    }
}

/// Extract criterion value from results
fn get_criterion_value(
    results: &HashMap<String, HashMap<String, f64>>,
    best_pq: (usize, usize),
    criterion: &str,
) -> f64 {
    let key = format!("CARMA({},{})", best_pq.0, best_pq.1);
    results.get(&key)
        .and_then(|model_results| model_results.get(criterion))
        .copied()
        .unwrap_or(f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::Python;
    
    #[test]
    fn test_simple_predict() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let values = vec![1.0, 2.0, 1.5, 2.5];
        
        let pred = simple_predict(&times, &values, 2.5);
        assert!(pred.is_some());
        let pred_val = pred.unwrap();
        assert!(pred_val.is_finite());
        assert!(pred_val >= 1.0 && pred_val <= 3.0); // Should be reasonable
    }
    
    #[test]
    fn test_split_data_for_fold() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let values = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1];
        let indices = vec![0, 1, 2, 3, 4, 5];
        
        let (train_times, train_values, test_times, test_values) = 
            split_data_for_fold(&times, &values, &indices, 2, 4);
        
        assert_eq!(train_times.len(), 4);
        assert_eq!(test_times.len(), 2);
        assert_eq!(train_values.len(), 4);
        assert_eq!(test_values.len(), 2);
        
        // Check that train_times are sorted
        for i in 1..train_times.len() {
            assert!(train_times[i] >= train_times[i-1]);
        }
    }
    
    #[test]
    fn test_model_selection_validation() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![1.1, 2.1, 1.9, 2.2, 1.8];
        
        // Test valid inputs
        assert!(select_best_model(&times, &values, None, 2, 1, "aic").is_ok());
        assert!(select_best_model(&times, &values, None, 2, 1, "bic").is_ok());
        
        // Test invalid criterion
        assert!(select_best_model(&times, &values, None, 2, 1, "invalid").is_err());
    }
    
    #[test]
    fn test_information_criteria_setup() {
        Python::with_gil(|py| {
            let times = PyArray1::from_vec(py, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let values = PyArray1::from_vec(py, vec![1.1, 1.9, 2.2, 1.8, 2.1, 1.95]);
            
            let result = carma_information_criteria(
                times.readonly(),
                values.readonly(),
                2,
                1,
                None,
            );
            
            assert!(result.is_ok());
            let ic_result = result.unwrap();
            
            assert!(!ic_result.results.is_empty());
            assert!(ic_result.best_aic.0 >= 1);
            assert!(ic_result.best_bic.0 >= 1);
            assert!(ic_result.best_aic.1 < ic_result.best_aic.0);
            assert!(ic_result.best_bic.1 < ic_result.best_bic.0);
        });
    }
    
    #[test]
    fn test_cross_validation_setup() {
        Python::with_gil(|py| {
            let times = PyArray1::from_vec(py, 
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
            let values = PyArray1::from_vec(py, 
                vec![1.1, 1.9, 2.2, 1.8, 2.1, 1.95, 2.3, 1.7, 2.0, 2.2]);
            
            let result = carma_cross_validation(
                times.readonly(),
                values.readonly(),
                2,
                1,
                3,
                Some(42),
            );
            
            assert!(result.is_ok());
            let cv_result = result.unwrap();
            
            assert!(cv_result.mean_score.is_finite());
            assert!(cv_result.std_score >= 0.0);
            assert!(!cv_result.fold_scores.is_empty());
            assert!(cv_result.fold_scores.iter().all(|&score| score.is_finite()));
        });
    }
}