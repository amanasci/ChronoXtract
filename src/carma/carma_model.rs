use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use nalgebra::{DVector, DMatrix};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur in CARMA operations
#[derive(Error, Debug)]
pub enum CarmaError {
    #[error("Invalid model parameters: {0}")]
    InvalidParameters(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("Convergence failed: {0}")]
    ConvergenceError(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

impl From<CarmaError> for PyErr {
    fn from(err: CarmaError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

impl From<PyErr> for CarmaError {
    fn from(err: PyErr) -> CarmaError {
        CarmaError::NumericalError(err.to_string())
    }
}

impl From<argmin::core::Error> for CarmaError {
    fn from(err: argmin::core::Error) -> CarmaError {
        CarmaError::ConvergenceError(err.to_string())
    }
}

/// CARMA model structure
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaModel {
    #[pyo3(get, set)]
    pub p: usize,
    #[pyo3(get, set)]
    pub q: usize,
    #[pyo3(get, set)]
    pub ar_coeffs: Vec<f64>,
    #[pyo3(get, set)]
    pub ma_coeffs: Vec<f64>,
    #[pyo3(get, set)]
    pub sigma: f64,
}

#[pymethods]
impl CarmaModel {
    #[new]
    pub fn new(p: usize, q: usize) -> PyResult<Self> {
        if p == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("p must be greater than 0"));
        }
        if q >= p {
            return Err(pyo3::exceptions::PyValueError::new_err("q must be less than p"));
        }
        
        Ok(CarmaModel {
            p,
            q,
            ar_coeffs: vec![0.0; p],
            ma_coeffs: vec![0.0; q + 1], // q+1 to include coefficient for current time
            sigma: 1.0,
        })
    }
    
    /// Check if the model is properly configured
    pub fn is_valid(&self) -> bool {
        self.p > 0 && 
        self.q < self.p && 
        self.ar_coeffs.len() == self.p && 
        self.ma_coeffs.len() == self.q + 1 &&
        self.sigma > 0.0
    }
    
    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.p + self.q + 1 + 1 // AR + MA + sigma
    }
    
    pub fn __repr__(&self) -> String {
        format!("CarmaModel(p={}, q={}, sigma={:.4})", self.p, self.q, self.sigma)
    }
}

/// Result structure for CARMA fitting
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaFitResult {
    #[pyo3(get)]
    pub model: CarmaModel,
    #[pyo3(get)]
    pub loglikelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub parameter_errors: Vec<f64>,
    #[pyo3(get)]
    pub convergence_info: HashMap<String, f64>,
}

#[pymethods]
impl CarmaFitResult {
    pub fn __repr__(&self) -> String {
        format!("CarmaFitResult(p={}, q={}, loglik={:.4}, AIC={:.4}, BIC={:.4})", 
                self.model.p, self.model.q, self.loglikelihood, self.aic, self.bic)
    }
}

/// Result structure for CARMA MCMC
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaMCMCResult {
    #[pyo3(get)]
    pub samples: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub acceptance_rate: f64,
    #[pyo3(get)]
    pub effective_sample_size: Vec<f64>,
    #[pyo3(get)]
    pub rhat: Vec<f64>,
}

#[pymethods]
impl CarmaMCMCResult {
    pub fn __repr__(&self) -> String {
        format!("CarmaMCMCResult(n_samples={}, acceptance_rate={:.4})", 
                self.samples.len(), self.acceptance_rate)
    }
}

/// Prediction result structure
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaPrediction {
    #[pyo3(get)]
    pub mean: Vec<f64>,
    #[pyo3(get)]
    pub variance: Vec<f64>,
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
}

#[pymethods]
impl CarmaPrediction {
    pub fn __repr__(&self) -> String {
        format!("CarmaPrediction(n_predictions={})", self.mean.len())
    }
}

/// Kalman filter result structure
#[pyclass]
#[derive(Clone, Debug)]
pub struct KalmanResult {
    #[pyo3(get)]
    pub filtered_mean: Vec<f64>,
    #[pyo3(get)]
    pub filtered_variance: Vec<f64>,
    #[pyo3(get)]
    pub predicted_mean: Vec<f64>,
    #[pyo3(get)]
    pub predicted_variance: Vec<f64>,
    #[pyo3(get)]
    pub loglikelihood: f64,
}

#[pymethods]
impl KalmanResult {
    pub fn __repr__(&self) -> String {
        format!("KalmanResult(n_points={}, loglik={:.4})", 
                self.filtered_mean.len(), self.loglikelihood)
    }
}

/// State-space model representation
#[pyclass]
#[derive(Clone, Debug)]
pub struct StateSpaceModel {
    #[pyo3(get)]
    pub transition_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub observation_vector: Vec<f64>,
    #[pyo3(get)]
    pub process_noise_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub observation_noise: f64,
}

#[pymethods]
impl StateSpaceModel {
    pub fn __repr__(&self) -> String {
        format!("StateSpaceModel(dimension={})", self.transition_matrix.len())
    }
}

/// Residuals analysis result
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaResiduals {
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub standardized_residuals: Vec<f64>,
    #[pyo3(get)]
    pub ljung_box_statistic: f64,
    #[pyo3(get)]
    pub ljung_box_pvalue: f64,
}

#[pymethods]
impl CarmaResiduals {
    pub fn __repr__(&self) -> String {
        format!("CarmaResiduals(n_residuals={}, ljung_box_pvalue={:.4})", 
                self.residuals.len(), self.ljung_box_pvalue)
    }
}

/// Information criteria result
#[pyclass]
#[derive(Clone, Debug)]
pub struct InformationCriteriaResult {
    #[pyo3(get)]
    pub results: HashMap<String, HashMap<String, f64>>,
    #[pyo3(get)]
    pub best_aic: (usize, usize),
    #[pyo3(get)]
    pub best_bic: (usize, usize),
}

#[pymethods]
impl InformationCriteriaResult {
    pub fn __repr__(&self) -> String {
        format!("InformationCriteriaResult(best_AIC=({}, {}), best_BIC=({}, {}))", 
                self.best_aic.0, self.best_aic.1, self.best_bic.0, self.best_bic.1)
    }
}

/// Cross-validation result
#[pyclass]
#[derive(Clone, Debug)]
pub struct CrossValidationResult {
    #[pyo3(get)]
    pub mean_score: f64,
    #[pyo3(get)]
    pub std_score: f64,
    #[pyo3(get)]
    pub fold_scores: Vec<f64>,
}

#[pymethods]
impl CrossValidationResult {
    pub fn __repr__(&self) -> String {
        format!("CrossValidationResult(mean_score={:.4}, std_score={:.4})", 
                self.mean_score, self.std_score)
    }
}

/// Create a new CARMA model
#[pyfunction]
pub fn carma_model(p: usize, q: usize) -> PyResult<CarmaModel> {
    CarmaModel::new(p, q)
}

/// Set CARMA model parameters
#[pyfunction]
pub fn set_carma_parameters(
    model: &mut CarmaModel,
    ar_coeffs: Vec<f64>,
    ma_coeffs: Vec<f64>,
    sigma: f64
) -> PyResult<()> {
    if ar_coeffs.len() != model.p {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("AR coefficients length {} does not match p={}", ar_coeffs.len(), model.p)
        ));
    }
    if ma_coeffs.len() != model.q + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("MA coefficients length {} does not match q+1={}", ma_coeffs.len(), model.q + 1)
        ));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("sigma must be positive"));
    }
    
    model.ar_coeffs = ar_coeffs;
    model.ma_coeffs = ma_coeffs;
    model.sigma = sigma;
    
    Ok(())
}

// Internal helper functions
impl CarmaModel {
    /// Convert to parameter vector for optimization
    pub fn to_param_vector(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.parameter_count());
        params.extend_from_slice(&self.ar_coeffs);
        params.extend_from_slice(&self.ma_coeffs);
        params.push(self.sigma.ln()); // log-transformed for positivity
        params
    }
    
    /// Create from parameter vector
    pub fn from_param_vector(&mut self, params: &[f64]) -> Result<(), CarmaError> {
        if params.len() != self.parameter_count() {
            return Err(CarmaError::InvalidParameters(
                format!("Expected {} parameters, got {}", self.parameter_count(), params.len())
            ));
        }
        
        let mut idx = 0;
        self.ar_coeffs.copy_from_slice(&params[idx..idx + self.p]);
        idx += self.p;
        
        self.ma_coeffs.copy_from_slice(&params[idx..idx + self.q + 1]);
        idx += self.q + 1;
        
        self.sigma = params[idx].exp(); // exp to ensure positivity
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_carma_model_creation() {
        let model = CarmaModel::new(3, 1).unwrap();
        assert_eq!(model.p, 3);
        assert_eq!(model.q, 1);
        assert_eq!(model.ar_coeffs.len(), 3);
        assert_eq!(model.ma_coeffs.len(), 2);
        assert!(model.sigma > 0.0);
    }
    
    #[test]
    fn test_carma_model_validation() {
        // Valid model
        let model = CarmaModel::new(3, 1).unwrap();
        assert!(model.is_valid());
        
        // Invalid: q >= p
        assert!(CarmaModel::new(2, 2).is_err());
        assert!(CarmaModel::new(2, 3).is_err());
        
        // Invalid: p = 0
        assert!(CarmaModel::new(0, 0).is_err());
    }
    
    #[test]
    fn test_parameter_vector_conversion() {
        let mut model = CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![0.5, -0.3];
        model.ma_coeffs = vec![1.0, 0.4];
        model.sigma = 2.0;
        
        let params = model.to_param_vector();
        assert_eq!(params.len(), 5); // 2 AR + 2 MA + 1 sigma
        
        let mut model2 = CarmaModel::new(2, 1).unwrap();
        model2.from_param_vector(&params).unwrap();
        
        assert_eq!(model2.ar_coeffs, model.ar_coeffs);
        assert_eq!(model2.ma_coeffs, model.ma_coeffs);
        assert!((model2.sigma - model.sigma).abs() < 1e-10);
    }
    
    #[test]
    fn test_set_carma_parameters() {
        let mut model = CarmaModel::new(2, 1).unwrap();
        
        // Valid parameters
        assert!(set_carma_parameters(
            &mut model,
            vec![0.5, -0.3],
            vec![1.0, 0.4],
            2.0
        ).is_ok());
        
        // Invalid AR length
        assert!(set_carma_parameters(
            &mut model,
            vec![0.5],
            vec![1.0, 0.4],
            2.0
        ).is_err());
        
        // Invalid MA length
        assert!(set_carma_parameters(
            &mut model,
            vec![0.5, -0.3],
            vec![1.0],
            2.0
        ).is_err());
        
        // Invalid sigma
        assert!(set_carma_parameters(
            &mut model,
            vec![0.5, -0.3],
            vec![1.0, 0.4],
            -1.0
        ).is_err());
    }
}