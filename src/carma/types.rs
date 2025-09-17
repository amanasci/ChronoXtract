//! Core data types and error handling for CARMA module
//!
//! This module defines the fundamental data structures used throughout the CARMA
//! implementation, including error types, model representations, and result structures.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use nalgebra::{DVector, DMatrix};
use num_complex::Complex64;
use thiserror::Error;

/// Comprehensive error types for CARMA operations
#[derive(Error, Debug)]
pub enum CarmaError {
    #[error("Invalid model parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Numerical instability: {0}")]
    NumericalError(String),
    
    #[error("MCMC convergence failed: {0}")]
    ConvergenceError(String),
    
    #[error("Invalid time series data: {0}")]
    InvalidData(String),
    
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    
    #[error("Matrix operation failed: {0}")]
    LinearAlgebraError(String),
    
    #[error("Model is not stationary: AR roots have positive real parts")]
    NonStationaryError,
    
    #[error("Invalid model order: p={p}, q={q}. Must have p > 0 and q < p")]
    InvalidOrder { p: usize, q: usize },
}

impl From<CarmaError> for PyErr {
    fn from(err: CarmaError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// CARMA model parameters in the standard parameterization
/// 
/// This represents a CARMA(p,q) model with the traditional AR and MA coefficients.
/// The model is defined by:
/// - AR polynomial: α(s) = s^p + α₁s^(p-1) + ... + αₚ
/// - MA polynomial: β(s) = β₀ + β₁s + ... + βₑs^q
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaParams {
    /// Autoregressive order
    #[pyo3(get, set)]
    pub p: usize,
    
    /// Moving average order
    #[pyo3(get, set)]
    pub q: usize,
    
    /// AR coefficients [α₁, α₂, ..., αₚ]
    #[pyo3(get, set)]
    pub ar_coeffs: Vec<f64>,
    
    /// MA coefficients [β₀, β₁, ..., βₑ]
    #[pyo3(get, set)]
    pub ma_coeffs: Vec<f64>,
    
    /// Process noise standard deviation
    #[pyo3(get, set)]
    pub sigma: f64,
}

#[pymethods]
impl CarmaParams {
    #[new]
    pub fn new(p: usize, q: usize) -> PyResult<Self> {
        if p == 0 {
            return Err(CarmaError::InvalidOrder { p, q }.into());
        }
        if q >= p {
            return Err(CarmaError::InvalidOrder { p, q }.into());
        }
        
        Ok(CarmaParams {
            p,
            q,
            ar_coeffs: vec![0.0; p],
            ma_coeffs: vec![0.0; q + 1],
            sigma: 1.0,
        })
    }
    
    /// Validate parameter consistency and stationarity
    pub fn validate(&self) -> PyResult<()> {
        if self.ar_coeffs.len() != self.p {
            return Err(CarmaError::InvalidParameters(
                format!("AR coefficients length {} != p={}", self.ar_coeffs.len(), self.p)
            ).into());
        }
        
        if self.ma_coeffs.len() != self.q + 1 {
            return Err(CarmaError::InvalidParameters(
                format!("MA coefficients length {} != q+1={}", self.ma_coeffs.len(), self.q + 1)
            ).into());
        }
        
        if self.sigma <= 0.0 {
            return Err(CarmaError::InvalidParameters(
                "sigma must be positive".to_string()
            ).into());
        }
        
        // Check stationarity by computing AR roots
        if !self.is_stationary()? {
            return Err(CarmaError::NonStationaryError.into());
        }
        
        Ok(())
    }
    
    /// Check if the model is stationary (all AR roots have negative real parts)
    pub fn is_stationary(&self) -> PyResult<bool> {
        let roots = self.ar_roots()?;
        Ok(roots.iter().all(|root| root.0 < 0.0))
    }
    
    /// Compute the roots of the AR polynomial
    pub fn ar_roots(&self) -> PyResult<Vec<(f64, f64)>> {
        // For now, return a placeholder - will implement proper root finding
        // in the math module
        Ok(vec![(1.0, 0.0); self.p])
    }
    
    pub fn __repr__(&self) -> String {
        format!("CarmaParams(p={}, q={}, sigma={:.4})", self.p, self.q, self.sigma)
    }
}

/// Specialized MCMC parameterization for improved sampling
/// 
/// This parameterization is designed to facilitate efficient MCMC sampling by:
/// - Using ysigma instead of sigma for better scaling
/// - Parameterizing AR polynomial via quadratic factors for stationarity
/// - Including measurement error scaling
#[pyclass]
#[derive(Clone, Debug)]
pub struct McmcParams {
    /// Standard deviation of the CARMA process
    #[pyo3(get, set)]
    pub ysigma: f64,
    
    /// Measurement error scaling factor
    #[pyo3(get, set)]
    pub measerr_scale: f64,
    
    /// Mean of the time series
    #[pyo3(get, set)]
    pub mu: f64,
    
    /// AR quadratic parameters for stationarity enforcement
    #[pyo3(get, set)]
    pub ar_params: Vec<f64>,
    
    /// MA parameters
    #[pyo3(get, set)]
    pub ma_params: Vec<f64>,
    
    /// Model orders
    #[pyo3(get, set)]
    pub p: usize,
    
    #[pyo3(get, set)]
    pub q: usize,
}

#[pymethods]
impl McmcParams {
    #[new]
    pub fn new(p: usize, q: usize) -> PyResult<Self> {
        if p == 0 || q >= p {
            return Err(CarmaError::InvalidOrder { p, q }.into());
        }
        
        Ok(McmcParams {
            ysigma: 1.0,
            measerr_scale: 1.0,
            mu: 0.0,
            ar_params: vec![0.0; p],
            ma_params: vec![0.0; q],
            p,
            q,
        })
    }
    
    /// Convert to standard CARMA parameterization
    pub fn to_carma_params(&self) -> PyResult<CarmaParams> {
        // Convert from MCMC parameterization to standard AR/MA coefficients
        let mut carma = CarmaParams::new(self.p, self.q)?;
        carma.sigma = self.ysigma;
        
        // For now, use a simple mapping
        // In a full implementation, this would involve converting from the
        // quadratic parameterization back to AR/MA polynomials
        for i in 0..self.p.min(self.ar_params.len()) {
            carma.ar_coeffs[i] = self.ar_params[i];
        }
        
        // Set MA coefficients
        carma.ma_coeffs[0] = 1.0; // Convention: first MA coefficient is 1
        for i in 0..self.q.min(self.ma_params.len()) {
            carma.ma_coeffs[i + 1] = self.ma_params[i];
        }
        
        Ok(carma)
    }
    
    pub fn __repr__(&self) -> String {
        format!("McmcParams(p={}, q={}, ysigma={:.4}, mu={:.4})", 
                self.p, self.q, self.ysigma, self.mu)
    }
}

/// Result of CARMA maximum likelihood estimation
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaMLEResult {
    /// Fitted model parameters
    #[pyo3(get)]
    pub params: CarmaParams,
    
    /// Maximum log-likelihood achieved
    #[pyo3(get)]
    pub loglikelihood: f64,
    
    /// Akaike Information Criterion
    #[pyo3(get)]
    pub aic: f64,
    
    /// Corrected Akaike Information Criterion
    #[pyo3(get)]
    pub aicc: f64,
    
    /// Bayesian Information Criterion
    #[pyo3(get)]
    pub bic: f64,
    
    /// Number of optimization iterations
    #[pyo3(get)]
    pub iterations: usize,
    
    /// Optimization convergence status
    #[pyo3(get)]
    pub converged: bool,
    
    /// Estimated parameter covariance matrix
    #[pyo3(get)]
    pub covariance: Option<Py<PyArray2<f64>>>,
}

#[pymethods]
impl CarmaMLEResult {
    pub fn __repr__(&self) -> String {
        format!("CarmaMLEResult(p={}, q={}, loglik={:.4}, AICc={:.4})", 
                self.params.p, self.params.q, self.loglikelihood, self.aicc)
    }
}

/// Result of CARMA MCMC sampling
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaMCMCResult {
    /// MCMC parameter samples
    #[pyo3(get)]
    pub samples: Py<PyArray2<f64>>,
    
    /// Log-likelihood values for each sample
    #[pyo3(get)]
    pub loglikelihoods: Py<PyArray1<f64>>,
    
    /// Acceptance rate during sampling
    #[pyo3(get)]
    pub acceptance_rate: f64,
    
    /// R-hat convergence diagnostic for each parameter
    #[pyo3(get)]
    pub rhat: Py<PyArray1<f64>>,
    
    /// Effective sample size for each parameter
    #[pyo3(get)]
    pub effective_sample_size: Py<PyArray1<f64>>,
    
    /// Number of samples (post burn-in)
    #[pyo3(get)]
    pub n_samples: usize,
    
    /// Number of burn-in samples discarded
    #[pyo3(get)]
    pub n_burn: usize,
    
    /// Model orders
    #[pyo3(get)]
    pub p: usize,
    
    #[pyo3(get)]
    pub q: usize,
}

#[pymethods]
impl CarmaMCMCResult {
    pub fn __repr__(&self) -> String {
        format!("CarmaMCMCResult(p={}, q={}, n_samples={}, acceptance_rate={:.3})", 
                self.p, self.q, self.n_samples, self.acceptance_rate)
    }
}

/// Result of CARMA model order selection
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaOrderResult {
    /// Best model orders
    #[pyo3(get)]
    pub best_p: usize,
    
    #[pyo3(get)]
    pub best_q: usize,
    
    /// AICc value for the best model
    #[pyo3(get)]
    pub best_aicc: f64,
    
    /// Grid of AICc values for all tested (p,q) combinations
    #[pyo3(get)]
    pub aicc_grid: Py<PyArray2<f64>>,
    
    /// Tested p values
    #[pyo3(get)]
    pub p_values: Py<PyArray1<usize>>,
    
    /// Tested q values  
    #[pyo3(get)]
    pub q_values: Py<PyArray1<usize>>,
}

#[pymethods]
impl CarmaOrderResult {
    pub fn __repr__(&self) -> String {
        format!("CarmaOrderResult(best_p={}, best_q={}, AICc={:.4})", 
                self.best_p, self.best_q, self.best_aicc)
    }
}

/// Prediction result from CARMA model
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaPrediction {
    /// Prediction times
    #[pyo3(get)]
    pub times: Py<PyArray1<f64>>,
    
    /// Predicted mean values
    #[pyo3(get)]
    pub means: Py<PyArray1<f64>>,
    
    /// Prediction standard deviations
    #[pyo3(get)]
    pub std_devs: Py<PyArray1<f64>>,
    
    /// Lower confidence bounds
    #[pyo3(get)]
    pub lower_bounds: Py<PyArray1<f64>>,
    
    /// Upper confidence bounds
    #[pyo3(get)]
    pub upper_bounds: Py<PyArray1<f64>>,
    
    /// Confidence level used
    #[pyo3(get)]
    pub confidence_level: f64,
}

#[pymethods]
impl CarmaPrediction {
    pub fn __repr__(&self) -> String {
        format!("CarmaPrediction(n_predictions={}, confidence={:.1}%)", 
                0, // Placeholder - proper implementation would get length
                self.confidence_level * 100.0)
    }
}

/// State-space representation of CARMA model
/// 
/// Represents the CARMA model in state-space form for Kalman filtering.
/// Uses the "rotated" representation where the transition matrix is diagonal.
#[derive(Clone, Debug)]
pub struct StateSpaceModel {
    /// Diagonal transition matrix eigenvalues (AR roots)
    pub lambda: Vec<Complex64>,
    
    /// Observation vector coefficients
    pub observation: DVector<f64>,
    
    /// Process noise covariance matrix
    pub process_noise_cov: DMatrix<f64>,
    
    /// State dimension (equals p)
    pub state_dim: usize,
    
    /// Stationary state covariance
    pub stationary_cov: DMatrix<f64>,
}

impl StateSpaceModel {
    /// Create new state-space model from CARMA parameters
    pub fn new(params: &CarmaParams) -> Result<Self, CarmaError> {
        let p = params.p;
        
        // Compute AR roots (eigenvalues)
        let lambda = compute_ar_roots(&params.ar_coeffs)?;
        
        // Validate stationarity
        if lambda.iter().any(|&root| root.re >= 0.0) {
            return Err(CarmaError::NonStationaryError);
        }
        
        // Compute observation vector from MA coefficients
        let observation = compute_observation_vector(&params.ma_coeffs, &lambda)?;
        
        // Compute process noise covariance
        let process_noise_cov = compute_process_noise_covariance(&lambda, params.sigma)?;
        
        // Compute stationary covariance
        let stationary_cov = compute_stationary_covariance(&lambda, &process_noise_cov)?;
        
        Ok(StateSpaceModel {
            lambda,
            observation,
            process_noise_cov,
            state_dim: p,
            stationary_cov,
        })
    }
}

// Placeholder functions - will be implemented in math module
fn compute_ar_roots(_ar_coeffs: &[f64]) -> Result<Vec<Complex64>, CarmaError> {
    // Placeholder implementation
    Ok(vec![Complex64::new(-1.0, 0.0)])
}

fn compute_observation_vector(_ma_coeffs: &[f64], _lambda: &[Complex64]) -> Result<DVector<f64>, CarmaError> {
    // Placeholder implementation
    Ok(DVector::from_element(1, 1.0))
}

fn compute_process_noise_covariance(_lambda: &[Complex64], _sigma: f64) -> Result<DMatrix<f64>, CarmaError> {
    // Placeholder implementation
    Ok(DMatrix::identity(1, 1))
}

fn compute_stationary_covariance(_lambda: &[Complex64], _process_noise: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    // Placeholder implementation
    Ok(DMatrix::identity(1, 1))
}