use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Error types for CARMA operations
#[derive(Error, Debug)]
pub enum CarmaError {
    #[error("Invalid model order: p must be > 0 and q < p, got p={p}, q={q}")]
    InvalidOrder { p: usize, q: usize },
    
    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },
    
    #[error("Data validation error: {message}")]
    DataValidationError { message: String },
    
    #[error("Numerical error: {message}")]
    NumericalError { message: String },
    
    #[error("Optimization failed: {message}")]
    OptimizationError { message: String },
    
    #[error("MCMC error: {message}")]
    McmcError { message: String },
}

/// CARMA model representation
/// 
/// This struct represents a CARMA(p,q) model with autoregressive and moving average
/// polynomials. The model is defined by the stochastic differential equation:
/// 
/// D^p Y(t) + α_{p-1} D^{p-1} Y(t) + ... + α_0 Y(t) = β_q D^q ε(t) + ... + β_0 ε(t)
/// 
/// where D is the differential operator and ε(t) is white noise.
#[pyclass]
#[derive(Clone, Debug)]
pub struct CarmaModel {
    /// Order of the autoregressive polynomial
    pub p: usize,
    /// Order of the moving average polynomial  
    pub q: usize,
    /// Autoregressive coefficients α_{p-1}, ..., α_0
    pub ar_coeffs: Vec<f64>,
    /// Moving average coefficients β_q, ..., β_0 (β_0 = 1 by convention)
    pub ma_coeffs: Vec<f64>,
    /// Driving noise amplitude σ
    pub sigma: f64,
    /// Mean of the process
    pub mu: f64,
}

#[pymethods]
impl CarmaModel {
    #[new]
    pub fn new(p: usize, q: usize) -> PyResult<Self> {
        if p == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Order p must be greater than 0"
            ));
        }
        
        if q >= p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Order q ({}) must be less than p ({})", q, p)
            ));
        }
        
        Ok(CarmaModel {
            p,
            q,
            ar_coeffs: vec![0.0; p],
            ma_coeffs: vec![0.0; q + 1],
            sigma: 1.0,
            mu: 0.0,
        })
    }
    
    /// Set the AR coefficients
    pub fn set_ar_coeffs(&mut self, coeffs: Vec<f64>) -> PyResult<()> {
        if coeffs.len() != self.p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} AR coefficients, got {}", self.p, coeffs.len())
            ));
        }
        self.ar_coeffs = coeffs;
        Ok(())
    }
    
    /// Set the MA coefficients
    pub fn set_ma_coeffs(&mut self, coeffs: Vec<f64>) -> PyResult<()> {
        if coeffs.len() != self.q + 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} MA coefficients, got {}", self.q + 1, coeffs.len())
            ));
        }
        self.ma_coeffs = coeffs;
        Ok(())
    }
    
    /// Set the driving noise amplitude
    pub fn set_sigma(&mut self, sigma: f64) -> PyResult<()> {
        if sigma <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sigma must be positive"
            ));
        }
        self.sigma = sigma;
        Ok(())
    }
    
    /// Set the mean of the process
    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu;
    }
    
    /// Get the total number of parameters
    pub fn num_params(&self) -> usize {
        self.p + self.q + 1 + 2  // AR + MA + sigma + mu
    }
    
    /// Check if the model is stable (all characteristic roots have negative real parts)
    pub fn is_stable(&self) -> bool {
        // Check stability by examining the characteristic polynomial roots
        let characteristic_poly = self.get_characteristic_polynomial();
        self.check_stability_from_polynomial(&characteristic_poly)
    }
    
    /// Get the characteristic polynomial coefficients
    pub fn get_characteristic_polynomial(&self) -> Vec<f64> {
        // The characteristic polynomial is s^p + α_{p-1} s^{p-1} + ... + α_0 = 0
        let mut poly = vec![0.0; self.p + 1];
        poly[0] = 1.0;  // Coefficient of s^p
        for i in 0..self.p {
            poly[i + 1] = self.ar_coeffs[self.p - 1 - i];
        }
        poly
    }
}

impl CarmaModel {
    /// Convert model parameters to a parameter vector for optimization
    pub fn to_param_vector(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_params());
        
        // Add mean
        params.push(self.mu);
        
        // Add AR coefficients
        params.extend_from_slice(&self.ar_coeffs);
        
        // Add MA coefficients
        params.extend_from_slice(&self.ma_coeffs);
        
        // Add log(sigma) for stability
        params.push(self.sigma.ln());
        
        params
    }
    
    /// Set model parameters from a parameter vector
    pub fn from_param_vector(&mut self, params: &[f64]) -> Result<(), CarmaError> {
        if params.len() != self.num_params() {
            return Err(CarmaError::InvalidParameter {
                message: format!("Expected {} parameters, got {}", self.num_params(), params.len())
            });
        }
        
        let mut idx = 0;
        
        // Set mean
        self.mu = params[idx];
        idx += 1;
        
        // Set AR coefficients
        for i in 0..self.p {
            self.ar_coeffs[i] = params[idx];
            idx += 1;
        }
        
        // Set MA coefficients  
        for i in 0..=self.q {
            self.ma_coeffs[i] = params[idx];
            idx += 1;
        }
        
        // Set sigma (from log space)
        self.sigma = params[idx].exp();
        
        Ok(())
    }
    
    fn check_stability_from_polynomial(&self, _poly: &[f64]) -> bool {
        // Use the Routh-Hurwitz criterion or root finding to check stability
        // For now, use a simple approach for low-order models
        if self.p == 1 {
            return self.ar_coeffs[0] > 0.0;
        } else if self.p == 2 {
            let a0 = self.ar_coeffs[0];
            let a1 = self.ar_coeffs[1];
            return a0 > 0.0 && a1 > 0.0;
        }
        
        // For higher orders, we need to actually compute the roots
        // This is a simplified check - in practice we'd use a proper root finder
        self.ar_coeffs.iter().all(|&x| x > 0.0)
    }
}

/// Maximum Likelihood Estimation result
#[pyclass]
#[derive(Clone, Debug)]
pub struct MleResult {
    /// Fitted CARMA model
    #[pyo3(get)]
    pub model: CarmaModel,
    
    /// Log-likelihood at the optimum
    #[pyo3(get)]
    pub loglikelihood: f64,
    
    /// Akaike Information Criterion
    #[pyo3(get)]
    pub aic: f64,
    
    /// Bayesian Information Criterion
    #[pyo3(get)]
    pub bic: f64,
    
    /// Number of data points
    #[pyo3(get)]
    pub ndata: usize,
    
    /// Convergence flag
    #[pyo3(get)]
    pub converged: bool,
    
    /// Optimization message
    #[pyo3(get)]
    pub message: String,
}

#[pymethods]
impl MleResult {
    #[new]
    pub fn new(
        model: CarmaModel,
        loglikelihood: f64,
        ndata: usize,
        converged: bool,
        message: String
    ) -> Self {
        let k = model.num_params() as f64;
        let n = ndata as f64;
        
        let aic = -2.0 * loglikelihood + 2.0 * k;
        let bic = -2.0 * loglikelihood + k * n.ln();
        
        MleResult {
            model,
            loglikelihood,
            aic,
            bic,
            ndata,
            converged,
            message,
        }
    }
}

/// MCMC sampling result  
#[pyclass]
#[derive(Clone, Debug)]
pub struct McmcResult {
    /// MCMC samples as a 2D array (n_samples x n_params)
    pub samples: Vec<Vec<f64>>,
    
    /// Parameter names
    #[pyo3(get)]
    pub param_names: Vec<String>,
    
    /// Log-likelihood values for each sample
    #[pyo3(get)]
    pub loglikelihoods: Vec<f64>,
    
    /// Acceptance rate
    #[pyo3(get)]
    pub acceptance_rate: f64,
    
    /// Number of burn-in samples
    #[pyo3(get)]
    pub burn_in: usize,
    
    /// Total number of samples generated
    #[pyo3(get)]
    pub total_samples: usize,
}

#[pymethods]
impl McmcResult {
    #[new]
    pub fn new(
        samples: Vec<Vec<f64>>,
        param_names: Vec<String>,
        loglikelihoods: Vec<f64>,
        acceptance_rate: f64,
        burn_in: usize,
        total_samples: usize,
    ) -> Self {
        McmcResult {
            samples,
            param_names,
            loglikelihoods,
            acceptance_rate,
            burn_in,
            total_samples,
        }
    }
    
    /// Get samples for a specific parameter
    pub fn get_samples(&self, param_name: &str) -> PyResult<Vec<f64>> {
        if let Some(idx) = self.param_names.iter().position(|name| name == param_name) {
            Ok(self.samples.iter().map(|sample| sample[idx]).collect())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Parameter '{}' not found", param_name)
            ))
        }
    }
    
    /// Get all samples as a 2D array
    #[getter]
    pub fn all_samples(&self) -> Vec<Vec<f64>> {
        self.samples.clone()
    }
}

/// Create a new CARMA model
#[pyfunction]
pub fn carma_model(p: usize, q: usize) -> PyResult<CarmaModel> {
    CarmaModel::new(p, q)
}

/// Set CARMA model parameters all at once
#[pyfunction]
pub fn set_carma_parameters(
    model: &mut CarmaModel,
    ar_coeffs: Vec<f64>,
    ma_coeffs: Vec<f64>,
    sigma: f64,
    mu: Option<f64>,
) -> PyResult<()> {
    model.set_ar_coeffs(ar_coeffs)?;
    model.set_ma_coeffs(ma_coeffs)?;
    model.set_sigma(sigma)?;
    
    if let Some(mu_val) = mu {
        model.set_mu(mu_val);
    }
    
    Ok(())
}