//! High-performance CARMA (Continuous-time AutoRegressive Moving Average) module
//! 
//! This module provides a comprehensive implementation of CARMA models for irregularly
//! sampled time series analysis. It is designed for maximum performance using Rust's
//! memory safety and parallelization capabilities.
//!
//! # Key Components
//! 
//! - **Kalman Filter**: Core engine for likelihood calculation and state estimation
//! - **MCMC Sampler**: Bayesian inference with adaptive Metropolis-Hastings and parallel tempering
//! - **Model Selection**: MLE optimization and AICc-based order selection
//! - **Mathematical Foundation**: Robust numerical algorithms for CARMA operations
//!
//! # Usage
//!
//! ```python
//! import chronoxtract as ct
//! import numpy as np
//!
//! # Create irregularly sampled time series
//! times = np.array([0.0, 1.2, 2.8, 4.1, 5.9])
//! values = np.array([1.0, 1.5, 0.8, 1.2, 0.9])
//! errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
//!
//! # Automatic model order selection
//! best_p, best_q = ct.carma_choose_order(times, values, errors, max_p=3, max_q=2)
//!
//! # Fit CARMA model with MLE
//! mle_result = ct.carma_mle(times, values, best_p, best_q, errors)
//!
//! # Bayesian inference with MCMC
//! mcmc_result = ct.carma_mcmc(times, values, best_p, best_q, errors, n_samples=10000)
//!
//! # Make predictions
//! pred_times = np.array([6.0, 7.0, 8.0])
//! predictions = ct.carma_predict(mcmc_result, pred_times)
//! ```

// Core modules
pub mod types;           // Data types and error handling
pub mod kalman;          // Kalman filter implementation
pub mod mcmc;            // MCMC sampler with parallel tempering
pub mod mle;             // Maximum likelihood estimation
pub mod selection;       // Model order selection and AICc
pub mod math;            // Mathematical utilities and algorithms
pub mod predict;         // Prediction functionality

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_carma_module_compiles() {
        // Basic compilation test
        assert!(true);
    }
}