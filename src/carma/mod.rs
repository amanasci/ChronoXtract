//! CARMA (Continuous AutoRegressive Moving Average) Module
//! 
//! This module provides a complete implementation of CARMA time series modeling,
//! including Maximum Likelihood Estimation (MLE) and Markov Chain Monte Carlo (MCMC)
//! parameter estimation methods.
//! 
//! The implementation is designed to match the results of the reference carma_pack
//! implementation by Brandon Kelly.

pub mod model;
pub mod likelihood;
pub mod mle;
pub mod mcmc;
pub mod simulation;
pub mod utils;

// Re-export main functions
pub use model::*;
pub use mle::carma_mle;
pub use mcmc::carma_mcmc;
pub use simulation::{simulate_carma, generate_carma_data, generate_stable_carma_parameters};
pub use utils::{check_carma_stability, carma_characteristic_roots, carma_model_selection, carma_power_spectrum, carma_autocovariance};