//! MCMC sampling for CARMA models with parallel tempering
//!
//! This module implements adaptive Metropolis-Hastings MCMC with parallel
//! tempering for robust Bayesian inference of CARMA model parameters.

use crate::carma::types::{CarmaMCMCResult, McmcParams, CarmaError};
use crate::carma::kalman::compute_loglikelihood;
use crate::carma::math::validate_time_series;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, PyArray2};
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{MultivariateNormal, Continuous};

/// MCMC Chain state for parallel tempering
#[derive(Clone, Debug)]
struct MCMCChain {
    /// Current parameter values
    params: McmcParams,
    /// Current log-likelihood
    loglikelihood: f64,
    /// Current log-prior
    logprior: f64,
    /// Temperature for this chain
    temperature: f64,
    /// Acceptance rate for this chain
    n_accepted: usize,
    /// Total proposals for this chain
    n_total: usize,
    /// Proposal covariance matrix
    proposal_cov: DMatrix<f64>,
    /// Random number generator
    rng: StdRng,
}

impl MCMCChain {
    fn new(params: McmcParams, temperature: f64, seed: u64) -> Self {
        let n_params = params.ar_params.len() + params.ma_params.len() + 3;
        let proposal_cov = DMatrix::identity(n_params, n_params) * 0.01; // Small initial covariance
        
        MCMCChain {
            params,
            loglikelihood: f64::NEG_INFINITY,
            logprior: 0.0,
            temperature,
            n_accepted: 0,
            n_total: 0,
            proposal_cov,
            rng: StdRng::seed_from_u64(seed),
        }
    }
    
    /// Convert McmcParams to parameter vector for sampling
    fn params_to_vector(&self) -> DVector<f64> {
        let mut vec = Vec::new();
        vec.extend_from_slice(&self.params.ar_params);
        vec.extend_from_slice(&self.params.ma_params);
        vec.push(self.params.ysigma.ln()); // Log-transform for positivity
        vec.push(self.params.measerr_scale.ln()); // Log-transform for positivity
        vec.push(self.params.mu);
        DVector::from_vec(vec)
    }
    
    /// Convert parameter vector back to McmcParams
    fn vector_to_params(&self, vec: &DVector<f64>) -> Result<McmcParams, CarmaError> {
        let mut params = self.params.clone();
        let mut idx = 0;
        
        // AR parameters
        for i in 0..params.ar_params.len() {
            params.ar_params[i] = vec[idx];
            idx += 1;
        }
        
        // MA parameters  
        for i in 0..params.ma_params.len() {
            params.ma_params[i] = vec[idx];
            idx += 1;
        }
        
        // ysigma (from log)
        params.ysigma = vec[idx].exp();
        idx += 1;
        
        // measerr_scale (from log)
        params.measerr_scale = vec[idx].exp();
        idx += 1;
        
        // mu
        params.mu = vec[idx];
        
        Ok(params)
    }
    
    /// Compute log-prior for parameters
    fn compute_logprior(&self) -> f64 {
        let mut logprior = 0.0;
        
        // Priors on AR parameters (normal with mean 0, std 2)
        for &ar_param in &self.params.ar_params {
            let normal = statrs::distribution::Normal::new(0.0, 2.0).unwrap();
            logprior += normal.ln_pdf(ar_param);
        }
        
        // Priors on MA parameters (normal with mean 0, std 2)
        for &ma_param in &self.params.ma_params {
            let normal = statrs::distribution::Normal::new(0.0, 2.0).unwrap();
            logprior += normal.ln_pdf(ma_param);
        }
        
        // Prior on ysigma (log-normal, so normal on log(ysigma))
        let normal1 = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        logprior += normal1.ln_pdf(self.params.ysigma.ln());
        
        // Prior on measerr_scale (log-normal)
        let normal2 = statrs::distribution::Normal::new(0.0, 0.5).unwrap();
        logprior += normal2.ln_pdf(self.params.measerr_scale.ln());
        
        // Prior on mu (normal with mean 0, std 5)
        let normal3 = statrs::distribution::Normal::new(0.0, 5.0).unwrap();
        logprior += normal3.ln_pdf(self.params.mu);
        
        logprior
    }
    
    /// Propose new parameters using multivariate normal
    fn propose_parameters(&mut self) -> Result<McmcParams, CarmaError> {
        let current_vec = self.params_to_vector();
        
        // Simple proposal: add multivariate normal noise
        let n_params = current_vec.len();
        let mut proposed_vec = current_vec.clone();
        
        for i in 0..n_params {
            let noise = Normal::new(0.0, self.proposal_cov[(i, i)].sqrt()).unwrap();
            proposed_vec[i] += noise.sample(&mut self.rng);
        }
        
        self.vector_to_params(&proposed_vec)
    }
    
    /// Accept or reject proposal
    fn accept_proposal(&mut self, 
                       new_params: McmcParams, 
                       new_loglikelihood: f64, 
                       new_logprior: f64) -> bool {
        let current_posterior = self.loglikelihood + self.logprior;
        let new_posterior = new_loglikelihood + new_logprior;
        
        // Metropolis acceptance probability with temperature
        let log_alpha = (new_posterior - current_posterior) / self.temperature;
        let alpha = log_alpha.exp().min(1.0);
        
        let accept = self.rng.gen::<f64>() < alpha;
        
        if accept {
            self.params = new_params;
            self.loglikelihood = new_loglikelihood;
            self.logprior = new_logprior;
            self.n_accepted += 1;
        }
        
        self.n_total += 1;
        accept
    }
    
    /// Update proposal covariance during adaptation phase
    fn update_proposal_covariance(&mut self, samples: &[DVector<f64>]) {
        if samples.len() < 10 {
            return; // Need enough samples to estimate covariance
        }
        
        let n = samples.len();
        let n_params = samples[0].len();
        
        // Compute sample mean
        let mut mean = DVector::zeros(n_params);
        for sample in samples {
            mean += sample;
        }
        mean /= n as f64;
        
        // Compute sample covariance
        let mut cov = DMatrix::zeros(n_params, n_params);
        for sample in samples {
            let centered = sample - &mean;
            cov += &centered * centered.transpose();
        }
        cov /= (n - 1) as f64;
        
        // Scale the covariance and add regularization
        self.proposal_cov = cov * 2.38_f64.powi(2) / (n_params as f64) + 
                           DMatrix::identity(n_params, n_params) * 1e-6;
    }
    
    fn acceptance_rate(&self) -> f64 {
        if self.n_total > 0 {
            self.n_accepted as f64 / self.n_total as f64
        } else {
            0.0
        }
    }
}

/// Parallel tempering MCMC sampler
struct ParallelTemperingSampler {
    chains: Vec<MCMCChain>,
    temperatures: Vec<f64>,
    times: Vec<f64>,
    values: Vec<f64>,
    errors: Vec<f64>,
    n_swaps_attempted: usize,
    n_swaps_accepted: usize,
}

impl ParallelTemperingSampler {
    fn new(
        initial_params: McmcParams,
        n_chains: usize,
        times: Vec<f64>,
        values: Vec<f64>,
        errors: Vec<f64>,
        seed: u64,
    ) -> Self {
        // Create temperature ladder
        let mut temperatures = Vec::with_capacity(n_chains);
        for i in 0..n_chains {
            let temp = 1.0 + i as f64 * 0.5; // Temperature ladder: 1.0, 1.5, 2.0, ...
            temperatures.push(temp);
        }
        
        // Initialize chains
        let mut chains = Vec::with_capacity(n_chains);
        for i in 0..n_chains {
            let chain_seed = seed + i as u64 * 1000;
            chains.push(MCMCChain::new(initial_params.clone(), temperatures[i], chain_seed));
        }
        
        ParallelTemperingSampler {
            chains,
            temperatures,
            times,
            values,
            errors,
            n_swaps_attempted: 0,
            n_swaps_accepted: 0,
        }
    }
    
    /// Evaluate log-likelihood for given parameters
    fn evaluate_loglikelihood(&self, params: &McmcParams) -> Result<f64, CarmaError> {
        // Convert MCMC parameters to CARMA parameters
        let carma_params = params.to_carma_params()
            .map_err(|_| CarmaError::InvalidParameters("Failed to convert MCMC params".to_string()))?;
        
        // Adjust for measurement error scaling
        let scaled_errors: Vec<f64> = self.errors.iter()
            .map(|&err| err * params.measerr_scale)
            .collect();
        
        // Subtract mean from values
        let centered_values: Vec<f64> = self.values.iter()
            .map(|&val| val - params.mu)
            .collect();
        
        compute_loglikelihood(&carma_params, &self.times, &centered_values, &scaled_errors)
    }
    
    /// Run one MCMC step for all chains
    fn step(&mut self) -> Result<(), CarmaError> {
        // Step each chain sequentially to avoid borrowing issues
        for i in 0..self.chains.len() {
            // Propose new parameters
            let proposed_params = self.chains[i].propose_parameters()?;
            
            // Evaluate new likelihood and prior
            let new_loglikelihood = self.evaluate_loglikelihood(&proposed_params)?;
            let new_logprior = self.chains[i].compute_logprior();
            
            // Accept or reject
            self.chains[i].accept_proposal(proposed_params, new_loglikelihood, new_logprior);
        }
        
        Ok(())
    }
    
    /// Propose temperature swaps between adjacent chains
    fn propose_swaps(&mut self) {
        for i in 0..(self.chains.len() - 1) {
            let beta_i = 1.0 / self.chains[i].temperature;
            let beta_j = 1.0 / self.chains[i + 1].temperature;
            
            let log_alpha = (beta_i - beta_j) * 
                (self.chains[i + 1].loglikelihood - self.chains[i].loglikelihood);
            
            let alpha = log_alpha.exp().min(1.0);
            
            self.n_swaps_attempted += 1;
            
            if self.chains[0].rng.gen::<f64>() < alpha {
                // Swap the parameter states using split_at_mut to avoid double borrow
                let (left, right) = self.chains.split_at_mut(i + 1);
                std::mem::swap(&mut left[i].params, &mut right[0].params);
                std::mem::swap(&mut left[i].loglikelihood, &mut right[0].loglikelihood);
                std::mem::swap(&mut left[i].logprior, &mut right[0].logprior);
                
                self.n_swaps_accepted += 1;
            }
        }
    }
    
    /// Run adaptation phase to tune proposal covariances
    fn adapt(&mut self, n_adapt: usize) -> Result<(), CarmaError> {
        let mut adaptation_samples: Vec<Vec<DVector<f64>>> = vec![Vec::new(); self.chains.len()];
        
        for iteration in 0..n_adapt {
            self.step()?;
            
            // Store samples for adaptation
            for (chain_idx, chain) in self.chains.iter().enumerate() {
                adaptation_samples[chain_idx].push(chain.params_to_vector());
            }
            
            // Propose swaps every 10 iterations
            if iteration % 10 == 0 {
                self.propose_swaps();
            }
            
            // Update proposal covariances every 100 iterations
            if iteration % 100 == 99 && iteration > 200 {
                for (chain_idx, chain) in self.chains.iter_mut().enumerate() {
                    let recent_samples = &adaptation_samples[chain_idx][adaptation_samples[chain_idx].len().saturating_sub(200)..];
                    chain.update_proposal_covariance(recent_samples);
                }
            }
        }
        
        Ok(())
    }
    
    /// Run sampling phase
    fn sample(&mut self, n_samples: usize) -> Result<Vec<Vec<DVector<f64>>>, CarmaError> {
        let mut samples: Vec<Vec<DVector<f64>>> = vec![Vec::new(); self.chains.len()];
        
        for iteration in 0..n_samples {
            self.step()?;
            
            // Store samples from all chains
            for (chain_idx, chain) in self.chains.iter().enumerate() {
                samples[chain_idx].push(chain.params_to_vector());
            }
            
            // Propose swaps every 5 iterations
            if iteration % 5 == 0 {
                self.propose_swaps();
            }
        }
        
        Ok(samples)
    }
    
    fn swap_acceptance_rate(&self) -> f64 {
        if self.n_swaps_attempted > 0 {
            self.n_swaps_accepted as f64 / self.n_swaps_attempted as f64
        } else {
            0.0
        }
    }
}
/// MCMC sampling with parallel tempering
/// 
/// # Arguments
/// * `times` - Observation times
/// * `values` - Observed values
/// * `errors` - Measurement error standard deviations
/// * `p` - Autoregressive order
/// * `q` - Moving average order
/// * `n_samples` - Number of samples to generate (post burn-in)
/// * `n_burn` - Number of burn-in samples
/// * `n_chains` - Number of parallel tempering chains
/// 
/// # Returns
/// MCMC sampling results
#[pyfunction]
pub fn carma_mcmc(
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    errors: PyReadonlyArray1<f64>,
    p: usize,
    q: usize,
    n_samples: usize,
    n_burn: Option<usize>,
    n_chains: Option<usize>,
    seed: Option<u64>,
) -> PyResult<CarmaMCMCResult> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let errors_slice = errors.as_slice()?;
    
    // Validate input data
    validate_time_series(times_slice, values_slice, Some(errors_slice))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    let n_burn = n_burn.unwrap_or(n_samples / 4);
    let n_chains = n_chains.unwrap_or(4);
    let seed = seed.unwrap_or(42);
    
    // Initialize MCMC parameters with reasonable defaults
    let mut initial_params = McmcParams::new(p, q)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    // Set reasonable initial values
    initial_params.ysigma = values_slice.iter().map(|x| x * x).sum::<f64>().sqrt() / (values_slice.len() as f64).sqrt();
    initial_params.mu = values_slice.iter().sum::<f64>() / values_slice.len() as f64;
    initial_params.measerr_scale = 1.0;
    
    // Initialize AR and MA parameters with small random values
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..p {
        initial_params.ar_params[i] = rng.gen_range(-0.1..0.1);
    }
    for i in 0..q {
        initial_params.ma_params[i] = rng.gen_range(-0.1..0.1);
    }
    
    // Create parallel tempering sampler
    let mut sampler = ParallelTemperingSampler::new(
        initial_params,
        n_chains,
        times_slice.to_vec(),
        values_slice.to_vec(),
        errors_slice.to_vec(),
        seed,
    );
    
    // Run adaptation phase
    println!("Running MCMC adaptation phase with {} iterations...", n_burn);
    sampler.adapt(n_burn)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // Run sampling phase
    println!("Running MCMC sampling phase with {} iterations...", n_samples);
    let all_samples = sampler.sample(n_samples)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // Extract samples from the cold chain (temperature = 1.0)
    let cold_chain_samples = &all_samples[0];
    let n_params = p + q + 3;
    
    // Convert samples to numpy arrays
    Python::with_gil(|py| {
        let mut samples_2d = Vec::with_capacity(n_samples * n_params);
        let mut loglikelihoods = Vec::with_capacity(n_samples);
        
        for sample_vec in cold_chain_samples {
            for &param in sample_vec.iter() {
                samples_2d.push(param);
            }
            
            // Compute log-likelihood for this sample
            let params = sampler.chains[0].vector_to_params(sample_vec)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let loglik = sampler.evaluate_loglikelihood(&params)
                .unwrap_or(f64::NEG_INFINITY);
            loglikelihoods.push(loglik);
        }
        
        let samples_array = PyArray2::from_vec2(py, &vec![samples_2d])
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to create samples array"))?
            .reshape((n_samples, n_params))
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to reshape samples array"))?;
            
        let loglikelihoods_array = PyArray1::from_vec(py, loglikelihoods);
        
        // Compute convergence diagnostics
        let (rhat_values, ess_values) = compute_convergence_diagnostics(&all_samples, n_params);
        let rhat_array = PyArray1::from_vec(py, rhat_values);
        let ess_array = PyArray1::from_vec(py, ess_values);
        
        // Get acceptance rate from the cold chain
        let acceptance_rate = sampler.chains[0].acceptance_rate();
        
        println!("MCMC completed successfully!");
        println!("Cold chain acceptance rate: {:.3}", acceptance_rate);
        println!("Temperature swap acceptance rate: {:.3}", sampler.swap_acceptance_rate());
        
        Ok(CarmaMCMCResult {
            samples: samples_array.into(),
            loglikelihoods: loglikelihoods_array.into(),
            acceptance_rate,
            rhat: rhat_array.into(),
            effective_sample_size: ess_array.into(),
            n_samples,
            n_burn,
            p,
            q,
        })
    })
}

/// Compute R-hat and effective sample size diagnostics
fn compute_convergence_diagnostics(
    all_samples: &[Vec<DVector<f64>>], 
    n_params: usize
) -> (Vec<f64>, Vec<f64>) {
    let n_chains = all_samples.len();
    let n_samples = all_samples[0].len();
    
    if n_chains < 2 {
        // Can't compute R-hat with only one chain
        return (vec![1.0; n_params], vec![n_samples as f64; n_params]);
    }
    
    let mut rhat_values = Vec::with_capacity(n_params);
    let mut ess_values = Vec::with_capacity(n_params);
    
    for param_idx in 0..n_params {
        // Extract parameter values across all chains
        let mut chain_means = Vec::with_capacity(n_chains);
        let mut chain_vars = Vec::with_capacity(n_chains);
        let mut all_values = Vec::new();
        
        for chain_samples in all_samples {
            let param_values: Vec<f64> = chain_samples.iter()
                .map(|sample| sample[param_idx])
                .collect();
            
            let mean = param_values.iter().sum::<f64>() / param_values.len() as f64;
            let variance = param_values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (param_values.len() - 1) as f64;
            
            chain_means.push(mean);
            chain_vars.push(variance);
            all_values.extend(param_values);
        }
        
        // Compute R-hat
        let overall_mean = chain_means.iter().sum::<f64>() / n_chains as f64;
        let between_var = n_samples as f64 * chain_means.iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>() / (n_chains - 1) as f64;
        let within_var = chain_vars.iter().sum::<f64>() / n_chains as f64;
        
        let var_est = ((n_samples - 1) as f64 * within_var + between_var) / n_samples as f64;
        let rhat = if within_var > 0.0 {
            (var_est / within_var).sqrt()
        } else {
            1.0
        };
        
        // Simple ESS estimate (can be improved)
        let ess = if var_est > 0.0 {
            (all_values.len() as f64) / (1.0 + 2.0 * rhat)
        } else {
            all_values.len() as f64
        };
        
        rhat_values.push(rhat);
        ess_values.push(ess);
    }
    
    (rhat_values, ess_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mcmc_placeholder() {
        // Basic test that the placeholder compiles
        assert!(true);
    }
}