use crate::carma::model::{CarmaModel, CarmaError, McmcResult};
use crate::carma::likelihood::compute_loglikelihood;
use crate::carma::mle::ParameterBounds;
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Configuration for MCMC sampling
#[derive(Clone, Debug)]
pub struct McmcConfig {
    /// Number of MCMC samples to generate
    pub n_samples: usize,
    /// Number of burn-in samples
    pub burn_in: usize,
    /// Thinning interval
    pub thin: usize,
    /// Number of parallel chains (for parallel tempering)
    pub n_chains: usize,
    /// Temperature ladder for parallel tempering
    pub temperatures: Vec<f64>,
    /// Proposal covariance adaptation parameters
    pub adapt_proposal: bool,
    pub adaptation_interval: usize,
    pub target_acceptance: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for McmcConfig {
    fn default() -> Self {
        McmcConfig {
            n_samples: 1000,
            burn_in: 500,
            thin: 1,
            n_chains: 4,
            temperatures: vec![1.0, 1.5, 2.0, 3.0],
            adapt_proposal: true,
            adaptation_interval: 50,
            target_acceptance: 0.44,
            seed: None,
        }
    }
}

/// MCMC chain state
#[derive(Clone, Debug)]
pub struct ChainState {
    /// Current parameter values
    pub params: Vec<f64>,
    /// Current log-likelihood
    pub loglikelihood: f64,
    /// Current log-prior
    pub logprior: f64,
    /// Current log-posterior
    pub logposterior: f64,
    /// Chain temperature (for parallel tempering)
    pub temperature: f64,
    /// Proposal covariance matrix (flattened)
    pub proposal_cov: Vec<f64>,
    /// Number of accepted proposals
    pub n_accepted: usize,
    /// Total number of proposals
    pub n_proposed: usize,
}

impl ChainState {
    /// Create new chain state with initial parameters
    pub fn new(params: Vec<f64>, temperature: f64) -> Self {
        let n_params = params.len();
        let proposal_cov = vec![0.0; n_params * n_params];
        
        ChainState {
            params,
            loglikelihood: f64::NEG_INFINITY,
            logprior: 0.0,
            logposterior: f64::NEG_INFINITY,
            temperature,
            proposal_cov,
            n_accepted: 0,
            n_proposed: 0,
        }
    }
    
    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_proposed == 0 {
            0.0
        } else {
            self.n_accepted as f64 / self.n_proposed as f64
        }
    }
    
    /// Adapt proposal covariance matrix
    pub fn adapt_proposal_covariance(&mut self, samples: &[Vec<f64>], target_acceptance: f64) {
        if samples.len() < 10 {
            return;
        }
        
        let n_params = self.params.len();
        let current_acceptance = self.acceptance_rate();
        
        // Compute sample covariance
        let sample_cov = compute_sample_covariance(samples);
        
        // Scale based on acceptance rate
        let scale_factor = if current_acceptance < target_acceptance {
            0.8
        } else {
            1.2
        };
        
        // Update proposal covariance
        for i in 0..n_params * n_params {
            self.proposal_cov[i] = sample_cov[i] * scale_factor * scale_factor;
        }
        
        // Add regularization to diagonal
        for i in 0..n_params {
            self.proposal_cov[i * n_params + i] += 1e-6;
        }
    }
}

/// Parallel tempering MCMC sampler for CARMA models
pub struct ParallelTemperingMcmc {
    /// Chain configurations
    pub chains: Vec<ChainState>,
    /// Data for likelihood computation
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub errors: Option<Vec<f64>>,
    /// Model specification
    pub p: usize,
    pub q: usize,
    /// Parameter bounds
    pub bounds: ParameterBounds,
    /// Random number generator
    pub rng: Xoshiro256PlusPlus,
    /// Configuration
    pub config: McmcConfig,
}

impl ParallelTemperingMcmc {
    /// Create new parallel tempering sampler
    pub fn new(
        times: Vec<f64>,
        values: Vec<f64>,
        errors: Option<Vec<f64>>,
        p: usize,
        q: usize,
        config: McmcConfig,
    ) -> Result<Self, CarmaError> {
        // Compute data statistics for bounds
        let mean_val = values.iter().sum::<f64>() / values.len() as f64;
        let var_val = values.iter().map(|&v| (v - mean_val).powi(2)).sum::<f64>() / values.len() as f64;
        let std_val = var_val.sqrt();
        let time_span = times[times.len() - 1] - times[0];
        let typical_time_scale = time_span / times.len() as f64;
        
        let bounds = ParameterBounds::for_carma_model(p, q, std_val, typical_time_scale);
        
        // Initialize random number generator
        let mut rng = if let Some(seed) = config.seed {
            Xoshiro256PlusPlus::seed_from_u64(seed)
        } else {
            Xoshiro256PlusPlus::from_entropy()
        };
        
        // Initialize chains with different starting points
        let mut chains = Vec::new();
        for (i, &temperature) in config.temperatures.iter().enumerate() {
            let start_params = generate_mcmc_starting_parameters(p, q, &bounds, mean_val, std_val, &mut rng);
            let mut chain = ChainState::new(start_params, temperature);
            
            // Initialize proposal covariance
            initialize_proposal_covariance(&mut chain, &bounds);
            
            chains.push(chain);
        }
        
        Ok(ParallelTemperingMcmc {
            chains,
            times,
            values,
            errors,
            p,
            q,
            bounds,
            rng,
            config,
        })
    }
    
    /// Run MCMC sampling
    pub fn sample(&mut self) -> Result<McmcResult, CarmaError> {
        let n_chains = self.chains.len();
        let total_iterations = self.config.burn_in + self.config.n_samples * self.config.thin;
        
        // Storage for samples (only from T=1 chain)
        let mut samples = Vec::new();
        let mut loglikelihoods = Vec::new();
        
        // Storage for adaptation
        let mut adaptation_samples: Vec<Vec<f64>> = Vec::new();
        
        // Evaluate initial states
        for i in 0..self.chains.len() {
            let (times, values, errors, p, q) = (&self.times, &self.values, &self.errors, self.p, self.q);
            evaluate_chain_state(&mut self.chains[i], times, values, errors.as_deref(), p, q)?;
        }
        
        // Main sampling loop
        for iteration in 0..total_iterations {
            // Metropolis-Hastings step for each chain
            for chain_idx in 0..n_chains {
                self.metropolis_hastings_step(chain_idx)?;
            }
            
            // Parallel tempering swaps every 10 iterations
            if iteration % 10 == 0 && n_chains > 1 {
                self.parallel_tempering_swap()?;
            }
            
            // Proposal adaptation
            if self.config.adapt_proposal && iteration % self.config.adaptation_interval == 0 && iteration > 0 {
                self.adapt_proposals(&adaptation_samples);
                adaptation_samples.clear();
            }
            
            // Store samples from T=1 chain after burn-in
            if iteration >= self.config.burn_in && (iteration - self.config.burn_in) % self.config.thin == 0 {
                if let Some(cold_chain) = self.chains.iter().find(|c| (c.temperature - 1.0).abs() < 1e-10) {
                    samples.push(cold_chain.params.clone());
                    loglikelihoods.push(cold_chain.loglikelihood);
                }
            }
            
            // Collect samples for adaptation
            if iteration < self.config.burn_in && iteration % 5 == 0 {
                if let Some(cold_chain) = self.chains.iter().find(|c| (c.temperature - 1.0).abs() < 1e-10) {
                    adaptation_samples.push(cold_chain.params.clone());
                }
            }
        }
        
        // Compute overall acceptance rate from T=1 chain
        let acceptance_rate = if let Some(cold_chain) = self.chains.iter().find(|c| (c.temperature - 1.0).abs() < 1e-10) {
            cold_chain.acceptance_rate()
        } else {
            0.0
        };
        
        // Generate parameter names
        let param_names = generate_parameter_names(self.p, self.q);
        
        Ok(McmcResult::new(
            samples,
            param_names,
            loglikelihoods,
            acceptance_rate,
            self.config.burn_in,
            total_iterations,
        ))
    }
    
    /// Perform one Metropolis-Hastings step
    fn metropolis_hastings_step(&mut self, chain_idx: usize) -> Result<(), CarmaError> {
        let n_params = self.chains[chain_idx].params.len();
        
        // Generate proposal
        let proposal = self.generate_proposal(chain_idx)?;
        
        // Check bounds
        if !self.bounds.is_feasible(&proposal) {
            self.chains[chain_idx].n_proposed += 1;
            return Ok(());
        }
        
        // Evaluate proposal
        let mut proposal_state = ChainState::new(proposal, self.chains[chain_idx].temperature);
        self.evaluate_state(&mut proposal_state)?;
        
        // Metropolis-Hastings acceptance criterion
        let current_posterior = self.chains[chain_idx].logposterior / self.chains[chain_idx].temperature;
        let proposal_posterior = proposal_state.logposterior / proposal_state.temperature;
        
        let log_alpha = proposal_posterior - current_posterior;
        let alpha = log_alpha.exp().min(1.0);
        
        self.chains[chain_idx].n_proposed += 1;
        
        // Accept or reject
        if self.rng.gen::<f64>() < alpha {
            self.chains[chain_idx] = proposal_state;
            self.chains[chain_idx].n_accepted += 1;
        }
        
        Ok(())
    }
    
    /// Generate proposal using multivariate normal
    fn generate_proposal(&mut self, chain_idx: usize) -> Result<Vec<f64>, CarmaError> {
        let current_params = &self.chains[chain_idx].params;
        let n_params = current_params.len();
        
        // Generate standard normal random vector
        let mut z = vec![0.0; n_params];
        for zi in &mut z {
            let normal = Normal::new(0.0, 1.0).unwrap();
            *zi = normal.sample(&mut self.rng);
        }
        
        // Transform using Cholesky decomposition of proposal covariance
        let chol = cholesky_decomposition(&self.chains[chain_idx].proposal_cov, n_params)?;
        let proposal_delta = cholesky_solve(&chol, &z);
        
        // Add to current parameters
        let mut proposal = vec![0.0; n_params];
        for i in 0..n_params {
            proposal[i] = current_params[i] + proposal_delta[i];
        }
        
        Ok(proposal)
    }
    
    /// Parallel tempering chain swap
    fn parallel_tempering_swap(&mut self) -> Result<(), CarmaError> {
        let n_chains = self.chains.len();
        if n_chains < 2 {
            return Ok(());
        }
        
        // Random pair of adjacent chains
        let i = self.rng.gen_range(0..n_chains - 1);
        let j = i + 1;
        
        let temp_i = self.chains[i].temperature;
        let temp_j = self.chains[j].temperature;
        
        let loglik_i = self.chains[i].loglikelihood + self.chains[i].logprior;
        let loglik_j = self.chains[j].loglikelihood + self.chains[j].logprior;
        
        // Swap criterion
        let log_swap_prob: f64 = (1.0 / temp_i - 1.0 / temp_j) * (loglik_j - loglik_i);
        let swap_prob = log_swap_prob.exp().min(1.0);
        
        if self.rng.gen::<f64>() < swap_prob {
            // Swap chain states (but keep temperatures)
            let params_i = self.chains[i].params.clone();
            let loglik_i = self.chains[i].loglikelihood;
            let logprior_i = self.chains[i].logprior;
            
            self.chains[i].params = self.chains[j].params.clone();
            self.chains[i].loglikelihood = self.chains[j].loglikelihood;
            self.chains[i].logprior = self.chains[j].logprior;
            self.chains[i].logposterior = self.chains[j].loglikelihood + self.chains[j].logprior;
            
            self.chains[j].params = params_i;
            self.chains[j].loglikelihood = loglik_i;
            self.chains[j].logprior = logprior_i;
            self.chains[j].logposterior = loglik_i + logprior_i;
        }
        
        Ok(())
    }
    
    /// Evaluate log-likelihood and log-prior for a state
    fn evaluate_state(&self, state: &mut ChainState) -> Result<(), CarmaError> {
        evaluate_chain_state(state, &self.times, &self.values, self.errors.as_deref(), self.p, self.q)
    }
    
    /// Adapt proposal covariances for all chains
    fn adapt_proposals(&mut self, adaptation_samples: &[Vec<f64>]) {
        for chain in &mut self.chains {
            chain.adapt_proposal_covariance(adaptation_samples, self.config.target_acceptance);
        }
    }
}

/// Perform MCMC sampling for CARMA model
#[pyfunction]
pub fn carma_mcmc(
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    p: usize,
    q: usize,
    n_samples: usize,
    errors: Option<PyReadonlyArray1<f64>>,
    burn_in: Option<usize>,
    thin: Option<usize>,
    n_chains: Option<usize>,
    seed: Option<u64>,
) -> PyResult<McmcResult> {
    let times_slice = times.as_slice()?.to_vec();
    let values_slice = values.as_slice()?.to_vec();
    let errors_slice = errors.as_ref().map(|e| e.as_slice().unwrap().to_vec());
    
    // Validate inputs
    if times_slice.len() != values_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Times and values must have the same length"
        ));
    }
    
    if times_slice.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot fit model to empty data"
        ));
    }
    
    if p == 0 || q >= p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid model order: p must be > 0 and q < p, got p={}, q={}", p, q)
        ));
    }
    
    // Set up configuration
    let burn_in_samples = burn_in.unwrap_or(n_samples / 2);
    let thin_interval = thin.unwrap_or(1);
    let num_chains = n_chains.unwrap_or(4);
    
    let temperatures = if num_chains == 1 {
        vec![1.0]
    } else {
        (0..num_chains).map(|i| 1.0 + 2.0 * (i as f64) / (num_chains - 1) as f64).collect()
    };
    
    let config = McmcConfig {
        n_samples,
        burn_in: burn_in_samples,
        thin: thin_interval,
        n_chains: num_chains,
        temperatures,
        seed,
        ..Default::default()
    };
    
    // Run MCMC
    let mut sampler = ParallelTemperingMcmc::new(times_slice, values_slice, errors_slice, p, q, config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    let result = sampler.sample()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(result)
}

/// Helper functions

/// Generate starting parameters for MCMC
fn generate_mcmc_starting_parameters(
    p: usize,
    q: usize,
    bounds: &ParameterBounds,
    data_mean: f64,
    data_std: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<f64> {
    let n_params = 2 + p + q + 1;
    let mut params = vec![0.0; n_params];
    let mut idx = 0;
    
    // Generate parameters using uniform distribution within bounds
    for i in 0..n_params {
        let lower = bounds.lower[i];
        let upper = bounds.upper[i];
        
        if lower.is_finite() && upper.is_finite() {
            let uniform = Uniform::new(lower, upper);
            params[i] = uniform.sample(rng);
        } else {
            // Use reasonable defaults for unbounded parameters
            match idx {
                0 => params[i] = data_mean + Normal::new(0.0, data_std).unwrap().sample(rng), // mu
                _ if idx <= p => params[i] = Normal::new(0.1, 0.3).unwrap().sample(rng), // AR coeffs
                _ if idx <= p + q => {
                    if idx == p + 1 { // First MA coeff
                        params[i] = Normal::new(1.0, 0.2).unwrap().sample(rng);
                    } else {
                        params[i] = Normal::new(0.0, 0.5).unwrap().sample(rng);
                    }
                }
                _ => params[i] = Normal::new((data_std * 0.5).ln(), 0.5).unwrap().sample(rng), // log(sigma)
            }
        }
        idx += 1;
    }
    
    params
}

/// Initialize proposal covariance matrix
fn initialize_proposal_covariance(chain: &mut ChainState, bounds: &ParameterBounds) {
    let n_params = chain.params.len();
    
    // Initialize with diagonal covariance
    for i in 0..n_params {
        for j in 0..n_params {
            if i == j {
                // Diagonal elements based on parameter range
                let range = bounds.upper[i] - bounds.lower[i];
                let scale = if range.is_finite() { range * 0.01 } else { 1.0 };
                chain.proposal_cov[i * n_params + j] = scale * scale;
            } else {
                chain.proposal_cov[i * n_params + j] = 0.0;
            }
        }
    }
}

/// Compute sample covariance matrix
fn compute_sample_covariance(samples: &[Vec<f64>]) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    
    let n_samples = samples.len();
    let n_params = samples[0].len();
    
    // Compute mean
    let mut mean = vec![0.0; n_params];
    for sample in samples {
        for i in 0..n_params {
            mean[i] += sample[i];
        }
    }
    for i in 0..n_params {
        mean[i] /= n_samples as f64;
    }
    
    // Compute covariance
    let mut cov = vec![0.0; n_params * n_params];
    for sample in samples {
        for i in 0..n_params {
            for j in 0..n_params {
                cov[i * n_params + j] += (sample[i] - mean[i]) * (sample[j] - mean[j]);
            }
        }
    }
    
    for i in 0..(n_params * n_params) {
        cov[i] /= (n_samples - 1) as f64;
    }
    
    cov
}

/// Cholesky decomposition
fn cholesky_decomposition(matrix: &[f64], n: usize) -> Result<Vec<f64>, CarmaError> {
    let mut l = vec![0.0; n * n];
    
    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal elements
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[j * n + k] * l[j * n + k];
                }
                let val = matrix[j * n + j] - sum;
                if val <= 0.0 {
                    return Err(CarmaError::NumericalError {
                        message: "Matrix is not positive definite".to_string()
                    });
                }
                l[j * n + j] = val.sqrt();
            } else {
                // Off-diagonal elements
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = (matrix[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    
    Ok(l)
}

/// Solve L * x = b where L is lower triangular
fn cholesky_solve(l: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    
    // Forward substitution
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * n + j] * x[j];
        }
        x[i] = (b[i] - sum) / l[i * n + i];
    }
    
    x
}

/// Generate parameter names for MCMC output
fn generate_parameter_names(p: usize, q: usize) -> Vec<String> {
    let mut names = Vec::new();
    
    names.push("mu".to_string());
    
    for i in 0..p {
        names.push(format!("ar_{}", i));
    }
    
    for i in 0..=q {
        names.push(format!("ma_{}", i));
    }
    
    names.push("log_sigma".to_string());
    
    names
}

/// Helper function to evaluate chain state
fn evaluate_chain_state(
    state: &mut ChainState,
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    p: usize,
    q: usize,
) -> Result<(), CarmaError> {
    // Create model from parameters
    let mut model = CarmaModel::new(p, q).unwrap();
    model.from_param_vector(&state.params)?;
    
    // Compute log-likelihood
    state.loglikelihood = compute_loglikelihood(&model, times, values, errors)?;
    
    // Compute log-prior (uniform within bounds - simplified)
    state.logprior = 0.0;
    
    state.logposterior = state.loglikelihood + state.logprior;
    
    Ok(())
}