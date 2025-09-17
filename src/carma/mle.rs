use crate::carma::model::{CarmaModel, CarmaError, MleResult};
use crate::carma::likelihood::compute_loglikelihood;
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Configuration for MLE optimization
#[derive(Clone, Debug)]
pub struct MleConfig {
    /// Maximum number of optimization iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of random starting points to try
    pub n_trials: usize,
    /// Number of parallel jobs (for future implementation)
    pub n_jobs: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for MleConfig {
    fn default() -> Self {
        MleConfig {
            max_iter: 1000,
            tolerance: 1e-6,
            n_trials: 10,
            n_jobs: 1,
            seed: None,
        }
    }
}

/// Parameter bounds for optimization
#[derive(Clone, Debug)]
pub struct ParameterBounds {
    /// Lower bounds for each parameter
    pub lower: Vec<f64>,
    /// Upper bounds for each parameter
    pub upper: Vec<f64>,
}

impl ParameterBounds {
    /// Create parameter bounds for a CARMA model
    pub fn for_carma_model(p: usize, q: usize, data_std: f64, time_scale: f64) -> Self {
        let n_params = 2 + p + q + 1;  // mu + AR + MA + log(sigma)
        let mut lower = vec![-f64::INFINITY; n_params];
        let mut upper = vec![f64::INFINITY; n_params];
        
        let mut idx = 0;
        
        // Mean parameter bounds (based on data range)
        lower[idx] = -10.0 * data_std;
        upper[idx] = 10.0 * data_std;
        idx += 1;
        
        // AR coefficient bounds (for stability)
        for _ in 0..p {
            lower[idx] = -100.0 / time_scale;
            upper[idx] = 100.0 / time_scale;
            idx += 1;
        }
        
        // MA coefficient bounds
        for i in 0..=q {
            if i == 0 {
                // First MA coefficient is typically normalized to 1
                lower[idx] = 0.1;
                upper[idx] = 10.0;
            } else {
                lower[idx] = -10.0;
                upper[idx] = 10.0;
            }
            idx += 1;
        }
        
        // Log(sigma) bounds
        lower[idx] = (0.01 * data_std).ln();
        upper[idx] = (10.0 * data_std).ln();
        
        ParameterBounds { lower, upper }
    }
    
    /// Project parameters to bounds
    pub fn project(&self, params: &mut [f64]) {
        for (i, param) in params.iter_mut().enumerate() {
            *param = param.max(self.lower[i]).min(self.upper[i]);
        }
    }
    
    /// Check if parameters are within bounds
    pub fn is_feasible(&self, params: &[f64]) -> bool {
        params.iter().enumerate().all(|(i, &param)| {
            param >= self.lower[i] && param <= self.upper[i]
        })
    }
}

/// Simple L-BFGS-B implementation for CARMA optimization
pub struct LbfgsOptimizer {
    /// Memory parameter for L-BFGS
    pub memory: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Line search parameters
    pub c1: f64,
    pub c2: f64,
}

impl Default for LbfgsOptimizer {
    fn default() -> Self {
        LbfgsOptimizer {
            memory: 10,
            max_iter: 1000,
            tolerance: 1e-6,
            c1: 1e-4,
            c2: 0.9,
        }
    }
}

impl LbfgsOptimizer {
    /// Optimize the negative log-likelihood
    pub fn minimize<F, G>(
        &self,
        mut x: Vec<f64>,
        bounds: &ParameterBounds,
        mut f: F,
        mut g: G,
    ) -> OptimizationResult
    where
        F: FnMut(&[f64]) -> f64,
        G: FnMut(&[f64], &mut [f64]),
    {
        let n = x.len();
        let mut gradient = vec![0.0; n];
        let mut search_direction = vec![0.0; n];
        
        // Storage for L-BFGS
        let mut s_history: Vec<Vec<f64>> = Vec::new();
        let mut y_history: Vec<Vec<f64>> = Vec::new();
        let mut rho_history: Vec<f64> = Vec::new();
        
        let mut prev_gradient = vec![0.0; n];
        let mut prev_x = x.clone();
        
        bounds.project(&mut x);
        let mut fx = f(&x);
        g(&x, &mut gradient);
        
        let mut converged = false;
        let mut iterations = 0;
        
        for iter in 0..self.max_iter {
            iterations = iter + 1;
            
            // Check convergence
            let grad_norm = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.tolerance {
                converged = true;
                break;
            }
            
            // Compute search direction using L-BFGS
            self.compute_search_direction(
                &gradient,
                &mut search_direction,
                &s_history,
                &y_history,
                &rho_history,
            );
            
            // Perform line search
            let alpha = self.line_search(&x, &search_direction, fx, &gradient, bounds, &mut f);
            
            // Update parameters
            prev_x.copy_from_slice(&x);
            prev_gradient.copy_from_slice(&gradient);
            
            for i in 0..n {
                x[i] += alpha * search_direction[i];
            }
            bounds.project(&mut x);
            
            // Evaluate function and gradient at new point
            let new_fx = f(&x);
            g(&x, &mut gradient);
            
            // Update L-BFGS history
            if iter > 0 {
                let s: Vec<f64> = x.iter().zip(&prev_x).map(|(&xi, &xi_prev)| xi - xi_prev).collect();
                let y: Vec<f64> = gradient.iter().zip(&prev_gradient).map(|(&gi, &gi_prev)| gi - gi_prev).collect();
                
                let sy: f64 = s.iter().zip(&y).map(|(&si, &yi)| si * yi).sum();
                
                if sy > 1e-10 {
                    let rho = 1.0 / sy;
                    
                    s_history.push(s);
                    y_history.push(y);
                    rho_history.push(rho);
                    
                    // Keep only the last 'memory' vectors
                    if s_history.len() > self.memory {
                        s_history.remove(0);
                        y_history.remove(0);
                        rho_history.remove(0);
                    }
                }
            }
            
            fx = new_fx;
        }
        
        OptimizationResult {
            x,
            fx,
            iterations,
            converged,
            message: if converged {
                "Optimization converged".to_string()
            } else {
                "Maximum iterations reached".to_string()
            },
        }
    }
    
    fn compute_search_direction(
        &self,
        gradient: &[f64],
        search_direction: &mut [f64],
        s_history: &[Vec<f64>],
        y_history: &[Vec<f64>],
        rho_history: &[f64],
    ) {
        let n = gradient.len();
        let m = s_history.len();
        
        // Start with steepest descent direction
        for i in 0..n {
            search_direction[i] = -gradient[i];
        }
        
        if m == 0 {
            return;
        }
        
        // Two-loop recursion for L-BFGS
        let mut alpha = vec![0.0; m];
        
        // First loop (backward)
        for i in (0..m).rev() {
            let rho = rho_history[i];
            alpha[i] = rho * s_history[i].iter().zip(search_direction.iter()).map(|(&s, &d)| s * d).sum::<f64>();
            
            for j in 0..n {
                search_direction[j] -= alpha[i] * y_history[i][j];
            }
        }
        
        // Scale by initial Hessian approximation
        if let (Some(s_last), Some(y_last)) = (s_history.last(), y_history.last()) {
            let sy: f64 = s_last.iter().zip(y_last).map(|(&s, &y)| s * y).sum();
            let yy: f64 = y_last.iter().map(|&y| y * y).sum();
            
            if yy > 1e-10 && sy > 1e-10 {
                let gamma = sy / yy;
                for d in search_direction.iter_mut() {
                    *d *= gamma;
                }
            }
        }
        
        // Second loop (forward)
        for i in 0..m {
            let rho = rho_history[i];
            let beta = rho * y_history[i].iter().zip(search_direction.iter()).map(|(&y, &d)| y * d).sum::<f64>();
            
            for j in 0..n {
                search_direction[j] += (alpha[i] - beta) * s_history[i][j];
            }
        }
    }
    
    fn line_search<F>(
        &self,
        x: &[f64],
        direction: &[f64],
        fx: f64,
        gradient: &[f64],
        bounds: &ParameterBounds,
        f: &mut F,
    ) -> f64
    where
        F: FnMut(&[f64]) -> f64,
    {
        let n = x.len();
        let mut x_new = vec![0.0; n];
        
        // Initial step size
        let mut alpha: f64 = 1.0;
        let max_alpha = compute_max_step_size(x, direction, bounds);
        alpha = alpha.min(max_alpha * 0.99);
        
        // Directional derivative
        let derphi0: f64 = gradient.iter().zip(direction).map(|(&g, &d)| g * d).sum();
        
        if derphi0 >= 0.0 {
            return 0.0;  // Not a descent direction
        }
        
        // Backtracking line search
        for _ in 0..20 {
            // Compute trial point
            for i in 0..n {
                x_new[i] = x[i] + alpha * direction[i];
            }
            bounds.project(&mut x_new);
            
            let fx_new = f(&x_new);
            
            // Armijo condition
            if fx_new <= fx + self.c1 * alpha * derphi0 {
                return alpha;
            }
            
            alpha *= 0.5;
            
            if alpha < 1e-10 {
                break;
            }
        }
        
        alpha.max(1e-10)
    }
}

/// Compute maximum step size to stay within bounds
fn compute_max_step_size(x: &[f64], direction: &[f64], bounds: &ParameterBounds) -> f64 {
    let mut max_alpha = f64::INFINITY;
    
    for (i, (&xi, &di)) in x.iter().zip(direction).enumerate() {
        if di > 1e-10 {
            let alpha_upper = (bounds.upper[i] - xi) / di;
            max_alpha = max_alpha.min(alpha_upper);
        } else if di < -1e-10 {
            let alpha_lower = (bounds.lower[i] - xi) / di;
            max_alpha = max_alpha.min(alpha_lower);
        }
    }
    
    max_alpha.max(0.0)
}

/// Result of optimization
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Optimized parameters
    pub x: Vec<f64>,
    /// Function value at optimum
    pub fx: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence flag
    pub converged: bool,
    /// Status message
    pub message: String,
}

/// Perform Maximum Likelihood Estimation for CARMA model
#[pyfunction]
pub fn carma_mle(
    times: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    p: usize,
    q: usize,
    errors: Option<PyReadonlyArray1<f64>>,
    max_iter: Option<usize>,
    tolerance: Option<f64>,
    n_trials: Option<usize>,
    seed: Option<u64>,
) -> PyResult<MleResult> {
    let times_slice = times.as_slice()?;
    let values_slice = values.as_slice()?;
    let errors_slice = errors.as_ref().map(|e| e.as_slice()).transpose()?;
    
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
    let config = MleConfig {
        max_iter: max_iter.unwrap_or(1000),
        tolerance: tolerance.unwrap_or(1e-6),
        n_trials: n_trials.unwrap_or(10),
        n_jobs: 1,
        seed,
    };
    
    // Perform MLE
    let result = fit_carma_mle(times_slice, values_slice, errors_slice, p, q, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(result)
}

/// Internal MLE fitting function
pub fn fit_carma_mle(
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    p: usize,
    q: usize,
    config: &MleConfig,
) -> Result<MleResult, CarmaError> {
    let n_data = times.len();
    
    // Compute data statistics for parameter bounds
    let mean_val = values.iter().sum::<f64>() / n_data as f64;
    let var_val = values.iter().map(|&v| (v - mean_val).powi(2)).sum::<f64>() / n_data as f64;
    let std_val = var_val.sqrt();
    
    let time_span = times[times.len() - 1] - times[0];
    let typical_time_scale = time_span / n_data as f64;
    
    // Set up parameter bounds
    let bounds = ParameterBounds::for_carma_model(p, q, std_val, typical_time_scale);
    
    // Set up random number generator
    let mut rng = if let Some(seed_val) = config.seed {
        Xoshiro256PlusPlus::seed_from_u64(seed_val)
    } else {
        Xoshiro256PlusPlus::from_entropy()
    };
    
    let mut best_result: Option<OptimizationResult> = None;
    let mut best_model: Option<CarmaModel> = None;
    
    // Try multiple random starting points
    for trial in 0..config.n_trials {
        // Generate random starting parameters
        let start_params = generate_starting_parameters(p, q, &bounds, mean_val, std_val, &mut rng);
        
        // Set up objective function (negative log-likelihood)
        let objective = |params: &[f64]| {
            let mut model = CarmaModel::new(p, q).unwrap();
            if model.from_param_vector(params).is_err() {
                return f64::INFINITY;
            }
            
            match compute_loglikelihood(&model, times, values, errors) {
                Ok(loglik) => -loglik,  // Minimize negative log-likelihood
                Err(_) => f64::INFINITY,
            }
        };
        
        // Set up gradient function (numerical differentiation)
        let gradient = |params: &[f64], grad: &mut [f64]| {
            let eps = 1e-8;
            let f0 = objective(params);
            
            for i in 0..params.len() {
                let mut params_plus = params.to_vec();
                params_plus[i] += eps;
                let f_plus = objective(&params_plus);
                
                grad[i] = (f_plus - f0) / eps;
            }
        };
        
        // Run optimization
        let optimizer = LbfgsOptimizer {
            max_iter: config.max_iter,
            tolerance: config.tolerance,
            ..Default::default()
        };
        
        let result = optimizer.minimize(start_params, &bounds, objective, gradient);
        
        // Check if this is the best result so far
        if let Some(ref best) = best_result {
            if result.fx < best.fx && result.fx.is_finite() {
                best_result = Some(result);
                
                // Create the best model
                let mut model = CarmaModel::new(p, q).unwrap();
                if model.from_param_vector(&best_result.as_ref().unwrap().x).is_ok() {
                    best_model = Some(model);
                }
            }
        } else if result.fx.is_finite() {
            best_result = Some(result);
            
            // Create the model
            let mut model = CarmaModel::new(p, q).unwrap();
            if model.from_param_vector(&best_result.as_ref().unwrap().x).is_ok() {
                best_model = Some(model);
            }
        }
    }
    
    // Return the best result
    if let (Some(opt_result), Some(model)) = (best_result, best_model) {
        let loglikelihood = -opt_result.fx;
        
        Ok(MleResult::new(
            model,
            loglikelihood,
            n_data,
            opt_result.converged,
            opt_result.message,
        ))
    } else {
        Err(CarmaError::OptimizationError {
            message: "All optimization trials failed".to_string()
        })
    }
}

/// Generate random starting parameters for optimization
fn generate_starting_parameters(
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
    
    // Mean parameter (close to data mean)
    let mean_dist = Normal::new(data_mean, data_std * 0.1).unwrap();
    params[idx] = mean_dist.sample(rng).max(bounds.lower[idx]).min(bounds.upper[idx]);
    idx += 1;
    
    // AR coefficients (small positive values for stability)
    for _ in 0..p {
        let ar_dist = Normal::new(0.1, 0.5).unwrap();
        let sample_val: f64 = ar_dist.sample(rng);
        params[idx] = sample_val.max(bounds.lower[idx]).min(bounds.upper[idx]);
        idx += 1;
    }
    
    // MA coefficients
    for i in 0..=q {
        if i == 0 {
            // First MA coefficient (close to 1)
            let ma_dist = Normal::new(1.0, 0.1).unwrap();
            let sample_val: f64 = ma_dist.sample(rng);
            params[idx] = sample_val.max(bounds.lower[idx]).min(bounds.upper[idx]);
        } else {
            let ma_dist = Normal::new(0.0, 0.5).unwrap();
            let sample_val: f64 = ma_dist.sample(rng);
            params[idx] = sample_val.max(bounds.lower[idx]).min(bounds.upper[idx]);
        }
        idx += 1;
    }
    
    // Log(sigma) parameter
    let log_sigma_dist = Normal::new((data_std * 0.5).ln(), 0.5).unwrap();
    params[idx] = log_sigma_dist.sample(rng).max(bounds.lower[idx]).min(bounds.upper[idx]);
    
    params
}