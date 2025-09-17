use crate::carma::model::{CarmaModel, CarmaError};
use nalgebra::{DMatrix, DVector, LU};
use std::f64::consts::PI;

/// State-space representation of a CARMA model
#[derive(Clone, Debug)]
pub struct StateSpaceModel {
    /// State transition matrix A (p x p)
    pub transition_matrix: DMatrix<f64>,
    /// Observation matrix C (1 x p) 
    pub observation_matrix: DVector<f64>,
    /// Process noise covariance Q (p x p)
    pub process_noise_cov: DMatrix<f64>,
    /// Observation noise variance R (scalar)
    pub observation_noise_var: f64,
    /// Steady-state covariance matrix P_∞ (p x p)
    pub steady_state_cov: DMatrix<f64>,
}

impl StateSpaceModel {
    /// Create state-space representation from CARMA model
    pub fn from_carma_model(model: &CarmaModel) -> Result<Self, CarmaError> {
        let p = model.p;
        
        // Build companion form transition matrix A
        let mut transition_matrix = DMatrix::zeros(p, p);
        
        // Fill the upper diagonal with 1s (companion form)
        for i in 0..p-1 {
            transition_matrix[(i, i+1)] = 1.0;
        }
        
        // Fill the last row with negative AR coefficients (characteristic polynomial)
        for j in 0..p {
            transition_matrix[(p-1, j)] = -model.ar_coeffs[p-1-j];
        }
        
        // Build observation matrix C = [1, 0, ..., 0]
        let mut observation_matrix = DVector::zeros(p);
        observation_matrix[0] = 1.0;
        
        // Build process noise covariance matrix Q
        let mut process_noise_cov = DMatrix::zeros(p, p);
        
        // For a CARMA(p,q) model, the process noise affects the last (q+1) states
        let noise_var = model.sigma * model.sigma;
        
        // Set up the MA polynomial contribution
        for i in 0..=model.q.min(p-1) {
            for j in 0..=model.q.min(p-1) {
                if i + j < p {
                    let coeff_i = if i <= model.q { model.ma_coeffs[i] } else { 0.0 };
                    let coeff_j = if j <= model.q { model.ma_coeffs[j] } else { 0.0 };
                    process_noise_cov[(p-1-i, p-1-j)] = noise_var * coeff_i * coeff_j;
                }
            }
        }
        
        // Calculate steady-state covariance using the discrete Lyapunov equation
        let steady_state_cov = solve_discrete_lyapunov(&transition_matrix, &process_noise_cov)?;
        
        Ok(StateSpaceModel {
            transition_matrix,
            observation_matrix,
            process_noise_cov,
            observation_noise_var: 0.0,  // Set separately based on measurement errors
            steady_state_cov,
        })
    }
    
    /// Propagate state from one time to another
    pub fn propagate_state(&self, dt: f64) -> DMatrix<f64> {
        // For small dt, use matrix exponential approximation
        if dt < 1e-10 {
            return DMatrix::identity(self.transition_matrix.nrows(), self.transition_matrix.ncols());
        }
        
        // Compute matrix exponential exp(A*dt) using scaling and squaring
        matrix_exponential(&(self.transition_matrix.clone() * dt))
    }
    
    /// Compute process noise covariance over time interval dt
    pub fn process_noise_over_interval(&self, dt: f64) -> DMatrix<f64> {
        if dt < 1e-10 {
            return DMatrix::zeros(self.transition_matrix.nrows(), self.transition_matrix.ncols());
        }
        
        // For CARMA models, this involves integrating the matrix exponential
        // This is a simplified approximation
        let phi = self.propagate_state(dt);
        let phi_t = phi.transpose();
        
        // Q_dt = ∫₀^dt exp(A*s) Q exp(A^T*s) ds
        // Approximation: Q_dt ≈ Q * dt for small dt, or use more sophisticated methods
        if dt < 0.1 {
            self.process_noise_cov.clone() * dt
        } else {
            // Use the integral formula: Q_dt = (I - Φ ⊗ Φ)^(-1) vec(Q)
            // For now, use a simpler approximation
            &phi * &self.process_noise_cov * &phi_t * dt
        }
    }
}

/// Kalman filter implementation for CARMA models
pub struct KalmanFilter {
    /// Current state estimate
    pub state: DVector<f64>,
    /// Current state covariance
    pub covariance: DMatrix<f64>,
    /// Log-likelihood accumulator
    pub loglikelihood: f64,
    /// Number of observations processed
    pub nobs: usize,
}

impl KalmanFilter {
    /// Initialize Kalman filter with steady-state values
    pub fn new(state_space: &StateSpaceModel, mu: f64) -> Self {
        let p = state_space.transition_matrix.nrows();
        let mut state = DVector::zeros(p);
        state[0] = mu;  // Initialize first state to mean
        
        KalmanFilter {
            state,
            covariance: state_space.steady_state_cov.clone(),
            loglikelihood: 0.0,
            nobs: 0,
        }
    }
    
    /// Perform one Kalman filter step
    pub fn filter_step(
        &mut self,
        observation: f64,
        dt: f64,
        measurement_error: f64,
        state_space: &StateSpaceModel,
    ) -> Result<(), CarmaError> {
        let p = state_space.transition_matrix.nrows();
        
        // Prediction step
        let phi = state_space.propagate_state(dt);
        let q_dt = state_space.process_noise_over_interval(dt);
        
        // Predict state: x_{k|k-1} = Φ * x_{k-1|k-1}
        self.state = &phi * &self.state;
        
        // Predict covariance: P_{k|k-1} = Φ * P_{k-1|k-1} * Φ^T + Q_dt
        self.covariance = &phi * &self.covariance * &phi.transpose() + &q_dt;
        
        // Update step
        let c = &state_space.observation_matrix;
        
        // Innovation: y_k - C * x_{k|k-1}
        let predicted_obs = c.dot(&self.state);
        let innovation = observation - predicted_obs;
        
        // Innovation covariance: S = C * P_{k|k-1} * C^T + R
        let innovation_var = c.dot(&(&self.covariance * c)) + measurement_error * measurement_error;
        
        if innovation_var <= 0.0 {
            return Err(CarmaError::NumericalError {
                message: "Non-positive innovation variance".to_string()
            });
        }
        
        // Kalman gain: K = P_{k|k-1} * C^T / S
        let kalman_gain = (&self.covariance * c) / innovation_var;
        
        // Update state: x_{k|k} = x_{k|k-1} + K * innovation
        self.state += &kalman_gain * innovation;
        
        // Update covariance: P_{k|k} = (I - K * C) * P_{k|k-1}
        let identity = DMatrix::identity(p, p);
        let update_matrix = &identity - &kalman_gain * c.transpose();
        self.covariance = &update_matrix * &self.covariance;
        
        // Update log-likelihood
        self.loglikelihood += -0.5 * (innovation * innovation / innovation_var + innovation_var.ln() + (2.0 * PI).ln());
        self.nobs += 1;
        
        Ok(())
    }
}

/// Compute log-likelihood of CARMA model given data
pub fn compute_loglikelihood(
    model: &CarmaModel,
    times: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
) -> Result<f64, CarmaError> {
    if times.len() != values.len() {
        return Err(CarmaError::DataValidationError {
            message: "Times and values must have the same length".to_string()
        });
    }
    
    if let Some(errs) = errors {
        if errs.len() != times.len() {
            return Err(CarmaError::DataValidationError {
                message: "Errors must have the same length as times and values".to_string()
            });
        }
    }
    
    if times.is_empty() {
        return Err(CarmaError::DataValidationError {
            message: "Cannot compute likelihood for empty data".to_string()
        });
    }
    
    // Create state-space representation
    let state_space = StateSpaceModel::from_carma_model(model)?;
    
    // Initialize Kalman filter
    let mut kalman = KalmanFilter::new(&state_space, model.mu);
    
    // Process each observation
    for i in 0..times.len() {
        let dt = if i == 0 { 0.0 } else { times[i] - times[i-1] };
        let measurement_error = if let Some(errs) = errors { errs[i] } else { 0.0 };
        
        kalman.filter_step(values[i], dt, measurement_error, &state_space)?;
    }
    
    Ok(kalman.loglikelihood)
}

/// Solve the discrete Lyapunov equation A*X*A^T - X + Q = 0
fn solve_discrete_lyapunov(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    let n = a.nrows();
    
    // For small matrices, use iterative method
    if n <= 4 {
        return solve_lyapunov_iterative(a, q);
    }
    
    // For larger matrices, use Kronecker product method
    solve_lyapunov_kronecker(a, q)
}

/// Iterative solution of discrete Lyapunov equation
fn solve_lyapunov_iterative(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    let n = a.nrows();
    let mut x = q.clone();
    let at = a.transpose();
    
    // Iterate: X_{k+1} = A * X_k * A^T + Q
    for _ in 0..100 {
        let x_new = a * &x * &at + q;
        let diff = (&x_new - &x).norm();
        x = x_new;
        
        if diff < 1e-12 {
            break;
        }
    }
    
    Ok(x)
}

/// Kronecker product solution of discrete Lyapunov equation
fn solve_lyapunov_kronecker(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    let n = a.nrows();
    
    // Form the system (I ⊗ I - A ⊗ A^T) vec(X) = vec(Q)
    let identity = DMatrix::identity(n, n);
    let a_kron_at = kronecker_product(a, &a.transpose());
    let i_kron_i = kronecker_product(&identity, &identity);
    
    let system_matrix = &i_kron_i - &a_kron_at;
    let q_vec = matrix_to_vector(q);
    
    // Solve the linear system
    let lu = LU::new(system_matrix);
    match lu.solve(&q_vec) {
        Some(x_vec) => {
            let x = vector_to_matrix(&x_vec, n, n);
            Ok(x)
        }
        None => Err(CarmaError::NumericalError {
            message: "Failed to solve Lyapunov equation".to_string()
        })
    }
}

/// Compute Kronecker product of two matrices
fn kronecker_product(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (p, q) = (b.nrows(), b.ncols());
    
    let mut result = DMatrix::zeros(m * p, n * q);
    
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                for l in 0..q {
                    result[(i * p + k, j * q + l)] = a[(i, j)] * b[(k, l)];
                }
            }
        }
    }
    
    result
}

/// Convert matrix to vector (column-major order)
fn matrix_to_vector(matrix: &DMatrix<f64>) -> DVector<f64> {
    let (m, n) = (matrix.nrows(), matrix.ncols());
    let mut vec = DVector::zeros(m * n);
    
    for j in 0..n {
        for i in 0..m {
            vec[j * m + i] = matrix[(i, j)];
        }
    }
    
    vec
}

/// Convert vector to matrix (column-major order)
fn vector_to_matrix(vec: &DVector<f64>, m: usize, n: usize) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(m, n);
    
    for j in 0..n {
        for i in 0..m {
            matrix[(i, j)] = vec[j * m + i];
        }
    }
    
    matrix
}

/// Compute matrix exponential using scaling and squaring
fn matrix_exponential(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    
    // Scale the matrix to reduce the norm
    let norm = a.norm();
    let scale = if norm > 1.0 {
        (norm.log2().ceil() as i32).max(0)
    } else {
        0
    };
    
    let scaled_a = a.clone() / (2.0_f64.powi(scale));
    
    // Compute matrix exponential using Padé approximation
    let exp_scaled = matrix_exponential_pade(&scaled_a);
    
    // Square the result scale times
    let mut result = exp_scaled;
    for _ in 0..scale {
        result = &result * &result;
    }
    
    result
}

/// Padé approximation for matrix exponential
fn matrix_exponential_pade(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let identity = DMatrix::identity(n, n);
    
    // Use order-6 Padé approximation
    let a2 = a * a;
    let a3 = &a2 * a;
    let a4 = &a2 * &a2;
    let a5 = &a4 * a;
    let a6 = &a3 * &a3;
    
    let u = a * (&identity * 17297280.0 + &a2 * 1995840.0 + &a4 * 25200.0 + &a6 * 56.0);
    let v = &identity * 17297280.0 + &a2 * 10810800.0 + &a4 * 302400.0 + &a6 * 840.0;
    
    // Solve (V - U) * exp(A) = V + U
    let lhs = &v - &u;
    let rhs = &v + &u;
    
    let lu = LU::new(lhs);
    match lu.solve(&rhs) {
        Some(result) => result,
        None => {
            // Fallback to first-order approximation
            &identity + a
        }
    }
}