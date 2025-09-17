use crate::carma::model::{CarmaModel, CarmaError};
use crate::carma::likelihood::StateSpaceModel;
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Simulate a CARMA process at given times
#[pyfunction]
pub fn simulate_carma(
    py: Python,
    model: &CarmaModel,
    times: PyReadonlyArray1<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let times_slice = times.as_slice()?;
    
    if times_slice.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Times array cannot be empty"
        ));
    }
    
    // Initialize random number generator
    let mut rng = if let Some(seed_val) = seed {
        Xoshiro256PlusPlus::seed_from_u64(seed_val)
    } else {
        Xoshiro256PlusPlus::from_entropy()
    };
    
    // Generate simulated values
    let values = simulate_carma_process(model, times_slice, &mut rng)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    // Convert to numpy array
    let py_array = PyArray1::from_vec(py, values);
    Ok(py_array.into())
}

/// Generate synthetic CARMA data with irregular sampling
#[pyfunction]
pub fn generate_carma_data(
    py: Python,
    model: &CarmaModel,
    duration: f64,
    mean_sampling_rate: f64,
    sampling_irregularity: Option<f64>,
    measurement_noise: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Option<Py<PyArray1<f64>>>)> {
    if duration <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Duration must be positive"
        ));
    }
    
    if mean_sampling_rate <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Mean sampling rate must be positive"
        ));
    }
    
    // Initialize random number generator
    let mut rng = if let Some(seed_val) = seed {
        Xoshiro256PlusPlus::seed_from_u64(seed_val)
    } else {
        Xoshiro256PlusPlus::from_entropy()
    };
    
    // Generate irregular sampling times
    let irregularity = sampling_irregularity.unwrap_or(0.3);
    let times = generate_irregular_times(duration, mean_sampling_rate, irregularity, &mut rng);
    
    // Simulate CARMA process
    let values = simulate_carma_process(model, &times, &mut rng)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    // Add measurement noise if specified
    let (final_values, errors) = if let Some(noise_std) = measurement_noise {
        let mut noisy_values = values.clone();
        let mut error_values = vec![noise_std; values.len()];
        
        let noise_dist = Normal::new(0.0, noise_std).unwrap();
        for value in &mut noisy_values {
            *value += noise_dist.sample(&mut rng);
        }
        
        (noisy_values, Some(error_values))
    } else {
        (values, None)
    };
    
    // Convert to numpy arrays
    let times_array = PyArray1::from_vec(py, times);
    let values_array = PyArray1::from_vec(py, final_values);
    let errors_array = errors.map(|e| PyArray1::from_vec(py, e).into());
    
    Ok((times_array.into(), values_array.into(), errors_array))
}

/// Internal function to simulate CARMA process
fn simulate_carma_process(
    model: &CarmaModel,
    times: &[f64],
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Vec<f64>, CarmaError> {
    if times.is_empty() {
        return Ok(Vec::new());
    }
    
    // Create state-space representation
    let state_space = StateSpaceModel::from_carma_model(model)?;
    let p = model.p;
    
    // Initialize state at steady-state distribution
    let mut state = sample_steady_state(&state_space, model.mu, rng)?;
    let mut values = Vec::with_capacity(times.len());
    
    // Simulate first observation
    let first_obs = state_space.observation_matrix.dot(&state) + model.mu;
    values.push(first_obs);
    
    // Simulate subsequent observations
    for i in 1..times.len() {
        let dt = times[i] - times[i-1];
        
        if dt < 0.0 {
            return Err(CarmaError::DataValidationError {
                message: "Times must be in ascending order".to_string()
            });
        }
        
        if dt > 0.0 {
            // Propagate state forward in time
            state = propagate_state_stochastic(&state_space, &state, dt, rng)?;
        }
        
        // Generate observation
        let obs = state_space.observation_matrix.dot(&state) + model.mu;
        values.push(obs);
    }
    
    Ok(values)
}

/// Sample from steady-state distribution
fn sample_steady_state(
    state_space: &StateSpaceModel,
    mu: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<DVector<f64>, CarmaError> {
    let p = state_space.steady_state_cov.nrows();
    
    // Cholesky decomposition of steady-state covariance
    let cov_chol = cholesky_decomposition_matrix(&state_space.steady_state_cov)?;
    
    // Generate standard normal random vector
    let mut z = DVector::zeros(p);
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    for i in 0..p {
        z[i] = normal_dist.sample(rng);
    }
    
    // Transform to correlated random vector
    let mut state = &cov_chol * &z;
    
    // Add mean to first component
    state[0] += mu;
    
    Ok(state)
}

/// Propagate state stochastically forward in time
fn propagate_state_stochastic(
    state_space: &StateSpaceModel,
    current_state: &DVector<f64>,
    dt: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<DVector<f64>, CarmaError> {
    let p = current_state.len();
    
    // Deterministic propagation
    let phi = state_space.propagate_state(dt);
    let mut new_state = &phi * current_state;
    
    // Add process noise
    let q_dt = state_space.process_noise_over_interval(dt);
    
    // Cholesky decomposition of process noise covariance
    let q_chol = cholesky_decomposition_matrix(&q_dt)?;
    
    // Generate noise
    let mut noise = DVector::zeros(p);
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    for i in 0..p {
        noise[i] = normal_dist.sample(rng);
    }
    
    // Add correlated noise
    let correlated_noise = &q_chol * &noise;
    new_state += &correlated_noise;
    
    Ok(new_state)
}

/// Generate irregular sampling times
fn generate_irregular_times(
    duration: f64,
    mean_rate: f64,
    irregularity: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<f64> {
    let expected_n_points = (duration * mean_rate) as usize;
    let mut times = Vec::with_capacity(expected_n_points + 10);
    
    let mut current_time = 0.0;
    times.push(current_time);
    
    let mean_interval = 1.0 / mean_rate;
    
    while current_time < duration {
        // Generate next interval with irregularity
        let base_interval = mean_interval;
        let noise_factor = if irregularity > 0.0 {
            let uniform_dist = Uniform::new(-irregularity, irregularity);
            1.0 + uniform_dist.sample(rng)
        } else {
            1.0
        };
        
        let interval = base_interval * noise_factor.max(0.1);
        current_time += interval;
        
        if current_time <= duration {
            times.push(current_time);
        }
    }
    
    times
}

/// Cholesky decomposition for DMatrix
fn cholesky_decomposition_matrix(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, CarmaError> {
    let n = matrix.nrows();
    let mut l = DMatrix::zeros(n, n);
    
    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal elements
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(j, k)] * l[(j, k)];
                }
                let val = matrix[(j, j)] - sum;
                if val <= 0.0 {
                    return Err(CarmaError::NumericalError {
                        message: format!("Matrix is not positive definite at element ({}, {}): {}", j, j, val)
                    });
                }
                l[(j, j)] = val.sqrt();
            } else {
                // Off-diagonal elements
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                if l[(j, j)].abs() < 1e-14 {
                    return Err(CarmaError::NumericalError {
                        message: "Near-singular matrix in Cholesky decomposition".to_string()
                    });
                }
                l[(i, j)] = (matrix[(i, j)] - sum) / l[(j, j)];
            }
        }
    }
    
    Ok(l)
}

/// Generate stable CARMA parameters for testing
#[pyfunction]
pub fn generate_stable_carma_parameters(
    p: usize,
    q: usize,
    seed: Option<u64>,
) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
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
    
    let mut rng = if let Some(seed_val) = seed {
        Xoshiro256PlusPlus::seed_from_u64(seed_val)
    } else {
        Xoshiro256PlusPlus::from_entropy()
    };
    
    let (ar_coeffs, ma_coeffs, sigma) = generate_stable_parameters(p, q, &mut rng);
    
    Ok((ar_coeffs, ma_coeffs, sigma))
}

/// Internal function to generate stable parameters
fn generate_stable_parameters(p: usize, q: usize, rng: &mut Xoshiro256PlusPlus) -> (Vec<f64>, Vec<f64>, f64) {
    // Generate stable AR coefficients
    let ar_coeffs = match p {
        1 => {
            // CAR(1): coefficient must be positive
            let coeff = rng.gen_range(0.1..2.0);
            vec![coeff]
        }
        2 => {
            // CARMA(2,q): coefficients must satisfy stability conditions
            let a1 = rng.gen_range(0.1..3.0);
            let a0 = rng.gen_range(0.1..a1 * 0.8);
            vec![a0, a1]
        }
        _ => {
            // Higher-order: use a more conservative approach
            let mut coeffs = Vec::with_capacity(p);
            for i in 0..p {
                let coeff = rng.gen_range(0.1..1.0) / (i + 1) as f64;
                coeffs.push(coeff);
            }
            coeffs
        }
    };
    
    // Generate MA coefficients
    let mut ma_coeffs = vec![1.0]; // β_0 = 1 by convention
    for _ in 0..q {
        let coeff = rng.gen_range(-1.0..1.0);
        ma_coeffs.push(coeff);
    }
    
    // Generate reasonable sigma
    let sigma = rng.gen_range(0.5..2.0);
    
    (ar_coeffs, ma_coeffs, sigma)
}

/// Validate CARMA simulation parameters
pub fn validate_simulation_params(
    model: &CarmaModel,
    times: &[f64],
) -> Result<(), CarmaError> {
    if times.is_empty() {
        return Err(CarmaError::DataValidationError {
            message: "Times array cannot be empty".to_string()
        });
    }
    
    // Check that times are sorted
    for i in 1..times.len() {
        if times[i] <= times[i-1] {
            return Err(CarmaError::DataValidationError {
                message: "Times must be strictly increasing".to_string()
            });
        }
    }
    
    // Check model stability
    if !model.is_stable() {
        return Err(CarmaError::InvalidParameter {
            message: "Model is not stable".to_string()
        });
    }
    
    // Check parameter validity
    if model.sigma <= 0.0 {
        return Err(CarmaError::InvalidParameter {
            message: "Sigma must be positive".to_string()
        });
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_irregular_times() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let times = generate_irregular_times(10.0, 1.0, 0.3, &mut rng);
        
        assert!(times.len() > 5);
        assert!(times[0] == 0.0);
        assert!(times[times.len() - 1] <= 10.0);
        
        // Check times are sorted
        for i in 1..times.len() {
            assert!(times[i] > times[i-1]);
        }
    }
    
    #[test]
    fn test_stable_parameter_generation() {
        let (ar_coeffs, ma_coeffs, sigma) = generate_stable_parameters(2, 1, &mut Xoshiro256PlusPlus::seed_from_u64(42));
        
        assert_eq!(ar_coeffs.len(), 2);
        assert_eq!(ma_coeffs.len(), 2);
        assert!(sigma > 0.0);
        assert_eq!(ma_coeffs[0], 1.0);
        
        // Check basic stability for CARMA(2,1)
        assert!(ar_coeffs[0] > 0.0);
        assert!(ar_coeffs[1] > 0.0);
    }
    
    #[test]
    fn test_carma_simulation_basic() {
        let mut model = CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![0.5, 1.0];
        model.ma_coeffs = vec![1.0, 0.3];
        model.sigma = 1.0;
        model.mu = 0.0;
        
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let result = simulate_carma_process(&model, &times, &mut rng);
        assert!(result.is_ok());
        
        let values = result.unwrap();
        assert_eq!(values.len(), times.len());
        
        // Values should be finite
        for value in values {
            assert!(value.is_finite());
        }
    }
}