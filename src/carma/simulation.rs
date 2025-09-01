use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use crate::carma::carma_model::{CarmaModel, CarmaError};
use crate::carma::utils::{matrix_exponential, carma_to_state_space, solve_lyapunov};

/// Simulate CARMA process at given times
#[pyfunction]
pub fn simulate_carma(
    py: Python,
    model: &CarmaModel,
    times: PyReadonlyArray1<f64>,
    initial_state: Option<PyReadonlyArray1<f64>>,
    seed: Option<u64>
) -> PyResult<Py<PyArray1<f64>>> {
    let times_slice = times.as_slice()?;
    
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    if times_slice.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Times cannot be empty"));
    }
    
    // Check times are sorted
    for i in 1..times_slice.len() {
        if times_slice[i] <= times_slice[i-1] {
            return Err(pyo3::exceptions::PyValueError::new_err("Times must be strictly increasing"));
        }
    }
    
    // Set up random number generator
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    
    // Convert to state space
    let ss = carma_to_state_space(model)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    let p = model.p;
    let transition = DMatrix::from_vec(p, p, ss.transition_matrix.into_iter().flatten().collect());
    let observation = DVector::from_vec(ss.observation_vector);
    let process_noise = DMatrix::from_vec(p, p, ss.process_noise_matrix.into_iter().flatten().collect());
    
    // Initialize state
    let mut state = if let Some(init_state) = initial_state {
        let init_slice = init_state.as_slice()?;
        if init_slice.len() != p {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Initial state must have length {}", p)
            ));
        }
        DVector::from_vec(init_slice.to_vec())
    } else {
        DVector::zeros(p)
    };
    
    // Simulate
    let values = simulate_carma_process(&transition, &observation, &process_noise, 
                                       &mut state, times_slice, &mut rng)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(PyArray1::from_vec(py, values).into())
}

/// Generate irregular sampling times and simulate CARMA process
#[pyfunction]
pub fn generate_irregular_carma(
    py: Python,
    model: &CarmaModel,
    duration: f64,
    mean_sampling_rate: f64,
    sampling_noise: f64,
    seed: Option<u64>
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid model"));
    }
    
    if duration <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Duration must be positive"));
    }
    
    if mean_sampling_rate <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Mean sampling rate must be positive"));
    }
    
    if sampling_noise < 0.0 || sampling_noise >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Sampling noise must be in [0, 1)"));
    }
    
    // Set up random number generator
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    
    // Generate irregular sampling times
    let times = generate_irregular_times(duration, mean_sampling_rate, sampling_noise, &mut rng)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    // Convert to state space
    let ss = carma_to_state_space(model)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    let p = model.p;
    let transition = DMatrix::from_vec(p, p, ss.transition_matrix.into_iter().flatten().collect());
    let observation = DVector::from_vec(ss.observation_vector);
    let process_noise = DMatrix::from_vec(p, p, ss.process_noise_matrix.into_iter().flatten().collect());
    
    // Initialize state
    let mut state = DVector::zeros(p);
    
    // Simulate
    let values = simulate_carma_process(&transition, &observation, &process_noise, 
                                       &mut state, &times, &mut rng)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok((
        PyArray1::from_vec(py, times).into(),
        PyArray1::from_vec(py, values).into()
    ))
}

/// Internal CARMA simulation implementation
fn simulate_carma_process(
    transition: &DMatrix<f64>,
    observation: &DVector<f64>,
    process_noise: &DMatrix<f64>,
    initial_state: &mut DVector<f64>,
    times: &[f64],
    rng: &mut StdRng,
) -> Result<Vec<f64>, CarmaError> {
    let n = times.len();
    let _p = transition.nrows();
    let mut values = Vec::with_capacity(n);
    let mut state = initial_state.clone();
    
    // Precompute steady-state covariance for process noise
    let steady_state_cov = solve_lyapunov(&(-transition), process_noise)?;
    
    // Normal distribution for noise generation
    let normal = Normal::new(0.0, 1.0).map_err(|_| CarmaError::NumericalError("Failed to create normal distribution".to_string()))?;
    
    for i in 0..n {
        // Time step
        let dt = if i == 0 { 0.0 } else { times[i] - times[i-1] };
        
        if dt > 0.0 {
            // Propagate state
            let transition_matrix = matrix_exponential(transition, dt)?;
            state = &transition_matrix * &state;
            
            // Add process noise
            let process_cov = compute_process_noise_cov(&transition_matrix, &steady_state_cov, dt)?;
            let noise = generate_multivariate_normal(&process_cov, rng, &normal)?;
            state += noise;
        }
        
        // Generate observation
        let obs_mean = observation.dot(&state);
        let obs_value = obs_mean; // For now, no observation noise in simulation
        
        values.push(obs_value);
    }
    
    Ok(values)
}

/// Generate irregular sampling times
fn generate_irregular_times(
    duration: f64,
    mean_rate: f64,
    noise_level: f64,
    rng: &mut StdRng,
) -> Result<Vec<f64>, CarmaError> {
    let expected_count = (duration * mean_rate) as usize;
    let mut times = Vec::with_capacity(expected_count + 10);
    
    let mean_interval = 1.0 / mean_rate;
    let uniform = Uniform::new(0.0, 1.0);
    
    let mut current_time = 0.0;
    
    while current_time < duration {
        times.push(current_time);
        
        // Generate next interval with noise
        let base_interval = mean_interval;
        let noise_factor = 1.0 + noise_level * (2.0 * rng.sample(uniform) - 1.0);
        let interval = base_interval * noise_factor.max(0.1); // Ensure positive interval
        
        current_time += interval;
    }
    
    // Remove any times beyond duration
    times.retain(|&t| t <= duration);
    
    if times.is_empty() {
        times.push(0.0);
    }
    
    Ok(times)
}

/// Compute process noise covariance for simulation
fn compute_process_noise_cov(
    transition_matrix: &DMatrix<f64>,
    steady_state_cov: &DMatrix<f64>,
    dt: f64,
) -> Result<DMatrix<f64>, CarmaError> {
    let p = transition_matrix.nrows();
    
    if dt <= 0.0 {
        return Ok(DMatrix::zeros(p, p));
    }
    
    // Simplified computation: use the fact that for small dt,
    // the process noise covariance is approximately steady_state_cov * dt
    if dt < 0.1 {
        Ok(steady_state_cov * dt)
    } else {
        // For larger dt, use more sophisticated computation
        // This is a simplified version - a full implementation would solve the integral
        let identity = DMatrix::identity(p, p);
        let cov_factor = &identity - transition_matrix * transition_matrix.transpose();
        
        if let Some(inverse) = cov_factor.try_inverse() {
            Ok(&inverse * steady_state_cov * dt)
        } else {
            // Fallback to simple scaling
            Ok(steady_state_cov * dt)
        }
    }
}

/// Generate multivariate normal random vector
fn generate_multivariate_normal(
    covariance: &DMatrix<f64>,
    rng: &mut StdRng,
    normal: &Normal<f64>,
) -> Result<DVector<f64>, CarmaError> {
    let p = covariance.nrows();
    
    // Cholesky decomposition for multivariate normal generation
    let chol = covariance.clone().cholesky();
    let l_matrix = if let Some(chol_decomp) = chol {
        chol_decomp.l()
    } else {
        // Fallback: use diagonal if Cholesky fails
        let mut diag = DMatrix::zeros(p, p);
        for i in 0..p {
            diag[(i, i)] = covariance[(i, i)].max(0.0).sqrt();
        }
        diag
    };
    
    // Generate standard normal vector
    let mut z = DVector::zeros(p);
    for i in 0..p {
        z[i] = rng.sample(normal);
    }
    
    // Transform to multivariate normal
    Ok(&l_matrix * z)
}

/// Validate simulation parameters
pub fn validate_simulation_params(
    model: &CarmaModel,
    times: &[f64],
    initial_state: Option<&[f64]>,
) -> Result<(), CarmaError> {
    if !model.is_valid() {
        return Err(CarmaError::InvalidParameters("Invalid model".to_string()));
    }
    
    if times.is_empty() {
        return Err(CarmaError::InvalidData("Times cannot be empty".to_string()));
    }
    
    if times.iter().any(|&t| !t.is_finite()) {
        return Err(CarmaError::InvalidData("Times must be finite".to_string()));
    }
    
    // Check times are sorted
    for i in 1..times.len() {
        if times[i] <= times[i-1] {
            return Err(CarmaError::InvalidData("Times must be strictly increasing".to_string()));
        }
    }
    
    if let Some(init_state) = initial_state {
        if init_state.len() != model.p {
            return Err(CarmaError::InvalidParameters(
                format!("Initial state must have length {}", model.p)
            ));
        }
        
        if init_state.iter().any(|&x| !x.is_finite()) {
            return Err(CarmaError::InvalidData("Initial state must be finite".to_string()));
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carma::carma_model::CarmaModel;
    use numpy::PyArray1;
    use pyo3::Python;
    
    #[test]
    fn test_irregular_time_generation() {
        let mut rng = StdRng::seed_from_u64(42);
        let times = generate_irregular_times(10.0, 1.0, 0.2, &mut rng).unwrap();
        
        assert!(!times.is_empty());
        assert!(times[0] >= 0.0);
        assert!(times.last().unwrap() <= &10.0);
        
        // Check times are sorted
        for i in 1..times.len() {
            assert!(times[i] > times[i-1]);
        }
    }
    
    #[test]
    fn test_multivariate_normal() {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let cov = DMatrix::identity(3, 3) * 2.0;
        
        let sample = generate_multivariate_normal(&cov, &mut rng, &normal).unwrap();
        assert_eq!(sample.len(), 3);
        assert!(sample.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_simulation_validation() {
        let model = CarmaModel::new(2, 1).unwrap();
        let times = vec![0.0, 1.0, 2.0];
        let init_state = vec![0.0, 0.0];
        
        assert!(validate_simulation_params(&model, &times, Some(&init_state)).is_ok());
        
        // Test various error conditions
        let empty_times = vec![];
        assert!(validate_simulation_params(&model, &empty_times, None).is_err());
        
        let unsorted_times = vec![1.0, 0.0, 2.0];
        assert!(validate_simulation_params(&model, &unsorted_times, None).is_err());
        
        let wrong_init_state = vec![0.0];
        assert!(validate_simulation_params(&model, &times, Some(&wrong_init_state)).is_err());
    }
    
    #[test]
    fn test_simulate_carma_setup() {
        Python::with_gil(|py| {
            let mut model = CarmaModel::new(2, 1).unwrap();
            model.ar_coeffs = vec![1.5, -0.5];
            model.ma_coeffs = vec![1.0, 0.3];
            model.sigma = 1.0;
            
            let times = PyArray1::from_vec(py, vec![0.0, 1.0, 2.0, 3.0]);
            let result = simulate_carma(py, &model, times.readonly(), None, Some(42));
            
            assert!(result.is_ok());
            let values = result.unwrap();
            let values_vec: Vec<f64> = values.as_ref(py).to_vec().unwrap();
            assert_eq!(values_vec.len(), 4);
            assert!(values_vec.iter().all(|&x| x.is_finite()));
        });
    }
    
    #[test]
    fn test_generate_irregular_carma_setup() {
        Python::with_gil(|py| {
            let mut model = CarmaModel::new(2, 1).unwrap();
            model.ar_coeffs = vec![1.5, -0.5];
            model.ma_coeffs = vec![1.0, 0.3];
            model.sigma = 1.0;
            
            let result = generate_irregular_carma(py, &model, 10.0, 1.0, 0.2, Some(42));
            
            assert!(result.is_ok());
            let (times, values) = result.unwrap();
            
            let times_vec: Vec<f64> = times.as_ref(py).to_vec().unwrap();
            let values_vec: Vec<f64> = values.as_ref(py).to_vec().unwrap();
            
            assert_eq!(times_vec.len(), values_vec.len());
            assert!(!times_vec.is_empty());
            assert!(times_vec.iter().all(|&x| x.is_finite() && x >= 0.0));
            assert!(values_vec.iter().all(|&x| x.is_finite()));
        });
    }
}