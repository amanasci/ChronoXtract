use numpy::ndarray::{Array1, Array2, Axis, s};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_series(data: &Array1<f64>) -> PyResult<()> {
    if data.is_empty() {
        return Err(PyValueError::new_err("Input time series cannot be empty"));
    }
    if data.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err(
            "Input time series must contain only finite values",
        ));
    }
    Ok(())
}

/// Build a Hankel-style time-delay embedding matrix from a 1D time series.
///
/// Given a sequence `x_0, x_1, ..., x_{N-1}` and window length `L`, this
/// computes the matrix `H in R^{(N-L+1) x L}` with entries:
///
/// `H[i, j] = x_{i + j}`
///
/// # Arguments
/// * `time_series` - Input time series `x` as a 1D NumPy array
/// * `window_length` - Embedding window length `L`
///
/// # Returns
/// A 2D NumPy array containing the time-delay embedding.
///
/// # Errors
/// Returns `PyValueError` if:
/// - The input is empty
/// - The input contains non-finite values
/// - `window_length == 0`
/// - `window_length > len(time_series)`
#[pyfunction]
pub fn time_delay_embedding<'py>(
    py: Python<'py>,
    time_series: PyReadonlyArray1<'py, f64>,
    window_length: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = time_series.as_array().to_owned();
    validate_series(&data)?;

    if window_length == 0 {
        return Err(PyValueError::new_err("window_length must be greater than 0"));
    }
    if window_length > data.len() {
        return Err(PyValueError::new_err(
            "window_length must be less than or equal to time series length",
        ));
    }

    let n_rows = data.len() - window_length + 1;
    let mut hankel = Array2::<f64>::zeros((n_rows, window_length));
    for i in 0..n_rows {
        hankel
            .slice_mut(s![i, ..])
            .assign(&data.slice(s![i..i + window_length]));
    }

    Ok(hankel.into_pyarray(py).to_owned())
}

/// Compute the Gramian Angular Summation Field (GASF) matrix.
///
/// Let `x` be the input series normalized to `x_t' in [-1, 1]`:
///
/// `x_t' = 2 * (x_t - min(x)) / (max(x) - min(x)) - 1`
///
/// and `phi_t = arccos(x_t')`. The GASF is:
///
/// `G[i, j] = cos(phi_i + phi_j)`
///
/// Computed in an optimized algebraic form:
///
/// `G[i, j] = x_i' * x_j' - sqrt(1 - x_i'^2) * sqrt(1 - x_j'^2)`
///
/// # Arguments
/// * `time_series` - Input time series `x` as a 1D NumPy array
///
/// # Returns
/// A 2D NumPy array with shape `(N, N)` containing the GASF matrix.
///
/// # Errors
/// Returns `PyValueError` if the input is empty or contains non-finite values.
#[pyfunction]
pub fn gramian_angular_summation_field<'py>(
    py: Python<'py>,
    time_series: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = time_series.as_array().to_owned();
    validate_series(&data)?;

    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;

    let normalized = if range <= f64::EPSILON {
        Array1::<f64>::zeros(data.len())
    } else {
        data.mapv(|x| (2.0 * (x - min) / range - 1.0).clamp(-1.0, 1.0))
    };

    let sin_component = normalized.mapv(|x| (1.0 - x * x).max(0.0).sqrt());
    let cos_i = normalized.view().insert_axis(Axis(1));
    let cos_j = normalized.view().insert_axis(Axis(0));
    let sin_i = sin_component.view().insert_axis(Axis(1));
    let sin_j = sin_component.view().insert_axis(Axis(0));

    let gasf = (&cos_i * &cos_j) - (&sin_i * &sin_j);
    Ok(gasf.into_pyarray(py).to_owned())
}

/// Compute the Markov Transition Field (MTF) of a 1D time series.
///
/// The series is discretized into `Q` bins over `[min(x), max(x)]` to obtain
/// states `q_t`. A first-order Markov transition matrix `P` is estimated:
///
/// `P[a, b] = count(q_t = a and q_{t+1} = b) / count(q_t = a)`
///
/// Then the field is constructed as:
///
/// `M[i, j] = P[q_i, q_j]`
///
/// # Arguments
/// * `time_series` - Input time series `x` as a 1D NumPy array
/// * `num_bins` - Number of discretization bins `Q` (must be >= 2)
///
/// # Returns
/// A 2D NumPy array with shape `(N, N)` containing the MTF matrix.
///
/// # Errors
/// Returns `PyValueError` if:
/// - The input is empty
/// - The input contains non-finite values
/// - `num_bins < 2`
#[pyfunction]
pub fn markov_transition_field<'py>(
    py: Python<'py>,
    time_series: PyReadonlyArray1<'py, f64>,
    num_bins: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = time_series.as_array().to_owned();
    validate_series(&data)?;

    if num_bins < 2 {
        return Err(PyValueError::new_err("num_bins must be at least 2"));
    }

    let n = data.len();
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;

    let bins: Vec<usize> = if range <= f64::EPSILON {
        vec![0; n]
    } else {
        data.iter()
            .map(|&x| {
                let scaled = ((x - min) / range).clamp(0.0, 1.0);
                let mut idx = (scaled * num_bins as f64).floor() as usize;
                if idx >= num_bins {
                    idx = num_bins - 1;
                }
                idx
            })
            .collect()
    };

    let mut transition = Array2::<f64>::zeros((num_bins, num_bins));
    for t in 0..n.saturating_sub(1) {
        let from = bins[t];
        let to = bins[t + 1];
        transition[[from, to]] += 1.0;
    }

    for i in 0..num_bins {
        let row_sum: f64 = transition.row(i).sum();
        if row_sum > 0.0 {
            transition.row_mut(i).mapv_inplace(|v| v / row_sum);
        }
    }

    let mut mtf = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            mtf[[i, j]] = transition[[bins[i], bins[j]]];
        }
    }

    Ok(mtf.into_pyarray(py).to_owned())
}
