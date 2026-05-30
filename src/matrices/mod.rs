use numpy::ndarray::{Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn validate_series(data: ArrayView1<'_, f64>) -> PyResult<()> {
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

fn min_max(data: ArrayView1<'_, f64>) -> (f64, f64) {
    data.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), &x| {
            (min_v.min(x), max_v.max(x))
        })
}

fn has_degenerate_range(min: f64, max: f64) -> bool {
    let scale = min.abs().max(max.abs()).max(1.0);
    (max - min).abs() <= 1e-12 * scale
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
    let data = time_series.as_array();
    validate_series(data)?;

    if window_length == 0 {
        return Err(PyValueError::new_err("window_length must be greater than 0"));
    }
    if window_length > data.len() {
        return Err(PyValueError::new_err(
            "window_length must be less than or equal to time series length",
        ));
    }

    let n_rows = data.len() - window_length + 1;
    let mut hankel = vec![0.0; n_rows * window_length];

    if let Some(slice) = data.as_slice() {
        // Parallelization threshold tuned from local benchmarks to avoid
        // threadpool overhead for small inputs.
        if n_rows >= 512 {
            hankel
                .par_chunks_mut(window_length)
                .enumerate()
                .for_each(|(i, row)| row.copy_from_slice(&slice[i..i + window_length]));
        } else {
            for (i, row) in hankel.chunks_mut(window_length).enumerate() {
                row.copy_from_slice(&slice[i..i + window_length]);
            }
        }
    } else {
        for i in 0..n_rows {
            for j in 0..window_length {
                hankel[i * window_length + j] = data[i + j];
            }
        }
    }

    let hankel = Array2::from_shape_vec((n_rows, window_length), hankel)
        .map_err(|e| PyValueError::new_err(format!("Failed to build embedding matrix: {e}")))?;

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
    let data = time_series.as_array();
    validate_series(data)?;

    let (min, max) = min_max(data);
    let range = max - min;
    let n = data.len();

    let normalized: Vec<f64> = if has_degenerate_range(min, max) {
        vec![0.0; n]
    } else {
        data.iter()
            .map(|&x| (2.0 * (x - min) / range - 1.0).clamp(-1.0, 1.0))
            .collect()
    };

    // sin(phi) where phi = arccos(x) and x is the normalized cosine component.
    let sin_component: Vec<f64> = normalized
        .iter()
        .map(|&x| (1.0 - x * x).max(0.0).sqrt())
        .collect();

    let mut gasf = vec![0.0; n * n];
    // Parallelization threshold tuned from local benchmarks to avoid
    // threadpool overhead for small matrices.
    if n >= 128 {
        gasf.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let ci = normalized[i];
            let si = sin_component[i];
            for j in 0..n {
                row[j] = ci * normalized[j] - si * sin_component[j];
            }
        });
    } else {
        for (i, row) in gasf.chunks_mut(n).enumerate() {
            let ci = normalized[i];
            let si = sin_component[i];
            for j in 0..n {
                row[j] = ci * normalized[j] - si * sin_component[j];
            }
        }
    }

    let gasf = Array2::from_shape_vec((n, n), gasf)
        .map_err(|e| PyValueError::new_err(format!("Failed to build GASF matrix: {e}")))?;
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
    let data = time_series.as_array();
    validate_series(data)?;

    if num_bins < 2 {
        return Err(PyValueError::new_err("num_bins must be at least 2"));
    }

    let n = data.len();
    let (min, max) = min_max(data);
    let range = max - min;

    let bins: Vec<usize> = if has_degenerate_range(min, max) {
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

    let mut transition = vec![0.0; num_bins * num_bins];
    for t in 0..n.saturating_sub(1) {
        let from = bins[t];
        let to = bins[t + 1];
        transition[from * num_bins + to] += 1.0;
    }

    for i in 0..num_bins {
        let row_start = i * num_bins;
        let row_end = row_start + num_bins;
        let row_sum: f64 = transition[row_start..row_end].iter().sum();
        if row_sum > 0.0 {
            for value in &mut transition[row_start..row_end] {
                *value /= row_sum;
            }
        }
    }

    let mut mtf = vec![0.0; n * n];
    // Parallelization threshold tuned from local benchmarks to avoid
    // threadpool overhead for small matrices.
    if n >= 128 {
        mtf.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let row_offset = bins[i] * num_bins;
            for j in 0..n {
                row[j] = transition[row_offset + bins[j]];
            }
        });
    } else {
        for (i, row) in mtf.chunks_mut(n).enumerate() {
            let row_offset = bins[i] * num_bins;
            for j in 0..n {
                row[j] = transition[row_offset + bins[j]];
            }
        }
    }

    let mtf = Array2::from_shape_vec((n, n), mtf)
        .map_err(|e| PyValueError::new_err(format!("Failed to build MTF matrix: {e}")))?;

    Ok(mtf.into_pyarray(py).to_owned())
}
