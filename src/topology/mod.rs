/// Topological time-series analysis.
///
/// This module exposes four families of features that characterise the
/// *shape* of attractors reconstructed from scalar time series, using tools
/// from Topological Data Analysis (TDA):
///
/// 1. **Takens embedding** – reconstructs a multi-dimensional attractor from a
///    scalar time series via delay-coordinate embedding.
/// 2. **Persistent homology summaries** – measures how connected components,
///    loops, and voids are born and die as the scale grows.
/// 3. **Betti curves** – counts of active topological features across scales.
/// 4. **Persistence landscapes** – function-valued summaries of the diagram.
///
/// ## Quick-start
/// ```python
/// import numpy as np
/// import chronoxtract as ct
///
/// ts = np.cumsum(np.random.randn(200))
///
/// # Full pipeline in one call
/// feats = ct.topological_features(ts, dimension=3, delay=2)
///
/// # Or step-by-step:
/// pts = ct.takens_embedding(ts, dimension=3, delay=2)
/// phom = ct.persistent_homology_summary(pts)
/// betti = ct.betti_curve_features(pts)
/// landscape = ct.persistence_landscape_features(pts)
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray2, IntoPyArray};
use numpy::ndarray::Array2;

mod takens;
mod homology;
mod features;

use homology::MAX_POINTS_FULL_HOMOLOGY;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a Python 2-D float64 numpy array to a Vec<Vec<f64>>.
fn py2d_to_rust(arr: PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    let view = arr.as_array();
    let nrows = view.shape()[0];
    let ncols = view.shape()[1];
    (0..nrows)
        .map(|i| (0..ncols).map(|j| view[[i, j]]).collect())
        .collect()
}

/// Compute the t_min/t_max range for Betti/landscape features from a set of
/// persistence pairs and an optional explicit max_scale.
fn t_range(pairs: &[homology::PersistencePair], max_scale: Option<f64>) -> (f64, f64) {
    let t_min = 0.0_f64;
    let t_max = max_scale.unwrap_or_else(|| {
        pairs
            .iter()
            .filter(|p| p.death.is_finite())
            .map(|p| p.death)
            .fold(0.0_f64, f64::max)
            * 1.05 // add 5% margin
    });
    // Guard against degenerate (t_max == 0) case.
    let t_max = if t_max <= t_min { t_min + 1.0 } else { t_max };
    (t_min, t_max)
}

// ─── Python functions ─────────────────────────────────────────────────────────

/// Compute the Takens delay-coordinate embedding of a 1-D time series.
///
/// Given a scalar time series x[0..n], produces the sequence of d-dimensional
/// embedded vectors:
///
///   p_i = (x[i], x[i + τ], x[i + 2τ], …, x[i + (d-1)τ])
///
/// for i = 0, stride, 2·stride, … as long as the last index is within bounds.
///
/// # Arguments
/// * `time_series` - Input 1-D time series (numpy array or Python list).
/// * `dimension`   - Embedding dimension d ≥ 1.
/// * `delay`       - Time delay τ ≥ 1 (default 1).
/// * `stride`      - Step between successive windows (default 1).
/// * `normalize`   - Whether to z-score normalise each embedded vector
///                   (default False).
///
/// # Returns
/// A 2-D numpy array of shape `(n_points, dimension)`.
///
/// # Errors
/// Raises `ValueError` for empty input, invalid parameters, or time series
/// too short for the given `dimension` and `delay`.
///
/// # Example
/// ```python
/// import numpy as np
/// import chronoxtract as ct
///
/// ts = np.sin(np.linspace(0, 4 * np.pi, 200))
/// pts = ct.takens_embedding(ts, dimension=3, delay=5)
/// print(pts.shape)  # (190, 3)
/// ```
#[pyfunction]
#[pyo3(signature = (time_series, dimension, delay=1, stride=1, normalize=false))]
pub fn takens_embedding(
    py: Python,
    time_series: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    stride: usize,
    normalize: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let ts = time_series.as_array();
    let ts_slice = ts.as_slice().ok_or_else(|| {
        PyValueError::new_err("Input array must be contiguous")
    })?;

    let points = takens::takens_embedding_internal(ts_slice, dimension, delay, stride, normalize)
        .map_err(PyValueError::new_err)?;

    if points.is_empty() {
        return Err(PyValueError::new_err(
            "Embedding produced no points. Check that dimension, delay, and stride \
             are compatible with the time series length.",
        ));
    }

    let nrows = points.len();
    let ncols = dimension;
    let flat: Vec<f64> = points.into_iter().flatten().collect();
    let arr = Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;

    Ok(arr.into_pyarray(py).to_owned())
}

/// Compute persistent homology summary features for a 2-D point cloud.
///
/// This function estimates the topological structure of the point cloud by
/// building a Vietoris–Rips filtration and computing persistent homology.
///
/// * For **small clouds** (≤ `max_h1_points`): both H0 (connected components)
///   and H1 (loops) are computed via boundary-matrix reduction.
/// * For **large clouds**: only H0 is computed via an efficient union-find
///   algorithm.
///
/// # Arguments
/// * `points`        - 2-D numpy array of shape `(n_points, ambient_dim)`.
/// * `max_scale`     - Optional upper bound on filtration values.
/// * `max_h1_points` - Threshold for switching to H0-only mode (default 50).
///
/// # Returns
/// Dictionary with scalar features:
/// * `n_pairs_h{k}`, `max_persistence_h{k}`, `total_persistence_h{k}`,
///   `mean_persistence_h{k}`, `persistence_entropy_h{k}`, `n_essential_h{k}`
///   for k ∈ {0, 1}.
/// * `max_finite_scale` – largest finite filtration value in the diagram.
///
/// # Example
/// ```python
/// import numpy as np
/// import chronoxtract as ct
///
/// ts = np.cumsum(np.random.randn(150))
/// pts = ct.takens_embedding(ts, dimension=2, delay=3)
/// feats = ct.persistent_homology_summary(pts)
/// print(feats['n_pairs_h0'], feats['max_persistence_h1'])
/// ```
#[pyfunction]
#[pyo3(signature = (points, max_scale=None, max_h1_points=50))]
pub fn persistent_homology_summary<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    max_scale: Option<f64>,
    max_h1_points: usize,
) -> PyResult<Py<PyDict>> {
    let pts = py2d_to_rust(points);
    if pts.is_empty() {
        return Err(PyValueError::new_err("Point cloud cannot be empty"));
    }

    let pairs = homology::compute_persistence(&pts, max_scale, max_h1_points)
        .map_err(PyValueError::new_err)?;

    let summary = features::persistence_summary(&pairs);

    let dict = PyDict::new(py);
    for (k, v) in &summary {
        dict.set_item(k, v)?;
    }
    Ok(dict.into())
}

/// Compute Betti curve features from a 2-D point cloud.
///
/// The Betti-k curve is β_k(t) = number of k-dimensional topological features
/// alive at filtration value t:
///   β_k(t) = #{(b, d) ∈ Dgm_k  |  b ≤ t < d}
///
/// Summary statistics of each curve are returned.
///
/// # Arguments
/// * `points`        - 2-D numpy array `(n_points, ambient_dim)`.
/// * `n_samples`     - Number of sample points for the curve (default 50).
/// * `max_scale`     - Optional filtration range upper bound.
/// * `max_h1_points` - Threshold for H0-only mode (default 50).
///
/// # Returns
/// Dictionary with:
/// * `betti_{k}_auc`  – trapezoidal area under the Betti-k curve (k ∈ {0,1}).
/// * `betti_{k}_peak` – maximum Betti-k value.
/// * `betti_{k}_mean` – mean Betti-k value.
///
/// # Example
/// ```python
/// import numpy as np
/// import chronoxtract as ct
///
/// ts = np.sin(np.linspace(0, 8 * np.pi, 300))
/// pts = ct.takens_embedding(ts, dimension=2, delay=10)
/// betti_feats = ct.betti_curve_features(pts)
/// print(betti_feats)
/// ```
#[pyfunction]
#[pyo3(signature = (points, n_samples=50, max_scale=None, max_h1_points=50))]
pub fn betti_curve_features<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    n_samples: usize,
    max_scale: Option<f64>,
    max_h1_points: usize,
) -> PyResult<Py<PyDict>> {
    let pts = py2d_to_rust(points);
    if pts.is_empty() {
        return Err(PyValueError::new_err("Point cloud cannot be empty"));
    }

    let pairs = homology::compute_persistence(&pts, max_scale, max_h1_points)
        .map_err(PyValueError::new_err)?;

    let (t_min, t_max) = t_range(&pairs, max_scale);
    let summary = features::betti_curve_summary(&pairs, n_samples, t_min, t_max);

    let dict = PyDict::new(py);
    for (k, v) in &summary {
        dict.set_item(k, v)?;
    }
    Ok(dict.into())
}

/// Compute persistence landscape features from a 2-D point cloud.
///
/// The λ-th persistence landscape layer is the λ-th largest tent-function
/// value at each scale t:
///   f_{b,d}(t) = max(0, min(t − b, d − t))
///   λ_k(t) = k-th largest over all persistence pairs
///
/// Scalar summaries (L1 norm, L2 norm, peak, mean) are computed per layer and
/// per homology dimension.
///
/// # Arguments
/// * `points`        - 2-D numpy array `(n_points, ambient_dim)`.
/// * `n_layers`      - Number of landscape layers to compute (default 3).
/// * `n_samples`     - Sample points per layer (default 50).
/// * `max_scale`     - Optional filtration range upper bound.
/// * `max_h1_points` - Threshold for H0-only mode (default 50).
///
/// # Returns
/// Dictionary with keys `landscape_h{k}_l{l}_{norm}` for k ∈ {0,1},
/// l ∈ {1…n_layers}, norm ∈ {l1, l2, peak, mean}.
///
/// # Example
/// ```python
/// import numpy as np
/// import chronoxtract as ct
///
/// ts = np.sin(np.linspace(0, 8 * np.pi, 300))
/// pts = ct.takens_embedding(ts, dimension=2, delay=10)
/// landscape_feats = ct.persistence_landscape_features(pts)
/// print(landscape_feats['landscape_h0_l1_l1'])
/// ```
#[pyfunction]
#[pyo3(signature = (points, n_layers=3, n_samples=50, max_scale=None, max_h1_points=50))]
pub fn persistence_landscape_features<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    n_layers: usize,
    n_samples: usize,
    max_scale: Option<f64>,
    max_h1_points: usize,
) -> PyResult<Py<PyDict>> {
    let pts = py2d_to_rust(points);
    if pts.is_empty() {
        return Err(PyValueError::new_err("Point cloud cannot be empty"));
    }

    let pairs = homology::compute_persistence(&pts, max_scale, max_h1_points)
        .map_err(PyValueError::new_err)?;

    let (t_min, t_max) = t_range(&pairs, max_scale);
    let summary = features::landscape_summary(&pairs, n_layers, n_samples, t_min, t_max);

    let dict = PyDict::new(py);
    for (k, v) in &summary {
        dict.set_item(k, v)?;
    }
    Ok(dict.into())
}

/// Compute a comprehensive set of topological features from a 1-D time series.
///
/// This is a convenience function that:
/// 1. Applies a Takens delay-coordinate embedding to the time series.
/// 2. Computes persistent homology of the resulting point cloud.
/// 3. Returns persistent homology summaries, Betti curve features, and
///    persistence landscape features in a single dictionary.
///
/// It is equivalent to calling `takens_embedding` followed by
/// `persistent_homology_summary`, `betti_curve_features`, and
/// `persistence_landscape_features` and merging the results.
///
/// # Arguments
/// * `time_series`    - Input 1-D time series.
/// * `dimension`      - Takens embedding dimension (default 2).
/// * `delay`          - Takens time delay (default 1).
/// * `stride`         - Embedding stride (default 1).
/// * `normalize`      - Normalise embedded vectors (default False).
/// * `max_scale`      - Optional filtration cutoff.
/// * `max_h1_points`  - H0-only threshold for large clouds (default 50).
/// * `n_betti_samples`    - Betti curve sample count (default 50).
/// * `n_landscape_layers` - Number of landscape layers (default 3).
/// * `n_landscape_samples`- Sample count for landscape (default 50).
///
/// # Returns
/// Dictionary combining all topological features plus:
/// * `n_embedding_points` – number of points in the embedding.
/// * `embedding_dim`      – ambient dimension of the embedding.
/// * `h1_computed`        – 1.0 if H1 was computed, 0.0 if H0-only.
///
/// # Example
/// ```python
/// import numpy as np
/// import chronoxtract as ct
///
/// np.random.seed(0)
/// ts = np.cumsum(np.random.randn(300))
/// feats = ct.topological_features(ts, dimension=3, delay=2)
/// print(sorted(feats.keys()))
/// ```
#[pyfunction]
#[pyo3(signature = (
    time_series,
    dimension = 2,
    delay = 1,
    stride = 1,
    normalize = false,
    max_scale = None,
    max_h1_points = 50,
    n_betti_samples = 50,
    n_landscape_layers = 3,
    n_landscape_samples = 50,
))]
pub fn topological_features<'py>(
    py: Python<'py>,
    time_series: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    stride: usize,
    normalize: bool,
    max_scale: Option<f64>,
    max_h1_points: usize,
    n_betti_samples: usize,
    n_landscape_layers: usize,
    n_landscape_samples: usize,
) -> PyResult<Py<PyDict>> {
    let ts = time_series.as_array();
    let ts_slice = ts.as_slice().ok_or_else(|| {
        PyValueError::new_err("Input array must be contiguous")
    })?;

    // Step 1: Takens embedding
    let points =
        takens::takens_embedding_internal(ts_slice, dimension, delay, stride, normalize)
            .map_err(PyValueError::new_err)?;

    if points.is_empty() {
        return Err(PyValueError::new_err(
            "Embedding produced no points. Check dimension, delay, and stride \
             against the time series length.",
        ));
    }

    let n_pts = points.len();
    let h1_computed = n_pts <= max_h1_points;

    // Step 2: Persistent homology
    let limit = if h1_computed {
        max_h1_points
    } else {
        MAX_POINTS_FULL_HOMOLOGY
    };
    let pairs = homology::compute_persistence(&points, max_scale, limit)
        .map_err(PyValueError::new_err)?;

    let (t_min, t_max) = t_range(&pairs, max_scale);

    // Step 3: All feature groups
    let phom_feats = features::persistence_summary(&pairs);
    let betti_feats =
        features::betti_curve_summary(&pairs, n_betti_samples, t_min, t_max);
    let landscape_feats =
        features::landscape_summary(&pairs, n_landscape_layers, n_landscape_samples, t_min, t_max);

    // Step 4: Assemble output dictionary
    let dict = PyDict::new(py);
    for (k, v) in phom_feats
        .iter()
        .chain(betti_feats.iter())
        .chain(landscape_feats.iter())
    {
        dict.set_item(k, v)?;
    }
    dict.set_item("n_embedding_points", n_pts as f64)?;
    dict.set_item("embedding_dim", dimension as f64)?;
    dict.set_item("h1_computed", if h1_computed { 1.0_f64 } else { 0.0 })?;

    Ok(dict.into())
}
