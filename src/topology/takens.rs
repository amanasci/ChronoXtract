/// Takens delay-coordinate embedding for scalar time series.
///
/// Given a 1-D scalar time series `x[0..n]`, the Takens embedding produces a
/// sequence of d-dimensional vectors:
///
///   p_i = (x[i], x[i + τ], x[i + 2τ], …, x[i + (d-1)τ])
///
/// for i = 0, stride, 2·stride, … while the last index `i + (d-1)τ < n`.
///
/// When `normalize = true`, every embedded vector is independently z-score
/// normalised (zero mean, unit variance) before being returned.  Vectors with
/// zero variance are left unchanged.

/// Compute the Takens embedding of a 1-D time series.
///
/// # Arguments
/// * `time_series` - Input scalar time series as a slice.
/// * `dimension`   - Embedding dimension d ≥ 1.
/// * `delay`       - Time delay τ ≥ 1.
/// * `stride`      - Step between successive embedding windows (≥ 1).
/// * `normalize`   - Whether to z-score normalise each embedded vector.
///
/// # Returns
/// `Ok(Vec<Vec<f64>>)` containing `n_points` rows, each of length `dimension`.
/// The number of points is `floor((n − (d-1)·τ − 1) / stride) + 1`.
///
/// # Errors
/// Returns an error string when:
/// * The time series is empty.
/// * `dimension`, `delay`, or `stride` is zero.
/// * The time series is too short for the chosen parameters.
pub(crate) fn takens_embedding_internal(
    time_series: &[f64],
    dimension: usize,
    delay: usize,
    stride: usize,
    normalize: bool,
) -> Result<Vec<Vec<f64>>, String> {
    if time_series.is_empty() {
        return Err("Input time series cannot be empty".to_string());
    }
    if dimension == 0 {
        return Err("Embedding dimension must be at least 1".to_string());
    }
    if delay == 0 {
        return Err("Delay must be at least 1".to_string());
    }
    if stride == 0 {
        return Err("Stride must be at least 1".to_string());
    }

    let n = time_series.len();
    // Minimum number of samples required: the last element of the last component
    // in the *first* window is at index (dimension - 1) * delay.
    let min_length = (dimension - 1) * delay + 1;
    if n < min_length {
        return Err(format!(
            "Time series length {} is too short for dimension={} and delay={} \
             (need at least {} points)",
            n, dimension, delay, min_length
        ));
    }

    // Number of valid starting positions (before stride is applied).
    let max_start = n - (dimension - 1) * delay; // exclusive upper bound

    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut i = 0;
    while i < max_start {
        let mut point: Vec<f64> = (0..dimension)
            .map(|d| time_series[i + d * delay])
            .collect();

        if normalize {
            let mean = point.iter().sum::<f64>() / dimension as f64;
            let var = point.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                / dimension as f64;
            let std = var.sqrt();
            // Use a data-scale-relative threshold to avoid dividing by tiny std.
            let scale = point.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
            if std > scale * 1e-8 {
                for v in &mut point {
                    *v = (*v - mean) / std;
                }
            }
        }

        points.push(point);
        i += stride;
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_embedding() {
        // dim=2, delay=1: pairs (x[0],x[1]), (x[1],x[2]), (x[2],x[3])
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = takens_embedding_internal(&ts, 2, 1, 1, false).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], vec![1.0, 2.0]);
        assert_eq!(result[3], vec![4.0, 5.0]);
    }

    #[test]
    fn test_embedding_with_delay() {
        // dim=2, delay=2: pairs (x[0],x[2]), (x[1],x[3]), (x[2],x[4])
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = takens_embedding_internal(&ts, 2, 2, 1, false).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![1.0, 3.0]);
        assert_eq!(result[1], vec![2.0, 4.0]);
        assert_eq!(result[2], vec![3.0, 5.0]);
    }

    #[test]
    fn test_embedding_with_stride() {
        // dim=2, delay=1, stride=2
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = takens_embedding_internal(&ts, 2, 1, 2, false).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0]);
        assert_eq!(result[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_embedding_3d() {
        // dim=3, delay=1: triples starting at 0,1,...
        let ts: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = takens_embedding_internal(&ts, 3, 1, 1, false).unwrap();
        // valid starts: 0..=7 (n - (d-1)*delay = 10 - 2 = 8)
        assert_eq!(result.len(), 8);
        assert_eq!(result[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(result[7], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_embedding_normalize() {
        let ts = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let result = takens_embedding_internal(&ts, 2, 1, 1, true).unwrap();
        // Each embedded vector should have mean ≈ 0 and std ≈ 1
        for point in &result {
            let mean = point.iter().sum::<f64>() / point.len() as f64;
            let var = point.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                / point.len() as f64;
            assert!((mean).abs() < 1e-10);
            assert!((var.sqrt() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_error_empty() {
        let result = takens_embedding_internal(&[], 2, 1, 1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_zero_dimension() {
        let ts = vec![1.0, 2.0, 3.0];
        let result = takens_embedding_internal(&ts, 0, 1, 1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_short_series() {
        // Need at least (dim-1)*delay+1 = 3 points for dim=2, delay=2
        let ts = vec![1.0, 2.0];
        let result = takens_embedding_internal(&ts, 2, 2, 1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_point_dimension_one() {
        let ts = vec![42.0];
        let result = takens_embedding_internal(&ts, 1, 1, 1, false).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![42.0]);
    }
}
