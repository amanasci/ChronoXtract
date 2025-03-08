use std::f64::consts::PI;

// lombscargle.rs
//
// This module implements a simple Lomb-Scargle periodogram method for unevenly-sampled data.
// The function lomb_scargle takes as input slices of time values t, signal values y, and a slice
// of frequencies at which the periodogram is evaluated. It returns a vector of power values.
//
// References:
// Lomb, N.R. "Least-squares frequency analysis of unequally spaced data." Astrophysics and Space
// Science 39.2 (1976): 447-462.
// Scargle, J.D. "Studies in astronomical time series analysis. II-Statistical aspects of spectral analysis
// of unevenly spaced data." The Astrophysical Journal 263 (1982): 835-853.


/// Computes the Lomb-Scargle periodogram power for a set of frequencies.
///
/// # Arguments
///
/// * `t` - A slice of time values.
/// * `y` - A slice of observed values corresponding to times in `t`.
/// * `freqs` - A slice of frequencies at which to evaluate the periodogram.
///
/// # Returns
///
/// A vector containing the periodogram power at each frequency.
///
/// # Panics
///
/// Panics if `t` and `y` have different lengths.
pub fn lomb_scargle(t: &[f64], y: &[f64], freqs: &[f64]) -> Vec<f64> {
    assert_eq!(t.len(), y.len(), "t and y must have the same length.");
    let n = t.len();
    let mut power = Vec::with_capacity(freqs.len());

    for &f in freqs {
        let omega = 2.0 * PI * f;

        // Compute sums needed for determining tau.
        let (mut sum_sin2wt, mut sum_cos2wt) = (0.0, 0.0);
        for &ti in t {
            let arg = 2.0 * omega * ti;
            sum_sin2wt += arg.sin();
            sum_cos2wt += arg.cos();
        }

        // Avoid division by zero in tau computation.
        let tau = if omega.abs() > std::f64::EPSILON {
            0.5 * (sum_sin2wt / sum_cos2wt).atan() / omega
        } else {
            0.0
        };

        // Accumulate sums for cosine and sine components.
        let (mut sum_yc, mut sum_ys, mut sum_c2, mut sum_s2) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let phi = omega * (t[i] - tau);
            let c = phi.cos();
            let s = phi.sin();
            sum_yc += y[i] * c;
            sum_ys += y[i] * s;
            sum_c2 += c * c;
            sum_s2 += s * s;
        }

        // Compute the power at this frequency.
        let p = 0.5 * (if sum_c2.abs() > std::f64::EPSILON { (sum_yc * sum_yc) / sum_c2 } else { 0.0 }
        + if sum_s2.abs() > std::f64::EPSILON { (sum_ys * sum_ys) / sum_s2 } else { 0.0 });

        power.push(p);
    }

    power
}