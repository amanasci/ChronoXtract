/// Feature extraction from persistence diagrams.
///
/// This module transforms raw persistence pairs into compact scalar feature
/// vectors suitable for downstream machine-learning or statistical analysis.
///
/// Three families of features are provided:
///
/// 1. **Persistence summaries** – scalar statistics of birth/death/persistence
///    values, computed per homology dimension.
///
/// 2. **Betti curves** – the Betti number β_k(t) counts how many k-dimensional
///    topological features are "alive" at filtration value t:
///      β_k(t) = #{(b,d) ∈ Dgm_k  |  b ≤ t < d}
///    Summary statistics (area, peak, mean) of each curve are returned.
///
/// 3. **Persistence landscapes** – for homology dimension k, the λ-th landscape
///    layer is the λ-th largest tent function value:
///      f_{b,d}(t) = max(0, min(t−b, d−t))
///    λ_1(t) ≥ λ_2(t) ≥ … ≥ 0.
///    Layer norms, peaks, and means are returned as scalar features.

use std::collections::HashMap;
use super::homology::PersistencePair;

// ─── Persistence summaries ────────────────────────────────────────────────────

/// Compute scalar summary statistics of a persistence diagram.
///
/// The returned `HashMap` contains the following keys, each suffixed with `_h0`
/// or `_h1` for the respective homology dimension:
///
/// * `n_pairs_h{k}`           – number of **finite** persistence pairs.
/// * `max_persistence_h{k}`   – maximum (death − birth) over finite pairs.
/// * `total_persistence_h{k}` – sum of (death − birth) over finite pairs.
/// * `mean_persistence_h{k}`  – arithmetic mean of persistence over finite pairs.
/// * `persistence_entropy_h{k}` – entropy of the normalised persistence lengths.
/// * `n_essential_h{k}`       – number of essential (death = ∞) classes.
///
/// `max_finite_scale` is the maximum *finite* filtration value in the diagram;
/// it is used to replace ∞ when computing statistics that require a finite
/// upper bound (it is **not** included in persistence computations but is
/// returned as `max_finite_scale`).
pub(crate) fn persistence_summary(
    pairs: &[PersistencePair],
) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();

    for dim in 0..=1 {
        let suffix = format!("_h{}", dim);
        let finite: Vec<f64> = pairs
            .iter()
            .filter(|p| p.dim == dim && p.death.is_finite())
            .map(|p| p.death - p.birth)
            .collect();
        let n_essential = pairs
            .iter()
            .filter(|p| p.dim == dim && p.death.is_infinite())
            .count() as f64;

        let n = finite.len() as f64;
        let total: f64 = finite.iter().sum();
        let max_pers = finite.iter().cloned().fold(0.0_f64, f64::max);
        let mean_pers = if n > 0.0 { total / n } else { 0.0 };

        // Persistence entropy: −Σ p_i log(p_i) where p_i = pers_i / total
        let entropy = if total > 1e-12 {
            finite
                .iter()
                .map(|&p| {
                    let pi = p / total;
                    if pi > 1e-12 {
                        -pi * pi.ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        } else {
            0.0
        };

        out.insert(format!("n_pairs{}", suffix), n);
        out.insert(format!("max_persistence{}", suffix), max_pers);
        out.insert(format!("total_persistence{}", suffix), total);
        out.insert(format!("mean_persistence{}", suffix), mean_pers);
        out.insert(format!("persistence_entropy{}", suffix), entropy);
        out.insert(format!("n_essential{}", suffix), n_essential);
    }

    // Overall max finite filtration value across all pairs.
    let max_scale = pairs
        .iter()
        .filter(|p| p.death.is_finite())
        .map(|p| p.death)
        .fold(0.0_f64, f64::max);
    out.insert("max_finite_scale".to_string(), max_scale);

    out
}

// ─── Betti curves ─────────────────────────────────────────────────────────────

/// Compute sampled Betti curves and their summary statistics.
///
/// `t_values` is a grid of `n_samples` evenly-spaced points in the interval
/// `[t_min, t_max]`.  At each t, β_k(t) = #{pairs (b,d) | b ≤ t < d}.
/// Essential pairs (d = ∞) are treated as d = +∞, so they are always alive.
///
/// # Returns
/// A tuple `(t_values, betti0_curve, betti1_curve)` of length `n_samples`.
pub(crate) fn compute_betti_curves(
    pairs: &[PersistencePair],
    n_samples: usize,
    t_min: f64,
    t_max: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = n_samples.max(2);
    let step = if t_max > t_min {
        (t_max - t_min) / (n - 1) as f64
    } else {
        0.0
    };

    let t_values: Vec<f64> = (0..n).map(|i| t_min + i as f64 * step).collect();
    let mut betti0 = vec![0.0_f64; n];
    let mut betti1 = vec![0.0_f64; n];

    for (idx, &t) in t_values.iter().enumerate() {
        for p in pairs {
            // Active iff b ≤ t < d  (essential pairs: d = ∞ → always alive once born)
            if p.birth <= t && t < p.death {
                match p.dim {
                    0 => betti0[idx] += 1.0,
                    1 => betti1[idx] += 1.0,
                    _ => {}
                }
            }
        }
    }

    (t_values, betti0, betti1)
}

/// Compute Betti-curve scalar summaries.
///
/// For each of β_0 and β_1, the returned map contains:
/// * `betti_{k}_auc`  – trapezoidal area under the Betti-k curve.
/// * `betti_{k}_peak` – maximum value of the Betti-k curve.
/// * `betti_{k}_mean` – arithmetic mean of the Betti-k curve.
pub(crate) fn betti_curve_summary(
    pairs: &[PersistencePair],
    n_samples: usize,
    t_min: f64,
    t_max: f64,
) -> HashMap<String, f64> {
    let (t, b0, b1) = compute_betti_curves(pairs, n_samples, t_min, t_max);

    let summarise = |curve: &[f64], t_vals: &[f64]| -> (f64, f64, f64) {
        let peak = curve.iter().cloned().fold(0.0_f64, f64::max);
        let mean = if !curve.is_empty() {
            curve.iter().sum::<f64>() / curve.len() as f64
        } else {
            0.0
        };
        // Trapezoidal AUC
        let auc = if t_vals.len() >= 2 {
            (0..t_vals.len() - 1)
                .map(|i| {
                    let dt = t_vals[i + 1] - t_vals[i];
                    0.5 * (curve[i] + curve[i + 1]) * dt
                })
                .sum()
        } else {
            0.0
        };
        (auc, peak, mean)
    };

    let (auc0, peak0, mean0) = summarise(&b0, &t);
    let (auc1, peak1, mean1) = summarise(&b1, &t);

    let mut out = HashMap::new();
    out.insert("betti_0_auc".to_string(), auc0);
    out.insert("betti_0_peak".to_string(), peak0);
    out.insert("betti_0_mean".to_string(), mean0);
    out.insert("betti_1_auc".to_string(), auc1);
    out.insert("betti_1_peak".to_string(), peak1);
    out.insert("betti_1_mean".to_string(), mean1);
    out
}

// ─── Persistence landscapes ───────────────────────────────────────────────────

/// Tent function for a single persistence pair (b, d) evaluated at t.
///
///   f_{b,d}(t) = max(0, min(t − b, d − t))
///
/// For essential pairs (d = ∞) the tent function is unbounded; we replace d
/// by `d_cap` to keep things finite.
#[inline]
fn tent(t: f64, b: f64, d: f64) -> f64 {
    let mid = (b + d) / 2.0;
    if t <= b || t >= d {
        0.0
    } else if t <= mid {
        t - b
    } else {
        d - t
    }
}

/// Sample the persistence landscape for a given homology dimension.
///
/// The λ-th landscape layer (1-indexed) is the λ-th largest tent-function
/// value over all persistence pairs in dimension `dim`.
///
/// # Returns
/// A `Vec<Vec<f64>>` of shape `(n_layers, n_samples)`.  Layers are
/// 0-indexed: index 0 = λ_1, index 1 = λ_2, etc.
pub(crate) fn compute_landscape(
    pairs: &[PersistencePair],
    dim: usize,
    n_layers: usize,
    n_samples: usize,
    t_min: f64,
    t_max: f64,
    d_cap: f64,
) -> Vec<Vec<f64>> {
    let n = n_samples.max(2);
    let step = if t_max > t_min {
        (t_max - t_min) / (n - 1) as f64
    } else {
        0.0
    };

    // Pairs for the requested dimension, with essential d capped.
    let dim_pairs: Vec<(f64, f64)> = pairs
        .iter()
        .filter(|p| p.dim == dim)
        .map(|p| {
            let d = if p.death.is_infinite() { d_cap } else { p.death };
            (p.birth, d)
        })
        .collect();

    let mut landscape: Vec<Vec<f64>> = vec![vec![0.0; n]; n_layers];

    for (s_idx, t) in (0..n).map(|i| t_min + i as f64 * step).enumerate() {
        // Collect tent values at t for all pairs.
        let mut vals: Vec<f64> = dim_pairs
            .iter()
            .map(|&(b, d)| tent(t, b, d))
            .collect();
        // Sort descending so vals[0] = λ_1(t), vals[1] = λ_2(t), etc.
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        for layer in 0..n_layers {
            landscape[layer][s_idx] = if layer < vals.len() { vals[layer] } else { 0.0 };
        }
    }

    landscape
}

/// Compute persistence-landscape scalar summaries.
///
/// For each homology dimension k ∈ {0, 1} and each landscape layer
/// l ∈ {1, …, n_layers}, the returned map contains:
/// * `landscape_h{k}_l{l}_l1`   – L1 norm (trapezoidal integral).
/// * `landscape_h{k}_l{l}_l2`   – L2 norm (sqrt of trapezoidal integral of λ²).
/// * `landscape_h{k}_l{l}_peak` – maximum landscape value in that layer.
/// * `landscape_h{k}_l{l}_mean` – arithmetic mean of that layer.
pub(crate) fn landscape_summary(
    pairs: &[PersistencePair],
    n_layers: usize,
    n_samples: usize,
    t_min: f64,
    t_max: f64,
) -> HashMap<String, f64> {
    let d_cap = if t_max.is_finite() {
        t_max * 1.1
    } else {
        // Fall back to the maximum finite death value in the diagram, or 1.0.
        pairs
            .iter()
            .filter(|p| p.death.is_finite())
            .map(|p| p.death)
            .fold(1.0_f64, f64::max)
            * 1.1
    };
    let n = n_samples.max(2);
    let step = if t_max > t_min {
        (t_max - t_min) / (n - 1) as f64
    } else {
        0.0
    };

    let mut out = HashMap::new();

    for dim in 0..=1_usize {
        let layers = compute_landscape(pairs, dim, n_layers, n, t_min, t_max, d_cap);
        for (l_idx, layer) in layers.iter().enumerate() {
            let l_label = l_idx + 1; // 1-indexed in the API
            let prefix = format!("landscape_h{}_l{}", dim, l_label);

            let peak = layer.iter().cloned().fold(0.0_f64, f64::max);
            let mean = layer.iter().sum::<f64>() / layer.len() as f64;

            // Trapezoidal integrals
            let l1: f64 = if layer.len() >= 2 {
                (0..layer.len() - 1)
                    .map(|i| 0.5 * (layer[i] + layer[i + 1]) * step)
                    .sum()
            } else {
                0.0
            };
            let l2: f64 = if layer.len() >= 2 {
                let sq_integral: f64 = (0..layer.len() - 1)
                    .map(|i| 0.5 * (layer[i].powi(2) + layer[i + 1].powi(2)) * step)
                    .sum();
                sq_integral.sqrt()
            } else {
                0.0
            };

            out.insert(format!("{}_l1", prefix), l1);
            out.insert(format!("{}_l2", prefix), l2);
            out.insert(format!("{}_peak", prefix), peak);
            out.insert(format!("{}_mean", prefix), mean);
        }
    }

    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pairs() -> Vec<PersistencePair> {
        vec![
            PersistencePair { dim: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dim: 0, birth: 0.0, death: 2.0 },
            PersistencePair { dim: 0, birth: 0.0, death: f64::INFINITY },
            PersistencePair { dim: 1, birth: 0.5, death: 1.5 },
        ]
    }

    // ── Persistence summary ──────────────────────────────────────────────────

    #[test]
    fn test_summary_counts() {
        let pairs = sample_pairs();
        let s = persistence_summary(&pairs);
        assert_eq!(s["n_pairs_h0"], 2.0);   // two finite H0 pairs
        assert_eq!(s["n_essential_h0"], 1.0);
        assert_eq!(s["n_pairs_h1"], 1.0);
        assert_eq!(s["n_essential_h1"], 0.0);
    }

    #[test]
    fn test_summary_max_persistence() {
        let pairs = sample_pairs();
        let s = persistence_summary(&pairs);
        assert!((s["max_persistence_h0"] - 2.0).abs() < 1e-10);
        assert!((s["max_persistence_h1"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_summary_total_mean() {
        let pairs = sample_pairs();
        let s = persistence_summary(&pairs);
        // H0 finite: pers = 1.0 and 2.0 → total = 3.0, mean = 1.5
        assert!((s["total_persistence_h0"] - 3.0).abs() < 1e-10);
        assert!((s["mean_persistence_h0"] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_summary_entropy_positive() {
        let pairs = sample_pairs();
        let s = persistence_summary(&pairs);
        assert!(s["persistence_entropy_h0"] > 0.0);
    }

    #[test]
    fn test_summary_empty() {
        let s = persistence_summary(&[]);
        assert_eq!(s["n_pairs_h0"], 0.0);
        assert_eq!(s["n_pairs_h1"], 0.0);
    }

    // ── Betti curves ────────────────────────────────────────────────────────

    #[test]
    fn test_betti_curve_basic() {
        let pairs = sample_pairs();
        let (t, b0, b1) = compute_betti_curves(&pairs, 10, 0.0, 3.0);
        assert_eq!(t.len(), 10);
        assert_eq!(b0.len(), 10);
        assert_eq!(b1.len(), 10);
        // At t=0: all three H0 pairs alive (b=0 ≤ 0 < d), β0=3
        assert!((b0[0] - 3.0).abs() < 1e-10);
        // At t=2.5: only the essential H0 pair survives, β0=1
        let t25_idx = t.iter().position(|&v| v >= 2.5).unwrap();
        assert!((b0[t25_idx] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_betti_curve_summary_keys() {
        let pairs = sample_pairs();
        let s = betti_curve_summary(&pairs, 20, 0.0, 3.0);
        for key in &[
            "betti_0_auc", "betti_0_peak", "betti_0_mean",
            "betti_1_auc", "betti_1_peak", "betti_1_mean",
        ] {
            assert!(s.contains_key(*key), "missing key: {}", key);
        }
    }

    #[test]
    fn test_betti_curve_auc_positive() {
        let pairs = sample_pairs();
        let s = betti_curve_summary(&pairs, 50, 0.0, 3.0);
        assert!(s["betti_0_auc"] > 0.0);
        assert!(s["betti_1_auc"] > 0.0);
    }

    // ── Persistence landscape ───────────────────────────────────────────────

    #[test]
    fn test_tent_function() {
        // Tent for pair (0, 2): peak at t=1 with height 1.0
        assert!((tent(1.0, 0.0, 2.0) - 1.0).abs() < 1e-10);
        assert!((tent(0.0, 0.0, 2.0)).abs() < 1e-10);
        assert!((tent(2.0, 0.0, 2.0)).abs() < 1e-10);
        // t < b or t >= d → 0
        assert!((tent(-0.1, 0.0, 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_landscape_shape() {
        let pairs = sample_pairs();
        let lsc = compute_landscape(&pairs, 0, 3, 20, 0.0, 3.0, 10.0);
        assert_eq!(lsc.len(), 3);   // n_layers
        assert_eq!(lsc[0].len(), 20); // n_samples
        // Layer 1 ≥ layer 2 ≥ layer 3 at every t
        for s in 0..20 {
            assert!(lsc[0][s] >= lsc[1][s]);
            assert!(lsc[1][s] >= lsc[2][s]);
        }
    }

    #[test]
    fn test_landscape_summary_keys() {
        let pairs = sample_pairs();
        let s = landscape_summary(&pairs, 2, 20, 0.0, 3.0);
        for key in &[
            "landscape_h0_l1_l1", "landscape_h0_l1_l2",
            "landscape_h0_l1_peak", "landscape_h0_l1_mean",
            "landscape_h1_l1_l1",
        ] {
            assert!(s.contains_key(*key), "missing key: {}", key);
        }
    }

    #[test]
    fn test_landscape_norms_non_negative() {
        let pairs = sample_pairs();
        let s = landscape_summary(&pairs, 2, 30, 0.0, 3.0);
        for (k, v) in &s {
            assert!(*v >= 0.0, "negative value for key: {}", k);
        }
    }
}
