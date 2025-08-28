// src/correlation/zdcf.rs

use super::dcf::{TimeSeriesPoint, CorrelationPoint};
use rand_distr::{Normal, Distribution};
use rand::thread_rng;

fn fishs(r: f64, n: f64) -> f64 {
    // Fisher's small sample approximation for s(z) (Kendall + Stuart Vol. 1 p.391)
    let r2 = r * r;
    let n_minus_1 = n - 1.0;
    let term1 = 1.0 / n_minus_1;
    let term2 = (4.0 - r2) / (2.0 * n_minus_1);
    let term3 = (22.0 - 6.0 * r2 - 3.0 * r2 * r2) / (6.0 * n_minus_1 * n_minus_1);
    (term1 * (1.0 + term2 + term3)).max(0.0).sqrt()
}

fn fishe(r: f64, n: f64) -> f64 {
    // Fisher's small sample approximation for E(z) (Kendall + Stuart Vol. 1 p.391)
    let r2 = r * r;
    let n_minus_1 = n - 1.0;
    let term1 = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let term2 = r / (2.0 * n_minus_1);
    let term3 = 1.0 + (5.0 + r2) / (4.0 * n_minus_1);
    let term4 = (11.0 + 2.0 * r2 + 3.0 * r2 * r2) / (8.0 * n_minus_1 * n_minus_1);
    term1 + term2 * (term3 + term4)
}

fn clcdcf(
    series1_mc: &[f64],
    series2_mc: &[f64],
    series1: &[TimeSeriesPoint],
    series2: &[TimeSeriesPoint],
    bins: &[Vec<(usize, usize)>],
) -> Vec<(f64, f64)> {
    let mut results = Vec::new();

    for bin in bins {
        let n = bin.len() as f64;
        if n < 2.0 {
            continue;
        }

        let mut lag_sum = 0.0;
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;
        let mut sum1_sq = 0.0;
        let mut sum2_sq = 0.0;
        let mut sum12 = 0.0;

        // Calculate sums for this bin
        for &(i, j) in bin {
            let x1 = series1_mc[i];
            let x2 = series2_mc[j];
            
            lag_sum += series2[j].time - series1[i].time;
            sum1 += x1;
            sum2 += x2;
            sum1_sq += x1 * x1;
            sum2_sq += x2 * x2;
            sum12 += x1 * x2;
        }

        let lag = lag_sum / n;
        
        // Calculate correlation coefficient for this bin
        let mean1 = sum1 / n;
        let mean2 = sum2 / n;
        
        let var1 = sum1_sq / n - mean1 * mean1;
        let var2 = sum2_sq / n - mean2 * mean2;
        let covar = sum12 / n - mean1 * mean2;
        
        let r = if var1 > 0.0 && var2 > 0.0 {
            covar / (var1.sqrt() * var2.sqrt())
        } else {
            0.0
        };
        
        results.push((lag, r));
    }
    results
}

fn alcbin(
    series1: &[TimeSeriesPoint],
    series2: &[TimeSeriesPoint],
    min_points: usize,
) -> Vec<Vec<(usize, usize)>> {
    let n1 = series1.len();
    let n2 = series2.len();
    
    // Create all possible lag pairs
    let mut time_lags: Vec<(f64, usize, usize)> = Vec::with_capacity(n1 * n2);
    for i in 0..n1 {
        for j in 0..n2 {
            time_lags.push((series2[j].time - series1[i].time, i, j));
        }
    }
    
    // Sort by time lag
    time_lags.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let n_pairs = time_lags.len();
    let mut bins: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut negative_bins: Vec<Vec<(usize, usize)>> = Vec::new();
    
    // Alexander's algorithm: Start from median and work in both directions
    // Phase 1: From median backwards (negative lags)
    // Phase 2: From median forwards (positive lags)
    
    let median_idx = n_pairs / 2;
    
    // Phase 1: Process negative lags (backwards from median)
    let mut pos = median_idx as i32 - 1;
    while pos >= 0 {
        let mut current_bin: Vec<(usize, usize)> = Vec::new();
        let mut used1 = vec![false; n1];
        let mut used2 = vec![false; n2];
        
        // Collect pairs for this bin, ensuring equal population
        let mut temp_pos = pos;
        while temp_pos >= 0 && current_bin.len() < min_points {
            let (_lag, idx1, idx2) = time_lags[temp_pos as usize];
            
            // Only use unused pairs in this bin
            if !used1[idx1] && !used2[idx2] {
                current_bin.push((idx1, idx2));
                used1[idx1] = true;
                used2[idx2] = true;
            }
            temp_pos -= 1;
        }
        
        // Add bin if it has enough points
        if current_bin.len() >= min_points {
            negative_bins.push(current_bin);
            pos = temp_pos;
        } else {
            // Not enough points left, stop
            break;
        }
    }
    
    // Phase 2: Process positive lags (forwards from median)
    pos = median_idx as i32;
    while pos < n_pairs as i32 {
        let mut current_bin: Vec<(usize, usize)> = Vec::new();
        let mut used1 = vec![false; n1];
        let mut used2 = vec![false; n2];
        
        // Collect pairs for this bin, ensuring equal population
        let mut temp_pos = pos;
        while temp_pos < n_pairs as i32 && current_bin.len() < min_points {
            let (_lag, idx1, idx2) = time_lags[temp_pos as usize];
            
            // Only use unused pairs in this bin
            if !used1[idx1] && !used2[idx2] {
                current_bin.push((idx1, idx2));
                used1[idx1] = true;
                used2[idx2] = true;
            }
            temp_pos += 1;
        }
        
        // Add bin if it has enough points
        if current_bin.len() >= min_points {
            bins.push(current_bin);
            pos = temp_pos;
        } else {
            // Not enough points left, stop
            break;
        }
    }
    
    // Combine bins in chronological order: negative lags (reversed) + positive lags
    negative_bins.reverse();
    negative_bins.extend(bins);
    
    negative_bins
}

pub fn zdcf(
    series1: &[TimeSeriesPoint],
    series2: &[TimeSeriesPoint],
    min_points: usize,
    num_mc: usize,
) -> Vec<CorrelationPoint> {
    let bins = alcbin(series1, series2, min_points);
    let mut rng = thread_rng();
    let mut mc_results: Vec<Vec<(f64, f64)>> = Vec::new();

    for _ in 0..num_mc {
        let mut series1_mc: Vec<f64> = Vec::with_capacity(series1.len());
        for p in series1 {
            let normal = Normal::new(p.value, p.error).unwrap();
            series1_mc.push(normal.sample(&mut rng));
        }

        let mut series2_mc: Vec<f64> = Vec::with_capacity(series2.len());
        for p in series2 {
            let normal = Normal::new(p.value, p.error).unwrap();
            series2_mc.push(normal.sample(&mut rng));
        }
        mc_results.push(clcdcf(&series1_mc, &series2_mc, series1, series2, &bins));
    }

    let mut correlation_points = Vec::new();
    for i in 0..bins.len() {
        let mut lag_sum = 0.0;
        let mut r_sum = 0.0;
        let n = mc_results.len() as f64;

        for mc_run in &mc_results {
            if i < mc_run.len() {
                lag_sum += mc_run[i].0;
                r_sum += mc_run[i].1;
            }
        }

        let lag = lag_sum / n;
        let r = r_sum / n;
        let r_clamped = r.max(-1.0 + 1e-7).min(1.0 - 1e-7);

        let n_bin = bins[i].len() as f64;
        let z = fishe(r_clamped, n_bin);
        let s = fishs(r_clamped, n_bin);

        let z_err_neg = r_clamped - (z - s).tanh();
        let z_err_pos = (z + s).tanh() - r_clamped;

        correlation_points.push(CorrelationPoint {
            lag,
            correlation: r,
            error: (z_err_neg + z_err_pos) / 2.0,
        });
    }

    correlation_points
}
