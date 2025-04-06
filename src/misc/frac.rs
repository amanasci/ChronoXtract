// frac.rs
// This program calculates the fractional variability of a lightcurve.
// Fractional Variability (Fvar) is defined as:
//    Fvar = sqrt((S² - ⟨σ_err²⟩)) / ⟨x⟩,
// where S² is the variance of the flux measurements, ⟨σ_err²⟩ is the mean squared flux error,
// and ⟨x⟩ is the average flux. If S² <= ⟨σ_err²⟩, the variability is indistinguishable from noise.


// Computes the fractional variability given slices of flux values and their measurement errors.
pub fn fractional_variability(flux: &[f64], flux_err: &[f64]) -> Option<f64> {
    if flux.is_empty() || flux_err.len() != flux.len() {
        return None;
    }
    
    let n = flux.len() as f64;
    let mean_flux = flux.iter().sum::<f64>() / n;
    if mean_flux == 0.0 {
        return None;
    }
    
    let variance = flux
        .iter()
        .map(|x| (x - mean_flux).powi(2))
        .sum::<f64>()
        / n;
    let mean_err_sq = flux_err
        .iter()
        .map(|e| e.powi(2))
        .sum::<f64>()
        / n;
    
    // If the computed variance is not greater than the measurement noise,
    // then variability cannot be reliably detected.
    if variance <= mean_err_sq {
        return Some(0.0);
    }
    
    Some(((variance - mean_err_sq).sqrt()) / mean_flux)
}

pub fn fractional_variability_error(flux: &[f64], flux_err: &[f64]) -> Option<f64> {
    if flux.is_empty() || flux.len() != flux_err.len() {
        return None;
    }
    
    let n = flux.len() as f64;
    let mean_flux = flux.iter().sum::<f64>() / n;
    if mean_flux == 0.0 {
        return None;
    }
    
    let variance = flux
        .iter()
        .map(|x| (x - mean_flux).powi(2))
        .sum::<f64>() / n;
    let mean_err_sq = flux_err
        .iter()
        .map(|e| e.powi(2))
        .sum::<f64>() / n;
    
    if variance <= mean_err_sq {
        return Some(0.0);
    }
    
    let fvar = ((variance - mean_err_sq).sqrt()) / mean_flux;
    
    // Error propagation as in Vaughan et al. (2003):
    let term1 = (1.0 / (2.0 * n)).sqrt() * mean_err_sq / (mean_flux * mean_flux * fvar);
    let term2 = (mean_err_sq / n).sqrt() / mean_flux;
    let fvar_err = (term1.powi(2) + term2.powi(2)).sqrt();
    
    Some(fvar_err)
}

/// Calculates the rolling fractional variability and its error over a specified window size.
/// Returns a tuple of vectors (fvar_values, fvar_err_values) for each window.
/// If the inputs are empty, lengths mismatch, window_size is zero, or the window_size is larger than the input length,
/// the function returns None.
pub fn rolling_fractional_variability(
    flux: &[f64],
    flux_err: &[f64],
    window_size: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if flux.is_empty() || flux_err.len() != flux.len() || window_size == 0 || flux.len() < window_size {
        return None;
    }

    let mut fvar_values = Vec::new();
    let mut fvar_err_values = Vec::new();

    for (flux_window, err_window) in flux.windows(window_size).zip(flux_err.windows(window_size)) {
        // Compute fractional variability and its error for the current window.
        // If the computation fails for any window, return None.
        match (fractional_variability(flux_window, err_window), fractional_variability_error(flux_window, err_window)) {
            (Some(fv), Some(err)) => {
                fvar_values.push(fv);
                fvar_err_values.push(err);
            }
            _ => return None,
        }
    }

    Some((fvar_values, fvar_err_values))
}