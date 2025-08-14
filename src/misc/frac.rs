// frac.rs
// This program calculates the fractional variability of a lightcurve.
// Fractional Variability (Fvar) is defined as:
//    Fvar = sqrt((S² - ⟨σ_err²⟩)) / ⟨x⟩,
// where S² is the variance of the flux measurements, ⟨σ_err²⟩ is the mean squared flux error,
// and ⟨x⟩ is the average flux. If S² <= ⟨σ_err²⟩, the variability is indistinguishable from noise.


// Computes the fractional variability given slices of flux values and their measurement errors.
use std::f64;

pub fn fractional_variability(flux: &[f64], flux_err: &[f64]) -> Result<f64, &'static str> {
    if flux.is_empty() || flux_err.len() != flux.len() {
        return Err("Input arrays are empty or have mismatched lengths.");
    }
    
    if flux.iter().any(|&x| x < 0.0) {
        return Err("Flux values cannot be negative.");
    }

    let n = flux.len() as f64;
    let mean_flux = flux.iter().sum::<f64>() / n;
    if mean_flux == 0.0 {
        return Ok(f64::NAN);
    }
    
    if n < 2.0 {
        return Err("Input arrays must have at least two elements.");
    }

    let variance = flux
        .iter()
        .map(|x| (x - mean_flux).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let mean_err_sq = flux_err
        .iter()
        .map(|e| e.powi(2))
        .sum::<f64>()
        / n;
    
    if variance <= mean_err_sq {
        return Ok(0.0);
    }
    
    Ok(((variance - mean_err_sq).sqrt()) / mean_flux)
}

pub fn fractional_variability_error(flux: &[f64], flux_err: &[f64]) -> Result<f64, &'static str> {
    if flux.is_empty() || flux_err.len() != flux.len() {
        return Err("Input arrays are empty or have mismatched lengths.");
    }
    
    if flux.iter().any(|&x| x < 0.0) {
        return Err("Flux values cannot be negative.");
    }

    let n = flux.len() as f64;
    let mean_flux = flux.iter().sum::<f64>() / n;
    if mean_flux == 0.0 {
        return Ok(f64::NAN);
    }
    
    if n < 2.0 {
        return Err("Input arrays must have at least two elements.");
    }

    let variance = flux
        .iter()
        .map(|x| (x - mean_flux).powi(2))
        .sum::<f64>() / (n - 1.0);
    let mean_err_sq = flux_err
        .iter()
        .map(|e| e.powi(2))
        .sum::<f64>() / n;
    
    if variance <= mean_err_sq {
        return Ok(0.0);
    }
    
    let fvar = ((variance - mean_err_sq).sqrt()) / mean_flux;
    
    // Error propagation as in Vaughan et al. (2003):
    let term1 = (1.0 / (2.0 * n)).sqrt() * mean_err_sq / (mean_flux * mean_flux * fvar);
    let term2 = (mean_err_sq / n).sqrt() / mean_flux;
    let fvar_err = (term1.powi(2) + term2.powi(2)).sqrt();
    
    Ok(fvar_err)
}

/// Calculates the rolling fractional variability and its error over a specified window size.
/// Returns a tuple of vectors (fvar_values, fvar_err_values) for each window.
/// If the inputs are empty, lengths mismatch, window_size is zero, or the window_size is larger than the input length,
/// the function returns None.
pub fn rolling_fractional_variability(
    flux: &[f64],
    flux_err: &[f64],
    window_size: usize,
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    if flux.is_empty() || flux_err.len() != flux.len() || window_size == 0 || flux.len() < window_size {
        return Err("Invalid input for rolling fractional variability.");
    }

    let mut fvar_values = Vec::new();
    let mut fvar_err_values = Vec::new();

    for (flux_window, err_window) in flux.windows(window_size).zip(flux_err.windows(window_size)) {
        let fv = fractional_variability(flux_window, err_window)?;
        let err = fractional_variability_error(flux_window, err_window)?;
        fvar_values.push(fv);
        fvar_err_values.push(err);
    }

    Ok((fvar_values, fvar_err_values))
}