// This file calculates the variability timescale from time, flux and flux_error arrays.
// The variability timescale is computed using adjacent pairs of measurements using the formula:
//    tau = Δtime / |ln(flux₂/flux₁)|
// We only consider measurements with significant flux changes (i.e. flux difference bigger than the combined error).
// A lower tau indicates more rapid variability.

pub fn calc_variability_timescale(time: &[f64], flux: &[f64], flux_error: &[f64]) -> Option<f64> {
    if time.len() < 2 || flux.len() < 2 || flux_error.len() < 2 {
        return None;
    }

    let mut min_tau: Option<f64> = None;

    for i in 0..(time.len() - 1) {
        let dt = time[i + 1] - time[i];
        if dt <= 0.0 {
            continue; // ensure time increases
        }

        let f1 = flux[i];
        let f2 = flux[i + 1];
        if f1 <= 0.0 || f2 <= 0.0 {
            continue; // flux must be positive for logarithm
        }

        let flux_ratio = f2 / f1;
        let dlnf = flux_ratio.ln().abs();
        if dlnf == 0.0 {
            continue; // no variability detected between these points
        }

        // Check if the flux difference is statistically significant against the errors
        let flux_diff = (f2 - f1).abs();
        let combined_error = (flux_error[i].powi(2) + flux_error[i + 1].powi(2)).sqrt();
        if flux_diff < combined_error {
            continue; // insignificant change; skip this pair
        }

        let tau = dt / dlnf;
        min_tau = match min_tau {
            Some(prev_tau) if tau < prev_tau => Some(tau),
            None => Some(tau),
            other => other,
        };
    }
    min_tau
}
