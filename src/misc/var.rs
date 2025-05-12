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


// Getting varaiability timescale statistics, getting average, mean, median, mode, maximum, minimum, standard deviation

// Define a struct to hold all statistics
pub struct VariabilityStatistics {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub std_dev: Option<f64>,
    pub count: usize,
}

pub fn variability_statistics(time: &[f64], flux: &[f64], flux_error: &[f64]) -> VariabilityStatistics {
    if time.len() < 2 || flux.len() < 2 || flux_error.len() < 2 {
        return VariabilityStatistics {
            min: None,
            max: None,
            mean: None,
            median: None,
            std_dev: None,
            count: 0,
        };
    }

    // Collect all valid tau values
    let mut tau_values = Vec::new();

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
        tau_values.push(tau);
    }

    // Calculate statistics
    if tau_values.is_empty() {
        return VariabilityStatistics {
            min: None,
            max: None,
            mean: None,
            median: None,
            std_dev: None,
            count: 0,
        };
    }

    // Sort values for median calculation
    tau_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let min = *tau_values.first().unwrap();
    let max = *tau_values.last().unwrap();
    let count = tau_values.len();
    
    // Calculate mean
    let sum: f64 = tau_values.iter().sum();
    let mean = sum / count as f64;
    
    // Calculate median
    let median = if count % 2 == 0 {
        (tau_values[count/2 - 1] + tau_values[count/2]) / 2.0
    } else {
        tau_values[count/2]
    };
    
    // Calculate standard deviation
    let variance: f64 = tau_values.iter()
        .map(|&value| (value - mean).powi(2))
        .sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();

    VariabilityStatistics {
        min: Some(min),
        max: Some(max),
        mean: Some(mean),
        median: Some(median),
        std_dev: Some(std_dev),
        count,
    }
}