// src/correlation/dcf.rs

// A method for measuring correlation functions without interpolating in the temporal domain is proposed
// which provides an assumption-free representation of the correlation measured in the data and allows
// meaningful error estimates.
//
// References:
// Edelson, R. A., & Krolik, J. H. (1988). The discrete correlation function.
// The Astrophysical Journal, 333, 646-659.

#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesPoint {
    pub time: f64,
    pub value: f64,
    pub error: f64,
}

#[derive(Debug)]
pub struct CorrelationPoint {
    pub lag: f64,
    pub correlation: f64,
    pub error: f64,
}

pub fn dcf(
    series1: &[TimeSeriesPoint],
    series2: &[TimeSeriesPoint],
    lag_min: f64,
    lag_max: f64,
    lag_bin_width: f64,
) -> Vec<CorrelationPoint> {
    let mut correlation_points = Vec::new();

    let mean1 = series1.iter().map(|p| p.value).sum::<f64>() / series1.len() as f64;
    let mean2 = series2.iter().map(|p| p.value).sum::<f64>() / series2.len() as f64;

    let std_dev1 = (series1.iter().map(|p| (p.value - mean1).powi(2)).sum::<f64>() / (series1.len() - 1) as f64).sqrt();
    let std_dev2 = (series2.iter().map(|p| (p.value - mean2).powi(2)).sum::<f64>() / (series2.len() - 1) as f64).sqrt();

    let mut lag_bins = Vec::new();
    let mut current_lag = lag_min;
    while current_lag <= lag_max {
        lag_bins.push(current_lag);
        current_lag += lag_bin_width;
    }

    for lag_bin in lag_bins.windows(2) {
        let bin_min = lag_bin[0];
        let bin_max = lag_bin[1];
        let mut udcf_values = Vec::new();

        for p1 in series1 {
            for p2 in series2 {
                let lag = p2.time - p1.time;
                if lag >= bin_min && lag < bin_max {
                    let udcf = (p1.value - mean1) * (p2.value - mean2) / (std_dev1 * std_dev2);
                    udcf_values.push(udcf);
                }
            }
        }

        if !udcf_values.is_empty() {
            let n = udcf_values.len() as f64;
            let mean_udcf = udcf_values.iter().sum::<f64>() / n;
            let std_err = (udcf_values.iter().map(|&x| (x - mean_udcf).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() / n.sqrt();
            correlation_points.push(CorrelationPoint {
                lag: (bin_min + bin_max) / 2.0,
                correlation: mean_udcf,
                error: std_err,
            });
        }
    }

    correlation_points
}
