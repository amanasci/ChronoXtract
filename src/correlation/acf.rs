// src/correlation/acf.rs

use super::dcf;

pub fn acf(
    series: &[dcf::TimeSeriesPoint],
    lag_min: f64,
    lag_max: f64,
    lag_bin_width: f64,
) -> Vec<dcf::CorrelationPoint> {
    dcf::dcf(series, series, lag_min, lag_max, lag_bin_width)
}
