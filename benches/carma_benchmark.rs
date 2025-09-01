use criterion::{black_box, criterion_group, criterion_main, Criterion};
use chronoxtract::carma::*;
use numpy::PyArray1;
use pyo3::Python;

fn benchmark_carma_simulation(c: &mut Criterion) {
    Python::with_gil(|py| {
        let mut model = carma_model::CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![0.3, 0.1];
        model.ma_coeffs = vec![1.0, 0.4];
        model.sigma = 1.0;
        
        let times: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
        let times_array = PyArray1::from_vec(py, times);
        
        c.bench_function("carma_simulation_1000", |b| {
            b.iter(|| {
                simulation::simulate_carma(
                    py,
                    black_box(&model),
                    black_box(times_array.readonly()),
                    None,
                    Some(42),
                ).unwrap()
            })
        });
    });
}

fn benchmark_carma_psd(c: &mut Criterion) {
    Python::with_gil(|py| {
        let mut model = carma_model::CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![0.3, 0.1];
        model.ma_coeffs = vec![1.0, 0.4];
        model.sigma = 1.0;
        
        let frequencies: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
        let freq_array = PyArray1::from_vec(py, frequencies);
        
        c.bench_function("carma_psd_100_freqs", |b| {
            b.iter(|| {
                analysis::carma_psd(
                    py,
                    black_box(&model),
                    black_box(freq_array.readonly()),
                ).unwrap()
            })
        });
    });
}

fn benchmark_method_of_moments(c: &mut Criterion) {
    Python::with_gil(|py| {
        let times: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = times.iter().map(|&t| (t * 0.5).sin() + 0.1 * t).collect();
        
        let times_array = PyArray1::from_vec(py, times);
        let values_array = PyArray1::from_vec(py, values);
        
        c.bench_function("method_of_moments_200", |b| {
            b.iter(|| {
                estimation::carma_method_of_moments(
                    black_box(times_array.readonly()),
                    black_box(values_array.readonly()),
                    2,
                    1,
                ).unwrap()
            })
        });
    });
}

fn benchmark_kalman_filter(c: &mut Criterion) {
    Python::with_gil(|py| {
        let mut model = carma_model::CarmaModel::new(2, 1).unwrap();
        model.ar_coeffs = vec![0.3, 0.1];
        model.ma_coeffs = vec![1.0, 0.4];
        model.sigma = 1.0;
        
        let times: Vec<f64> = (0..500).map(|i| i as f64 * 0.02).collect();
        let values: Vec<f64> = times.iter().map(|&t| (t * 0.3).sin() + 0.05 * t).collect();
        
        let times_array = PyArray1::from_vec(py, times);
        let values_array = PyArray1::from_vec(py, values);
        
        c.bench_function("kalman_filter_500", |b| {
            b.iter(|| {
                kalman::carma_kalman_filter(
                    black_box(&model),
                    black_box(times_array.readonly()),
                    black_box(values_array.readonly()),
                    None,
                ).unwrap()
            })
        });
    });
}

criterion_group!(
    benches,
    benchmark_carma_simulation,
    benchmark_carma_psd,
    benchmark_method_of_moments,
    benchmark_kalman_filter
);
criterion_main!(benches);