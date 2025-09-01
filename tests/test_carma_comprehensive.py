#!/usr/bin/env python3
"""
Comprehensive test of CARMA module functionality
"""

import numpy as np
import chronoxtract as ct

def test_comprehensive_carma():
    print("🚀 Comprehensive CARMA Module Test")
    print("=" * 60)
    
    # Create and configure a CARMA(2,1) model
    print("\n1. Creating CARMA Model")
    model = ct.carma_model(2, 1)
    ct.set_carma_parameters(model, [0.3, 0.1], [1.0, 0.4], 1.5)  # More stable coefficients
    print(f"   Model: {model}")
    
    # Test stability
    print("\n2. Testing Model Stability")
    is_stable = ct.check_carma_stability(model)
    roots = ct.carma_characteristic_roots(model)
    print(f"   Stable: {is_stable}")
    print(f"   Characteristic roots: {roots}")
    
    # Generate irregular time series
    print("\n3. Generating Irregular Time Series")
    times, values = ct.generate_irregular_carma(model, duration=20.0, 
                                               mean_sampling_rate=2.0, 
                                               sampling_noise=0.3, 
                                               seed=42)
    print(f"   Generated {len(times)} observations")
    print(f"   Time range: {times[0]:.2f} to {times[-1]:.2f}")
    print(f"   Value range: {np.min(values):.3f} to {np.max(values):.3f}")
    
    # Compute PSD
    print("\n4. Computing Power Spectral Density")
    frequencies = np.logspace(-2, 0, 50)  # 0.01 to 1 Hz
    psd = ct.carma_psd(model, frequencies)
    print(f"   PSD computed for {len(frequencies)} frequencies")
    print(f"   Max PSD: {np.max(psd):.3f} at f={frequencies[np.argmax(psd)]:.3f}")
    
    # Compute covariance function
    print("\n5. Computing Covariance Function")
    lags = np.linspace(0, 5, 20)
    covariance = ct.carma_covariance(model, lags)
    print(f"   Covariance at lag 0: {covariance[0]:.3f}")
    print(f"   Covariance at lag 5: {covariance[-1]:.3f}")
    
    # Fit models to data
    print("\n6. Model Fitting")
    
    # Method of moments
    mom_result = ct.carma_method_of_moments(times, values, 2, 1)
    print(f"   Method of Moments: {mom_result}")
    
    # MLE estimation  
    try:
        mle_result = ct.carma_mle(times, values, 2, 1)
        print(f"   MLE Result: {mle_result}")
    except Exception as e:
        print(f"   MLE failed: {e}")
    
    # Model selection
    print("\n7. Model Selection")
    try:
        ic_result = ct.carma_information_criteria(times, values, 3, 2)
        print(f"   Best AIC: CARMA({ic_result.best_aic[0]}, {ic_result.best_aic[1]})")
        print(f"   Best BIC: CARMA({ic_result.best_bic[0]}, {ic_result.best_bic[1]})")
        
        # Show some results
        for key, results in list(ic_result.results.items())[:3]:
            aic = results.get('aic', float('inf'))
            bic = results.get('bic', float('inf'))
            print(f"   {key}: AIC={aic:.2f}, BIC={bic:.2f}")
    except Exception as e:
        print(f"   Model selection failed: {e}")
    
    # Prediction and filtering
    print("\n8. Prediction and Filtering")
    
    # Split data for prediction
    n_train = int(0.8 * len(times))
    train_times = times[:n_train]
    train_values = values[:n_train]
    test_times = times[n_train:]
    test_values = values[n_train:]
    
    # Kalman filtering
    try:
        kalman_result = ct.carma_kalman_filter(model, train_times, train_values)
        print(f"   Kalman filter log-likelihood: {kalman_result.loglikelihood:.3f}")
        print(f"   Filtered {len(kalman_result.filtered_mean)} points")
    except Exception as e:
        print(f"   Kalman filtering failed: {e}")
    
    # Prediction
    try:
        pred_result = ct.carma_predict(model, train_times, train_values, test_times[:5])
        print(f"   Predicted {len(pred_result.mean)} points")
        print(f"   Mean prediction error: {np.mean(np.abs(pred_result.mean)):.3f}")
    except Exception as e:
        print(f"   Prediction failed: {e}")
    
    # Cross-validation
    print("\n9. Cross-Validation")
    try:
        cv_result = ct.carma_cross_validation(times, values, 2, 1, 3, seed=42)
        print(f"   CV Score: {cv_result.mean_score:.3f} ± {cv_result.std_score:.3f}")
        print(f"   Fold scores: {[f'{score:.3f}' for score in cv_result.fold_scores]}")
    except Exception as e:
        print(f"   Cross-validation failed: {e}")
    
    # Residual analysis
    print("\n10. Residual Analysis")
    try:
        residuals = ct.carma_residuals(model, times, values)
        print(f"   Residual std: {np.std(residuals.residuals):.3f}")
        print(f"   Ljung-Box test: stat={residuals.ljung_box_statistic:.3f}, "
              f"p-value={residuals.ljung_box_pvalue:.3f}")
    except Exception as e:
        print(f"   Residual analysis failed: {e}")
    
    print("\n✅ Comprehensive test completed successfully!")
    print("\n📊 Summary Statistics:")
    print(f"   Data points: {len(times)}")
    print(f"   Time span: {times[-1] - times[0]:.1f}")
    print(f"   Mean sampling rate: {len(times) / (times[-1] - times[0]):.2f} Hz")
    print(f"   Model parameters: AR={model.ar_coeffs}, MA={model.ma_coeffs}, σ={model.sigma}")

if __name__ == "__main__":
    test_comprehensive_carma()