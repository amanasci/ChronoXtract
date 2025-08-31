#!/usr/bin/env python3
"""
ChronoXtract Advanced Features Examples

This example demonstrates the advanced analytical capabilities of ChronoXtract
including higher-order statistics, entropy measures, seasonality analysis,
and shape features.

Requirements: chronoxtract, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import chronoxtract as ct
    print("✅ ChronoXtract imported successfully!")
except ImportError:
    print("❌ Please install ChronoXtract: pip install chronoxtract")
    exit(1)

def generate_sample_signals():
    """Generate various sample signals for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    # Time vector
    t = np.linspace(0, 10, 1000)
    
    # Signal 1: EEG-like signal with multiple frequency components
    eeg_like = (np.sin(2*np.pi*8*t) +  # Alpha wave (8 Hz)
                0.5*np.sin(2*np.pi*13*t) +  # Beta wave (13 Hz)
                0.3*np.sin(2*np.pi*30*t) +  # Gamma wave (30 Hz)
                0.2*np.random.randn(1000))   # Noise
    
    # Signal 2: Seasonal time series
    seasonal_trend = 0.02 * np.arange(1000)  # Linear trend
    seasonal_component = 2 * np.sin(2*np.pi*np.arange(1000)/100)  # Period of 100
    seasonal_ts = seasonal_trend + seasonal_component + 0.5*np.random.randn(1000)
    
    # Signal 3: Financial-like signal with volatility clustering
    returns = np.random.randn(1000) * 0.02
    volatility = np.abs(np.random.randn(1000) * 0.01 + 0.02)
    financial_ts = np.cumsum(returns * volatility)
    
    return t, eeg_like, seasonal_ts, financial_ts

def demonstrate_hjorth_parameters(signal, signal_name):
    """Demonstrate Hjorth parameter analysis"""
    print(f"\n🧠 Hjorth Parameters Analysis for {signal_name}")
    print("=" * 50)
    
    # Calculate all Hjorth parameters at once
    hjorth = ct.hjorth_parameters(signal.tolist())
    
    print(f"Activity (variance): {hjorth['activity']:.6f}")
    print(f"Mobility (mean frequency): {hjorth['mobility']:.6f}")
    print(f"Complexity (frequency spread): {hjorth['complexity']:.6f}")
    
    # Calculate individual parameters
    activity = ct.hjorth_activity(signal.tolist())
    mobility = ct.hjorth_mobility(signal.tolist())
    complexity = ct.hjorth_complexity(signal.tolist())
    
    print(f"\nVerification (individual calculations):")
    print(f"Activity: {activity:.6f}")
    print(f"Mobility: {mobility:.6f}")
    print(f"Complexity: {complexity:.6f}")
    
    return hjorth

def demonstrate_entropy_analysis(signal, signal_name):
    """Demonstrate entropy and information theory measures"""
    print(f"\n🔍 Entropy Analysis for {signal_name}")
    print("=" * 40)
    
    # Sample entropy
    try:
        sample_ent = ct.sample_entropy(signal.tolist(), m=2, r=0.1)
        print(f"Sample Entropy: {sample_ent:.6f}")
    except Exception as e:
        print(f"Sample Entropy: Error - {e}")
    
    # Approximate entropy
    try:
        approx_ent = ct.approximate_entropy(signal.tolist(), m=2, r=0.1)
        print(f"Approximate Entropy: {approx_ent:.6f}")
    except Exception as e:
        print(f"Approximate Entropy: Error - {e}")
    
    # Permutation entropy
    try:
        perm_ent = ct.permutation_entropy(signal.tolist(), order=3)
        print(f"Permutation Entropy: {perm_ent:.6f}")
    except Exception as e:
        print(f"Permutation Entropy: Error - {e}")
    
    # Lempel-Ziv complexity
    try:
        lz_comp = ct.lempel_ziv_complexity(signal.tolist())
        print(f"Lempel-Ziv Complexity: {lz_comp:.6f}")
    except Exception as e:
        print(f"Lempel-Ziv Complexity: Error - {e}")
    
    # Multiscale entropy
    try:
        mse = ct.multiscale_entropy(signal.tolist(), max_scale=5, m=2, r=0.1)
        print(f"Multiscale Entropy (scales 1-5): {[f'{x:.4f}' for x in mse]}")
    except Exception as e:
        print(f"Multiscale Entropy: Error - {e}")

def demonstrate_seasonality_analysis(signal, signal_name):
    """Demonstrate seasonality and trend analysis"""
    print(f"\n📅 Seasonality Analysis for {signal_name}")
    print("=" * 45)
    
    # Seasonal and trend strength
    try:
        strength = ct.seasonal_trend_strength(signal.tolist(), period=100)
        print(f"Seasonal Strength: {strength['seasonal_strength']:.6f}")
        print(f"Trend Strength: {strength['trend_strength']:.6f}")
    except Exception as e:
        print(f"Seasonal/Trend Strength: Error - {e}")
    
    # Individual components
    try:
        seasonal_str = ct.seasonal_strength(signal.tolist(), period=100)
        trend_str = ct.trend_strength(signal.tolist())
        print(f"Seasonal Strength (individual): {seasonal_str:.6f}")
        print(f"Trend Strength (individual): {trend_str:.6f}")
    except Exception as e:
        print(f"Individual components: Error - {e}")
    
    # STL decomposition
    try:
        decomp = ct.simple_stl_decomposition(signal.tolist(), period=100)
        print(f"STL Decomposition completed:")
        print(f"  Trend component length: {len(decomp['trend'])}")
        print(f"  Seasonal component variance: {np.var(decomp['seasonal']):.6f}")
        print(f"  Remainder variance: {np.var(decomp['remainder']):.6f}")
    except Exception as e:
        print(f"STL Decomposition: Error - {e}")
    
    # Detrended fluctuation analysis
    try:
        dfa = ct.detrended_fluctuation_analysis(signal.tolist())
        print(f"DFA Scaling Exponent: {dfa.get('scaling_exponent', 'N/A')}")
    except Exception as e:
        print(f"DFA: Error - {e}")

def demonstrate_shape_features(signal, signal_name):
    """Demonstrate shape and peak features"""
    print(f"\n🏔️ Shape Features Analysis for {signal_name}")
    print("=" * 45)
    
    # Zero crossing rate
    try:
        zcr = ct.zero_crossing_rate(signal.tolist())
        print(f"Zero Crossing Rate: {zcr:.6f}")
    except Exception as e:
        print(f"Zero Crossing Rate: Error - {e}")
    
    # Slope features
    try:
        slopes = ct.slope_features(signal.tolist())
        print(f"Slope Features: {slopes}")
        
        mean_slope = ct.mean_slope(signal.tolist())
        slope_var = ct.slope_variance(signal.tolist())
        max_slope = ct.max_slope(signal.tolist())
        
        print(f"Mean Slope: {mean_slope:.8f}")
        print(f"Slope Variance: {slope_var:.8f}")
        print(f"Max Slope: {max_slope:.6f}")
    except Exception as e:
        print(f"Slope Features: Error - {e}")
    
    # Peak detection and analysis
    try:
        # Find peaks
        peaks = ct.find_peaks(signal.tolist(), height=np.std(signal)*0.5, distance=10)
        print(f"Number of peaks found: {len(peaks)}")
        
        if peaks:
            # Peak prominence
            prominence = ct.peak_prominence(signal.tolist(), peaks)
            print(f"Average peak prominence: {np.mean(prominence):.6f}")
            
            # Peak-to-peak amplitude
            p2p = ct.peak_to_peak_amplitude(signal.tolist())
            print(f"Peak-to-peak amplitude: {p2p:.6f}")
        
        # Enhanced peak statistics
        peak_stats = ct.enhanced_peak_stats(signal.tolist())
        print(f"Enhanced peak statistics available: {list(peak_stats.keys())}")
        
    except Exception as e:
        print(f"Peak Analysis: Error - {e}")
    
    # Variability and turning points
    try:
        variability = ct.variability_features(signal.tolist())
        turning_pts = ct.turning_points(signal.tolist())
        energy_dist = ct.energy_distribution(signal.tolist())
        
        print(f"Variability features: {list(variability.keys())}")
        print(f"Turning points: {turning_pts.get('count', 'N/A')}")
        print(f"Energy distribution: {list(energy_dist.keys())}")
        
    except Exception as e:
        print(f"Variability/Turning Points: Error - {e}")

def demonstrate_higher_order_statistics(signal, signal_name):
    """Demonstrate higher-order statistics"""
    print(f"\n📊 Higher-order Statistics for {signal_name}")
    print("=" * 45)
    
    try:
        # Higher moments
        higher_moments = ct.higher_moments(signal.tolist())
        print(f"Higher moments: {higher_moments}")
        
        # Individual central moments
        moment_5 = ct.central_moment_5(signal.tolist())
        moment_6 = ct.central_moment_6(signal.tolist())
        moment_7 = ct.central_moment_7(signal.tolist())
        moment_8 = ct.central_moment_8(signal.tolist())
        
        print(f"5th Central Moment: {moment_5:.8f}")
        print(f"6th Central Moment: {moment_6:.8f}")
        print(f"7th Central Moment: {moment_7:.8f}")
        print(f"8th Central Moment: {moment_8:.8f}")
        
    except Exception as e:
        print(f"Higher-order Statistics: Error - {e}")

def create_visualization(t, signals, signal_names):
    """Create visualization of the sample signals"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, (signal, name) in enumerate(zip(signals, signal_names)):
        axes[i].plot(t, signal, linewidth=1)
        axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_features_signals.png', dpi=150, bbox_inches='tight')
    print("\n📊 Signal plots saved as 'advanced_features_signals.png'")

def main():
    """Main demonstration function"""
    print("🚀 ChronoXtract Advanced Features Demonstration")
    print("=" * 60)
    
    # Generate sample signals
    t, eeg_like, seasonal_ts, financial_ts = generate_sample_signals()
    signals = [eeg_like, seasonal_ts, financial_ts]
    signal_names = ["EEG-like Signal", "Seasonal Time Series", "Financial-like Signal"]
    
    # Create visualization
    create_visualization(t, signals, signal_names)
    
    # Demonstrate features for each signal
    for signal, name in zip(signals, signal_names):
        print(f"\n" + "="*60)
        print(f"ANALYZING: {name}")
        print("="*60)
        
        # Run all analyses
        demonstrate_hjorth_parameters(signal, name)
        demonstrate_entropy_analysis(signal, name)
        demonstrate_seasonality_analysis(signal, name)
        demonstrate_shape_features(signal, name)
        demonstrate_higher_order_statistics(signal, name)
    
    print(f"\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("Check the generated plots and console output for comprehensive")
    print("analysis results across all advanced ChronoXtract features.")

if __name__ == "__main__":
    main()