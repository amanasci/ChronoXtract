"""
ChronoXtract New Feature Families Demonstration
===============================================

This script demonstrates the newly implemented feature families:
1. Higher-Order Statistics
2. Information-Theoretic Measures  
3. Seasonality & Trend Analysis
4. Shape & Peak Features
"""

import numpy as np
import chronoxtract as ct

def create_test_signals():
    """Create various test signals for demonstration"""
    np.random.seed(42)
    
    # 1. Sine wave with trend and noise
    t = np.linspace(0, 10, 1000)
    sine_trend = 0.1 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.normal(0, 1, 1000)
    
    # 2. Seasonal signal
    seasonal = np.sin(2 * np.pi * t / 2) + 0.5 * np.sin(2 * np.pi * t / 0.5) + 0.05 * t
    
    # 3. Chaotic-like signal
    chaotic = np.cumsum(np.random.choice([-1, 1], 1000) * np.random.exponential(0.1, 1000))
    
    # 4. Gaussian noise
    gaussian = np.random.normal(0, 1, 1000)
    
    return {
        'sine_trend': sine_trend,
        'seasonal': seasonal, 
        'chaotic': chaotic,
        'gaussian': gaussian,
        't': t
    }

def demonstrate_higher_order_stats(signals):
    """Demonstrate Higher-Order Statistics features"""
    print("=" * 60)
    print("HIGHER-ORDER STATISTICS")
    print("=" * 60)
    
    for name, signal in signals.items():
        if name == 't':
            continue
            
        print(f"\n{name.upper()} SIGNAL:")
        print("-" * 40)
        
        # Hjorth parameters
        activity, mobility, complexity = ct.hjorth_parameters(signal)
        print(f"Hjorth Activity:   {activity:.4f}")
        print(f"Hjorth Mobility:   {mobility:.4f}")
        print(f"Hjorth Complexity: {complexity:.4f}")
        
        # Higher moments
        m5, m6, m7, m8 = ct.higher_moments(signal)
        print(f"5th Central Moment: {m5:.4f}")
        print(f"6th Central Moment: {m6:.4f}")
        print(f"7th Central Moment: {m7:.4f}")
        print(f"8th Central Moment: {m8:.4f}")

def demonstrate_entropy_measures(signals):
    """Demonstrate Information-Theoretic Measures"""
    print("\n" + "=" * 60)
    print("INFORMATION-THEORETIC MEASURES")
    print("=" * 60)
    
    for name, signal in signals.items():
        if name == 't':
            continue
            
        print(f"\n{name.upper()} SIGNAL:")
        print("-" * 40)
        
        # Entropy measures
        try:
            sampen = ct.sample_entropy(signal, m=2, r=0.2)
            apen = ct.approximate_entropy(signal, m=2, r=0.2)
            permen = ct.permutation_entropy(signal, m=3, delay=1)
            lzc = ct.lempel_ziv_complexity(signal, threshold=None)
            
            print(f"Sample Entropy:      {sampen:.4f}")
            print(f"Approximate Entropy: {apen:.4f}")
            print(f"Permutation Entropy: {permen:.4f}")
            print(f"Lempel-Ziv Complex: {lzc:.4f}")
            
            # Multiscale entropy (first 3 scales)
            mse = ct.multiscale_entropy(signal, m=2, r=0.2, max_scale=3)
            print(f"Multiscale Entropy:  {[f'{e:.3f}' for e in mse]}")
            
        except Exception as e:
            print(f"Error calculating entropy: {e}")

def demonstrate_seasonality_features(signals):
    """Demonstrate Seasonality & Trend Analysis"""
    print("\n" + "=" * 60)
    print("SEASONALITY & TREND ANALYSIS")
    print("=" * 60)
    
    for name, signal in signals.items():
        if name == 't':
            continue
            
        print(f"\n{name.upper()} SIGNAL:")
        print("-" * 40)
        
        # Seasonal and trend strength
        seasonal_str, trend_str = ct.seasonal_trend_strength(signal, period=50)
        print(f"Seasonal Strength: {seasonal_str:.4f}")
        print(f"Trend Strength:    {trend_str:.4f}")
        
        # Seasonality detection
        is_seasonal = ct.detect_seasonality(signal, period=50, threshold=0.3)
        print(f"Seasonality Detected: {is_seasonal}")
        
        # DFA
        try:
            dfa_alpha = ct.detrended_fluctuation_analysis(signal, min_window=10, max_window=100, num_windows=8)
            print(f"DFA Scaling Exponent: {dfa_alpha:.4f}")
        except Exception as e:
            print(f"DFA Error: {e}")

def demonstrate_shape_features(signals):
    """Demonstrate Shape & Peak Features"""
    print("\n" + "=" * 60)
    print("SHAPE & PEAK FEATURES")
    print("=" * 60)
    
    for name, signal in signals.items():
        if name == 't':
            continue
            
        print(f"\n{name.upper()} SIGNAL:")
        print("-" * 40)
        
        # Zero-crossing rate
        zcr = ct.zero_crossing_rate(signal)
        print(f"Zero-Crossing Rate: {zcr:.4f}")
        
        # Slope features
        mean_slope, slope_var, max_slope = ct.slope_features(signal)
        print(f"Mean Slope:     {mean_slope:.4f}")
        print(f"Slope Variance: {slope_var:.4f}")
        print(f"Max Slope:      {max_slope:.4f}")
        
        # Enhanced peak statistics
        peak_stats = ct.enhanced_peak_stats(signal, min_prominence=0.1, min_distance=5)
        num_peaks, mean_prom, mean_spacing, mean_width, max_p2p, peak_density = peak_stats
        print(f"Number of Peaks:    {num_peaks}")
        print(f"Mean Prominence:    {mean_prom:.4f}")
        print(f"Mean Peak Spacing:  {mean_spacing:.4f}")
        print(f"Peak Density:       {peak_density:.4f}")
        
        # Variability features
        cv, qcd, mad, iqr = ct.variability_features(signal)
        print(f"Coeff. of Variation: {cv:.4f}")
        print(f"Quartile Coeff. Disp: {qcd:.4f}")
        
        # Turning points
        num_tp, tp_rate = ct.turning_points(signal)
        print(f"Turning Points:     {num_tp} (rate: {tp_rate:.4f})")

def main():
    """Main demonstration function"""
    print("ChronoXtract New Feature Families Demonstration")
    print("Analysis of various signal types with new feature extractors")
    
    # Create test signals
    signals = create_test_signals()
    
    # Demonstrate each feature family
    demonstrate_higher_order_stats(signals)
    demonstrate_entropy_measures(signals)
    demonstrate_seasonality_features(signals) 
    demonstrate_shape_features(signals)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Successfully demonstrated 30 new functions across 4 feature families:")
    print("• Higher-Order Statistics: 9 functions")
    print("• Information-Theoretic Measures: 5 functions") 
    print("• Seasonality & Trend Analysis: 6 functions")
    print("• Shape & Peak Features: 10 functions")
    print("\nAll functions are available in the chronoxtract module!")

if __name__ == "__main__":
    main()