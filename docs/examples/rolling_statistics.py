#!/usr/bin/env python3
"""
Rolling Statistics Examples with ChronoXtract

This script demonstrates rolling window calculations including rolling mean,
variance, exponential moving averages, and sliding window entropy.
"""

import chronoxtract as ct
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def demonstrate_rolling_mean():
    """
    Demonstrate rolling mean with different window sizes and signal types
    """
    print("=" * 60)
    print("ROLLING MEAN DEMONSTRATION")
    print("=" * 60)
    
    # Generate signal with noise
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    clean_signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
    noise = np.random.normal(0, 0.3, len(t))
    noisy_signal = clean_signal + noise
    
    # Different window sizes
    windows = [5, 20, 50]
    
    print(f"📊 Signal length: {len(noisy_signal)} points")
    print(f"🔧 Testing window sizes: {windows}")
    
    # Create visualization
    fig, axes = plt.subplots(len(windows) + 1, 1, figsize=(12, 10))
    fig.suptitle('Rolling Mean with Different Window Sizes', fontsize=16)
    
    # Original signal
    axes[0].plot(t, noisy_signal, alpha=0.6, color='gray', label='Noisy Signal')
    axes[0].plot(t, clean_signal, color='blue', linewidth=2, label='True Signal')
    axes[0].set_title('Original Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling means
    for i, window in enumerate(windows):
        rolling_avg = ct.rolling_mean(noisy_signal.tolist(), window=window)
        
        # Adjust time array for plotting (rolling mean has fewer points)
        t_adjusted = t[window-1:]
        
        print(f"\nWindow size {window}:")
        print(f"  Original length: {len(noisy_signal)}")
        print(f"  Rolling mean length: {len(rolling_avg)}")
        print(f"  Smoothing effect: {np.std(noisy_signal) / np.std(rolling_avg):.2f}x reduction in variance")
        
        axes[i+1].plot(t, noisy_signal, alpha=0.3, color='gray', label='Noisy Signal')
        axes[i+1].plot(t, clean_signal, alpha=0.7, color='blue', label='True Signal')
        axes[i+1].plot(t_adjusted, rolling_avg, color='red', linewidth=2, 
                      label=f'Rolling Mean (window={window})')
        axes[i+1].set_title(f'Rolling Mean - Window Size: {window}')
        axes[i+1].legend()
        axes[i+1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('rolling_mean_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📈 Visualization saved as 'rolling_mean_demo.png'")

def demonstrate_rolling_variance():
    """
    Demonstrate rolling variance for volatility analysis
    """
    print("\n" + "=" * 60)
    print("ROLLING VARIANCE DEMONSTRATION")
    print("=" * 60)
    
    # Generate data with changing volatility
    np.random.seed(42)
    n_points = 1000
    
    # Create signal with periods of different volatility
    signal = []
    volatility_periods = [
        (0.1, 250),    # Low volatility
        (0.5, 250),    # High volatility
        (0.2, 250),    # Medium volatility
        (0.8, 250)     # Very high volatility
    ]
    
    for vol, length in volatility_periods:
        period_data = np.random.normal(0, vol, length)
        signal.extend(period_data)
    
    signal = signal[:n_points]  # Trim to exact length
    
    # Calculate rolling variance
    window = 50
    rolling_var = ct.rolling_variance(signal, window=window)
    
    print(f"📊 Signal analysis:")
    print(f"  Total length: {len(signal)} points")
    print(f"  Rolling variance window: {window} points")
    print(f"  Rolling variance length: {len(rolling_var)} points")
    print(f"  Min variance: {min(rolling_var):.4f}")
    print(f"  Max variance: {max(rolling_var):.4f}")
    print(f"  Mean variance: {sum(rolling_var) / len(rolling_var):.4f}")
    
    # Detect high volatility periods
    variance_threshold = np.percentile(rolling_var, 75)  # 75th percentile
    high_vol_indices = [i for i, var in enumerate(rolling_var) if var > variance_threshold]
    
    print(f"\n🚨 High volatility detection:")
    print(f"  Threshold (75th percentile): {variance_threshold:.4f}")
    print(f"  High volatility periods: {len(high_vol_indices)} windows")
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original signal
    ax1.plot(signal, color='blue', alpha=0.7)
    ax1.set_title('Signal with Changing Volatility')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Mark different volatility periods
    colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    start_idx = 0
    for i, (vol, length) in enumerate(volatility_periods):
        end_idx = min(start_idx + length, len(signal))
        ax1.axvspan(start_idx, end_idx, alpha=0.3, color=colors[i % len(colors)], 
                   label=f'σ={vol}')
        start_idx = end_idx
    ax1.legend()
    
    # Rolling variance
    ax2.plot(range(window-1, len(signal)), rolling_var, color='red', linewidth=2)
    ax2.axhline(variance_threshold, color='orange', linestyle='--', 
               label=f'Threshold: {variance_threshold:.3f}')
    ax2.set_title('Rolling Variance (Volatility)')
    ax2.set_ylabel('Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Highlight high volatility periods
    for idx in high_vol_indices:
        ax2.axvspan(idx + window - 1, idx + window, alpha=0.5, color='red')
    
    # Histogram of variance values
    ax3.hist(rolling_var, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(variance_threshold, color='orange', linestyle='--', 
               label=f'Threshold: {variance_threshold:.3f}')
    ax3.set_title('Distribution of Rolling Variance')
    ax3.set_xlabel('Variance')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rolling_variance_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'rolling_variance_demo.png'")

def demonstrate_exponential_moving_average():
    """
    Demonstrate exponential moving average with different alpha values
    """
    print("\n" + "=" * 60)
    print("EXPONENTIAL MOVING AVERAGE DEMONSTRATION")
    print("=" * 60)
    
    # Generate step signal with noise
    np.random.seed(42)
    n_points = 300
    
    # Create signal with step changes
    signal = []
    step_values = [0, 5, 2, 8, 3]
    points_per_step = n_points // len(step_values)
    
    for value in step_values:
        step_data = value + np.random.normal(0, 0.5, points_per_step)
        signal.extend(step_data)
    
    signal = signal[:n_points]
    
    # Different alpha values
    alphas = [0.05, 0.1, 0.3, 0.5]
    
    print(f"📊 Testing alpha values: {alphas}")
    print("💡 Higher alpha = more responsive to recent changes")
    print("💡 Lower alpha = more smoothing, less responsive")
    
    # Calculate EMAs
    emas = {}
    for alpha in alphas:
        ema = ct.exponential_moving_average(signal, alpha=alpha)
        emas[alpha] = ema
        
        # Calculate responsiveness metric
        signal_changes = np.diff(signal)
        ema_changes = np.diff(ema)
        responsiveness = np.corrcoef(signal_changes, ema_changes)[0, 1]
        
        print(f"\nAlpha {alpha}:")
        print(f"  Responsiveness correlation: {responsiveness:.3f}")
        print(f"  Smoothing effect: {np.std(signal) / np.std(ema):.2f}x variance reduction")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Main plot with all EMAs
    ax1.plot(signal, alpha=0.5, color='gray', label='Original Signal', linewidth=1)
    
    colors = ['blue', 'red', 'green', 'purple']
    for alpha, color in zip(alphas, colors):
        ax1.plot(emas[alpha], color=color, linewidth=2, label=f'EMA α={alpha}')
    
    ax1.set_title('Exponential Moving Average with Different Alpha Values')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view of a step change
    zoom_start, zoom_end = 120, 180
    ax2.plot(range(zoom_start, zoom_end), signal[zoom_start:zoom_end], 
            alpha=0.7, color='gray', label='Original Signal', linewidth=2)
    
    for alpha, color in zip(alphas, colors):
        ax2.plot(range(zoom_start, zoom_end), emas[alpha][zoom_start:zoom_end], 
                color=color, linewidth=2, label=f'EMA α={alpha}')
    
    ax2.set_title('Zoomed View: Response to Step Change')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ema_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'ema_demo.png'")

def demonstrate_sliding_window_entropy():
    """
    Demonstrate sliding window entropy for complexity analysis
    """
    print("\n" + "=" * 60)
    print("SLIDING WINDOW ENTROPY DEMONSTRATION")
    print("=" * 60)
    
    # Generate signals with different complexity levels
    np.random.seed(42)
    n_each = 200
    
    # Different signal types
    signals = []
    labels = []
    
    # 1. Constant signal (low entropy)
    constant = np.full(n_each, 5.0)
    signals.extend(constant)
    labels.extend(['Constant'] * n_each)
    
    # 2. Sine wave (medium entropy)
    t = np.linspace(0, 4*np.pi, n_each)
    sine = 5 + 2 * np.sin(t)
    signals.extend(sine)
    labels.extend(['Sine Wave'] * n_each)
    
    # 3. Random noise (high entropy)
    noise = np.random.normal(5, 2, n_each)
    signals.extend(noise)
    labels.extend(['Random Noise'] * n_each)
    
    # 4. Chaotic signal (very high entropy)
    chaotic = []
    x = 0.1
    for _ in range(n_each):
        x = 4 * x * (1 - x)  # Logistic map
        chaotic.append(5 + 4 * x)
    signals.extend(chaotic)
    labels.extend(['Chaotic'] * n_each)
    
    # Calculate sliding window entropy
    window = 50
    bins = 10
    entropy = ct.sliding_window_entropy(signals, window=window, bins=bins)
    
    print(f"📊 Signal analysis:")
    print(f"  Total length: {len(signals)} points")
    print(f"  Window size: {window} points")
    print(f"  Number of bins: {bins}")
    print(f"  Entropy values: {len(entropy)}")
    
    # Analyze entropy for each signal type
    entropy_by_type = {}
    start_idx = 0
    for signal_type in ['Constant', 'Sine Wave', 'Random Noise', 'Chaotic']:
        end_idx = start_idx + n_each - window + 1
        if end_idx <= len(entropy):
            type_entropy = entropy[start_idx:end_idx]
            entropy_by_type[signal_type] = type_entropy
            
            if type_entropy:
                print(f"\n{signal_type}:")
                print(f"  Mean entropy: {np.mean(type_entropy):.3f}")
                print(f"  Std entropy:  {np.std(type_entropy):.3f}")
                print(f"  Max entropy:  {max(type_entropy):.3f}")
                print(f"  Min entropy:  {min(type_entropy):.3f}")
        
        start_idx = end_idx
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original signals
    ax1.plot(signals, color='blue', alpha=0.7)
    ax1.set_title('Combined Signals with Different Complexity Levels')
    ax1.set_ylabel('Value')
    
    # Mark different signal regions
    colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
    for i, (label, color) in enumerate(zip(['Constant', 'Sine Wave', 'Random Noise', 'Chaotic'], colors)):
        start = i * n_each
        end = (i + 1) * n_each
        ax1.axvspan(start, end, alpha=0.3, color=color, label=label)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Entropy
    entropy_x = range(window-1, len(signals))
    ax2.plot(entropy_x, entropy, color='red', linewidth=2)
    ax2.set_title(f'Sliding Window Entropy (window={window}, bins={bins})')
    ax2.set_ylabel('Entropy')
    ax2.grid(True, alpha=0.3)
    
    # Mark signal regions on entropy plot
    for i, color in enumerate(colors):
        start = i * n_each + window - 1
        end = (i + 1) * n_each + window - 1
        if start < len(entropy_x):
            ax2.axvspan(start, min(end, len(entropy_x)+window-1), alpha=0.3, color=color)
    
    # Entropy distribution
    all_entropy_values = [val for vals in entropy_by_type.values() for val in vals]
    ax3.hist(all_entropy_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_title('Distribution of Entropy Values')
    ax3.set_xlabel('Entropy')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Add vertical lines for mean entropy of each type
    for signal_type, color in zip(entropy_by_type.keys(), colors):
        if entropy_by_type[signal_type]:
            mean_entropy = np.mean(entropy_by_type[signal_type])
            ax3.axvline(mean_entropy, color=color, linestyle='--', linewidth=2, 
                       label=f'{signal_type}: {mean_entropy:.2f}')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('entropy_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'entropy_demo.png'")

def demonstrate_expanding_sum():
    """
    Demonstrate expanding sum (cumulative sum)
    """
    print("\n" + "=" * 60)
    print("EXPANDING SUM DEMONSTRATION")
    print("=" * 60)
    
    # Generate data representing daily sales
    np.random.seed(42)
    daily_sales = np.random.lognormal(4, 0.5, 365)  # Log-normal distribution for sales
    
    # Calculate cumulative sales
    cumulative_sales = ct.expanding_sum(daily_sales.tolist())
    
    print(f"📊 Sales Analysis (365 days):")
    print(f"  Average daily sales: ${np.mean(daily_sales):,.2f}")
    print(f"  Total annual sales: ${cumulative_sales[-1]:,.2f}")
    print(f"  Median daily sales: ${np.median(daily_sales):,.2f}")
    print(f"  Best day sales: ${max(daily_sales):,.2f}")
    print(f"  Worst day sales: ${min(daily_sales):,.2f}")
    
    # Calculate monthly totals
    monthly_totals = []
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    start_day = 0
    
    print(f"\n📅 Monthly Sales:")
    for month, days in enumerate(days_per_month, 1):
        end_day = start_day + days
        if end_day <= len(cumulative_sales):
            month_start = cumulative_sales[start_day-1] if start_day > 0 else 0
            month_end = cumulative_sales[end_day-1]
            monthly_total = month_end - month_start
            monthly_totals.append(monthly_total)
            print(f"  Month {month:2d}: ${monthly_total:,.2f}")
        start_day = end_day
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Daily sales
    ax1.plot(daily_sales, color='blue', alpha=0.7)
    ax1.set_title('Daily Sales')
    ax1.set_ylabel('Sales ($)')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative sales
    ax2.plot(cumulative_sales, color='green', linewidth=2)
    ax2.set_title('Cumulative Sales (Expanding Sum)')
    ax2.set_ylabel('Cumulative Sales ($)')
    ax2.grid(True, alpha=0.3)
    
    # Add quarterly markers
    quarters = [91, 182, 273, 365]
    for i, day in enumerate(quarters):
        if day <= len(cumulative_sales):
            ax2.axvline(day, color='red', linestyle='--', alpha=0.7)
            ax2.text(day, cumulative_sales[day-1], f'Q{i+1}', rotation=90, 
                    verticalalignment='bottom')
    
    # Monthly sales bar chart
    ax3.bar(range(1, len(monthly_totals)+1), monthly_totals, color='orange', alpha=0.7)
    ax3.set_title('Monthly Sales Totals')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Monthly Sales ($)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('expanding_sum_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'expanding_sum_demo.png'")

def comprehensive_rolling_analysis():
    """
    Apply multiple rolling statistics to the same dataset
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ROLLING ANALYSIS")
    print("=" * 60)
    
    # Generate complex time series (e.g., sensor readings)
    np.random.seed(42)
    t = np.linspace(0, 24, 1440)  # 24 hours, minute resolution
    
    # Base signal with daily pattern
    daily_pattern = 20 + 5 * np.sin(2 * np.pi * t / 24)  # Daily temperature cycle
    
    # Add weekly pattern
    weekly_pattern = 2 * np.sin(2 * np.pi * t / (24 * 7))
    
    # Add noise and occasional spikes
    noise = np.random.normal(0, 1, len(t))
    spikes = np.random.poisson(0.01, len(t)) * 10  # Occasional spikes
    
    signal = daily_pattern + weekly_pattern + noise + spikes
    
    # Apply all rolling statistics
    window = 60  # 1-hour window
    
    rolling_avg = ct.rolling_mean(signal.tolist(), window=window)
    rolling_var = ct.rolling_variance(signal.tolist(), window=window)
    ema = ct.exponential_moving_average(signal.tolist(), alpha=0.1)
    entropy = ct.sliding_window_entropy(signal.tolist(), window=window, bins=10)
    cumsum = ct.expanding_sum(signal.tolist())
    
    print(f"📊 Sensor Data Analysis (24 hours, minute resolution):")
    print(f"  Data points: {len(signal)}")
    print(f"  Rolling window: {window} minutes (1 hour)")
    print(f"  Mean temperature: {np.mean(signal):.1f}°C")
    print(f"  Temperature range: {np.min(signal):.1f}°C to {np.max(signal):.1f}°C")
    
    # Detect anomalies using multiple criteria
    anomalies = []
    for i in range(len(signal)):
        is_anomaly = False
        
        # Spike detection (more than 3 standard deviations from EMA)
        if i < len(ema):
            if abs(signal[i] - ema[i]) > 3 * np.std(signal):
                is_anomaly = True
        
        # Variance-based detection (if rolling variance available)
        if i >= window - 1:
            var_idx = i - window + 1
            if var_idx < len(rolling_var):
                if rolling_var[var_idx] > np.percentile(rolling_var, 95):
                    is_anomaly = True
        
        if is_anomaly:
            anomalies.append(i)
    
    print(f"\n🚨 Anomaly Detection:")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Anomaly rate: {len(anomalies)/len(signal)*100:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    
    # Original signal with anomalies
    axes[0].plot(t, signal, color='blue', alpha=0.7, linewidth=1)
    for anomaly_idx in anomalies:
        axes[0].scatter(t[anomaly_idx], signal[anomaly_idx], color='red', s=20, alpha=0.7)
    axes[0].set_title('Temperature Sensor Readings with Anomalies')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].grid(True, alpha=0.3)
    
    # Rolling mean and EMA
    t_rolling = t[window-1:]
    axes[1].plot(t, signal, alpha=0.3, color='gray', label='Original')
    axes[1].plot(t_rolling, rolling_avg, color='blue', linewidth=2, label=f'Rolling Mean ({window}min)')
    axes[1].plot(t, ema, color='red', linewidth=2, label='EMA (α=0.1)')
    axes[1].set_title('Smoothed Signals')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Rolling variance
    axes[2].plot(t_rolling, rolling_var, color='green', linewidth=2)
    axes[2].set_title('Rolling Variance (Temperature Stability)')
    axes[2].set_ylabel('Variance')
    axes[2].grid(True, alpha=0.3)
    
    # Entropy
    axes[3].plot(t[window-1:], entropy, color='purple', linewidth=2)
    axes[3].set_title('Sliding Window Entropy (Signal Complexity)')
    axes[3].set_ylabel('Entropy')
    axes[3].grid(True, alpha=0.3)
    
    # Cumulative sum
    axes[4].plot(t, cumsum, color='orange', linewidth=2)
    axes[4].set_title('Cumulative Temperature Sum')
    axes[4].set_xlabel('Time (hours)')
    axes[4].set_ylabel('Cumulative Sum')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_rolling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'comprehensive_rolling_analysis.png'")

if __name__ == "__main__":
    print("🚀 ChronoXtract Rolling Statistics Examples")
    print("This script demonstrates rolling window calculations and their applications.\n")
    
    # Run all demonstrations
    demonstrate_rolling_mean()
    demonstrate_rolling_variance()
    demonstrate_exponential_moving_average()
    demonstrate_sliding_window_entropy()
    demonstrate_expanding_sum()
    comprehensive_rolling_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All rolling statistics examples completed!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)