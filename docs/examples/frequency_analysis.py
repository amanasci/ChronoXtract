#!/usr/bin/env python3
"""
Frequency Analysis Examples with ChronoXtract

This script demonstrates frequency domain analysis capabilities including
FFT analysis and Lomb-Scargle periodogram for irregular data.
"""

import chronoxtract as ct
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def demonstrate_fft_analysis():
    """
    Demonstrate FFT analysis for spectral decomposition
    """
    print("=" * 60)
    print("FFT ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Generate complex signal with multiple frequency components
    np.random.seed(42)
    fs = 1000  # Sampling frequency (Hz)
    t = np.arange(0, 2, 1/fs)  # 2 seconds of data
    
    # Create signal with known frequency components
    f1, f2, f3 = 50, 120, 200  # Frequencies in Hz
    signal = (2.0 * np.sin(2*np.pi*f1*t) +
              1.0 * np.sin(2*np.pi*f2*t) +
              0.5 * np.sin(2*np.pi*f3*t) +
              0.2 * np.random.randn(len(t)))  # Add noise
    
    print(f"📊 Signal Properties:")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Duration: {len(t)/fs:.1f} seconds")
    print(f"  Data points: {len(signal)}")
    print(f"  True frequencies: {f1}, {f2}, {f3} Hz")
    
    # Perform FFT
    fft_result = ct.perform_fft_py(signal.tolist())
    
    # Calculate power spectrum and frequencies
    power = [abs(c)**2 for c in fft_result]
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Find peaks in positive frequencies only
    positive_idx = freqs > 0
    positive_freqs = freqs[positive_idx]
    positive_power = np.array(power)[positive_idx]
    
    # Peak detection (simple approach)
    peak_threshold = np.max(positive_power) * 0.1  # 10% of maximum
    peaks = []
    for i in range(1, len(positive_power)-1):
        if (positive_power[i] > positive_power[i-1] and 
            positive_power[i] > positive_power[i+1] and
            positive_power[i] > peak_threshold):
            peaks.append((positive_freqs[i], positive_power[i]))
    
    print(f"\n🔍 Detected Peaks:")
    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)  # Sort by power
    for i, (freq, power_val) in enumerate(peaks_sorted[:5]):
        print(f"  Peak {i+1}: {freq:.1f} Hz (power: {power_val:.1f})")
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time domain signal
    ax1.plot(t[:500], signal[:500], color='blue', linewidth=1)  # Show first 0.5 seconds
    ax1.set_title('Time Domain Signal (First 0.5 seconds)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Full power spectrum
    ax2.plot(positive_freqs, positive_power, color='red', linewidth=1)
    ax2.set_title('Power Spectrum (FFT)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    ax2.set_xlim(0, 500)  # Focus on relevant frequency range
    ax2.grid(True, alpha=0.3)
    
    # Mark detected peaks
    for freq, power_val in peaks_sorted[:5]:
        if freq <= 500:  # Only mark peaks in visible range
            ax2.axvline(freq, color='orange', linestyle='--', alpha=0.7)
            ax2.text(freq, power_val, f'{freq:.0f}Hz', rotation=90, 
                    verticalalignment='bottom')
    
    # Zoomed view of main peaks
    ax3.plot(positive_freqs, positive_power, color='red', linewidth=1)
    ax3.set_title('Power Spectrum - Zoomed View (0-300 Hz)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_xlim(0, 300)
    ax3.grid(True, alpha=0.3)
    
    # Highlight true frequencies
    for freq in [f1, f2, f3]:
        ax3.axvline(freq, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax3.legend(['Power Spectrum', 'True Frequencies'])
    
    plt.tight_layout()
    plt.savefig('fft_analysis_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'fft_analysis_demo.png'")

def demonstrate_lomb_scargle():
    """
    Demonstrate Lomb-Scargle periodogram for irregularly sampled data
    """
    print("\n" + "=" * 60)
    print("LOMB-SCARGLE PERIODOGRAM DEMONSTRATION")
    print("=" * 60)
    
    # Generate irregularly sampled astronomical-like data
    np.random.seed(42)
    
    # True signal parameters
    true_period = 12.5  # hours
    true_freq = 1.0 / true_period
    
    # Create irregular time sampling (like astronomical observations)
    total_time = 100  # hours
    n_observations = 200
    
    # Irregular sampling (some gaps for "bad weather")
    time_regular = np.linspace(0, total_time, n_observations * 2)
    
    # Remove some observations randomly (simulate weather, equipment issues)
    keep_fraction = 0.6
    keep_indices = np.random.choice(len(time_regular), 
                                   size=int(len(time_regular) * keep_fraction), 
                                   replace=False)
    time_irregular = np.sort(time_regular[keep_indices])
    
    # Generate signal with known period
    amplitude = 2.0
    phase = 0.3
    mean_level = 5.0
    noise_level = 0.3
    
    true_signal = mean_level + amplitude * np.sin(2*np.pi*true_freq*time_irregular + phase)
    noise = np.random.normal(0, noise_level, len(time_irregular))
    observed_signal = true_signal + noise
    
    # Simulate measurement errors (typical for astronomical data)
    measurement_errors = np.random.uniform(0.1, 0.5, len(observed_signal))
    
    print(f"📊 Irregular Sampling Analysis:")
    print(f"  Total time span: {total_time} hours")
    print(f"  Regular grid points: {len(time_regular)}")
    print(f"  Actual observations: {len(time_irregular)}")
    print(f"  Sampling completeness: {len(time_irregular)/len(time_regular)*100:.1f}%")
    print(f"  True period: {true_period} hours")
    print(f"  Signal amplitude: {amplitude}")
    print(f"  Noise level: {noise_level}")
    
    # Define frequency grid for Lomb-Scargle
    # Search from 0.01 to 1 cycles per hour
    min_freq = 0.01
    max_freq = 1.0
    n_freqs = 1000
    frequencies = np.linspace(min_freq, max_freq, n_freqs)
    
    print(f"\n🔍 Frequency Analysis:")
    print(f"  Frequency range: {min_freq} to {max_freq} cycles/hour")
    print(f"  Frequency resolution: {(max_freq-min_freq)/n_freqs:.4f} cycles/hour")
    print(f"  Period range: {1/max_freq:.1f} to {1/min_freq:.1f} hours")
    
    # Compute Lomb-Scargle periodogram
    power = ct.lomb_scargle_py(time_irregular.tolist(), 
                               observed_signal.tolist(), 
                               frequencies.tolist())
    
    # Convert frequencies to periods for easier interpretation
    periods = 1.0 / frequencies
    
    # Find the peak (most significant period)
    max_power_idx = np.argmax(power)
    detected_period = periods[max_power_idx]
    detected_freq = frequencies[max_power_idx]
    max_power = power[max_power_idx]
    
    print(f"\n🎯 Results:")
    print(f"  Detected period: {detected_period:.1f} hours")
    print(f"  True period: {true_period:.1f} hours")
    print(f"  Error: {abs(detected_period - true_period):.1f} hours")
    print(f"  Relative error: {abs(detected_period - true_period)/true_period*100:.1f}%")
    print(f"  Peak power: {max_power:.2f}")
    
    # Find other significant peaks
    power_array = np.array(power)
    threshold = np.percentile(power_array, 95)  # 95th percentile
    significant_peaks = []
    
    for i in range(1, len(power)-1):
        if (power[i] > power[i-1] and power[i] > power[i+1] and 
            power[i] > threshold):
            significant_peaks.append((periods[i], power[i]))
    
    significant_peaks.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Significant Peaks (> 95th percentile):")
    for i, (period, power_val) in enumerate(significant_peaks[:5]):
        print(f"  {i+1}. Period: {period:.1f} hours (power: {power_val:.2f})")
    
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series data
    ax1.errorbar(time_irregular, observed_signal, yerr=measurement_errors, 
                fmt='o', alpha=0.7, markersize=3, capsize=2, color='blue')
    ax1.plot(time_irregular, true_signal, color='red', linewidth=2, 
            label=f'True signal (P={true_period}h)')
    ax1.set_title('Irregularly Sampled Time Series')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Signal Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sampling pattern
    ax2.plot(time_irregular, np.ones_like(time_irregular), '|', markersize=10, 
            color='blue', alpha=0.7)
    ax2.set_title('Sampling Pattern')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Observations')
    ax2.set_ylim(0.5, 1.5)
    ax2.grid(True, alpha=0.3)
    
    # Lomb-Scargle periodogram (frequency space)
    ax3.plot(frequencies, power, color='green', linewidth=1)
    ax3.axvline(true_freq, color='red', linestyle='--', linewidth=2, 
               label=f'True freq: {true_freq:.3f} cyc/h')
    ax3.axvline(detected_freq, color='orange', linestyle=':', linewidth=2,
               label=f'Detected: {detected_freq:.3f} cyc/h')
    ax3.set_title('Lomb-Scargle Periodogram')
    ax3.set_xlabel('Frequency (cycles/hour)')
    ax3.set_ylabel('Power')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Lomb-Scargle periodogram (period space)
    ax4.plot(periods, power, color='green', linewidth=1)
    ax4.axvline(true_period, color='red', linestyle='--', linewidth=2,
               label=f'True period: {true_period}h')
    ax4.axvline(detected_period, color='orange', linestyle=':', linewidth=2,
               label=f'Detected: {detected_period:.1f}h')
    ax4.set_title('Lomb-Scargle Periodogram (Period View)')
    ax4.set_xlabel('Period (hours)')
    ax4.set_ylabel('Power')
    ax4.set_xlim(1, 50)  # Focus on reasonable period range
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lomb_scargle_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'lomb_scargle_demo.png'")

def compare_fft_vs_lomb_scargle():
    """
    Compare FFT and Lomb-Scargle on the same dataset
    """
    print("\n" + "=" * 60)
    print("FFT vs LOMB-SCARGLE COMPARISON")
    print("=" * 60)
    
    # Generate signal with regular sampling
    np.random.seed(42)
    fs = 100  # Hz
    t_regular = np.arange(0, 10, 1/fs)  # 10 seconds
    
    # Signal with two clear frequencies
    f1, f2 = 2, 7  # Hz
    signal_regular = (np.sin(2*np.pi*f1*t_regular) + 
                     0.5*np.sin(2*np.pi*f2*t_regular) + 
                     0.1*np.random.randn(len(t_regular)))
    
    # Create irregular sampling from the same signal
    keep_ratio = 0.3  # Keep only 30% of points
    irregular_indices = np.sort(np.random.choice(len(t_regular), 
                                               size=int(len(t_regular)*keep_ratio), 
                                               replace=False))
    t_irregular = t_regular[irregular_indices]
    signal_irregular = signal_regular[irregular_indices]
    
    print(f"📊 Comparison Setup:")
    print(f"  Regular sampling: {len(t_regular)} points at {fs} Hz")
    print(f"  Irregular sampling: {len(t_irregular)} points ({keep_ratio*100:.0f}% of regular)")
    print(f"  True frequencies: {f1} Hz and {f2} Hz")
    
    # FFT analysis (regular data)
    fft_result = ct.perform_fft_py(signal_regular.tolist())
    fft_power = [abs(c)**2 for c in fft_result]
    fft_freqs = np.fft.fftfreq(len(signal_regular), 1/fs)
    
    # Lomb-Scargle analysis (irregular data)
    ls_freqs = np.linspace(0.1, 20, 500)
    ls_power = ct.lomb_scargle_py(t_irregular.tolist(), 
                                  signal_irregular.tolist(), 
                                  ls_freqs.tolist())
    
    # Find peaks in both methods
    def find_peaks_simple(freqs, power, min_freq=0.5, max_freq=15):
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs_range = freqs[mask]
        power_range = np.array(power)[mask] if isinstance(power, list) else power[mask]
        
        peaks = []
        threshold = np.max(power_range) * 0.2
        for i in range(1, len(power_range)-1):
            if (power_range[i] > power_range[i-1] and 
                power_range[i] > power_range[i+1] and
                power_range[i] > threshold):
                peaks.append((freqs_range[i], power_range[i]))
        return sorted(peaks, key=lambda x: x[1], reverse=True)
    
    # FFT peaks (positive frequencies only)
    fft_pos_mask = fft_freqs > 0
    fft_peaks = find_peaks_simple(fft_freqs[fft_pos_mask], 
                                 np.array(fft_power)[fft_pos_mask])
    
    # Lomb-Scargle peaks
    ls_peaks = find_peaks_simple(ls_freqs, ls_power)
    
    print(f"\n🔍 FFT Results (Regular Sampling):")
    for i, (freq, power) in enumerate(fft_peaks[:3]):
        print(f"  Peak {i+1}: {freq:.1f} Hz (power: {power:.1f})")
    
    print(f"\n🔍 Lomb-Scargle Results (Irregular Sampling):")
    for i, (freq, power) in enumerate(ls_peaks[:3]):
        print(f"  Peak {i+1}: {freq:.1f} Hz (power: {power:.2f})")
    
    # Accuracy assessment
    true_freqs = [f1, f2]
    
    def assess_accuracy(peaks, true_freqs, tolerance=0.2):
        detected = []
        for true_f in true_freqs:
            best_match = None
            best_error = float('inf')
            for peak_f, power in peaks:
                error = abs(peak_f - true_f)
                if error < tolerance and error < best_error:
                    best_match = peak_f
                    best_error = error
            detected.append((true_f, best_match, best_error))
        return detected
    
    fft_accuracy = assess_accuracy(fft_peaks, true_freqs)
    ls_accuracy = assess_accuracy(ls_peaks, true_freqs)
    
    print(f"\n📈 Accuracy Assessment:")
    print("FFT (Regular sampling):")
    for true_f, detected_f, error in fft_accuracy:
        if detected_f is not None:
            print(f"  {true_f} Hz → {detected_f:.2f} Hz (error: {error:.2f} Hz)")
        else:
            print(f"  {true_f} Hz → Not detected")
    
    print("Lomb-Scargle (Irregular sampling):")
    for true_f, detected_f, error in ls_accuracy:
        if detected_f is not None:
            print(f"  {true_f} Hz → {detected_f:.2f} Hz (error: {error:.2f} Hz)")
        else:
            print(f"  {true_f} Hz → Not detected")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Regular time series
    axes[0,0].plot(t_regular, signal_regular, 'b-', alpha=0.7, linewidth=1)
    axes[0,0].set_title('Regular Sampling')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Signal')
    axes[0,0].grid(True, alpha=0.3)
    
    # Irregular time series
    axes[0,1].plot(t_irregular, signal_irregular, 'ro', markersize=3, alpha=0.7)
    axes[0,1].plot(t_regular, signal_regular, 'b-', alpha=0.3, linewidth=1, 
                   label='Original regular')
    axes[0,1].set_title('Irregular Sampling (30% of points)')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Signal')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # FFT power spectrum
    pos_mask = fft_freqs > 0
    axes[1,0].plot(fft_freqs[pos_mask], np.array(fft_power)[pos_mask], 'b-', linewidth=1)
    for true_f in true_freqs:
        axes[1,0].axvline(true_f, color='red', linestyle='--', alpha=0.7)
    axes[1,0].set_title('FFT Power Spectrum')
    axes[1,0].set_xlabel('Frequency (Hz)')
    axes[1,0].set_ylabel('Power')
    axes[1,0].set_xlim(0, 15)
    axes[1,0].grid(True, alpha=0.3)
    
    # Lomb-Scargle periodogram
    axes[1,1].plot(ls_freqs, ls_power, 'g-', linewidth=1)
    for true_f in true_freqs:
        axes[1,1].axvline(true_f, color='red', linestyle='--', alpha=0.7)
    axes[1,1].set_title('Lomb-Scargle Periodogram')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].set_ylabel('Power')
    axes[1,1].set_xlim(0, 15)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fft_vs_lomb_scargle.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Comparison visualization saved as 'fft_vs_lomb_scargle.png'")

def real_world_example_ecg():
    """
    Simulate ECG-like signal analysis using frequency domain methods
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: ECG-LIKE SIGNAL ANALYSIS")
    print("=" * 60)
    
    # Simulate ECG signal
    np.random.seed(42)
    fs = 250  # Hz (typical ECG sampling rate)
    duration = 10  # seconds
    t = np.arange(0, duration, 1/fs)
    
    # Heart rate parameters
    heart_rate = 72  # beats per minute
    beat_freq = heart_rate / 60  # Hz
    
    # Create ECG-like signal
    # Main QRS complex
    qrs_signal = np.zeros_like(t)
    beat_interval = fs / beat_freq  # samples per beat
    
    for i in range(int(len(t) / beat_interval)):
        beat_center = int(i * beat_interval)
        if beat_center < len(t):
            # Simple QRS complex simulation
            qrs_width = int(0.1 * fs)  # 100ms width
            start = max(0, beat_center - qrs_width//2)
            end = min(len(t), beat_center + qrs_width//2)
            
            # QRS shape (simplified)
            qrs_samples = end - start
            qrs_pattern = np.exp(-np.linspace(-2, 2, qrs_samples)**2) * 2  # Gaussian-like
            qrs_signal[start:end] += qrs_pattern[:qrs_samples]
    
    # Add baseline wander (typical ECG artifact)
    baseline_freq = 0.5  # Hz
    baseline_wander = 0.3 * np.sin(2*np.pi*baseline_freq*t)
    
    # Add high-frequency noise (muscle artifact, electrical interference)
    noise_50hz = 0.05 * np.sin(2*np.pi*50*t)  # 50Hz power line interference
    noise_60hz = 0.03 * np.sin(2*np.pi*60*t)  # 60Hz power line interference
    random_noise = 0.1 * np.random.randn(len(t))
    
    # Combine all components
    ecg_signal = qrs_signal + baseline_wander + noise_50hz + noise_60hz + random_noise
    
    print(f"📊 ECG Signal Simulation:")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Heart rate: {heart_rate} BPM")
    print(f"  Beat frequency: {beat_freq:.2f} Hz")
    print(f"  Expected harmonics: {beat_freq:.2f}, {2*beat_freq:.2f}, {3*beat_freq:.2f} Hz")
    
    # Frequency analysis
    fft_result = ct.perform_fft_py(ecg_signal.tolist())
    power = [abs(c)**2 for c in fft_result]
    freqs = np.fft.fftfreq(len(ecg_signal), 1/fs)
    
    # Analyze different frequency bands
    frequency_bands = {
        'Baseline wander': (0, 1),
        'Heart rate': (0.8, 3),
        'QRS complex': (3, 40),
        'Muscle artifact': (40, 100),
        'Power line interference': (45, 65)
    }
    
    print(f"\n🔍 Frequency Band Analysis:")
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = np.array(power)[pos_mask]
    
    for band_name, (f_min, f_max) in frequency_bands.items():
        band_mask = (pos_freqs >= f_min) & (pos_freqs <= f_max)
        band_power = pos_power[band_mask]
        if len(band_power) > 0:
            total_power = np.sum(band_power)
            max_freq = pos_freqs[band_mask][np.argmax(band_power)]
            print(f"  {band_name:20s}: Peak at {max_freq:.1f} Hz, Total power: {total_power:.1f}")
    
    # Heart rate detection
    hr_band_mask = (pos_freqs >= 0.8) & (pos_freqs <= 3.0)
    hr_freqs = pos_freqs[hr_band_mask]
    hr_power = pos_power[hr_band_mask]
    
    if len(hr_power) > 0:
        detected_hr_freq = hr_freqs[np.argmax(hr_power)]
        detected_hr_bpm = detected_hr_freq * 60
        print(f"\n💓 Heart Rate Detection:")
        print(f"  True heart rate: {heart_rate} BPM")
        print(f"  Detected frequency: {detected_hr_freq:.2f} Hz")
        print(f"  Detected heart rate: {detected_hr_bpm:.0f} BPM")
        print(f"  Error: {abs(detected_hr_bpm - heart_rate):.1f} BPM")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Time domain signal
    axes[0].plot(t[:fs*3], ecg_signal[:fs*3], 'b-', linewidth=1)  # Show 3 seconds
    axes[0].set_title('Simulated ECG Signal (First 3 seconds)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].grid(True, alpha=0.3)
    
    # Full power spectrum
    axes[1].plot(pos_freqs, pos_power, 'r-', linewidth=1)
    axes[1].set_title('Full Power Spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power')
    axes[1].set_xlim(0, 100)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Mark frequency bands
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    for i, (band_name, (f_min, f_max)) in enumerate(frequency_bands.items()):
        axes[1].axvspan(f_min, f_max, alpha=0.3, color=colors[i % len(colors)], 
                       label=band_name)
    axes[1].legend()
    
    # Heart rate band detail
    axes[2].plot(hr_freqs, hr_power, 'g-', linewidth=2)
    axes[2].axvline(beat_freq, color='red', linestyle='--', linewidth=2, 
                   label=f'True HR: {heart_rate} BPM')
    if len(hr_power) > 0:
        axes[2].axvline(detected_hr_freq, color='orange', linestyle=':', linewidth=2,
                       label=f'Detected: {detected_hr_bpm:.0f} BPM')
    axes[2].set_title('Heart Rate Band Detail (0.8-3 Hz)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Power')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ecg_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 ECG analysis visualization saved as 'ecg_frequency_analysis.png'")

if __name__ == "__main__":
    print("🚀 ChronoXtract Frequency Analysis Examples")
    print("This script demonstrates frequency domain analysis capabilities.\n")
    
    # Run all demonstrations
    demonstrate_fft_analysis()
    demonstrate_lomb_scargle()
    compare_fft_vs_lomb_scargle()
    real_world_example_ecg()
    
    print("\n" + "=" * 60)
    print("✅ All frequency analysis examples completed!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)