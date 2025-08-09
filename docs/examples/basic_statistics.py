#!/usr/bin/env python3
"""
Basic Statistics Examples with ChronoXtract

This script demonstrates the core statistical functions available in ChronoXtract,
including descriptive statistics, distribution analysis, and data interpretation.
"""

import chronoxtract as ct
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def demonstrate_basic_statistics():
    """
    Demonstrate the time_series_summary function with different data distributions
    """
    print("=" * 60)
    print("BASIC STATISTICS DEMONSTRATION")
    print("=" * 60)
    
    # Generate different types of data
    np.random.seed(42)
    
    datasets = {
        "Normal Distribution": np.random.normal(0, 1, 1000),
        "Exponential (Right-skewed)": np.random.exponential(2, 1000),
        "Uniform Distribution": np.random.uniform(-5, 5, 1000),
        "Bimodal Distribution": np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ])
    }
    
    # Analyze each dataset
    for name, data in datasets.items():
        print(f"\n📊 {name}:")
        print("-" * 40)
        
        stats = ct.time_series_summary(data.tolist())
        
        print(f"Mean:               {stats['mean']:.4f}")
        print(f"Median:             {stats['median']:.4f}")
        print(f"Mode:               {stats['mode']:.4f}")
        print(f"Standard Deviation: {stats['standard_deviation']:.4f}")
        print(f"Variance:           {stats['variance']:.4f}")
        print(f"Skewness:           {stats['skewness']:.4f}")
        print(f"Kurtosis:           {stats['kurtosis']:.4f}")
        print(f"Range:              {stats['range']:.4f}")
        print(f"Q25-Q75 (IQR):      {stats['q75'] - stats['q25']:.4f}")
        
        # Interpret the results
        interpret_distribution(stats)

def interpret_distribution(stats: Dict[str, float]):
    """
    Provide interpretation of statistical measures
    """
    print("  📈 Interpretation:")
    
    # Skewness interpretation
    skew = stats['skewness']
    if abs(skew) < 0.5:
        print("    • Distribution: Approximately symmetric")
    elif skew > 0.5:
        print("    • Distribution: Right-skewed (tail extends right)")
    else:
        print("    • Distribution: Left-skewed (tail extends left)")
    
    # Kurtosis interpretation
    kurt = stats['kurtosis']
    if kurt > 3.5:
        print("    • Tail shape: Heavy-tailed (more extreme values)")
    elif kurt < 2.5:
        print("    • Tail shape: Light-tailed (fewer extreme values)")
    else:
        print("    • Tail shape: Normal-like")
    
    # Mean vs Median comparison
    mean_median_diff = abs(stats['mean'] - stats['median'])
    if mean_median_diff > 0.1 * stats['standard_deviation']:
        print("    • Asymmetry: Mean and median differ significantly")
    else:
        print("    • Asymmetry: Mean and median are close (good symmetry)")

def demonstrate_central_tendencies():
    """
    Compare different measures of central tendency
    """
    print("\n" + "=" * 60)
    print("CENTRAL TENDENCIES COMPARISON")
    print("=" * 60)
    
    # Create different scenarios
    scenarios = {
        "Symmetric Data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Data with Outliers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "Repeated Values": [1, 2, 2, 3, 3, 3, 4, 4, 5],
        "Skewed Data": [1, 1, 1, 2, 2, 3, 4, 5, 8, 15]
    }
    
    for name, data in scenarios.items():
        print(f"\n📊 {name}: {data}")
        
        mean, median, mode = ct.time_series_mean_median_mode(data)
        
        print(f"  Mean:   {mean:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Mode:   {mode:.2f}")
        
        # Determine best measure
        if name == "Data with Outliers":
            print("  💡 Best measure: Median (robust to outliers)")
        elif name == "Repeated Values":
            print("  💡 Best measure: Mode (shows most common value)")
        elif name == "Symmetric Data":
            print("  💡 Best measure: Mean (all measures are equivalent)")
        else:
            print("  💡 Best measure: Depends on analysis goal")

def visualize_distributions():
    """
    Create visualizations to understand statistical concepts
    """
    print("\n" + "=" * 60)
    print("CREATING DISTRIBUTION VISUALIZATIONS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create subplot for different distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution Shapes and Their Statistics', fontsize=16)
    
    distributions = [
        ("Normal", np.random.normal(0, 1, 1000)),
        ("Right-skewed", np.random.exponential(1, 1000)),
        ("Left-skewed", -np.random.exponential(1, 1000)),
        ("Heavy-tailed", np.random.laplace(0, 0.7, 1000))
    ]
    
    for idx, (name, data) in enumerate(distributions):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Calculate statistics
        stats = ct.time_series_summary(data.tolist())
        
        # Create histogram
        ax.hist(data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add vertical lines for mean, median
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}")
        ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.2f}")
        
        # Add statistics text
        textstr = f"Skewness: {stats['skewness']:.2f}\nKurtosis: {stats['kurtosis']:.2f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f"{name} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_statistics_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualization saved as 'basic_statistics_distributions.png'")

def analyze_real_world_example():
    """
    Analyze a realistic dataset (simulated stock returns)
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: STOCK RETURNS ANALYSIS")
    print("=" * 60)
    
    # Simulate daily stock returns (realistic parameters)
    np.random.seed(42)
    n_days = 252  # One trading year
    
    # Stock returns typically follow a distribution with:
    # - Small positive mean (market growth)
    # - Fat tails (more extreme movements than normal distribution)
    daily_returns = np.random.laplace(0.0005, 0.015, n_days)  # Laplace distribution
    
    # Convert to cumulative returns (stock price)
    initial_price = 100
    prices = [initial_price]
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Analyze returns
    print("📈 Daily Returns Analysis:")
    returns_stats = ct.time_series_summary(daily_returns.tolist())
    
    print(f"Average daily return:    {returns_stats['mean']*100:.3f}%")
    print(f"Daily volatility:        {returns_stats['standard_deviation']*100:.2f}%")
    print(f"Annualized volatility:   {returns_stats['standard_deviation']*np.sqrt(252)*100:.1f}%")
    print(f"Skewness:               {returns_stats['skewness']:.3f}")
    print(f"Kurtosis:               {returns_stats['kurtosis']:.3f}")
    print(f"Worst day:              {returns_stats['minimum']*100:.2f}%")
    print(f"Best day:               {returns_stats['maximum']*100:.2f}%")
    
    # Risk metrics
    var_5 = returns_stats['q05']  # 5% Value at Risk
    print(f"5% Value at Risk:       {var_5*100:.2f}%")
    
    # Analyze price levels
    print(f"\n💰 Price Analysis:")
    price_stats = ct.time_series_summary(prices)
    print(f"Starting price:         ${initial_price:.2f}")
    print(f"Ending price:           ${prices[-1]:.2f}")
    print(f"Total return:           {(prices[-1]/initial_price - 1)*100:.1f}%")
    print(f"Price volatility:       ${price_stats['standard_deviation']:.2f}")
    print(f"Maximum price:          ${price_stats['maximum']:.2f}")
    print(f"Minimum price:          ${price_stats['minimum']:.2f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price chart
    ax1.plot(prices, color='blue', linewidth=1)
    ax1.set_title('Stock Price Over Time')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Returns histogram
    ax2.hist(daily_returns * 100, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(returns_stats['mean'] * 100, color='red', linestyle='--', 
                label=f"Mean: {returns_stats['mean']*100:.3f}%")
    ax2.set_title('Daily Returns Distribution')
    ax2.set_xlabel('Daily Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stock_analysis_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Stock analysis charts saved as 'stock_analysis_example.png'")

def performance_comparison():
    """
    Compare performance of ChronoXtract vs pure Python
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    import time
    
    # Generate test data
    data_sizes = [1000, 10000, 100000]
    
    for size in data_sizes:
        data = np.random.randn(size).tolist()
        
        print(f"\nDataset size: {size:,} points")
        
        # ChronoXtract timing
        start_time = time.time()
        ct_result = ct.time_series_summary(data)
        ct_time = time.time() - start_time
        
        # Pure Python timing (just mean calculation for comparison)
        start_time = time.time()
        python_mean = sum(data) / len(data)
        python_time = time.time() - start_time
        
        print(f"ChronoXtract (full summary): {ct_time*1000:.2f} ms")
        print(f"Pure Python (mean only):     {python_time*1000:.2f} ms")
        print(f"ChronoXtract mean:           {ct_result['mean']:.6f}")
        print(f"Pure Python mean:            {python_mean:.6f}")
        print(f"Results match: {abs(ct_result['mean'] - python_mean) < 1e-10}")

if __name__ == "__main__":
    print("🚀 ChronoXtract Basic Statistics Examples")
    print("This script demonstrates core statistical analysis capabilities.\n")
    
    # Run all demonstrations
    demonstrate_basic_statistics()
    demonstrate_central_tendencies()
    visualize_distributions()
    analyze_real_world_example()
    performance_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)