import timeit
import numpy as np
import chronoxtract as ct
import pandas as pd
from scipy import stats

def benchmark_summary(data_size):
    data = np.random.randn(data_size)

    # ChronoXtract
    ct_time = timeit.timeit(lambda: ct.time_series_summary(data), number=10)

    # Numpy
    def numpy_summary(data):
        np.mean(data)
        np.median(data)
        np.std(data)
        np.min(data)
        np.max(data)
        # For fair comparison, add the other stats
        stats.mode(data)
        stats.skew(data)
        stats.kurtosis(data)
        np.quantile(data, [0.05, 0.25, 0.75, 0.95])
        np.sum(data)
        np.sum(data**2) # absolute energy

    np_time = timeit.timeit(lambda: numpy_summary(data), number=10)

    return ct_time, np_time

def benchmark_rolling_mean(data_size):
    data = np.random.randn(data_size)
    window = 50

    # ChronoXtract
    ct_time = timeit.timeit(lambda: ct.rolling_mean(data, window), number=10)

    # Pandas
    df = pd.Series(data)
    pd_time = timeit.timeit(lambda: df.rolling(window=window).mean(), number=10)

    return ct_time, pd_time

def benchmark_fft(data_size):
    data = np.random.randn(data_size)

    # ChronoXtract
    ct_time = timeit.timeit(lambda: ct.perform_fft_py(data), number=10)

    # Numpy
    np_time = timeit.timeit(lambda: np.fft.fft(data), number=10)

    return ct_time, np_time

def main():
    data_sizes = [100, 1000, 10_000, 100_000, 1_000_000, 5_000_000,10_000_000]

    print("| Function          | Data Size | ChronoXtract (s) | Numpy/Pandas (s) | Speedup      |")
    print("|-------------------|-----------|------------------|------------------|--------------|")

    for size in data_sizes:
        ct_sum_time, np_sum_time = benchmark_summary(size)
        speedup = np_sum_time / ct_sum_time if ct_sum_time > 0 else float('inf')
        print(f"| time_series_summary | {size:<9} | {ct_sum_time:<16.6f} | {np_sum_time:<16.6f} | {speedup:<12.2f}x |")

        if size >= 50:
            ct_roll_time, pd_roll_time = benchmark_rolling_mean(size)
            speedup = pd_roll_time / ct_roll_time if ct_roll_time > 0 else float('inf')
            print(f"| rolling_mean        | {size:<9} | {ct_roll_time:<16.6f} | {pd_roll_time:<16.6f} | {speedup:<12.2f}x |")

        # FFT data size should be a power of 2 for optimal performance
        fft_size = 2**int(np.log2(size))
        if fft_size > 0:
            ct_fft_time, np_fft_time = benchmark_fft(fft_size)
            speedup = np_fft_time / ct_fft_time if ct_fft_time > 0 else float('inf')
            print(f"| perform_fft_py      | {fft_size:<9} | {ct_fft_time:<16.6f} | {np_fft_time:<16.6f} | {speedup:<12.2f}x |")

if __name__ == "__main__":
    main()
