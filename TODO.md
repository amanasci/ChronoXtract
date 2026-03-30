# TODO

This is list of features to implement.

## Topological Features (TDA) — Completed ✅

- [x] **Takens delay-coordinate embedding** (`takens_embedding`)
  - Configurable dimension, delay, stride, and optional z-score normalisation
  - Validates inputs; clear errors for too-short series or zero parameters
  - Returns 2-D numpy array
- [x] **Persistent homology summaries** (`persistent_homology_summary`)
  - H0 via union-find (always fast)
  - H0+H1 via boundary-matrix reduction (Edelsbrunner–Zomorodian 2002) over ℤ/2ℤ
  - Returns n_pairs, max/total/mean persistence, persistence entropy, n_essential
- [x] **Betti curve features** (`betti_curve_features`)
  - Samples β_0 and β_1 curves at configurable resolution
  - Returns AUC, peak, mean for each curve
- [x] **Persistence landscape features** (`persistence_landscape_features`)
  - Computes λ_k(t) = k-th largest tent function per filtration value
  - Returns L1 norm, L2 norm, peak, mean per layer and homology dimension
- [x] **Combined pipeline** (`topological_features`)
  - Single-call full pipeline: embedding → homology → all features
  - Returns 38 scalar features in one dict
- [x] Rust unit tests (31 tests in `src/topology/`)
- [x] Python integration tests (50 tests in `tests/test_topology.py`)
- [x] Example script (`docs/examples/topology_analysis.py`)
- [x] API documentation added to `docs/api_reference.md`

## Features to extract

    - [X] Mean, Median, Mode, Variance, Standard Deviation 
    - [X] Kurtosis, Skewness, Min, Max, Range, Quantiles, Sum, Total Energy
    - [X] Rolling Mean, Rolling Variance, Expanding Sum, Exponential_moving_average, Sliding_window_entropy
    - [X] Frequency Domnain Analysis 
    - [ ] Peak Count, Peak Prominence, Peak Width, Burst Detection, 
    - [ ] Seasonality detection
    - [ ] Trend analysis
    - [ ] Anomaly detection
    - [ ] Time series decomposition
    - [ ] Add a textual summary function based on stats.

## TOP PRIORITY

- [X] Add documentation. With examples.
  - [X] Enhanced README.md with comprehensive feature overview
  - [X] API Reference documentation with detailed function descriptions
  - [X] User Guide with tutorials and best practices
  - [X] Examples Gallery with practical use cases
  - [X] Contributing guidelines for developers
  - [X] License and Changelog files
- [ ] Create a separate weekly report.

## DOCUMENTATION COMPLETED ✅

- [X] **API Reference** (`docs/api_reference.md`)
  - Detailed documentation for all functions
  - Parameter descriptions and return types
  - Comprehensive examples for each function
  - Error handling information

- [X] **Rust Function Documentation** (Rust doc comments ///)
  - All Python-exposed functions now have proper Rust documentation
  - Complete parameter descriptions and return types
  - Examples with proper data types
  - Error condition documentation
  - Accessible via Python's help() system

- [X] **User Guide** (`docs/user_guide.md`)
  - Getting started tutorial
  - Core concepts explanation
  - Best practices and performance tips
  - Real-world application examples

- [X] **Examples Gallery** (`docs/examples/`)
  - Basic statistics examples
  - Rolling statistics demonstrations
  - Frequency analysis tutorials
  - Variability analysis examples
  - Integration examples with pandas/numpy

- [X] **Project Documentation**
  - Enhanced README.md with feature showcase
  - Contributing guidelines (CONTRIBUTING.md)
  - MIT License (LICENSE)
