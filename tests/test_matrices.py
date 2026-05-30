"""Tests for matrix-based time-series transformations.

The tested methods are:
- Time-Delay Embedding (Hankel matrix): H[i, j] = x[i + j]
- Gramian Angular Summation Field (GASF):
  G[i, j] = cos(phi_i + phi_j), phi_i = arccos(x_i')
- Markov Transition Field (MTF):
  M[i, j] = P[q_i, q_j], where P is the row-normalized transition matrix
"""

import numpy as np
import pytest

import chronoxtract as ct


def _expected_gasf(series: np.ndarray) -> np.ndarray:
    """Reference GASF using normalized cosine-sum formulation."""
    s_min = np.min(series)
    s_max = np.max(series)
    if np.isclose(s_max - s_min, 0.0):
        x = np.zeros_like(series, dtype=float)
    else:
        x = 2.0 * (series - s_min) / (s_max - s_min) - 1.0
    x = np.clip(x, -1.0, 1.0)
    y = np.sqrt(np.maximum(1.0 - x**2, 0.0))
    return np.outer(x, x) - np.outer(y, y)


def test_time_delay_embedding_known_case():
    """Hankel embedding matches known hand-computed example."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    got = ct.time_delay_embedding(x, 3)
    expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    assert got.shape == (2, 3)
    assert np.allclose(got, expected)


def test_time_delay_embedding_window_equals_length():
    """Window equal to series length yields a single embedding row."""
    x = np.array([3.0, -1.0, 2.0])
    got = ct.time_delay_embedding(x, 3)
    assert got.shape == (1, 3)
    assert np.allclose(got[0], x)


def test_gramian_angular_summation_field_known_case():
    """GASF agrees with reference implementation for a simple signal."""
    x = np.array([0.0, 1.0, 2.0])
    got = ct.gramian_angular_summation_field(x)
    expected = _expected_gasf(x)
    assert got.shape == (3, 3)
    assert np.allclose(got, expected, atol=1e-12)
    assert np.allclose(got, got.T, atol=1e-12)


def test_gramian_angular_summation_field_constant_series():
    """Constant input is handled safely and produces a finite matrix."""
    x = np.array([5.0, 5.0, 5.0])
    got = ct.gramian_angular_summation_field(x)
    expected = -np.ones((3, 3))
    assert np.all(np.isfinite(got))
    assert np.allclose(got, expected, atol=1e-12)


def test_markov_transition_field_known_case():
    """MTF matches expected transition-probability field for 2-bin sequence."""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    got = ct.markov_transition_field(x, 2)
    expected = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    assert got.shape == (4, 4)
    assert np.allclose(got, expected, atol=1e-12)


def test_markov_transition_field_constant_series():
    """Constant series collapses to one state and yields an all-ones field."""
    x = np.array([7.0, 7.0, 7.0, 7.0])
    got = ct.markov_transition_field(x, 3)
    assert np.allclose(got, np.ones((4, 4)), atol=1e-12)


@pytest.mark.parametrize(
    "fn,args",
    [
        (ct.time_delay_embedding, (np.array([]), 2)),
        (ct.gramian_angular_summation_field, (np.array([]),)),
        (ct.markov_transition_field, (np.array([]), 3)),
    ],
)
def test_empty_input_errors(fn, args):
    """All matrix transforms reject empty input."""
    with pytest.raises(ValueError):
        fn(*args)


def test_invalid_parameter_errors():
    """Invalid window length and bin count raise ValueError."""
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ct.time_delay_embedding(x, 0)
    with pytest.raises(ValueError):
        ct.time_delay_embedding(x, 4)
    with pytest.raises(ValueError):
        ct.markov_transition_field(x, 1)


@pytest.mark.parametrize(
    "fn,args",
    [
        (ct.time_delay_embedding, (np.array([1.0, np.nan, 2.0]), 2)),
        (ct.gramian_angular_summation_field, (np.array([1.0, np.inf]),)),
        (ct.markov_transition_field, (np.array([1.0, np.nan]), 2)),
    ],
)
def test_non_finite_input_errors(fn, args):
    """NaN/Inf values are rejected by all matrix transforms."""
    with pytest.raises(ValueError):
        fn(*args)
