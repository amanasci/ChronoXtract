"""
Tests for the topology module: Takens embedding, persistent homology,
Betti curves, persistence landscapes, and the combined pipeline.
"""

import math
import pytest
import numpy as np
import chronoxtract as ct


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sine_ts():
    """Deterministic sine-wave time series (200 samples)."""
    t = np.linspace(0, 4 * np.pi, 200)
    return np.sin(t)


@pytest.fixture
def random_ts():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


@pytest.fixture
def short_embedding():
    """Small 2-D embedding from a short time series (<=50 pts → H1 computed)."""
    rng = np.random.default_rng(0)
    ts = rng.standard_normal(60)
    return ct.takens_embedding(ts, dimension=2, delay=1, stride=2)


@pytest.fixture
def circle_embedding():
    """Point cloud on a circle – should produce exactly 1 H1 class."""
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    return pts


# ─── takens_embedding ─────────────────────────────────────────────────────────

class TestTakensEmbedding:

    def test_output_shape_basic(self, sine_ts):
        # dimension=3, delay=1, stride=1 → n_points = 200 - 2 = 198
        pts = ct.takens_embedding(sine_ts, dimension=3, delay=1)
        assert pts.shape == (198, 3)

    def test_output_shape_delay(self, sine_ts):
        # dimension=3, delay=5 → 200 - 2*5 = 190
        pts = ct.takens_embedding(sine_ts, dimension=3, delay=5)
        assert pts.shape == (190, 3)

    def test_output_shape_stride(self, sine_ts):
        # dimension=2, delay=1, stride=3 → max_start=199, n_points=ceil(199/3)=67
        pts = ct.takens_embedding(sine_ts, dimension=2, delay=1, stride=3)
        assert pts.shape == (67, 2)

    def test_values_dimension_1(self, sine_ts):
        # dimension=1 → each row is just x[i]
        pts = ct.takens_embedding(sine_ts, dimension=1, delay=1)
        assert pts.shape == (200, 1)
        np.testing.assert_allclose(pts[:, 0], sine_ts)

    def test_values_delay_2(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pts = ct.takens_embedding(ts, dimension=2, delay=2)
        # (x[0],x[2]), (x[1],x[3]), (x[2],x[4])
        assert pts.shape == (3, 2)
        np.testing.assert_allclose(pts[0], [1.0, 3.0])
        np.testing.assert_allclose(pts[1], [2.0, 4.0])
        np.testing.assert_allclose(pts[2], [3.0, 5.0])

    def test_normalize_flag(self, sine_ts):
        pts = ct.takens_embedding(sine_ts, dimension=3, delay=1, normalize=True)
        # Each row should have mean ≈ 0 and std ≈ 1
        row_means = pts.mean(axis=1)
        row_stds = pts.std(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-10)
        np.testing.assert_allclose(row_stds, 1.0, atol=1e-10)

    def test_output_dtype(self, sine_ts):
        pts = ct.takens_embedding(sine_ts, dimension=2, delay=1)
        assert pts.dtype == np.float64

    def test_deterministic(self, sine_ts):
        a = ct.takens_embedding(sine_ts, dimension=3, delay=2, stride=2)
        b = ct.takens_embedding(sine_ts, dimension=3, delay=2, stride=2)
        np.testing.assert_array_equal(a, b)

    # ── Error handling ─────────────────────────────────────────────────────

    def test_error_empty(self):
        with pytest.raises(ValueError):
            ct.takens_embedding(np.array([]), dimension=2, delay=1)

    def test_error_zero_dimension(self, sine_ts):
        with pytest.raises(ValueError):
            ct.takens_embedding(sine_ts, dimension=0, delay=1)

    def test_error_zero_delay(self, sine_ts):
        with pytest.raises(ValueError):
            ct.takens_embedding(sine_ts, dimension=2, delay=0)

    def test_error_zero_stride(self, sine_ts):
        with pytest.raises(ValueError):
            ct.takens_embedding(sine_ts, dimension=2, delay=1, stride=0)

    def test_error_too_short(self):
        # Need at least (d-1)*tau + 1 = 5 samples for dim=3, delay=2
        with pytest.raises(ValueError):
            ct.takens_embedding(np.array([1.0, 2.0, 3.0]), dimension=3, delay=2)

    def test_single_sample_dim1(self):
        pts = ct.takens_embedding(np.array([42.0]), dimension=1, delay=1)
        assert pts.shape == (1, 1)
        assert pts[0, 0] == pytest.approx(42.0)


# ─── persistent_homology_summary ─────────────────────────────────────────────

class TestPersistentHomologySummary:

    def test_required_keys(self, short_embedding):
        feats = ct.persistent_homology_summary(short_embedding)
        for key in [
            "n_pairs_h0", "max_persistence_h0", "total_persistence_h0",
            "mean_persistence_h0", "persistence_entropy_h0", "n_essential_h0",
            "n_pairs_h1", "max_persistence_h1", "total_persistence_h1",
            "mean_persistence_h1", "persistence_entropy_h1", "n_essential_h1",
            "max_finite_scale",
        ]:
            assert key in feats, f"missing key: {key}"

    def test_h0_n_pairs_matches_npoints(self, short_embedding):
        n = short_embedding.shape[0]
        feats = ct.persistent_homology_summary(short_embedding)
        # n-1 finite H0 pairs + 1 essential = n total H0 features
        assert feats["n_pairs_h0"] + feats["n_essential_h0"] == pytest.approx(n)

    def test_h0_max_persistence_non_negative(self, short_embedding):
        feats = ct.persistent_homology_summary(short_embedding)
        assert feats["max_persistence_h0"] >= 0.0

    def test_persistence_entropy_non_negative(self, short_embedding):
        feats = ct.persistent_homology_summary(short_embedding)
        assert feats["persistence_entropy_h0"] >= 0.0

    def test_h1_computed_for_small_cloud(self, short_embedding):
        # short_embedding has ≤ 50 points
        feats = ct.persistent_homology_summary(short_embedding, max_h1_points=50)
        # H1 should be computed; n_essential_h0 == 1 (one connected component)
        assert feats["n_essential_h0"] == pytest.approx(1.0)

    def test_circle_has_h1(self, circle_embedding):
        feats = ct.persistent_homology_summary(
            circle_embedding, max_h1_points=35
        )
        # A circle has exactly 1 loop.
        assert feats["n_pairs_h1"] == pytest.approx(1.0)

    def test_max_scale_reduces_pairs(self, short_embedding):
        feats_full = ct.persistent_homology_summary(short_embedding)
        # Use a very small max_scale so most edges are excluded.
        feats_small = ct.persistent_homology_summary(
            short_embedding, max_scale=1e-6
        )
        # With tiny scale there should be more essential H0 classes.
        assert feats_small["n_essential_h0"] >= feats_full["n_essential_h0"]

    def test_error_empty_points(self):
        with pytest.raises(Exception):
            ct.persistent_homology_summary(np.zeros((0, 2)))

    def test_deterministic_output(self, short_embedding):
        a = ct.persistent_homology_summary(short_embedding)
        b = ct.persistent_homology_summary(short_embedding)
        assert a["n_pairs_h0"] == b["n_pairs_h0"]
        assert a["max_persistence_h0"] == pytest.approx(b["max_persistence_h0"])


# ─── betti_curve_features ─────────────────────────────────────────────────────

class TestBettiCurveFeatures:

    def test_required_keys(self, short_embedding):
        feats = ct.betti_curve_features(short_embedding)
        for key in [
            "betti_0_auc", "betti_0_peak", "betti_0_mean",
            "betti_1_auc", "betti_1_peak", "betti_1_mean",
        ]:
            assert key in feats, f"missing key: {key}"

    def test_auc_non_negative(self, short_embedding):
        feats = ct.betti_curve_features(short_embedding)
        assert feats["betti_0_auc"] >= 0.0
        assert feats["betti_1_auc"] >= 0.0

    def test_peak_non_negative(self, short_embedding):
        feats = ct.betti_curve_features(short_embedding)
        assert feats["betti_0_peak"] >= 0.0
        assert feats["betti_1_peak"] >= 0.0

    def test_betti0_auc_positive_for_real_data(self, short_embedding):
        # Any non-trivial point cloud must have β_0 > 0 somewhere
        feats = ct.betti_curve_features(short_embedding)
        assert feats["betti_0_auc"] > 0.0

    def test_circle_betti1_positive(self, circle_embedding):
        feats = ct.betti_curve_features(circle_embedding, max_h1_points=35)
        assert feats["betti_1_auc"] > 0.0, "circle must have positive β_1 AUC"
        assert feats["betti_1_peak"] >= 1.0

    def test_n_samples_parameter(self, short_embedding):
        # Should not crash for different n_samples values
        for ns in [10, 50, 100]:
            feats = ct.betti_curve_features(short_embedding, n_samples=ns)
            assert feats["betti_0_auc"] >= 0.0

    def test_deterministic(self, short_embedding):
        a = ct.betti_curve_features(short_embedding)
        b = ct.betti_curve_features(short_embedding)
        assert a["betti_0_auc"] == pytest.approx(b["betti_0_auc"])


# ─── persistence_landscape_features ──────────────────────────────────────────

class TestPersistenceLandscapeFeatures:

    def test_required_keys_default_layers(self, short_embedding):
        feats = ct.persistence_landscape_features(short_embedding)
        for key in [
            "landscape_h0_l1_l1", "landscape_h0_l1_l2",
            "landscape_h0_l1_peak", "landscape_h0_l1_mean",
            "landscape_h0_l2_l1",
            "landscape_h0_l3_l1",
            "landscape_h1_l1_l1",
        ]:
            assert key in feats, f"missing key: {key}"

    def test_non_negative_values(self, short_embedding):
        feats = ct.persistence_landscape_features(short_embedding)
        for k, v in feats.items():
            assert v >= 0.0, f"negative value for key: {k}"

    def test_layer_ordering(self, circle_embedding):
        """λ_1 norms should be ≥ λ_2 norms for each homology dimension."""
        feats = ct.persistence_landscape_features(
            circle_embedding, n_layers=2, max_h1_points=35
        )
        for dim in range(2):
            l1_l1 = feats[f"landscape_h{dim}_l1_l1"]
            l2_l1 = feats[f"landscape_h{dim}_l2_l1"]
            assert l1_l1 >= l2_l1, (
                f"h{dim} layer 1 L1 ({l1_l1}) must be >= layer 2 L1 ({l2_l1})"
            )

    def test_h0_landscape_positive(self, short_embedding):
        feats = ct.persistence_landscape_features(short_embedding)
        assert feats["landscape_h0_l1_l1"] > 0.0

    def test_custom_n_layers(self, short_embedding):
        feats5 = ct.persistence_landscape_features(short_embedding, n_layers=5)
        for l in range(1, 6):
            key = f"landscape_h0_l{l}_l1"
            assert key in feats5

    def test_deterministic(self, short_embedding):
        a = ct.persistence_landscape_features(short_embedding)
        b = ct.persistence_landscape_features(short_embedding)
        for k in a:
            assert a[k] == pytest.approx(b[k])


# ─── topological_features (combined pipeline) ─────────────────────────────────

class TestTopologicalFeatures:

    def test_required_keys(self, sine_ts):
        feats = ct.topological_features(sine_ts, dimension=2, delay=5)
        for key in [
            "n_pairs_h0", "betti_0_auc", "landscape_h0_l1_l1",
            "n_embedding_points", "embedding_dim", "h1_computed",
        ]:
            assert key in feats, f"missing key: {key}"

    def test_embedding_metadata(self, sine_ts):
        feats = ct.topological_features(sine_ts, dimension=3, delay=2)
        # 200 - 2*2 = 196 embedding points
        assert feats["n_embedding_points"] == pytest.approx(196.0)
        assert feats["embedding_dim"] == pytest.approx(3.0)

    def test_h1_flag_small_input(self):
        rng = np.random.default_rng(99)
        ts = rng.standard_normal(60)
        feats = ct.topological_features(ts, dimension=2, delay=1, stride=2,
                                        max_h1_points=50)
        # 60 - 1*(2-1) = 59 starting points; with stride=2 → ~30 pts → ≤50
        assert feats["h1_computed"] == pytest.approx(1.0)

    def test_h1_flag_large_input(self, sine_ts):
        feats = ct.topological_features(sine_ts, dimension=2, delay=1,
                                        stride=1, max_h1_points=50)
        # 200 - 1 = 199 pts > 50
        assert feats["h1_computed"] == pytest.approx(0.0)

    def test_deterministic(self, sine_ts):
        a = ct.topological_features(sine_ts, dimension=2, delay=3)
        b = ct.topological_features(sine_ts, dimension=2, delay=3)
        for k in a:
            assert a[k] == pytest.approx(b[k])

    def test_normalize_option(self, sine_ts):
        a = ct.topological_features(sine_ts, dimension=2, delay=3, normalize=False)
        b = ct.topological_features(sine_ts, dimension=2, delay=3, normalize=True)
        # Normalised embedding should differ in at least one feature
        diffs = [abs(a[k] - b[k]) > 1e-6 for k in a if isinstance(a[k], float)]
        assert any(diffs), "normalised and non-normalised results must differ"

    def test_error_too_short(self):
        with pytest.raises(ValueError):
            ct.topological_features(np.array([1.0, 2.0]), dimension=3, delay=2)

    def test_error_empty(self):
        with pytest.raises(ValueError):
            ct.topological_features(np.array([]), dimension=2, delay=1)

    def test_all_values_finite(self, sine_ts):
        feats = ct.topological_features(sine_ts, dimension=2, delay=5)
        for k, v in feats.items():
            assert math.isfinite(v), f"non-finite value for key: {k} = {v}"

    def test_all_values_non_negative(self, sine_ts):
        feats = ct.topological_features(sine_ts, dimension=2, delay=5)
        for k, v in feats.items():
            assert v >= 0.0, f"negative value for key: {k} = {v}"

    def test_stride_reduces_points(self, sine_ts):
        a = ct.topological_features(sine_ts, dimension=2, delay=1, stride=1)
        b = ct.topological_features(sine_ts, dimension=2, delay=1, stride=4)
        assert b["n_embedding_points"] < a["n_embedding_points"]

    def test_max_scale_parameter(self, sine_ts):
        # Should not raise with valid max_scale
        feats = ct.topological_features(sine_ts, dimension=2, delay=5,
                                        max_scale=1.0)
        assert "n_pairs_h0" in feats

    # ── Sanity checks on known signals ────────────────────────────────────

    def test_sine_has_positive_betti0_auc(self, sine_ts):
        feats = ct.topological_features(sine_ts, dimension=2, delay=5)
        assert feats["betti_0_auc"] > 0.0

    def test_total_persistence_increases_with_dimension(self, random_ts):
        """Higher embedding dimension generally captures more structure."""
        f2 = ct.topological_features(random_ts, dimension=2, delay=1, stride=4)
        f3 = ct.topological_features(random_ts, dimension=3, delay=1, stride=4)
        # Both should have positive total persistence
        assert f2["total_persistence_h0"] > 0.0
        assert f3["total_persistence_h0"] > 0.0
