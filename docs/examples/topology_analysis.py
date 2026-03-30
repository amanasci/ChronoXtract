"""
Topological Time-Series Analysis with ChronoXtract
====================================================

This example demonstrates the topological feature-extraction capabilities of
ChronoXtract, showing how tools from Topological Data Analysis (TDA) can
reveal the geometric structure of time-series attractors.

The workflow is:
  1. Generate or load a scalar time series.
  2. Apply Takens delay-coordinate embedding to reconstruct a point cloud.
  3. Compute persistent homology of the point cloud.
  4. Extract compact scalar features (Betti curves, persistence landscapes).

These features complement classical statistics by capturing global
topological properties – such as the number of loops in an attractor – that
purely distributional features cannot see.
"""

import numpy as np
import sys
import os

# Try to import matplotlib for optional visualisation
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import chronoxtract as ct

# ─── 1. Generate synthetic time series ─────────────────────────────────────

def make_signals():
    """Return three contrasting scalar time series."""
    t = np.linspace(0, 8 * np.pi, 500)

    # Periodic sine: maps to a closed curve (circle) in 2-D/3-D phase space,
    # which produces an H1 feature when the embedding dimension ≥ 2 and
    # enough of the cycle is captured.
    sine = np.sin(t)

    # Lorenz x-coordinate (chaotic): computed via simple Euler integration
    dt, sigma, rho, beta = 0.01, 10.0, 28.0, 8 / 3
    xs, ys, zs = [1.0], [1.0], [1.0]
    for _ in range(len(t) - 1):
        dx = sigma * (ys[-1] - xs[-1]) * dt
        dy = (xs[-1] * (rho - zs[-1]) - ys[-1]) * dt
        dz = (xs[-1] * ys[-1] - beta * zs[-1]) * dt
        xs.append(xs[-1] + dx)
        ys.append(ys[-1] + dy)
        zs.append(zs[-1] + dz)
    lorenz = np.array(xs)

    # White noise: no geometric structure
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(t))

    return {"sine": sine, "lorenz": lorenz, "noise": noise}


# ─── 2. Takens embedding demo ───────────────────────────────────────────────

def demo_takens_embedding():
    """Show how the embedding parameters affect the reconstructed cloud."""
    print("=" * 60)
    print("DEMO 1: Takens delay-coordinate embedding")
    print("=" * 60)

    ts = np.sin(np.linspace(0, 6 * np.pi, 200))

    for dim, delay in [(2, 1), (2, 10), (3, 5)]:
        pts = ct.takens_embedding(ts, dimension=dim, delay=delay)
        print(
            f"  dim={dim}, delay={delay}: "
            f"{len(ts)} samples → {pts.shape[0]} embedded points in ℝ^{dim}"
        )

    # Normalised embedding
    pts_norm = ct.takens_embedding(ts, dimension=3, delay=5, normalize=True)
    row_means = pts_norm.mean(axis=1)
    print(
        f"\n  Normalised embedding: row means ~ 0 "
        f"(max |mean| = {np.abs(row_means).max():.2e})"
    )


# ─── 3. Persistent homology demo ────────────────────────────────────────────

def demo_persistent_homology(signals):
    """Compute and compare persistence features across three signals."""
    print("\n" + "=" * 60)
    print("DEMO 2: Persistent homology summary")
    print("=" * 60)

    print(f"\n{'Signal':<10}  {'H0 pairs':>8}  {'max pers H0':>12}  "
          f"{'H1 pairs':>8}  {'max pers H1':>12}  {'pers entropy H0':>16}")
    print("-" * 70)

    for name, ts in signals.items():
        # Use stride=4 to cap embedding points ≤ 50 so H1 is always computed
        pts = ct.takens_embedding(ts, dimension=2, delay=5, stride=4)
        feats = ct.persistent_homology_summary(pts, max_h1_points=50)
        print(
            f"{name:<10}  {feats['n_pairs_h0']:>8.0f}  "
            f"{feats['max_persistence_h0']:>12.4f}  "
            f"{feats['n_pairs_h1']:>8.0f}  "
            f"{feats['max_persistence_h1']:>12.4f}  "
            f"{feats['persistence_entropy_h0']:>16.4f}"
        )


# ─── 4. Betti curve demo ─────────────────────────────────────────────────────

def demo_betti_curves(signals):
    """Show Betti-curve area under curve for three signals."""
    print("\n" + "=" * 60)
    print("DEMO 3: Betti curve features")
    print("=" * 60)

    print(f"\n{'Signal':<10}  {'β₀ AUC':>10}  {'β₀ peak':>9}  "
          f"{'β₁ AUC':>10}  {'β₁ peak':>9}")
    print("-" * 55)

    for name, ts in signals.items():
        pts = ct.takens_embedding(ts, dimension=2, delay=5, stride=4)
        feats = ct.betti_curve_features(pts, n_samples=100, max_h1_points=50)
        print(
            f"{name:<10}  {feats['betti_0_auc']:>10.4f}  "
            f"{feats['betti_0_peak']:>9.1f}  "
            f"{feats['betti_1_auc']:>10.4f}  "
            f"{feats['betti_1_peak']:>9.1f}"
        )


# ─── 5. Persistence landscape demo ──────────────────────────────────────────

def demo_persistence_landscape(signals):
    """Compare landscape L1 norms across signals."""
    print("\n" + "=" * 60)
    print("DEMO 4: Persistence landscape features")
    print("=" * 60)

    print(f"\n{'Signal':<10}  {'H0-λ1 L1':>10}  {'H0-λ2 L1':>10}  "
          f"{'H1-λ1 L1':>10}")
    print("-" * 45)

    for name, ts in signals.items():
        pts = ct.takens_embedding(ts, dimension=2, delay=5, stride=4)
        feats = ct.persistence_landscape_features(
            pts, n_layers=2, n_samples=100, max_h1_points=50
        )
        print(
            f"{name:<10}  {feats['landscape_h0_l1_l1']:>10.4f}  "
            f"{feats['landscape_h0_l2_l1']:>10.4f}  "
            f"{feats['landscape_h1_l1_l1']:>10.4f}"
        )


# ─── 6. Combined pipeline ───────────────────────────────────────────────────

def demo_full_pipeline(signals):
    """topological_features() – single-call comprehensive features."""
    print("\n" + "=" * 60)
    print("DEMO 5: Combined pipeline (topological_features)")
    print("=" * 60)

    for name, ts in signals.items():
        feats = ct.topological_features(
            ts,
            dimension=2,
            delay=5,
            stride=4,
            max_h1_points=50,
            n_betti_samples=50,
            n_landscape_layers=2,
        )
        print(f"\n  {name}: {len(feats)} features computed")
        print(f"    n_embedding_points = {feats['n_embedding_points']:.0f}")
        print(f"    h1_computed        = {bool(feats['h1_computed'])}")
        print(f"    n_pairs_h0         = {feats['n_pairs_h0']:.0f}")
        print(f"    n_pairs_h1         = {feats['n_pairs_h1']:.0f}")
        print(f"    betti_0_auc        = {feats['betti_0_auc']:.4f}")
        print(f"    landscape_h0_l1_l1 = {feats['landscape_h0_l1_l1']:.4f}")


# ─── 7. Optional visualisation ──────────────────────────────────────────────

def demo_visualisation(signals):
    """Save a 2-panel figure: embedding point clouds and Betti-0 curves."""
    if not HAS_MATPLOTLIB:
        print("\n  (matplotlib not available – skipping plots)")
        return

    print("\n" + "=" * 60)
    print("DEMO 6: Visualisation (saved to /tmp/topology_example.png)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Topological features of three time series", fontsize=14)

    for col, (name, ts) in enumerate(signals.items()):
        pts = ct.takens_embedding(ts, dimension=2, delay=5)
        pairs_phom = ct.persistent_homology_summary(
            pts[:50], max_h1_points=50  # first 50 pts for speed
        )

        # Row 0: embedding scatter
        ax = axes[0, col]
        ax.scatter(pts[:200, 0], pts[:200, 1], s=2, alpha=0.5)
        ax.set_title(f"{name}: 2-D Takens embedding")
        ax.set_xlabel("x[i]")
        ax.set_ylabel("x[i+5]")

        # Row 1: Betti summary for first 50 embedding points (approximation for demo)
        pts_small = pts[:50]
        betti_feats = ct.betti_curve_features(
            pts_small, n_samples=100, max_h1_points=50
        )
        t_vals = np.linspace(0, t_max, 100)
        ax = axes[1, col]
        # Approximate Betti curve shape using AUC / range as scale
        # (exact curve values require lower-level access)
        ax.text(
            0.5, 0.5,
            f"β₀ AUC = {betti_feats['betti_0_auc']:.3f}\n"
            f"β₁ AUC = {betti_feats['betti_1_auc']:.3f}",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_title(f"{name}: Betti AUC summary")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("/tmp/topology_example.png", dpi=100, bbox_inches="tight")
    print("  Figure saved to /tmp/topology_example.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signals = make_signals()

    demo_takens_embedding()
    demo_persistent_homology(signals)
    demo_betti_curves(signals)
    demo_persistence_landscape(signals)
    demo_full_pipeline(signals)
    demo_visualisation(signals)

    print("\n" + "=" * 60)
    print("Example completed successfully.")
    print("=" * 60)
