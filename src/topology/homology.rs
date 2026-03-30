/// Persistent homology over Vietoris–Rips filtrations.
///
/// This module provides two related algorithms:
///
/// 1. **H0-only (union-find)** – An efficient O(n² α(n)) algorithm that
///    tracks connected-component birth/death pairs as edges are added in
///    order of increasing length.  This is always fast.
///
/// 2. **Full H0+H1 (boundary-matrix reduction)** – Implements the standard
///    column-reduction algorithm (Edelsbrunner–Zomorodian, 2002) over ℤ/2ℤ
///    on the Vietoris–Rips complex up to dimension 2 (vertices, edges,
///    triangles).  This computes both H0 and H1 persistence pairs.
///    Because the triangle count grows as O(n³), this path is guarded by a
///    configurable `max_points` limit (default 50).
///
/// All filtration values are Euclidean distances.

use std::collections::{HashMap, HashSet};

/// A persistence pair (b, d) recording the birth and death filtration values
/// of a topological feature together with its homology dimension.
///
/// `death = f64::INFINITY` denotes an *essential* class that never dies.
#[derive(Debug, Clone)]
pub(crate) struct PersistencePair {
    /// Homology dimension (0 = connected component, 1 = loop).
    pub dim: usize,
    /// Filtration value at which the feature is born.
    pub birth: f64,
    /// Filtration value at which the feature dies, or `f64::INFINITY`.
    pub death: f64,
}

impl PersistencePair {
    /// Persistence length (death − birth).  Returns 0 for essential classes.
    pub fn persistence(&self) -> f64 {
        if self.death.is_infinite() {
            0.0
        } else {
            self.death - self.birth
        }
    }
}

// ─── Union-Find helpers (path-compression + rank) ───────────────────────────

fn find(parent: &mut Vec<usize>, x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }
    parent[x]
}

fn union_by_rank(
    parent: &mut Vec<usize>,
    rank: &mut Vec<usize>,
    survivor: usize,
    dying: usize,
) {
    // `survivor` absorbs `dying`
    if rank[survivor] < rank[dying] {
        parent[survivor] = dying;
    } else if rank[survivor] > rank[dying] {
        parent[dying] = survivor;
    } else {
        parent[dying] = survivor;
        rank[survivor] += 1;
    }
}

// ─── Euclidean distance helpers ──────────────────────────────────────────────

pub(crate) fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn compute_distance_matrix(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut dist = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&points[i], &points[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

// ─── H0 via union-find ───────────────────────────────────────────────────────

/// Compute H0 persistence pairs using Kruskal's-style union-find.
///
/// All vertices are born at filtration value 0.  When edge (i, j) merges
/// two components, the component with the higher root index "dies" at
/// `dist(i, j)`.  The surviving component (lowest root overall) receives
/// an essential pair with `death = f64::INFINITY`.
///
/// Returns `n` persistence pairs: `n - 1` finite and exactly 1 essential.
pub(crate) fn compute_h0_persistence(points: &[Vec<f64>]) -> Vec<PersistencePair> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![PersistencePair {
            dim: 0,
            birth: 0.0,
            death: f64::INFINITY,
        }];
    }

    // Build and sort all edges by distance.
    let dist = compute_distance_matrix(points);
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((dist[i][j], i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];
    let mut pairs: Vec<PersistencePair> = Vec::new();

    for &(d, i, j) in &edges {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            // Elder rule: the component with the smaller root index survives.
            let (survivor, dying) = if ri < rj { (ri, rj) } else { (rj, ri) };
            union_by_rank(&mut parent, &mut rank, survivor, dying);
            pairs.push(PersistencePair {
                dim: 0,
                birth: 0.0,
                death: d,
            });
        }
    }

    // Add the one essential H0 class (the last surviving component).
    pairs.push(PersistencePair {
        dim: 0,
        birth: 0.0,
        death: f64::INFINITY,
    });

    pairs
}

// ─── Z/2ℤ column arithmetic ──────────────────────────────────────────────────

/// Symmetric difference of two **sorted** index lists (XOR over ℤ/2ℤ).
fn z2_add(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // Cancel in ℤ/2ℤ
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// ─── Full H0+H1 via boundary-matrix reduction ────────────────────────────────

/// Internal simplex representation used during boundary-matrix reduction.
#[derive(Clone)]
struct Simplex {
    /// Sorted vertex indices.
    vertices: Vec<usize>,
    /// Vietoris–Rips filtration value (0 for vertices; max edge for triangles).
    filtration: f64,
    /// 0 = vertex, 1 = edge, 2 = triangle.
    dim: usize,
}

/// Compute H0 and H1 persistence pairs via full boundary-matrix reduction
/// (Edelsbrunner–Zomorodian 2002) over the Vietoris–Rips complex.
///
/// # Arguments
/// * `points`       - Point cloud (each entry is an ambient-space coordinate vector).
/// * `max_scale`    - Optional upper bound on filtration values.  Simplices
///                    whose filtration value exceeds `max_scale` are omitted.
///
/// # Returns
/// All H0 and H1 persistence pairs (finite and essential) sorted by birth.
///
/// # Complexity
/// The triangle count is O(n³), so this function is suitable for small
/// point clouds (≤ 50 points).  Call [`compute_h0_persistence`] for larger ones.
pub(crate) fn compute_full_persistence(
    points: &[Vec<f64>],
    max_scale: Option<f64>,
) -> Result<Vec<PersistencePair>, String> {
    let n = points.len();
    if n == 0 {
        return Err("Point cloud cannot be empty".to_string());
    }

    // Validate ambient dimension consistency.
    let ambient_dim = points[0].len();
    for (k, p) in points.iter().enumerate() {
        if p.len() != ambient_dim {
            return Err(format!(
                "All points must have the same dimension: point 0 has {} \
                 coordinates, point {} has {}",
                ambient_dim,
                k,
                p.len()
            ));
        }
    }

    let dist = compute_distance_matrix(points);
    let eff_max = max_scale.unwrap_or(f64::INFINITY);

    // ── Build simplex list ────────────────────────────────────────────────

    let mut simplices: Vec<Simplex> = Vec::new();

    // 0-simplices (vertices) – born at filtration 0.
    for i in 0..n {
        simplices.push(Simplex {
            vertices: vec![i],
            filtration: 0.0,
            dim: 0,
        });
    }

    // 1-simplices (edges).
    for i in 0..n {
        for j in (i + 1)..n {
            let f = dist[i][j];
            if f <= eff_max {
                simplices.push(Simplex {
                    vertices: vec![i, j],
                    filtration: f,
                    dim: 1,
                });
            }
        }
    }

    // 2-simplices (triangles) – filtration = max edge length.
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let f = dist[i][j].max(dist[i][k]).max(dist[j][k]);
                if f <= eff_max {
                    simplices.push(Simplex {
                        vertices: vec![i, j, k],
                        filtration: f,
                        dim: 2,
                    });
                }
            }
        }
    }

    // Sort: ascending filtration, then ascending dimension, then lex vertices.
    simplices.sort_by(|a, b| {
        a.filtration
            .partial_cmp(&b.filtration)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.dim.cmp(&b.dim))
            .then_with(|| a.vertices.cmp(&b.vertices))
    });

    let m = simplices.len();

    // ── Build index lookup ────────────────────────────────────────────────

    let mut simplex_to_idx: HashMap<Vec<usize>, usize> = HashMap::with_capacity(m);
    for (idx, s) in simplices.iter().enumerate() {
        simplex_to_idx.insert(s.vertices.clone(), idx);
    }

    // ── Build initial boundary columns ───────────────────────────────────
    // Each column is stored as a sorted Vec<usize> of row indices.

    let mut columns: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (j, s) in simplices.iter().enumerate() {
        columns[j] = match s.dim {
            0 => vec![],
            1 => {
                let v0 = s.vertices[0];
                let v1 = s.vertices[1];
                let i0 = simplex_to_idx[&vec![v0]];
                let i1 = simplex_to_idx[&vec![v1]];
                let mut b = [i0, i1];
                b.sort_unstable();
                b.to_vec()
            }
            2 => {
                let v0 = s.vertices[0];
                let v1 = s.vertices[1];
                let v2 = s.vertices[2];
                let e01 = simplex_to_idx[&vec![v0, v1]];
                let e02 = simplex_to_idx[&vec![v0, v2]];
                let e12 = simplex_to_idx[&vec![v1, v2]];
                let mut b = [e01, e02, e12];
                b.sort_unstable();
                b.to_vec()
            }
            _ => vec![],
        };
    }

    // ── Standard column reduction (Edelsbrunner–Zomorodian 2002) ─────────

    // `pivot_to_col[r] = j` means reduced column j currently has pivot at row r.
    let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();
    // Finite persistence pairs as (birth_simplex_idx, death_simplex_idx).
    let mut finite_pairs: Vec<(usize, usize)> = Vec::new();

    for j in 0..m {
        loop {
            if columns[j].is_empty() {
                break;
            }
            let low = *columns[j].last().unwrap();
            if let Some(&prev) = pivot_to_col.get(&low) {
                let col_prev = columns[prev].clone();
                columns[j] = z2_add(&columns[j], &col_prev);
            } else {
                pivot_to_col.insert(low, j);
                finite_pairs.push((low, j));
                break;
            }
        }
    }

    // ── Extract persistence pairs ─────────────────────────────────────────

    let mut birth_set: HashSet<usize> = HashSet::new();
    let mut death_set: HashSet<usize> = HashSet::new();
    for &(b, d) in &finite_pairs {
        birth_set.insert(b);
        death_set.insert(d);
    }

    let mut result: Vec<PersistencePair> = Vec::new();

    for &(birth_idx, death_idx) in &finite_pairs {
        let b = simplices[birth_idx].filtration;
        let d = simplices[death_idx].filtration;
        let dim = simplices[birth_idx].dim;
        // Discard zero-persistence pairs (b == d).
        if dim <= 1 && b < d {
            result.push(PersistencePair { dim, birth: b, death: d });
        }
    }

    // Essential classes: column is empty after reduction AND not the birth
    // index of any finite pair.
    for j in 0..m {
        if columns[j].is_empty() && !birth_set.contains(&j) && !death_set.contains(&j) {
            let dim = simplices[j].dim;
            if dim <= 1 {
                result.push(PersistencePair {
                    dim,
                    birth: simplices[j].filtration,
                    death: f64::INFINITY,
                });
            }
        }
    }

    result.sort_by(|a, b| {
        a.dim
            .cmp(&b.dim)
            .then_with(|| a.birth.partial_cmp(&b.birth).unwrap_or(std::cmp::Ordering::Equal))
    });

    Ok(result)
}

// ─── Public entry-point: choose algorithm automatically ──────────────────────

/// The maximum number of points for which the full H0+H1 boundary-matrix
/// reduction is attempted.  Larger point clouds fall back to H0-only.
pub(crate) const MAX_POINTS_FULL_HOMOLOGY: usize = 50;

/// Compute persistence pairs for a point cloud.
///
/// * If `n_points ≤ max_h1_points` (default [`MAX_POINTS_FULL_HOMOLOGY`]),
///   both H0 and H1 pairs are computed via boundary-matrix reduction.
/// * Otherwise only H0 pairs are computed via the faster union-find algorithm.
///
/// # Arguments
/// * `points`       - Point cloud.
/// * `max_scale`    - Optional filtration cutoff.
/// * `max_h1_points`- Threshold above which H1 computation is skipped.
pub(crate) fn compute_persistence(
    points: &[Vec<f64>],
    max_scale: Option<f64>,
    max_h1_points: usize,
) -> Result<Vec<PersistencePair>, String> {
    if points.is_empty() {
        return Err("Point cloud cannot be empty".to_string());
    }
    if points.len() <= max_h1_points {
        compute_full_persistence(points, max_scale)
    } else {
        // H0 only — filter by max_scale
        let mut pairs = compute_h0_persistence(points);
        if let Some(ms) = max_scale {
            pairs.retain(|p| p.death <= ms || p.death.is_infinite());
        }
        Ok(pairs)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_clusters() -> Vec<Vec<f64>> {
        // Two tight clusters far apart.
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![10.0, 0.1],
        ]
    }

    fn triangle_points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.866],
        ]
    }

    // ── union-find H0 ───────────────────────────────────────────────────────

    #[test]
    fn test_h0_single_point() {
        let pts = vec![vec![0.0, 0.0]];
        let pairs = compute_h0_persistence(&pts);
        // One essential H0 class.
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].death.is_infinite());
        assert_eq!(pairs[0].dim, 0);
    }

    #[test]
    fn test_h0_two_clusters() {
        let pts = two_clusters();
        let pairs = compute_h0_persistence(&pts);
        // 6 points → 5 finite pairs + 1 essential.
        let finite: Vec<_> = pairs.iter().filter(|p| p.death.is_finite()).collect();
        let essential: Vec<_> = pairs.iter().filter(|p| p.death.is_infinite()).collect();
        assert_eq!(finite.len(), 5);
        assert_eq!(essential.len(), 1);
        // The five shortest edges are all within a cluster (< 0.15).
        // The long inter-cluster edge merges the two clusters.
        let max_finite = finite.iter().map(|p| p.death).fold(0.0_f64, f64::max);
        assert!(max_finite > 9.0, "inter-cluster merge must exceed 9.0");
    }

    #[test]
    fn test_h0_all_births_zero() {
        let pts = two_clusters();
        let pairs = compute_h0_persistence(&pts);
        for p in &pairs {
            assert_eq!(p.birth, 0.0);
            assert_eq!(p.dim, 0);
        }
    }

    // ── full boundary-matrix H0+H1 ──────────────────────────────────────────

    #[test]
    fn test_full_triangle_h0() {
        // Three vertices of an equilateral-ish triangle.
        // H0: 2 finite pairs (two vertices die when edges connect them).
        // H1: 0 until the triangle closes → 1 H1 pair born at the longest edge,
        //     dead at the triangle filtration value (= same longest edge → skip).
        // But wait: for an equilateral triangle all edges equal d.
        // Use a scalene triangle so edge lengths differ.
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 2.0],
        ];
        let pairs = compute_full_persistence(&pts, None).unwrap();
        let h0: Vec<_> = pairs.iter().filter(|p| p.dim == 0).collect();
        let finite_h0: Vec<_> = h0.iter().filter(|p| p.death.is_finite()).collect();
        let essential_h0: Vec<_> = h0.iter().filter(|p| p.death.is_infinite()).collect();
        assert_eq!(finite_h0.len(), 2);
        assert_eq!(essential_h0.len(), 1);
    }

    #[test]
    fn test_full_h1_cycle() {
        // A square: 4 corners.  The loop around the square should give 1 H1 pair.
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ];
        let pairs = compute_full_persistence(&pts, None).unwrap();
        let h1: Vec<_> = pairs.iter().filter(|p| p.dim == 1).collect();
        // The 4 outer edges form a cycle; the two diagonals fill it.
        // Expected: exactly 1 H1 pair.
        assert_eq!(h1.len(), 1);
        // The loop is born when the 4th outer edge closes it.
        // It dies when a diagonal (triangle) fills the cycle.
        let pair = h1[0];
        assert!(pair.birth <= pair.death);
    }

    #[test]
    fn test_z2_add_basic() {
        let a = vec![0, 2, 4];
        let b = vec![1, 2, 3];
        let result = z2_add(&a, &b);
        assert_eq!(result, vec![0, 1, 3, 4]);
    }

    #[test]
    fn test_z2_add_cancel() {
        let a = vec![0, 1];
        let b = vec![0, 1];
        let result = z2_add(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_persistence_dispatch_h0_only() {
        // Point cloud larger than MAX_POINTS_FULL_HOMOLOGY → H0 only.
        let pts: Vec<Vec<f64>> = (0..60).map(|i| vec![i as f64, 0.0]).collect();
        let pairs = compute_persistence(&pts, None, MAX_POINTS_FULL_HOMOLOGY).unwrap();
        let h1: Vec<_> = pairs.iter().filter(|p| p.dim == 1).collect();
        assert!(h1.is_empty(), "H1 should not be computed for >50 points");
    }

    #[test]
    fn test_compute_persistence_full() {
        let pts = triangle_points();
        let pairs = compute_persistence(&pts, None, MAX_POINTS_FULL_HOMOLOGY).unwrap();
        // At minimum we have H0 pairs.
        let h0: Vec<_> = pairs.iter().filter(|p| p.dim == 0).collect();
        assert!(!h0.is_empty());
    }

    #[test]
    fn test_max_scale_filters_edges() {
        let pts = two_clusters();
        // max_scale = 1.0 should cut the inter-cluster edges.
        let pairs = compute_full_persistence(&pts, Some(1.0)).unwrap();
        // Within each cluster edges are < 0.15, so 6-1=5... actually
        // max_scale=1 includes the within-cluster edges but not the inter-cluster edge.
        // Result: two components → one finite H0 pair (within each cluster)
        // and two essential H0 classes (the two surviving cluster roots).
        let finite: Vec<_> = pairs.iter().filter(|p| p.dim == 0 && p.death.is_finite()).collect();
        let essential: Vec<_> = pairs.iter().filter(|p| p.dim == 0 && p.death.is_infinite()).collect();
        assert_eq!(finite.len(), 4, "4 finite within-cluster merges");
        assert_eq!(essential.len(), 2, "2 essential: two separate clusters");
    }
}
