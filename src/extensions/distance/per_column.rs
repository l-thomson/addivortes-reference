//! The compound per-column metric: mixes assignment geometries column by
//! column (the fit-time default, driven by `Vec<Metric>`), optionally
//! composed with researcher-supplied per-column groups
//! ([`ColumnMetrics::with_group`]).

use std::sync::Arc;

use crate::engine::data::Metric;
use crate::engine::error::{AddiVortesError, Result};
use crate::extensions::distance::PairwiseDistance;
use crate::extensions::distance::spherical::great_circle;

/// The built-in compound metric (the fit-time default): squared-distance keys
/// from one Euclidean group plus at most one joint spherical great-circle
/// group.
///
/// - **Euclidean group**: every non-spherical active dimension contributes its
///   squared difference (Σ diff²; squared distance is a valid monotone key),
///   the standalone [`Euclidean`](crate::extensions::distance::Euclidean) geometry.
/// - **Spherical group** (the paper's model is Euclidean and gives no
///   spherical equation): all
///   spherical columns form one group, interpreted as the hyperspherical
///   coordinates of a point on the unit s-sphere in ℝ^{s+1}, the last
///   spherical column being the azimuthal angle; the group contributes the
///   squared great-circle arc length, commensurate with the squared
///   Euclidean group: the standalone
///   [`Spherical`](crate::extensions::distance::Spherical) geometry (see the
///   `great_circle` helper there for the embedding and derivation).
///
/// Each group participates only when at least one of its columns is active (an
/// inactive-only group is identical on both rows by the synthesis contract and
/// would contribute exactly zero).
///
/// Researcher-supplied geometries snap in per column via
/// [`with_group`](ColumnMetrics::with_group): a group's columns leave the
/// built-in treatment and its key joins the compound sum like the spherical
/// group's does.
#[derive(Debug, Clone)]
pub struct ColumnMetrics {
    metrics: Vec<Metric>,
    /// Cached ascending indices of the spherical columns (the group);
    /// custom-owned columns are excluded.
    spherical: Vec<usize>,
    /// Researcher-supplied per-column groups, in insertion order.
    groups: Vec<MetricGroup>,
    /// Column → index into `groups` (`None` = built-in treatment).
    owner: Vec<Option<usize>>,
}

/// One researcher-supplied group: the ascending columns it owns and the
/// geometry that scores them.
#[derive(Debug, Clone)]
struct MetricGroup {
    columns: Vec<usize>,
    geometry: Arc<dyn PairwiseDistance>,
}

impl ColumnMetrics {
    /// Build from per-encoded-column metrics (the scaler's expanded list;
    /// any `Categorical` entry is treated as Euclidean, which is what one-hot
    /// groups are post-encoding).
    pub fn new(metrics: Vec<Metric>) -> Self {
        let spherical = metrics
            .iter()
            .enumerate()
            .filter(|(_, m)| **m == Metric::Spherical)
            .map(|(i, _)| i)
            .collect();
        let owner = vec![None; metrics.len()];
        Self {
            metrics,
            spherical,
            groups: Vec::new(),
            owner,
        }
    }

    /// Hand a subset of encoded columns to a researcher-supplied geometry,
    /// composed with the built-in treatment of the remaining columns: the
    /// group's key joins the compound sum exactly like the built-in spherical
    /// group's, and its columns leave the built-in Euclidean/spherical
    /// handling entirely.
    ///
    /// The geometry receives full p-length rows and the active dims that
    /// belong to this group, and must obey the [`PairwiseDistance`] contract;
    /// its key is added to the other groups' squared-scale contributions, so
    /// return a squared-distance-scale key to stay commensurate (as the
    /// built-in groups do).
    ///
    /// Errors with [`AddiVortesError::InvalidMetricGroup`] when `columns` is
    /// empty, out of range, duplicated, or already owned by an earlier group.
    pub fn with_group(
        mut self,
        mut columns: Vec<usize>,
        geometry: Arc<dyn PairwiseDistance>,
    ) -> Result<Self> {
        let bad = |reason: String| Err(AddiVortesError::InvalidMetricGroup { reason });
        if columns.is_empty() {
            return bad("a group must own at least one column".into());
        }
        columns.sort_unstable();
        if let Some(pair) = columns.windows(2).find(|pair| pair[0] == pair[1]) {
            return bad(format!("column {} appears twice", pair[0]));
        }
        for &col in &columns {
            if col >= self.metrics.len() {
                return bad(format!(
                    "column {col} is out of range for {p} encoded columns",
                    p = self.metrics.len()
                ));
            }
            if self.owner[col].is_some() {
                return bad(format!("column {col} already belongs to another group"));
            }
        }
        let index = self.groups.len();
        for &col in &columns {
            self.owner[col] = Some(index);
        }
        self.spherical.retain(|col| self.owner[*col].is_none());
        self.groups.push(MetricGroup { columns, geometry });
        Ok(self)
    }
}

/// Value equality on the metric list and each group's columns; group
/// geometries compare by pointer identity (`Arc::ptr_eq`), the same
/// precedent as the config's Arc-held extension points.
impl PartialEq for ColumnMetrics {
    fn eq(&self, other: &Self) -> bool {
        // `spherical` and `owner` are derived from `metrics` + `groups`.
        self.metrics == other.metrics
            && self.groups.len() == other.groups.len()
            && self
                .groups
                .iter()
                .zip(&other.groups)
                .all(|(a, b)| a.columns == b.columns && Arc::ptr_eq(&a.geometry, &b.geometry))
    }
}

impl PairwiseDistance for ColumnMetrics {
    fn all_euclidean(&self) -> bool {
        self.spherical.is_empty() && self.groups.is_empty()
    }

    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        let mut key = 0.0_f64;
        let mut spherical_active = false;
        if self.groups.is_empty() {
            // The fit-time default's hot path: no custom groups, no allocation.
            for &dim in active_dims {
                if self.metrics.get(dim) == Some(&Metric::Spherical) {
                    spherical_active = true;
                } else {
                    let diff = x_row[dim] - centre_row[dim];
                    key += diff * diff;
                }
            }
        } else {
            for &dim in active_dims {
                if self.owner.get(dim).copied().flatten().is_some() {
                    continue; // scored by its group below
                }
                if self.metrics.get(dim) == Some(&Metric::Spherical) {
                    spherical_active = true;
                } else {
                    let diff = x_row[dim] - centre_row[dim];
                    key += diff * diff;
                }
            }
            // One buffer reused across groups (researcher path only; the
            // default hot path above stays allocation-free).
            let mut group_active: Vec<usize> = Vec::new();
            for (index, group) in self.groups.iter().enumerate() {
                group_active.clear();
                group_active.extend(
                    active_dims
                        .iter()
                        .copied()
                        .filter(|dim| self.owner.get(*dim).copied().flatten() == Some(index)),
                );
                if !group_active.is_empty() {
                    key += group.geometry.distance(x_row, centre_row, &group_active);
                }
            }
        }
        if spherical_active {
            let gc = great_circle(x_row, centre_row, &self.spherical);
            key += gc * gc;
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;
    use crate::test_support::assert_abs_eq;

    // ---- hand-derived closed-form oracles (1e-12, single-target tightness) ----

    #[test]
    fn euclidean_group_is_sum_of_squared_active_diffs() {
        let cm = ColumnMetrics::new(vec![Metric::Euclidean; 3]);
        let x = [0.1, 0.5, -0.2];
        let c = [0.4, 0.5, 0.3]; // dim 1 inactive ⇒ synthesised equal anyway
        // Hand-derived: (0.1−0.4)² + (−0.2−0.3)² = 0.09 + 0.25 = 0.34.
        assert_abs_eq(cm.distance(&x, &c, &[0, 2]), 0.34, 1e-12);
        // Active set restricts the sum: only dim 0 ⇒ 0.09.
        assert_abs_eq(cm.distance(&x, &c, &[0]), 0.09, 1e-12);
    }

    #[test]
    fn single_spherical_column_is_wrapped_angle_difference() {
        // s = 1 reduces to |θ − φ| wrapped: π − 0.1 vs −π + 0.1 are 0.2 apart
        // across the wrap (NOT 2π − 0.2). Hand-derived key = 0.2² = 0.04.
        let cm = ColumnMetrics::new(vec![Metric::Spherical]);
        let x = [PI - 0.1];
        let c = [-PI + 0.1];
        assert_abs_eq(cm.distance(&x, &c, &[0]), 0.04, 1e-12);
        // Antipodal points: acos(cos π) = π ⇒ key = π².
        assert_abs_eq(cm.distance(&[0.0], &[PI], &[0]), PI * PI, 1e-12);
    }

    #[test]
    fn joint_spherical_group_matches_hand_embedding() {
        // Two spherical columns (θ polar-like, φ azimuthal):
        // A = (π/2, 0)  → (cos π/2, sin π/2·cos 0, sin π/2·sin 0) = (0, 1, 0)
        // B = (π/2, π/2) → (0, 0, 1); ⟨A,B⟩ = 0 ⇒ great circle = π/2.
        let cm = ColumnMetrics::new(vec![Metric::Spherical, Metric::Spherical]);
        let x = [FRAC_PI_2, 0.0];
        let c = [FRAC_PI_2, FRAC_PI_2];
        assert_abs_eq(cm.distance(&x, &c, &[0, 1]), FRAC_PI_2 * FRAC_PI_2, 1e-12);
    }

    #[test]
    fn mixed_metric_key_adds_squared_groups() {
        // Metrics [E, E, S, S]; active {0, 2, 3}.
        // Euclidean part: (0.1 − 0.4)² = 0.09.
        // Spherical part: A = (π/2, 0) vs B = (π/2, π/2) ⇒ (π/2)² as above.
        // Hand-derived total: 0.09 + π²/4.
        let cm = ColumnMetrics::new(vec![
            Metric::Euclidean,
            Metric::Euclidean,
            Metric::Spherical,
            Metric::Spherical,
        ]);
        let x = [0.1, 0.9, FRAC_PI_2, 0.0];
        let c = [0.4, 0.9, FRAC_PI_2, FRAC_PI_2];
        assert_abs_eq(
            cm.distance(&x, &c, &[0, 2, 3]),
            0.09 + FRAC_PI_2 * FRAC_PI_2,
            1e-12,
        );
    }

    #[test]
    fn near_boundary_dot_products_are_clamped() {
        // Identical spherical coordinates: dot may land a few ULP above 1;
        // clamped, acos(1) = 0, contribution exactly 0.
        let cm = ColumnMetrics::new(vec![Metric::Spherical, Metric::Spherical]);
        let x = [2.0, -3.0];
        assert_abs_eq(cm.distance(&x, &x, &[0, 1]), 0.0, 0.0);
    }

    // ---- researcher-supplied per-column groups ----

    /// Manhattan on its owned columns (the researcher geometry of the tests).
    #[derive(Debug)]
    struct ManhattanGroup;
    impl PairwiseDistance for ManhattanGroup {
        fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
            active_dims
                .iter()
                .map(|&d| (x_row[d] - centre_row[d]).abs())
                .sum()
        }
    }

    #[test]
    fn custom_group_composes_with_the_euclidean_group() {
        // Metrics [E, E, E]; Manhattan owns {1, 2}. Active {0, 1, 2}:
        // hand-derived key = (0.1−0.4)² + |0.5−0.2| + |−0.2−0.3|
        //                  = 0.09 + 0.3 + 0.5 = 0.89.
        let cm = ColumnMetrics::new(vec![Metric::Euclidean; 3])
            .with_group(vec![1, 2], Arc::new(ManhattanGroup))
            .unwrap();
        let x = [0.1, 0.5, -0.2];
        let c = [0.4, 0.2, 0.3];
        assert_abs_eq(cm.distance(&x, &c, &[0, 1, 2]), 0.89, 1e-12);
        // Group inactive (only dim 0 active): pure Euclidean part.
        assert_abs_eq(cm.distance(&x, &c, &[0]), 0.09, 1e-12);
        // Only one group column active: the group sees just that dim.
        assert_abs_eq(cm.distance(&x, &c, &[0, 2]), 0.09 + 0.5, 1e-12);
        // A compound with groups must not claim the all-Euclidean fast path.
        assert!(!cm.all_euclidean());
    }

    #[test]
    fn custom_group_removes_its_columns_from_the_spherical_group() {
        // Metrics [S, S]; the group takes column 1, so the built-in spherical
        // group shrinks to {0}: wrapped |θ−φ| on column 0 squared, plus the
        // group's |x₁−c₁|. Hand-derived: 0.2² + 0.4 = 0.44.
        let cm = ColumnMetrics::new(vec![Metric::Spherical, Metric::Spherical])
            .with_group(vec![1], Arc::new(ManhattanGroup))
            .unwrap();
        let x = [PI - 0.1, 1.0];
        let c = [-PI + 0.1, 0.6];
        assert_abs_eq(cm.distance(&x, &c, &[0, 1]), 0.04 + 0.4, 1e-12);
    }

    #[test]
    fn with_group_rejects_misconfigured_groups() {
        use crate::engine::error::AddiVortesError;
        let base = || ColumnMetrics::new(vec![Metric::Euclidean; 3]);
        let geometry = || -> Arc<dyn PairwiseDistance> { Arc::new(ManhattanGroup) };
        for (columns, fragment) in [
            (vec![], "at least one column"),
            (vec![0, 0], "appears twice"),
            (vec![3], "out of range"),
        ] {
            let err = base().with_group(columns, geometry()).unwrap_err();
            match err {
                AddiVortesError::InvalidMetricGroup { reason } => {
                    assert!(reason.contains(fragment), "{reason}");
                }
                other => panic!("expected InvalidMetricGroup, got {other:?}"),
            }
        }
        // Overlap with an earlier group.
        let err = base()
            .with_group(vec![0, 1], geometry())
            .unwrap()
            .with_group(vec![1, 2], geometry())
            .unwrap_err();
        assert!(matches!(
            err,
            AddiVortesError::InvalidMetricGroup { reason } if reason.contains("another group")
        ));
    }

    #[test]
    fn group_equality_is_pointer_identity_on_the_geometry() {
        let geometry: Arc<dyn PairwiseDistance> = Arc::new(ManhattanGroup);
        let a = ColumnMetrics::new(vec![Metric::Euclidean; 2])
            .with_group(vec![1], Arc::clone(&geometry))
            .unwrap();
        let b = ColumnMetrics::new(vec![Metric::Euclidean; 2])
            .with_group(vec![1], Arc::clone(&geometry))
            .unwrap();
        assert_eq!(a, b); // same Arc
        let c = ColumnMetrics::new(vec![Metric::Euclidean; 2])
            .with_group(vec![1], Arc::new(ManhattanGroup))
            .unwrap();
        assert_ne!(a, c); // equal value, different Arc
        assert_ne!(a, ColumnMetrics::new(vec![Metric::Euclidean; 2]));
    }

    #[test]
    fn inactive_spherical_group_contributes_nothing() {
        // Spherical columns exist but no spherical dim is active: only the
        // Euclidean part counts (the group is identical under synthesis).
        let cm = ColumnMetrics::new(vec![Metric::Euclidean, Metric::Spherical]);
        let x = [0.1, 1.0];
        let c = [0.4, 1.0];
        assert_abs_eq(cm.distance(&x, &c, &[0]), 0.09, 1e-12);
    }
}
