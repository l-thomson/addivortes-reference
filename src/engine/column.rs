//! Column semantics: the per-column geometry the core consults instead of
//! branching on `Metric`. It owns the three behaviours the scaler and the
//! boundary validator decide per column (valid input domain, fit-time scaling
//! range, and per-value scaling), so a new geometry supplies its own without
//! editing `scale.rs` / `data.rs`.
//!
//! The built-in `Metric` variants map (via [`semantics_for`]) to built-in impls
//! whose arithmetic is bit-identical to the pre-seam `match metric` code; the
//! golden chain freezes this. `Metric` stays the built-in, `Copy` column
//! vocabulary; it merely maps to a semantics.
//!
//! This is Role A (column semantics, the engine's concern). Role B, distance
//! grouping, stays with the distance approach: `distance/per_column.rs` reads
//! `Metric::Spherical` to choose the great-circle group, and is untouched.

use std::sync::Arc;

use crate::engine::data::Metric;
use crate::extensions::coord::{CoordinateDistribution, EuclideanNormal, WrappedNormal};

/// How the core treats one column's values: valid input domain, fit-time scaling
/// range, and per-value scaling. Consulted by the scaler ([`crate::engine::scaler`]) and
/// the boundary validator ([`crate::engine::data`]) so neither branches on a concrete
/// geometry. Object-safe by construction.
pub(crate) trait ColumnSemantics: std::fmt::Debug {
    /// Whether one raw input value lies in this column's valid domain. Called
    /// after the non-finite scan, so `value` is finite. Real/categorical columns
    /// are unbounded (always `true`); an angle column is `[−π, π]`.
    fn in_domain(&self, value: f64) -> bool;

    /// The fit-time scaling range `(min, max)` for an encoded column of this
    /// geometry, or `None` for a degenerate column with no spread, on which the
    /// caller raises `DegenerateFeature` with the caller-visible index. An angle
    /// column has the fixed range `[−π, π]` and is never degenerate.
    fn fit_range(&self, column: &[f64]) -> Option<(f64, f64)>;

    /// Scale one encoded value given its fitted `(min, max)`. A real column maps
    /// its range affinely onto `[−0.5, 0.5]`; an angle column is identity (its
    /// values already sit on `[−π, π]`).
    fn scale(&self, value: f64, min: f64, max: f64) -> f64;

    /// The default centre-coordinate law for a column of this
    /// geometry, used when the caller supplies no explicit `with_coords`. Real
    /// columns pair with `EuclideanNormal`, angle columns with
    /// `WrappedNormal`.
    fn default_coord_law(&self, sigma_c: f64) -> Arc<dyn CoordinateDistribution>;
}

/// Real-valued column: the `Euclidean` metric (and every one-hot column a
/// `Categorical` metric expands to), min–max scaled to `[−0.5, 0.5]`, unbounded
/// domain, constant columns reported degenerate.
#[derive(Debug)]
struct RealColumn;

impl ColumnSemantics for RealColumn {
    fn in_domain(&self, _value: f64) -> bool {
        true
    }

    fn fit_range(&self, column: &[f64]) -> Option<(f64, f64)> {
        // Reproduces `scale::min_max` exactly (same fold, same init sentinels),
        // then the pass-3 `lo == hi` degeneracy test.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in column {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if lo == hi { None } else { Some((lo, hi)) }
    }

    fn scale(&self, value: f64, min: f64, max: f64) -> f64 {
        (value - min) / (max - min) - 0.5
    }

    fn default_coord_law(&self, sigma_c: f64) -> Arc<dyn CoordinateDistribution> {
        Arc::new(EuclideanNormal::new(sigma_c))
    }
}

/// Angle column, the `Spherical` metric: fixed range `[−π, π]`, identity
/// scaling (values already lie in that range), domain-checked at the boundary.
#[derive(Debug)]
struct AngleColumn;

impl ColumnSemantics for AngleColumn {
    fn in_domain(&self, value: f64) -> bool {
        (-std::f64::consts::PI..=std::f64::consts::PI).contains(&value)
    }

    fn fit_range(&self, _column: &[f64]) -> Option<(f64, f64)> {
        Some((-std::f64::consts::PI, std::f64::consts::PI))
    }

    fn scale(&self, value: f64, _min: f64, _max: f64) -> f64 {
        value
    }

    fn default_coord_law(&self, sigma_c: f64) -> Arc<dyn CoordinateDistribution> {
        Arc::new(WrappedNormal::new(sigma_c))
    }
}

/// Caller-prepared column, the `Prepared` metric: unbounded domain, identity
/// scaling, never degenerate. The stored fit range is the column's observed
/// (min, max), informative only, never applied (`scale` ignores it).
#[derive(Debug)]
struct PreparedColumn;

impl ColumnSemantics for PreparedColumn {
    fn in_domain(&self, _value: f64) -> bool {
        true
    }

    fn fit_range(&self, column: &[f64]) -> Option<(f64, f64)> {
        // Observed range, kept for the scaler's accessors; a constant prepared
        // column is legitimate (the caller owns its preparation), so this is
        // never `None`.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in column {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        Some((lo, hi))
    }

    fn scale(&self, value: f64, _min: f64, _max: f64) -> f64 {
        value
    }

    fn default_coord_law(&self, sigma_c: f64) -> Arc<dyn CoordinateDistribution> {
        Arc::new(EuclideanNormal::new(sigma_c))
    }
}

static REAL: RealColumn = RealColumn;
static ANGLE: AngleColumn = AngleColumn;
static PREPARED: PreparedColumn = PreparedColumn;

/// The column semantics for a built-in [`Metric`].
///
/// `Categorical` maps to the real semantics: a categorical value has no
/// domain constraint (like a real value), and each column it one-hot expands to
/// is real-scaled. So a categorical metric only ever reaches `in_domain` (during
/// boundary validation, where "unbounded" is correct), never `fit_range`/`scale`
/// directly, because the expansion into real columns happens first.
pub(crate) fn semantics_for(metric: &Metric) -> &'static dyn ColumnSemantics {
    match metric {
        Metric::Spherical => &ANGLE,
        Metric::Euclidean | Metric::Categorical => &REAL,
        Metric::Prepared => &PREPARED,
    }
}
