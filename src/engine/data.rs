//! The input matrix (`Data`), per-column metrics, typed warnings, and the shared
//! boundary-validation helpers used by both fit and predict.
//!
//! Callers pass raw, unscaled data; scaling and encoding are owned by the
//! library. Everything here works in the caller's coordinate system.

use crate::engine::column::semantics_for;
use crate::engine::error::{AddiVortesError, Result};

/// How a caller-visible (pre-encoding) column is treated by scaling, validation,
/// and the built-in distance.
///
/// One entry per raw column; the one-hot encoder expands the list alongside
/// the columns it expands. This enum is the scaling/validation contract; custom
/// distance geometry is the `PairwiseDistance` extension point instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Metric {
    /// Ordinary real-valued column: min-max scaled to [−0.5, 0.5], Euclidean
    /// distance group, degenerate (constant) columns rejected at fit.
    Euclidean,
    /// Angle in radians on [−π, π]: bypasses scaling, validated against that
    /// domain, joins the great-circle distance group.
    Spherical,
    /// Categorical level codes: one-hot encoded by the library; unseen levels at
    /// predict error with `UnseenCategory`.
    Categorical,
    /// Caller-prepared column: the bring-your-own-data-type escape valve
    /// (the walkthrough is on the `distance` module docs). The library validates finiteness only, imposes no
    /// domain, and never rescales (identity pass-through into the sampler's
    /// scaled space). The caller owns the column's preparation and must map it
    /// onto a range commensurate with the other scaled columns ([−0.5, 0.5]);
    /// constant columns are permitted (the caller's concern). The built-in
    /// defaults treat it as Euclidean: squared-difference assignment geometry
    /// and the `EuclideanNormal` centre law, typically overridden together via
    /// `with_distance`/`with_coords`.
    Prepared,
}

/// A non-fatal condition noticed at fit time, surfaced through
/// `warnings() -> &[Warning]` on the fitted model (never stdout, never an error).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Warning {
    /// More features than observations (p > n): the fit proceeds, but the model
    /// is weakly identified. Counts are in pre-encoding (caller) columns.
    MoreFeaturesThanObservations {
        /// Number of caller-visible (pre-encoding) features.
        p: usize,
        /// Number of observations.
        n: usize,
    },
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Warning::MoreFeaturesThanObservations { p, n } => {
                write!(f, "more features ({p}) than observations ({n})")
            }
        }
    }
}

/// A row-major n×p `f64` matrix: observations as rows, features as columns, in
/// the caller's raw, unscaled coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Data {
    values: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

impl Data {
    /// Build a matrix from a row-major buffer and its shape.
    ///
    /// Errors with [`AddiVortesError::InvalidDataShape`] when
    /// `values.len() != n_rows * n_cols`. Values are not inspected here;
    /// finiteness and domain checks happen at the fit/predict boundary.
    pub fn new(values: Vec<f64>, n_rows: usize, n_cols: usize) -> Result<Self> {
        let expected =
            n_rows
                .checked_mul(n_cols)
                .ok_or_else(|| AddiVortesError::InvalidDataShape {
                    reason: format!("{n_rows} by {n_cols} matrix overflows usize"),
                })?;
        if values.len() != expected {
            return Err(AddiVortesError::InvalidDataShape {
                reason: format!(
                    "{len} values cannot form a {n_rows} by {n_cols} matrix",
                    len = values.len()
                ),
            });
        }
        Ok(Self {
            values,
            n_rows,
            n_cols,
        })
    }

    /// Build a matrix from one slice per row.
    ///
    /// The first row fixes the column count; errors with
    /// [`AddiVortesError::InvalidDataShape`] on a ragged row. An empty `rows`
    /// gives the 0×0 matrix (valid at predict; fit requires n ≥ 2).
    pub fn from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self> {
        let n_rows = rows.len();
        let n_cols = rows.first().map_or(0, |r| r.as_ref().len());
        let mut values = Vec::with_capacity(n_rows * n_cols);
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_ref();
            if row.len() != n_cols {
                return Err(AddiVortesError::InvalidDataShape {
                    reason: format!(
                        "row {i} has {found} values but row 0 has {n_cols}",
                        found = row.len()
                    ),
                });
            }
            values.extend_from_slice(row);
        }
        Ok(Self {
            values,
            n_rows,
            n_cols,
        })
    }

    /// Number of observations (rows).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of features (columns).
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Observation `i` as a slice of its `n_cols` feature values (raw caller
    /// coordinates).
    ///
    /// # Panics
    ///
    /// Panics if `i >= n_rows()`. An out-of-range index is a programming error,
    /// like slice indexing, not a data-validation condition.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.values[i * self.n_cols..(i + 1) * self.n_cols]
    }

    /// The whole row-major buffer, zero-copy (raw caller coordinates;
    /// `values()[r * n_cols() + c]` is row `r`, column `c`).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Decompose into `(values, n_rows, n_cols)` without copying (row-major,
    /// raw caller coordinates).
    pub fn into_parts(self) -> (Vec<f64>, usize, usize) {
        (self.values, self.n_rows, self.n_cols)
    }
}

/// Shared fit-boundary validation, in the canonical order:
/// `RowCountMismatch` → `InsufficientObservations` (n ≥ 2) → `NonFiniteResponse`
/// (y scanned first) / `NonFiniteFeature` (row-major, first offence) →
/// `DegenerateResponse` / `DegenerateFeature` (Euclidean columns only) →
/// `SphericalOutOfDomain` ([−π, π]) → `MetricLengthMismatch` →
/// `InvalidHyperparameter` for ω ≥ p (only when p ≥ 2; the stated
/// dimension prior is degenerate there, so it is a hard error, never a
/// silent clamp).
///
/// The two metric-driven scans run before the metric-length check (canonical
/// order); they zip columns with metrics, so a wrong-length list cannot
/// mis-index; it is reported by the later `MetricLengthMismatch`.
pub(crate) fn validate_fit(x: &Data, y: &[f64], metrics: &[Metric], omega: f64) -> Result<()> {
    if y.len() != x.n_rows {
        return Err(AddiVortesError::RowCountMismatch {
            y_len: y.len(),
            x_rows: x.n_rows,
        });
    }
    if x.n_rows < 2 {
        return Err(AddiVortesError::InsufficientObservations {
            found: x.n_rows,
            required: 2,
        });
    }
    if let Some(row) = y.iter().position(|v| !v.is_finite()) {
        return Err(AddiVortesError::NonFiniteResponse { row });
    }
    scan_finite(x)?;

    // Degenerate response: n ≥ 2 held above, so y is non-empty.
    if y.iter().all(|&v| v == y[0]) {
        return Err(AddiVortesError::DegenerateResponse {});
    }
    // Degenerate features: constant Euclidean columns only (categorical columns
    // are the encoder's concern, spherical columns have no scale to collapse).
    for (col, metric) in metrics.iter().enumerate().take(x.n_cols) {
        if *metric == Metric::Euclidean {
            let first = x.values[col];
            let constant = (1..x.n_rows).all(|r| x.values[r * x.n_cols + col] == first);
            if constant {
                return Err(AddiVortesError::DegenerateFeature { col });
            }
        }
    }
    scan_domains(x, metrics)?;

    if metrics.len() != x.n_cols {
        return Err(AddiVortesError::MetricLengthMismatch {
            metric_len: metrics.len(),
            x_cols: x.n_cols,
        });
    }
    // ω ≥ p makes the stated dimension prior degenerate (a log(0) in the
    // Binomial pricing), so it is a hard error, never a silent clamp. Only
    // meaningful when p ≥ 2 (at p = 1 the dimension count is degenerate anyway).
    // A NaN ω is rejected explicitly.
    let p = x.n_cols;
    if p >= 2 && (omega.is_nan() || omega >= p as f64) {
        return Err(AddiVortesError::InvalidHyperparameter {
            name: "omega".into(),
            reason: format!("must be less than the number of features p = {p}, got {omega}"),
        });
    }
    Ok(())
}

/// Shared predict-boundary validation, in the canonical order:
/// `FeatureCountMismatch` → `NonFiniteFeature` (row-major, first offence) →
/// `SphericalOutOfDomain`. An empty matrix (0 rows) is valid at predict.
///
/// `UnseenCategory` is raised by the encoder when it applies the fitted
/// encoding, and quantile-probability checks live with `predict_quantiles`;
/// both are downstream of this helper, preserving the canonical order.
// Consumed by predict; fully exercised by the tests below.
#[allow(dead_code)]
pub(crate) fn validate_predict(x: &Data, expected_cols: usize, metrics: &[Metric]) -> Result<()> {
    if x.n_cols != expected_cols {
        return Err(AddiVortesError::FeatureCountMismatch {
            expected: expected_cols,
            found: x.n_cols,
        });
    }
    scan_finite(x)?;
    scan_domains(x, metrics)
}

/// Fit-time warnings (never errors, never stdout). Currently the
/// single p > n condition; counts are pre-encoding columns.
pub(crate) fn fit_warnings(x: &Data) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if x.n_cols > x.n_rows {
        warnings.push(Warning::MoreFeaturesThanObservations {
            p: x.n_cols,
            n: x.n_rows,
        });
    }
    warnings
}

/// Row-major scan for the first non-finite feature value. Deterministic first
/// offence: lowest row, then lowest column within it.
fn scan_finite(x: &Data) -> Result<()> {
    if let Some(offset) = x.values.iter().position(|v| !v.is_finite()) {
        return Err(AddiVortesError::NonFiniteFeature {
            row: offset / x.n_cols,
            col: offset % x.n_cols,
        });
    }
    Ok(())
}

/// Row-major scan of every column against its [`ColumnSemantics`] domain
/// ([`crate::engine::column`]). In the built-in set only the angle (spherical) geometry
/// is bounded, so the sole possible violation is a spherical value outside
/// [−π, π] → `SphericalOutOfDomain`; the scan is written generically so a future
/// bounded geometry validates through the same path. `metrics.get(col)` keeps it
/// index-safe even before the `MetricLengthMismatch` check has run.
fn scan_domains(x: &Data, metrics: &[Metric]) -> Result<()> {
    for row in 0..x.n_rows {
        for col in 0..x.n_cols {
            let Some(metric) = metrics.get(col) else {
                break;
            };
            let value = x.values[row * x.n_cols + col];
            if !semantics_for(metric).in_domain(value) {
                return Err(AddiVortesError::SphericalOutOfDomain { row, col, value });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    fn euclidean(p: usize) -> Vec<Metric> {
        vec![Metric::Euclidean; p]
    }

    /// A valid 3×2 fixture that passes every fit check with ω = 1.
    fn valid_fixture() -> (Data, Vec<f64>) {
        let x = Data::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        let y = vec![0.5, 1.5, 2.5];
        (x, y)
    }

    // ---- Data construction ----

    #[test]
    fn new_rejects_wrong_buffer_length() {
        let err = Data::new(vec![0.0; 6], 2, 4).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidDataShape {
                reason: "6 values cannot form a 2 by 4 matrix".into()
            }
        );
    }

    #[test]
    fn from_rows_rejects_ragged_rows() {
        let err = Data::from_rows(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidDataShape {
                reason: "row 1 has 1 values but row 0 has 2".into()
            }
        );
    }

    #[test]
    fn construction_and_accessors_round_trip() {
        let x = Data::from_rows(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();
        assert_eq!((x.n_rows(), x.n_cols()), (3, 2));
        assert_eq!(x.row(1), &[3.0, 4.0]);
        assert_eq!(x.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (values, n_rows, n_cols) = x.into_parts();
        assert_eq!(
            Data::new(values, n_rows, n_cols).unwrap().row(2),
            &[5.0, 6.0]
        );
    }

    #[test]
    fn empty_matrix_is_constructible() {
        let x = Data::from_rows::<&[f64]>(&[]).unwrap();
        assert_eq!((x.n_rows(), x.n_cols()), (0, 0));
        // …and valid at the predict boundary.
        assert!(validate_predict(&x, 0, &[]).is_ok());
    }

    // ---- fit validation: every reachable variant, in order ----

    #[test]
    fn fit_row_count_mismatch() {
        let (x, _) = valid_fixture();
        let err = validate_fit(&x, &[1.0, 2.0], &euclidean(2), 1.0).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::RowCountMismatch {
                y_len: 2,
                x_rows: 3
            }
        );
    }

    #[test]
    fn fit_insufficient_observations() {
        let x = Data::new(vec![1.0, 2.0], 1, 2).unwrap();
        let err = validate_fit(&x, &[1.0], &euclidean(2), 1.0).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InsufficientObservations {
                found: 1,
                required: 2
            }
        );
    }

    #[test]
    fn fit_non_finite_response_scanned_before_features() {
        // Both y and X contain a NaN: the response offence must win (canonical order).
        let x = Data::new(vec![1.0, f64::NAN, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        let err = validate_fit(&x, &[0.5, f64::NEG_INFINITY, 2.5], &euclidean(2), 1.0).unwrap_err();
        assert_eq!(err, AddiVortesError::NonFiniteResponse { row: 1 });
    }

    #[test]
    fn fit_non_finite_feature_first_offence_is_row_major() {
        // Offences at (1, 1) and (2, 0): row-major scan must report (1, 1), and
        // report it identically on every call (first-offence determinism).
        let x = Data::new(vec![1.0, 10.0, 2.0, f64::INFINITY, f64::NAN, 30.0], 3, 2).unwrap();
        let y = vec![0.5, 1.5, 2.5];
        for _ in 0..3 {
            let err = validate_fit(&x, &y, &euclidean(2), 1.0).unwrap_err();
            assert_eq!(err, AddiVortesError::NonFiniteFeature { row: 1, col: 1 });
        }
    }

    #[test]
    fn fit_degenerate_response() {
        let (x, _) = valid_fixture();
        let err = validate_fit(&x, &[7.0, 7.0, 7.0], &euclidean(2), 1.0).unwrap_err();
        assert_eq!(err, AddiVortesError::DegenerateResponse {});
    }

    #[test]
    fn fit_degenerate_feature_euclidean_only() {
        // Column 0 constant. As Euclidean it must error; declared Spherical
        // (constant angles are legitimate) it must pass that check.
        let x = Data::new(vec![1.0, 10.0, 1.0, 20.0, 1.0, 30.0], 3, 2).unwrap();
        let y = vec![0.5, 1.5, 2.5];
        let err = validate_fit(&x, &y, &euclidean(2), 1.0).unwrap_err();
        assert_eq!(err, AddiVortesError::DegenerateFeature { col: 0 });

        let metrics = vec![Metric::Spherical, Metric::Euclidean];
        assert!(validate_fit(&x, &y, &metrics, 1.0).is_ok());
    }

    #[test]
    fn fit_prepared_columns_skip_degeneracy_and_domain_checks() {
        // Column 0 constant and far outside [−π, π]: declared Prepared it
        // passes both the degeneracy and the domain scan (finiteness only);
        // declared Euclidean the same data is degenerate.
        let x = Data::new(vec![9.0, 10.0, 9.0, 20.0, 9.0, 30.0], 3, 2).unwrap();
        let y = vec![0.5, 1.5, 2.5];
        let metrics = vec![Metric::Prepared, Metric::Euclidean];
        assert!(validate_fit(&x, &y, &metrics, 1.0).is_ok());
        assert!(validate_predict(&x, 2, &metrics).is_ok());
        let err = validate_fit(&x, &y, &euclidean(2), 1.0).unwrap_err();
        assert_eq!(err, AddiVortesError::DegenerateFeature { col: 0 });
        // Non-finite values are still rejected: the one check Prepared keeps.
        let nan = Data::new(vec![f64::NAN, 10.0, 9.0, 20.0, 9.0, 30.0], 3, 2).unwrap();
        assert_eq!(
            validate_fit(&nan, &y, &metrics, 1.0).unwrap_err(),
            AddiVortesError::NonFiniteFeature { row: 0, col: 0 }
        );
    }

    #[test]
    fn fit_spherical_domain_boundaries() {
        // ±π are valid (closed interval); just outside is not.
        let metrics = vec![Metric::Spherical, Metric::Euclidean];
        let ok = Data::new(vec![PI, 10.0, -PI, 20.0, 0.0, 30.0], 3, 2).unwrap();
        let y = vec![0.5, 1.5, 2.5];
        assert!(validate_fit(&ok, &y, &metrics, 1.0).is_ok());

        let bad = Data::new(vec![PI, 10.0, 3.5, 20.0, 0.0, 30.0], 3, 2).unwrap();
        let err = validate_fit(&bad, &y, &metrics, 1.0).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::SphericalOutOfDomain {
                row: 1,
                col: 0,
                value: 3.5
            }
        );
    }

    #[test]
    fn fit_metric_length_mismatch() {
        let (x, y) = valid_fixture();
        let err = validate_fit(&x, &y, &euclidean(3), 1.0).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::MetricLengthMismatch {
                metric_len: 3,
                x_cols: 2
            }
        );
    }

    #[test]
    fn fit_omega_must_be_below_p() {
        let (x, y) = valid_fixture();
        // p = 2: ω = 2 errors (ω ≥ p, no silent clamp), ω just below passes.
        let err = validate_fit(&x, &y, &euclidean(2), 2.0).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidHyperparameter {
                name: "omega".into(),
                reason: "must be less than the number of features p = 2, got 2".into(),
            }
        );
        assert!(validate_fit(&x, &y, &euclidean(2), 1.999).is_ok());
        // NaN ω is never accepted.
        assert!(validate_fit(&x, &y, &euclidean(2), f64::NAN).is_err());
    }

    #[test]
    fn fit_omega_unchecked_when_p_is_one() {
        // p = 1: the ω < p check is skipped.
        let x = Data::new(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        let y = vec![0.5, 1.5, 2.5];
        assert!(validate_fit(&x, &y, &euclidean(1), 3.0).is_ok());
    }

    #[test]
    fn fit_ordering_metric_scans_precede_length_check() {
        // Canonical order: a spherical-domain offence is reported even though the
        // metric list is also the wrong length (scans zip, so no mis-indexing).
        let x = Data::new(vec![9.0, 10.0, 0.0, 20.0, 0.0, 30.0], 3, 2).unwrap();
        let y = vec![0.5, 1.5, 2.5];
        let metrics = vec![Metric::Spherical]; // wrong length AND col 0 out of domain
        let err = validate_fit(&x, &y, &metrics, 1.0).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::SphericalOutOfDomain {
                row: 0,
                col: 0,
                value: 9.0
            }
        );
    }

    // ---- predict validation ----

    #[test]
    fn predict_feature_count_mismatch() {
        let (x, _) = valid_fixture();
        let err = validate_predict(&x, 5, &euclidean(5)).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::FeatureCountMismatch {
                expected: 5,
                found: 2
            }
        );
    }

    #[test]
    fn predict_non_finite_feature_first_offence() {
        let x = Data::new(vec![1.0, 10.0, 2.0, f64::NAN, f64::NAN, 30.0], 3, 2).unwrap();
        let err = validate_predict(&x, 2, &euclidean(2)).unwrap_err();
        assert_eq!(err, AddiVortesError::NonFiniteFeature { row: 1, col: 1 });
    }

    #[test]
    fn predict_spherical_domain() {
        let metrics = vec![Metric::Euclidean, Metric::Spherical];
        let x = Data::new(vec![1.0, -4.0], 1, 2).unwrap();
        let err = validate_predict(&x, 2, &metrics).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::SphericalOutOfDomain {
                row: 0,
                col: 1,
                value: -4.0
            }
        );
    }

    #[test]
    fn predict_accepts_degenerate_and_small_inputs() {
        // Constant columns and a single row are fine at predict (fit-only checks).
        let x = Data::new(vec![1.0, 1.0], 1, 2).unwrap();
        assert!(validate_predict(&x, 2, &euclidean(2)).is_ok());
    }

    // ---- warnings ----

    #[test]
    fn p_greater_than_n_yields_exactly_one_warning() {
        let x = Data::new(vec![0.0; 2 * 3], 2, 3).unwrap();
        let warnings = fit_warnings(&x);
        assert_eq!(
            warnings,
            vec![Warning::MoreFeaturesThanObservations { p: 3, n: 2 }]
        );
        assert_eq!(
            warnings[0].to_string(),
            "more features (3) than observations (2)"
        );
    }

    #[test]
    fn p_not_greater_than_n_yields_no_warning() {
        let (x, _) = valid_fixture();
        assert!(fit_warnings(&x).is_empty()); // p = 2 < n = 3
        let square = Data::new(vec![0.0; 4], 2, 2).unwrap();
        assert!(fit_warnings(&square).is_empty()); // p = n
    }

    // ---- property: validation never panics ----

    mod props {
        use proptest::prelude::*;

        use super::*;

        fn metric_strategy() -> impl Strategy<Value = Metric> {
            prop_oneof![
                Just(Metric::Euclidean),
                Just(Metric::Spherical),
                Just(Metric::Categorical),
                Just(Metric::Prepared),
            ]
        }

        proptest! {
            /// `Data::new` never panics for any (buffer, shape) triple.
            #[test]
            fn data_new_never_panics(
                values in prop::collection::vec(prop::num::f64::ANY, 0..64),
                n_rows in 0usize..20,
                n_cols in 0usize..20,
            ) {
                let _ = Data::new(values, n_rows, n_cols);
            }

            /// The fit/predict validators never panic on arbitrary matrices
            /// (including NaN/±∞ values), arbitrary metric lists, and arbitrary ω.
            #[test]
            fn validators_never_panic(
                values in prop::collection::vec(prop::num::f64::ANY, 0..64),
                n_cols in 1usize..8,
                y in prop::collection::vec(prop::num::f64::ANY, 0..12),
                metrics in prop::collection::vec(metric_strategy(), 0..10),
                omega in prop::num::f64::ANY,
                expected_cols in 0usize..8,
            ) {
                let n_rows = values.len() / n_cols;
                let x = Data::new(values[..n_rows * n_cols].to_vec(), n_rows, n_cols).unwrap();
                let _ = validate_fit(&x, &y, &metrics, omega);
                let _ = validate_predict(&x, expected_cols, &metrics);
                let _ = fit_warnings(&x);
            }
        }
    }
}
