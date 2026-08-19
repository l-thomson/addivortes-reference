//! The crate's single error type, returned by `fit` and `predict`.

use std::sync::Arc;

use thiserror::Error;

/// Every error AddiVortes can return. One error channel for all boundary
/// validation; no panic is reachable from user input in release builds.
// Field names are self-describing and every variant carries both a doc comment
// and a Display message; per-field rustdoc would only repeat the names.
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Debug, Clone, Error)]
pub enum AddiVortesError {
    /// The response value in `row` is not a finite number (it is NaN or infinite).
    #[error("response value in row {row} is not finite")]
    NonFiniteResponse { row: usize },

    /// The feature value at (`row`, `col`) is not a finite number (it is NaN or infinite).
    #[error("feature value at row {row}, col {col} is not finite")]
    NonFiniteFeature { row: usize, col: usize },

    /// The response and the feature matrix disagree on the number of observations.
    #[error("response has {y_len} rows but the feature matrix has {x_rows}")]
    RowCountMismatch { y_len: usize, x_rows: usize },

    /// The per-column metric list does not match the number of features.
    #[error("metric list has {metric_len} entries but the feature matrix has {x_cols} columns")]
    MetricLengthMismatch { metric_len: usize, x_cols: usize },

    /// A [`Data`](crate::Data) constructor was given values that cannot form the
    /// requested matrix (buffer length vs shape, or ragged rows).
    #[error("invalid data shape: {reason}")]
    InvalidDataShape { reason: String },

    /// There are too few observations to fit the model.
    #[error("found {found} observations but at least {required} are required")]
    InsufficientObservations { found: usize, required: usize },

    /// A hyperparameter was given an invalid value.
    #[error("invalid hyperparameter `{name}`: {reason}")]
    InvalidHyperparameter { name: String, reason: String },

    /// Prediction input has a different number of features than the fitted model.
    #[error("model expected {expected} features but got {found}")]
    FeatureCountMismatch { expected: usize, found: usize },

    /// The response is constant, so there is no variance to model.
    #[error("response is constant (zero variance)")]
    DegenerateResponse {},

    /// A feature column is constant, so it carries no information.
    #[error("feature in column {col} is constant (zero range)")]
    DegenerateFeature { col: usize },

    /// The response is an exact linear function of the features, so the
    /// residual variance is zero and the σ² prior scale calibrates to λ = 0,
    /// which the selected scale model cannot accept (the H variance
    /// calibration would pin every variance cell at 0).
    #[error("response is an exact linear function of the features (zero residual variance)")]
    DegenerateResidual {},

    /// A categorical column holds a value that was not present during fitting.
    #[error("column {col} contains category {value} not seen during fit")]
    UnseenCategory { col: usize, value: f64 },

    /// A quantile probability was outside the open interval (0, 1).
    #[error("quantile probability {value} is not in the open interval (0, 1)")]
    InvalidQuantileProb { value: f64 },

    /// A coordinate handed to the spherical metric is outside its valid domain.
    #[error("spherical coordinate at row {row}, col {col} is {value}, outside the valid domain")]
    SphericalOutOfDomain { row: usize, col: usize, value: f64 },

    /// A computed distance between an observation and a centre is not finite.
    #[error("distance between observation {observation} and centre {centre} is not finite")]
    NonFiniteDistance { observation: usize, centre: usize },

    /// A custom move set is misconfigured.
    #[error("invalid move set: move `{move_name}` {reason}")]
    InvalidMoveSet { move_name: String, reason: String },

    /// A compound per-column metric group is misconfigured (the distance point:
    /// `ColumnMetrics::with_group`).
    #[error("invalid metric group: {reason}")]
    InvalidMetricGroup { reason: String },

    /// Soft membership was selected but the configured assigner
    /// provides no dense per-cell keys (`CellAssigner::membership_keys`;
    /// every `PairwiseDistance` provides them through the blanket impl;
    /// batch-level custom assigners must opt in).
    #[error("assigner `{assigner}` does not support soft membership (no dense per-cell keys)")]
    MembershipUnsupported { assigner: String },

    /// A calibration battery's successive-conditional simulator never emitted
    /// a statistic the marginal-conditional draw produced
    /// ([`calibration::getting_it_right`](crate::calibration::getting_it_right)).
    #[error("successive-conditional simulator never emitted statistic `{statistic}`")]
    MissingStatistic { statistic: String },

    /// A user-supplied extension (`CellModel`/`ResponseModel`) returned an error.
    #[error("error from a user extension")]
    Extension {
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

// Hand-written equality. We cannot `#[derive(PartialEq)]` because the
// `Extension` variant holds an `Arc<dyn Error>`, which has no notion of value equality;
// instead we compare two `Extension`s by whether they point to the *same* inner error
// (`Arc::ptr_eq`). Every other variant is compared field by field, exactly what derive
// would have generated. No `Eq` is provided, because the `f64` payloads are not `Eq`.
impl PartialEq for AddiVortesError {
    fn eq(&self, other: &Self) -> bool {
        use AddiVortesError::*;
        match (self, other) {
            (Extension { source: a }, Extension { source: b }) => Arc::ptr_eq(a, b),

            (NonFiniteResponse { row: a }, NonFiniteResponse { row: b }) => a == b,
            (NonFiniteFeature { row: r1, col: c1 }, NonFiniteFeature { row: r2, col: c2 }) => {
                r1 == r2 && c1 == c2
            }
            (
                RowCountMismatch {
                    y_len: a,
                    x_rows: b,
                },
                RowCountMismatch {
                    y_len: c,
                    x_rows: d,
                },
            ) => a == c && b == d,
            (
                MetricLengthMismatch {
                    metric_len: a,
                    x_cols: b,
                },
                MetricLengthMismatch {
                    metric_len: c,
                    x_cols: d,
                },
            ) => a == c && b == d,
            (InvalidDataShape { reason: a }, InvalidDataShape { reason: b }) => a == b,
            (
                InsufficientObservations {
                    found: a,
                    required: b,
                },
                InsufficientObservations {
                    found: c,
                    required: d,
                },
            ) => a == c && b == d,
            (
                InvalidHyperparameter { name: a, reason: b },
                InvalidHyperparameter { name: c, reason: d },
            ) => a == c && b == d,
            (
                FeatureCountMismatch {
                    expected: a,
                    found: b,
                },
                FeatureCountMismatch {
                    expected: c,
                    found: d,
                },
            ) => a == c && b == d,
            (DegenerateResponse {}, DegenerateResponse {}) => true,
            (DegenerateFeature { col: a }, DegenerateFeature { col: b }) => a == b,
            (DegenerateResidual {}, DegenerateResidual {}) => true,
            (UnseenCategory { col: a, value: b }, UnseenCategory { col: c, value: d }) => {
                a == c && b == d
            }
            (InvalidQuantileProb { value: a }, InvalidQuantileProb { value: b }) => a == b,
            (
                SphericalOutOfDomain {
                    row: r1,
                    col: c1,
                    value: v1,
                },
                SphericalOutOfDomain {
                    row: r2,
                    col: c2,
                    value: v2,
                },
            ) => r1 == r2 && c1 == c2 && v1 == v2,
            (
                NonFiniteDistance {
                    observation: a,
                    centre: b,
                },
                NonFiniteDistance {
                    observation: c,
                    centre: d,
                },
            ) => a == c && b == d,
            (
                InvalidMoveSet {
                    move_name: a,
                    reason: b,
                },
                InvalidMoveSet {
                    move_name: c,
                    reason: d,
                },
            ) => a == c && b == d,
            (InvalidMetricGroup { reason: a }, InvalidMetricGroup { reason: b }) => a == b,
            (MembershipUnsupported { assigner: a }, MembershipUnsupported { assigner: b }) => {
                a == b
            }
            (MissingStatistic { statistic: a }, MissingStatistic { statistic: b }) => a == b,

            // Any two *different* variants are never equal.
            _ => false,
        }
    }
}

/// Shorthand: `Result<T>` means `Result<T, AddiVortesError>`, so functions in this crate
/// can write `-> Result<Foo>` instead of spelling out the error type every time.
pub type Result<T> = std::result::Result<T, AddiVortesError>;

/// Lift an extension-point error into the crate's error: a component whose
/// error type is already [`AddiVortesError`] (every shelf component that can
/// fail) surfaces its own variant; any other error is carried as
/// [`AddiVortesError::Extension`].
pub(crate) fn from_extension<E: std::error::Error + Send + Sync + 'static>(
    error: E,
) -> AddiVortesError {
    let any: &dyn std::any::Any = &error;
    match any.downcast_ref::<AddiVortesError>() {
        Some(native) => native.clone(),
        None => AddiVortesError::Extension {
            source: Arc::new(error),
        },
    }
}

/// The shelf constructors' shared argument check: `value` must be finite and
/// strictly positive, otherwise [`AddiVortesError::InvalidHyperparameter`]
/// naming the argument.
pub(crate) fn require_positive_finite(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(AddiVortesError::InvalidHyperparameter {
            name: name.into(),
            reason: format!("must be finite and positive, got {value}"),
        })
    }
}

/// As [`require_positive_finite`], but zero is allowed.
pub(crate) fn require_non_negative_finite(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(AddiVortesError::InvalidHyperparameter {
            name: name.into(),
            reason: format!("must be finite and non-negative, got {value}"),
        })
    }
}

/// The count sibling of [`require_positive_finite`]: `value` must be at least
/// 1.
pub(crate) fn require_at_least_one(name: &str, value: usize) -> Result<usize> {
    if value >= 1 {
        Ok(value)
    } else {
        Err(AddiVortesError::InvalidHyperparameter {
            name: name.into(),
            reason: "must be at least 1".into(),
        })
    }
}

/// Validation failure while deserialising a saved value (`serde` feature): a
/// corrupt or hand-edited payload was rejected. Never public; serde surfaces
/// it through the deserialiser's own error type via `Display`.
#[cfg(feature = "serde")]
#[derive(Debug, thiserror::Error)]
#[error("invalid saved model: {0}")]
pub(crate) struct SavedModelError(pub(crate) String);

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    #[derive(Debug, thiserror::Error, PartialEq)]
    #[error("boom")]
    struct FakeExtensionError;

    /// Every variant's user-facing message, pinned exactly (house style:
    /// lowercase, no trailing punctuation). A change here is a deliberate,
    /// reviewed act.
    #[test]
    fn display_messages_are_pinned() {
        let cases: Vec<(AddiVortesError, &str)> = vec![
            (
                AddiVortesError::NonFiniteResponse { row: 3 },
                "response value in row 3 is not finite",
            ),
            (
                AddiVortesError::NonFiniteFeature { row: 3, col: 7 },
                "feature value at row 3, col 7 is not finite",
            ),
            (
                AddiVortesError::RowCountMismatch {
                    y_len: 10,
                    x_rows: 12,
                },
                "response has 10 rows but the feature matrix has 12",
            ),
            (
                AddiVortesError::MetricLengthMismatch {
                    metric_len: 4,
                    x_cols: 5,
                },
                "metric list has 4 entries but the feature matrix has 5 columns",
            ),
            (
                AddiVortesError::InvalidDataShape {
                    reason: "6 values cannot form a 2 by 4 matrix".into(),
                },
                "invalid data shape: 6 values cannot form a 2 by 4 matrix",
            ),
            (
                AddiVortesError::InsufficientObservations {
                    found: 2,
                    required: 10,
                },
                "found 2 observations but at least 10 are required",
            ),
            (
                AddiVortesError::InvalidHyperparameter {
                    name: "m".into(),
                    reason: "must be at least 1".into(),
                },
                "invalid hyperparameter `m`: must be at least 1",
            ),
            (
                AddiVortesError::FeatureCountMismatch {
                    expected: 5,
                    found: 4,
                },
                "model expected 5 features but got 4",
            ),
            (
                AddiVortesError::DegenerateResponse {},
                "response is constant (zero variance)",
            ),
            (
                AddiVortesError::DegenerateFeature { col: 2 },
                "feature in column 2 is constant (zero range)",
            ),
            (
                AddiVortesError::DegenerateResidual {},
                "response is an exact linear function of the features (zero residual variance)",
            ),
            (
                AddiVortesError::UnseenCategory { col: 1, value: 9.0 },
                "column 1 contains category 9 not seen during fit",
            ),
            (
                AddiVortesError::InvalidQuantileProb { value: 1.5 },
                "quantile probability 1.5 is not in the open interval (0, 1)",
            ),
            (
                AddiVortesError::SphericalOutOfDomain {
                    row: 0,
                    col: 3,
                    value: 7.0,
                },
                "spherical coordinate at row 0, col 3 is 7, outside the valid domain",
            ),
            (
                AddiVortesError::NonFiniteDistance {
                    observation: 5,
                    centre: 2,
                },
                "distance between observation 5 and centre 2 is not finite",
            ),
            (
                AddiVortesError::InvalidMoveSet {
                    move_name: "AddCentre".into(),
                    reason: "has zero weight".into(),
                },
                "invalid move set: move `AddCentre` has zero weight",
            ),
            (
                AddiVortesError::InvalidMetricGroup {
                    reason: "column 3 is out of range".into(),
                },
                "invalid metric group: column 3 is out of range",
            ),
            (
                AddiVortesError::MissingStatistic {
                    statistic: "sigma_sq".into(),
                },
                "successive-conditional simulator never emitted statistic `sigma_sq`",
            ),
            (
                AddiVortesError::Extension {
                    source: Arc::new(FakeExtensionError),
                },
                "error from a user extension",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    /// The `Extension` variant exposes its inner error through `source()`.
    #[test]
    fn extension_source_chain_is_wired() {
        let error = AddiVortesError::Extension {
            source: Arc::new(FakeExtensionError),
        };
        let source = error.source().expect("Extension must expose a source");
        assert_eq!(source.to_string(), "boom");
    }

    /// Value equality for ordinary variants; pointer identity for `Extension`.
    #[test]
    fn partial_eq_semantics() {
        // same fields: equal
        assert_eq!(
            AddiVortesError::NonFiniteResponse { row: 1 },
            AddiVortesError::NonFiniteResponse { row: 1 },
        );
        // different fields: unequal
        assert_ne!(
            AddiVortesError::NonFiniteResponse { row: 1 },
            AddiVortesError::NonFiniteResponse { row: 2 },
        );
        // different variants: unequal
        assert_ne!(
            AddiVortesError::DegenerateResponse {},
            AddiVortesError::DegenerateFeature { col: 0 },
        );

        // Extension: two Arcs to *equal* inner errors are still unequal…
        let a = AddiVortesError::Extension {
            source: Arc::new(FakeExtensionError),
        };
        let b = AddiVortesError::Extension {
            source: Arc::new(FakeExtensionError),
        };
        assert_ne!(a, b);
        // …but a clone shares the same Arc, so it is equal (Arc::ptr_eq).
        let c = a.clone();
        assert_eq!(a, c);
    }
}
