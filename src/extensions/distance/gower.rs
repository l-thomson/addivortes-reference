//! Gower assignment geometry: the classic mixed-type distance
//! (Gower 1971) over the engine's scaled, encoded design.
//!
//! Gower's coefficient averages per-column dissimilarities: numeric columns
//! contribute their range-normalised absolute difference, categorical
//! columns a 0/1 mismatch. Two facts make the engine version simple:
//!
//! - the scaler has already range-normalised every numeric column (min–max
//!   onto [−0.5, 0.5], range 1), so the numeric contribution is `|diff|`;
//! - a categorical mismatch flips exactly two ±0.5 one-hot columns by 1
//!   each, so weighting every one-hot column's `|diff|` by ½ makes a full
//!   mismatch contribute exactly 1, commensurate with a full-range numeric
//!   difference, which is precisely Gower's normalisation. (Centres are
//!   continuous, not snapped to category vertices; the ½-weighted L1 is the
//!   natural extension, and it restricts to the 0/1 indicator on vertices.)
//!
//! Gower's denominator (the count of participating columns) is constant
//! across the centres being compared for one observation, so it drops out
//! of the arg-min: the key is the weighted L1 sum alone (strictly monotone
//! in the true Gower dissimilarity; the [`PairwiseDistance`] contract asks
//! no more).

use crate::engine::error::{AddiVortesError, Result};
use crate::extensions::distance::PairwiseDistance;

/// One raw (pre-encoding) column of the design, as [`Gower`] needs to
/// understand it: enough to reconstruct the encoded layout and weight each
/// encoded column correctly.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GowerKind {
    /// A numeric column (`Metric::Euclidean` or `Metric::Prepared`): one
    /// encoded column, contributing its absolute difference.
    Numeric,
    /// A categorical column (`Metric::Categorical`) one-hot encoded into
    /// exactly `levels` columns (the distinct levels seen at fit),
    /// each contributing half its absolute difference so a full mismatch
    /// counts 1.
    Categorical {
        /// The number of distinct levels the fit will see (a count).
        levels: usize,
    },
}

/// The Gower mixed-type assignment geometry (Gower 1971), selected with
/// [`with_distance`](crate::AddiVortesConfig::with_distance). Construct it
/// from the raw column layout (the same order as `with_metrics`, with
/// each categorical column's level count), so it can weight the encoded
/// columns the scaler will produce.
///
/// The angular (`Metric::Spherical`) columns are not part of classic Gower;
/// a design that mixes angles in keeps the built-in compound metric or a
/// custom composition instead.
#[derive(Debug, Clone)]
pub struct Gower {
    /// Per encoded column: the L1 weight (1 for numeric, ½ for one-hot
    /// members), derived once at construction.
    weights: Vec<f64>,
}

impl Gower {
    /// A Gower geometry over the given raw-column layout (one entry per
    /// pre-encoding column, in `with_metrics` order).
    ///
    /// The derived encoded width must match the fitted design's: a
    /// categorical level count that disagrees with the levels the fit
    /// actually sees shifts every later column, so [`distance`] guards the
    /// row width and returns a non-finite key on mismatch, which the
    /// assigner's always-on finiteness check surfaces as
    /// [`NonFiniteDistance`](crate::AddiVortesError::NonFiniteDistance)
    /// instead of silently assigning with the wrong geometry.
    ///
    /// Fails with
    /// [`AddiVortesError::InvalidHyperparameter`](crate::AddiVortesError::InvalidHyperparameter)
    /// if a categorical column declares zero levels.
    ///
    /// [`distance`]: PairwiseDistance::distance
    pub fn new(columns: Vec<GowerKind>) -> Result<Self> {
        let mut weights = Vec::new();
        for (i, column) in columns.into_iter().enumerate() {
            match column {
                GowerKind::Numeric => weights.push(1.0),
                GowerKind::Categorical { levels } => {
                    if levels == 0 {
                        return Err(AddiVortesError::InvalidHyperparameter {
                            name: "levels".into(),
                            reason: format!(
                                "column {i} is categorical with 0 levels: it needs at least 1"
                            ),
                        });
                    }
                    weights.extend(std::iter::repeat_n(0.5, levels));
                }
            }
        }
        Ok(Self { weights })
    }
}

impl PairwiseDistance for Gower {
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        // The declared layout must match the fitted encoding: a mismatch
        // means every weight after the disagreement is wrong, so refuse
        // loudly (NaN → the assigner's NonFiniteDistance) rather than
        // assign with the wrong geometry.
        if x_row.len() != self.weights.len() {
            return f64::NAN;
        }
        active_dims
            .iter()
            .map(|&d| self.weights[d] * (x_row[d] - centre_row[d]).abs())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full category mismatch weighs exactly like a full-range numeric
    /// difference: Gower's normalisation, on the ±0.5 encoding.
    #[test]
    fn category_mismatch_weighs_one_full_numeric_range() {
        // Layout: one numeric, one 3-level categorical → 4 encoded columns.
        let gower = Gower::new(vec![
            GowerKind::Numeric,
            GowerKind::Categorical { levels: 3 },
        ])
        .unwrap();
        // Observation: numeric −0.5, category A. Encoded: [−0.5 | 0.5, −0.5, −0.5].
        let x = [-0.5, 0.5, -0.5, -0.5];
        // Centre 1: numeric 0.5 (full range), same category.
        let full_numeric = [0.5, 0.5, -0.5, -0.5];
        // Centre 2: same numeric, category B (full mismatch).
        let full_mismatch = [-0.5, -0.5, 0.5, -0.5];
        let dims = [0usize, 1, 2, 3];
        let a = gower.distance(&x, &full_numeric, &dims);
        let b = gower.distance(&x, &full_mismatch, &dims);
        assert_eq!(a, 1.0);
        assert_eq!(b, 1.0);
    }

    #[test]
    fn a_zero_level_categorical_column_is_an_error() {
        assert!(matches!(
            Gower::new(vec![GowerKind::Numeric, GowerKind::Categorical { levels: 0 }]),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "levels"
        ));
    }

    /// On an all-numeric layout Gower is plain L1 over the active dims.
    #[test]
    fn all_numeric_is_manhattan() {
        let gower = Gower::new(vec![GowerKind::Numeric; 3]).unwrap();
        let x = [0.1_f64, -0.3, 0.4];
        let c = [-0.2_f64, 0.2, 0.4];
        let l1 = (x[0] - c[0]).abs() + (x[1] - c[1]).abs();
        assert_eq!(gower.distance(&x, &c, &[0, 1, 2]), l1);
        // Inactive dims do not participate.
        assert_eq!(gower.distance(&x, &c, &[0]), (x[0] - c[0]).abs());
    }

    /// A row width that disagrees with the declared layout is refused with a
    /// non-finite key (surfaced by the assigner as `NonFiniteDistance`).
    #[test]
    fn wrong_encoded_width_returns_non_finite() {
        let gower = Gower::new(vec![GowerKind::Categorical { levels: 3 }]).unwrap();
        let x = [0.5, -0.5]; // two columns, layout says three
        let c = [0.5, -0.5];
        assert!(gower.distance(&x, &c, &[0]).is_nan());
    }

    /// The one-command conformance check on a mixed fixture: metric-level correctness
    /// (determinism, finiteness, self-minimality, digest) and the batch
    /// checks, exactly as a researcher would run them from the template.
    #[test]
    fn passes_the_distance_and_assigner_checks() {
        let gower = Gower::new(vec![
            GowerKind::Numeric,
            GowerKind::Categorical { levels: 2 },
        ])
        .unwrap();
        // Encoded fixture: numeric + 2 one-hot columns, rows on vertices.
        let x = crate::engine::data::Data::from_rows(&[
            [-0.4, 0.5, -0.5],
            [0.4, -0.5, 0.5],
            [0.05, 0.5, -0.5],
        ])
        .unwrap();
        let tessellation = crate::engine::tessellation::Tessellation::new(
            vec![-0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
            vec![0, 1, 2],
            vec![0.0, 0.0],
        )
        .unwrap();
        let mut results = crate::conformance::check_distance(&gower, &x, &tessellation);
        results.extend(crate::conformance::check_assigner(
            &gower,
            &x,
            &tessellation,
        ));
        assert!(
            results.iter().all(|r| r.passed),
            "conformance failures: {results:?}"
        );
    }

    /// End to end: a mixed numeric + categorical fit through
    /// `with_distance(Gower)`: raw layout declared once, predictions finite.
    #[test]
    fn mixed_fit_through_the_config_seam() {
        let n = 24;
        let mut rows = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let numeric = i as f64 / (n - 1) as f64;
            let category = f64::from(u8::from(i % 3 == 0)); // two levels
            rows.push([numeric, category]);
            y.push(2.0 * numeric + if i % 3 == 0 { 0.5 } else { 0.0 });
        }
        let x = crate::engine::data::Data::from_rows(&rows).unwrap();
        let model = crate::AddiVortesConfig::new(42)
            .with_m(10)
            .with_burn_in(20)
            .with_draws(30)
            .with_omega(1.5)
            .with_metrics(vec![
                crate::engine::data::Metric::Euclidean,
                crate::engine::data::Metric::Categorical,
            ])
            .with_distance(
                Gower::new(vec![
                    GowerKind::Numeric,
                    GowerKind::Categorical { levels: 2 },
                ])
                .unwrap(),
            )
            .fit(&x, &y)
            .unwrap();
        let predictions = model.predict(&x).unwrap();
        assert!(predictions.iter().all(|p| p.is_finite()));
        assert!(
            model.in_sample_rmse() < 0.4,
            "rmse {}",
            model.in_sample_rmse()
        );
    }

    /// The declared-layout guard end to end: a level count that disagrees
    /// with the fitted encoding fails the fit with `NonFiniteDistance`
    /// instead of silently assigning with the wrong geometry.
    #[test]
    fn wrong_declared_levels_fail_the_fit_loudly() {
        let n = 12;
        let mut rows = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            rows.push([i as f64 / (n - 1) as f64, f64::from(u8::from(i % 2 == 0))]);
            y.push(i as f64);
        }
        let x = crate::engine::data::Data::from_rows(&rows).unwrap();
        let err = crate::AddiVortesConfig::new(7)
            .with_m(5)
            .with_burn_in(5)
            .with_draws(5)
            .with_omega(1.5)
            .with_metrics(vec![
                crate::engine::data::Metric::Euclidean,
                crate::engine::data::Metric::Categorical,
            ])
            .with_distance(
                Gower::new(vec![
                    GowerKind::Numeric,
                    GowerKind::Categorical { levels: 5 }, // the fit sees 2
                ])
                .unwrap(),
            )
            .fit(&x, &y)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::engine::error::AddiVortesError::NonFiniteDistance { .. }
        ));
    }
}
