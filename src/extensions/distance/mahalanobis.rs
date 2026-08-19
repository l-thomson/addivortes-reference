//! Standalone Mahalanobis assignment geometry.

use crate::engine::error::{AddiVortesError, Result};
use crate::engine::mathsfn;
use crate::extensions::distance::PairwiseDistance;

/// Mahalanobis distance (Mahalanobis 1936) over the active dimensions:
/// the quadratic form `diffᵀ P diff`, where `P` is a caller-supplied
/// precision matrix (typically the inverse of a covariance estimated on
/// the scaled design). Squared-distance form: a valid strictly-monotone
/// key, matching the built-in squared-Euclidean convention.
///
/// Correlated or unequally-informative coordinates stop being treated as
/// independent and equally weighted: the precision matrix whitens the
/// geometry, so cells form along the data's own correlation structure. The
/// identity precision reproduces the paper's Euclidean key exactly.
///
/// # Construction and guards
///
/// `P` is validated once, loudly, at construction: row-major `width ×
/// width`, all entries finite, symmetric to within a small relative
/// tolerance (then symmetrised exactly, deterministically, to
/// `(P + Pᵀ)/2`), and positive definite, checked by the pinned
/// [`mathsfn::cholesky`] factorisation, so the check itself is
/// bit-deterministic across platforms.
///
/// `width` must equal the encoded design width the fit will see (the
/// post-one-hot column count). Like [`Gower`], a mismatched row width is
/// refused at assignment time with a non-finite key, which the assigner
/// surfaces as [`NonFiniteDistance`] instead of silently measuring with
/// misaligned entries.
///
/// Only the active dimensions' rows/columns of `P` participate in a given
/// tessellation's key; cross-terms into inactive dimensions are never read
/// (the synthesised centre row equals the observation there, so their
/// differences are zero regardless).
///
/// [`Gower`]: crate::extensions::distance::Gower
/// [`NonFiniteDistance`]: crate::engine::error::AddiVortesError::NonFiniteDistance
#[derive(Debug, Clone, PartialEq)]
pub struct Mahalanobis {
    /// Row-major `width × width`, exactly symmetric after construction.
    precision: Vec<f64>,
    width: usize,
}

impl Mahalanobis {
    /// A Mahalanobis geometry from a row-major `width × width` precision
    /// matrix. Fails with [`AddiVortesError::InvalidHyperparameter`] when
    /// the matrix is the wrong size, non-finite, asymmetric beyond a
    /// `1e-9`-relative tolerance, or not positive definite.
    pub fn new(precision: Vec<f64>, width: usize) -> Result<Self> {
        let invalid = |reason: &str| AddiVortesError::InvalidHyperparameter {
            name: "mahalanobis_precision".into(),
            reason: reason.into(),
        };
        if width == 0 || precision.len() != width * width {
            return Err(invalid(&format!(
                "must be row-major width × width with width ≥ 1 (got {} entries for width {width})",
                precision.len()
            )));
        }
        if !precision.iter().all(|v| v.is_finite()) {
            return Err(invalid("every entry must be finite"));
        }
        // Symmetry within a relative tolerance, then exact deterministic
        // symmetrisation: numerically-inverted covariances are rarely
        // bit-symmetric, but a materially asymmetric matrix is a caller bug.
        let mut precision = precision;
        for i in 0..width {
            for j in 0..i {
                let a = precision[i * width + j];
                let b = precision[j * width + i];
                if (a - b).abs() > 1e-9 * f64::max(1.0, f64::max(a.abs(), b.abs())) {
                    return Err(invalid(&format!(
                        "must be symmetric: entries ({i},{j}) = {a} and ({j},{i}) = {b} disagree"
                    )));
                }
                let mean = 0.5 * (a + b);
                precision[i * width + j] = mean;
                precision[j * width + i] = mean;
            }
        }
        // Positive definiteness via the pinned Cholesky (bit-deterministic).
        let mut factor = precision.clone();
        if !mathsfn::cholesky(&mut factor, width) {
            return Err(invalid("must be positive definite (Cholesky failed)"));
        }
        Ok(Self { precision, width })
    }
}

impl PairwiseDistance for Mahalanobis {
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        // The declared width must match the fitted encoding: refuse loudly
        // (NaN → the assigner's NonFiniteDistance) rather than measure with
        // misaligned matrix entries.
        if x_row.len() != self.width {
            return f64::NAN;
        }
        let mut key = 0.0_f64;
        for &a in active_dims {
            let diff_a = x_row[a] - centre_row[a];
            for &b in active_dims {
                let diff_b = x_row[b] - centre_row[b];
                key += diff_a * self.precision[a * self.width + b] * diff_b;
            }
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::distance::Euclidean;
    use crate::test_support::assert_abs_eq;

    fn identity(width: usize) -> Vec<f64> {
        let mut m = vec![0.0; width * width];
        for i in 0..width {
            m[i * width + i] = 1.0;
        }
        m
    }

    #[test]
    fn identity_precision_matches_euclidean_bit_for_bit() {
        // Under the identity only the diagonal contributes; the zero
        // cross-terms add exactly 0.0 and diff·1.0·diff == diff·diff, so the
        // accumulated key is the Euclidean key's bits, not just close.
        let m = Mahalanobis::new(identity(4), 4).unwrap();
        let x = [0.13, -0.7, 2.5, 0.01];
        let c = [1.9, 0.4, -0.3, 0.02];
        for dims in [vec![0usize, 2, 3], vec![1], vec![0, 1, 2, 3]] {
            assert_eq!(
                m.distance(&x, &c, &dims).to_bits(),
                Euclidean.distance(&x, &c, &dims).to_bits()
            );
        }
    }

    #[test]
    fn quadratic_form_matches_a_hand_computation() {
        // P = [[2, −1], [−1, 2]], diff = (0.3, −0.1):
        // 2·0.09 + 2·(−1)·0.3·(−0.1) + 2·0.01 = 0.18 + 0.06 + 0.02 = 0.26.
        let m = Mahalanobis::new(vec![2.0, -1.0, -1.0, 2.0], 2).unwrap();
        let x = [0.4, 0.1];
        let c = [0.1, 0.2];
        assert_abs_eq(m.distance(&x, &c, &[0, 1]), 0.26, 1e-12);
        // Restricted to dim 0 only the (0,0) entry participates: 2·0.09.
        assert_abs_eq(m.distance(&x, &c, &[0]), 0.18, 1e-12);
    }

    #[test]
    fn construction_guards_reject_bad_matrices() {
        let name_is_precision = |err: &AddiVortesError| {
            matches!(
                err,
                AddiVortesError::InvalidHyperparameter { name, .. } if name == "mahalanobis_precision"
            )
        };
        // Wrong size.
        assert!(name_is_precision(
            &Mahalanobis::new(vec![1.0, 0.0, 1.0], 2).unwrap_err()
        ));
        // Zero width.
        assert!(name_is_precision(&Mahalanobis::new(vec![], 0).unwrap_err()));
        // Non-finite entry.
        assert!(name_is_precision(
            &Mahalanobis::new(vec![1.0, 0.0, 0.0, f64::NAN], 2).unwrap_err()
        ));
        // Materially asymmetric.
        assert!(name_is_precision(
            &Mahalanobis::new(vec![1.0, 0.5, -0.5, 1.0], 2).unwrap_err()
        ));
        // Symmetric but indefinite (eigenvalues 3 and −1).
        assert!(name_is_precision(
            &Mahalanobis::new(vec![1.0, 2.0, 2.0, 1.0], 2).unwrap_err()
        ));
        // Semi-definite (rank 1) is refused too: strictly positive pivots.
        assert!(name_is_precision(
            &Mahalanobis::new(vec![1.0, 1.0, 1.0, 1.0], 2).unwrap_err()
        ));
    }

    #[test]
    fn tiny_asymmetry_is_symmetrised_deterministically() {
        let eps = 1e-13;
        let m = Mahalanobis::new(vec![2.0, -1.0 + eps, -1.0 - eps, 2.0], 2).unwrap();
        let exact = Mahalanobis::new(vec![2.0, -1.0, -1.0, 2.0], 2).unwrap();
        // (a + b)/2 restores −1.0 exactly here, so the two keys are bit-equal.
        let x = [0.4, 0.1];
        let c = [0.1, 0.2];
        assert_eq!(
            m.distance(&x, &c, &[0, 1]).to_bits(),
            exact.distance(&x, &c, &[0, 1]).to_bits()
        );
    }

    #[test]
    fn wrong_row_width_returns_non_finite() {
        let m = Mahalanobis::new(identity(3), 3).unwrap();
        let x = [0.1, 0.2];
        let c = [0.3, 0.4];
        assert!(m.distance(&x, &c, &[0]).is_nan());
    }

    #[test]
    fn passes_the_distance_and_assigner_checks() {
        let m = Mahalanobis::new(vec![2.0, -0.5, -0.5, 1.0], 2).unwrap();
        let x =
            crate::engine::data::Data::from_rows(&[[-0.4, 0.2], [0.4, -0.1], [0.05, 0.3]]).unwrap();
        let tessellation = crate::engine::tessellation::Tessellation::new(
            vec![-0.5, 0.0, 0.5, 0.0],
            vec![0, 1],
            vec![0.0, 0.0],
        )
        .unwrap();
        let mut results = crate::conformance::check_distance(&m, &x, &tessellation);
        results.extend(crate::conformance::check_assigner(&m, &x, &tessellation));
        assert!(
            results.iter().all(|r| r.passed),
            "conformance failures: {results:?}"
        );
    }

    #[test]
    fn fits_through_the_config_seam() {
        let n = 24;
        let mut rows = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = i as f64 / (n - 1) as f64;
            rows.push([a, 1.0 - a]);
            y.push(3.0 * a);
        }
        let x = crate::engine::data::Data::from_rows(&rows).unwrap();
        let model = crate::engine::builder::SamplerBuilder::new(
            crate::AddiVortesConfig::new(42)
                .with_m(10)
                .with_burn_in(20)
                .with_draws(30)
                .with_omega(1.5),
        )
        .with_distance(Mahalanobis::new(vec![2.0, -0.5, -0.5, 1.0], 2).unwrap())
        .fit(&x, &y)
        .unwrap();
        let predictions = model.predict(&x).unwrap();
        assert!(predictions.iter().all(|p| p.is_finite()));
        assert!(
            model.in_sample_rmse() < 0.6,
            "rmse {}",
            model.in_sample_rmse()
        );
    }

    #[test]
    fn wrong_declared_width_fails_the_fit_loudly() {
        // Declared for width 3, fitted design has 2 encoded columns: the
        // guard must abort the fit with NonFiniteDistance, not mis-measure.
        let n = 12;
        let mut rows = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            rows.push([i as f64 / (n - 1) as f64, (i % 3) as f64]);
            y.push(i as f64);
        }
        let x = crate::engine::data::Data::from_rows(&rows).unwrap();
        let err = crate::engine::builder::SamplerBuilder::new(
            crate::AddiVortesConfig::new(7)
                .with_m(5)
                .with_burn_in(5)
                .with_draws(5)
                .with_omega(1.5),
        )
        .with_distance(Mahalanobis::new(identity(3), 3).unwrap())
        .fit(&x, &y)
        .unwrap_err();
        assert!(matches!(
            err,
            crate::engine::error::AddiVortesError::NonFiniteDistance { .. }
        ));
    }
}
