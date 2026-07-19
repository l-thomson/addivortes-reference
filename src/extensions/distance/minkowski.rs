//! Standalone Minkowski (L_p) assignment geometry.

use crate::engine::error::{AddiVortesError, Result};
use crate::engine::mathsfn;
use crate::extensions::distance::PairwiseDistance;

/// Minkowski (L_p) distance over the active dimensions, for a fixed order
/// `p ≥ 1`. The key is `Σ |diff|^p`, the p-th power of the L_p metric,
/// which is a strictly increasing transform of it and therefore a valid
/// comparison key (the same convention as the built-in squared-Euclidean
/// key; the p-th root is never taken).
///
/// `p` interpolates the assignment geometry between [`Manhattan`] (`p = 1`)
/// and, as `p → ∞`, the Chebyshev max-coordinate limit; `p = 2` reproduces
/// the paper's Euclidean geometry (up to the last floating-point bit: the
/// power routes through `mathsfn::powf`, not a bare squaring).
///
/// [`Manhattan`]: crate::extensions::distance::Manhattan
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Minkowski {
    p: f64,
}

impl Minkowski {
    /// A Minkowski geometry of order `p`. Fails with
    /// [`AddiVortesError::InvalidHyperparameter`] unless `p` is finite and
    /// at least 1 (orders below 1 do not satisfy the triangle inequality,
    /// so they are not metrics and not offered).
    pub fn new(p: f64) -> Result<Self> {
        if !(p.is_finite() && p >= 1.0) {
            return Err(AddiVortesError::InvalidHyperparameter {
                name: "minkowski_p".into(),
                reason: "must be finite and at least 1".into(),
            });
        }
        Ok(Self { p })
    }

    /// The order `p` this geometry was constructed with: a dimensionless
    /// exponent (no coordinate system; it is not a scaled-space quantity).
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl PairwiseDistance for Minkowski {
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        let mut key = 0.0_f64;
        for &dim in active_dims {
            key += mathsfn::powf((x_row[dim] - centre_row[dim]).abs(), self.p);
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::distance::{Euclidean, Manhattan};
    use crate::test_support::assert_abs_eq;

    #[test]
    fn rejects_invalid_orders() {
        for bad in [0.5, 0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = Minkowski::new(bad).unwrap_err();
            assert!(matches!(
                err,
                AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "minkowski_p"
            ));
        }
    }

    #[test]
    fn key_is_sum_of_active_diffs_to_the_p() {
        let m = Minkowski::new(3.0).unwrap();
        let x = [0.1, 0.5, -0.2];
        let c = [0.4, 0.5, 0.3];
        // Hand-derived: 0.3³ + 0.5³ = 0.027 + 0.125 = 0.152.
        assert_abs_eq(m.distance(&x, &c, &[0, 2]), 0.152, 1e-12);
        assert_abs_eq(m.distance(&x, &c, &[0]), 0.027, 1e-12);
    }

    #[test]
    fn order_one_matches_manhattan_and_order_two_matches_euclidean() {
        let x = [0.13, -0.7, 2.5, 0.01];
        let c = [1.9, 0.4, -0.3, 0.02];
        let dims = [0usize, 1, 2, 3];
        let l1 = Minkowski::new(1.0).unwrap();
        assert_abs_eq(
            l1.distance(&x, &c, &dims),
            Manhattan.distance(&x, &c, &dims),
            1e-12,
        );
        let l2 = Minkowski::new(2.0).unwrap();
        assert_abs_eq(
            l2.distance(&x, &c, &dims),
            Euclidean.distance(&x, &c, &dims),
            1e-12,
        );
    }

    #[test]
    fn passes_the_distance_and_assigner_checks() {
        let m = Minkowski::new(1.5).unwrap();
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
        let model = crate::AddiVortesConfig::new(42)
            .with_m(10)
            .with_burn_in(20)
            .with_draws(30)
            .with_omega(1.5)
            .with_distance(Minkowski::new(3.0).unwrap())
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
}
