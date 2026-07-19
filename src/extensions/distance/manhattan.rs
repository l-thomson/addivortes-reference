//! Standalone Manhattan (L1) assignment geometry.

use crate::extensions::distance::PairwiseDistance;

/// Manhattan (L1, taxicab) distance over the active dimensions: the sum of
/// absolute coordinate differences. Compared with the paper's squared
/// Euclidean key it grows linearly rather than quadratically in each
/// coordinate, so no single dimension's large difference dominates the
/// assignment: the L1 analogue of the usual robustness trade.
///
/// On an all-numeric design this is exactly the key [`Gower`] computes
/// (Gower's per-column weights are all 1 there), bit for bit.
///
/// [`Gower`]: crate::extensions::distance::Gower
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Manhattan;

impl PairwiseDistance for Manhattan {
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        let mut key = 0.0_f64;
        for &dim in active_dims {
            key += (x_row[dim] - centre_row[dim]).abs();
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::distance::{Gower, GowerKind};
    use crate::test_support::assert_abs_eq;

    #[test]
    fn key_is_sum_of_absolute_active_diffs() {
        let x = [0.1, 0.5, -0.2];
        let c = [0.4, 0.5, 0.3];
        // Hand-derived: |0.1−0.4| + |−0.2−0.3| = 0.3 + 0.5 = 0.8.
        assert_abs_eq(Manhattan.distance(&x, &c, &[0, 2]), 0.8, 1e-12);
        assert_abs_eq(Manhattan.distance(&x, &c, &[0]), 0.3, 1e-12);
    }

    #[test]
    fn matches_all_numeric_gower_bit_for_bit() {
        // Same arithmetic in the same order (Gower's numeric weight is 1.0,
        // and 1.0 * x == x exactly): identical bits, not just close.
        let gower = Gower::new(vec![GowerKind::Numeric; 4]);
        let x = [0.13, -0.7, 2.5, 0.01];
        let c = [1.9, 0.4, -0.3, 0.02];
        let dims = [0usize, 2, 3];
        assert_eq!(
            Manhattan.distance(&x, &c, &dims).to_bits(),
            gower.distance(&x, &c, &dims).to_bits()
        );
    }

    #[test]
    fn passes_the_distance_and_assigner_checks() {
        let x =
            crate::engine::data::Data::from_rows(&[[-0.4, 0.2], [0.4, -0.1], [0.05, 0.3]]).unwrap();
        let tessellation = crate::engine::tessellation::Tessellation::new(
            vec![-0.5, 0.0, 0.5, 0.0],
            vec![0, 1],
            vec![0.0, 0.0],
        )
        .unwrap();
        let mut results = crate::conformance::check_distance(&Manhattan, &x, &tessellation);
        results.extend(crate::conformance::check_assigner(
            &Manhattan,
            &x,
            &tessellation,
        ));
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
            .with_distance(Manhattan)
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
