//! Standalone cosine assignment geometry.

use crate::extensions::distance::PairwiseDistance;

/// Cosine distance over the active dimensions: `1 − cos θ` between the
/// observation's and the centre's active-coordinate vectors, in `[0, 2]`
/// (up to rounding at the ends).
///
/// The angle is measured from the origin of scaled space (the mid-range
/// point of every column after the scaler's min–max map onto [−0.5, 0.5]),
/// so this geometry groups observations by their direction from the middle
/// of the data rather than by proximity: two points far apart along the same
/// ray key as identical. Reach for it when relative profile (which
/// coordinates are high together) matters more than magnitude.
///
/// A vector with zero norm on the active dimensions has no direction; its
/// key against every centre is defined as `1.0` (no alignment information,
/// the same convention scikit-learn uses), so assignment stays finite and
/// deterministic (ties break to the lowest centre index, as always).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cosine;

impl PairwiseDistance for Cosine {
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        let mut dot = 0.0_f64;
        let mut norm_x = 0.0_f64;
        let mut norm_c = 0.0_f64;
        for &dim in active_dims {
            let a = x_row[dim];
            let b = centre_row[dim];
            dot += a * b;
            norm_x += a * a;
            norm_c += b * b;
        }
        if norm_x == 0.0 || norm_c == 0.0 {
            return 1.0;
        }
        1.0 - dot / (norm_x.sqrt() * norm_c.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_abs_eq;

    #[test]
    fn key_is_one_minus_cosine_similarity() {
        // Orthogonal vectors: cos θ = 0, key 1.
        assert_abs_eq(
            Cosine.distance(&[1.0, 0.0], &[0.0, 1.0], &[0, 1]),
            1.0,
            1e-12,
        );
        // Opposite vectors: cos θ = −1, key 2.
        assert_abs_eq(
            Cosine.distance(&[0.3, 0.4], &[-0.3, -0.4], &[0, 1]),
            2.0,
            1e-12,
        );
        // Same direction, different magnitude: key 0.
        assert_abs_eq(
            Cosine.distance(&[0.1, 0.2], &[0.3, 0.6], &[0, 1]),
            0.0,
            1e-12,
        );
        // 45°: cos θ = 1/√2.
        assert_abs_eq(
            Cosine.distance(&[1.0, 0.0], &[1.0, 1.0], &[0, 1]),
            1.0 - 1.0 / 2.0_f64.sqrt(),
            1e-12,
        );
    }

    #[test]
    fn only_active_dims_participate() {
        // Restricted to dim 0 the vectors are parallel, whatever dim 1 holds.
        assert_abs_eq(Cosine.distance(&[0.2, 9.0], &[0.4, -3.0], &[0]), 0.0, 1e-12);
    }

    #[test]
    fn zero_norm_keys_one_against_everything() {
        assert_eq!(Cosine.distance(&[0.0, 0.0], &[0.3, 0.4], &[0, 1]), 1.0);
        assert_eq!(Cosine.distance(&[0.3, 0.4], &[0.0, 0.0], &[0, 1]), 1.0);
        assert_eq!(Cosine.distance(&[0.0, 0.0], &[0.0, 0.0], &[0, 1]), 1.0);
    }

    #[test]
    fn passes_the_distance_and_assigner_checks() {
        // Rows and centres away from the origin (a zero vector has no
        // direction and would key 1 everywhere: legal, but self-minimality
        // is only meaningful off the origin).
        let x =
            crate::engine::data::Data::from_rows(&[[-0.4, 0.2], [0.4, -0.1], [0.05, 0.3]]).unwrap();
        let tessellation = crate::engine::tessellation::Tessellation::new(
            vec![-0.5, 0.1, 0.5, 0.2],
            vec![0, 1],
            vec![0.0, 0.0],
        )
        .unwrap();
        let mut results = crate::conformance::check_distance(&Cosine, &x, &tessellation);
        results.extend(crate::conformance::check_assigner(
            &Cosine,
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
        // A directional signal: the response depends on which of the two
        // coordinates dominates, not on distance from the middle.
        let n = 24;
        let mut rows = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = i as f64 / (n - 1) as f64;
            rows.push([a, 1.0 - a]);
            y.push(if a > 0.5 { 2.0 } else { -1.0 });
        }
        let x = crate::engine::data::Data::from_rows(&rows).unwrap();
        let model = crate::engine::builder::SamplerBuilder::new(
            crate::AddiVortesConfig::new(42)
                .with_m(10)
                .with_burn_in(20)
                .with_draws(30)
                .with_omega(1.5),
        )
        .with_distance(Cosine)
        .fit(&x, &y)
        .unwrap();
        let predictions = model.predict(&x).unwrap();
        assert!(predictions.iter().all(|p| p.is_finite()));
    }
}
