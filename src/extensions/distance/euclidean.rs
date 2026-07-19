//! Standalone Euclidean assignment geometry: the paper's metric.

use crate::extensions::distance::PairwiseDistance;

/// Squared Euclidean distance over the active dimensions, exactly the
/// paper's assignment geometry (squared distance is a valid strictly-monotone
/// key). The all-Euclidean [`ColumnMetrics`](crate::extensions::distance::ColumnMetrics)
/// default computes the identical key; this standalone form is the shelf
/// entry to reach for when composing or comparing whole geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Euclidean;

impl PairwiseDistance for Euclidean {
    // All-Euclidean by definition: same fast blanket path as the
    // all-Euclidean ColumnMetrics default.
    fn all_euclidean(&self) -> bool {
        true
    }

    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        let mut key = 0.0_f64;
        for &dim in active_dims {
            let diff = x_row[dim] - centre_row[dim];
            key += diff * diff;
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::data::Metric;
    use crate::extensions::distance::ColumnMetrics;
    use crate::test_support::assert_abs_eq;

    #[test]
    fn key_is_sum_of_squared_active_diffs() {
        let x = [0.1, 0.5, -0.2];
        let c = [0.4, 0.5, 0.3];
        // Hand-derived: (0.1−0.4)² + (−0.2−0.3)² = 0.09 + 0.25 = 0.34.
        assert_abs_eq(Euclidean.distance(&x, &c, &[0, 2]), 0.34, 1e-12);
        assert_abs_eq(Euclidean.distance(&x, &c, &[0]), 0.09, 1e-12);
    }

    #[test]
    fn matches_all_euclidean_column_metrics_bit_for_bit() {
        // The standalone key and the all-Euclidean compound key are the same
        // arithmetic in the same order: identical bits, not just close.
        let cm = ColumnMetrics::new(vec![Metric::Euclidean; 4]);
        let x = [0.13, -0.7, 2.5, 0.01];
        let c = [1.9, 0.4, -0.3, 0.02];
        let dims = [0usize, 2, 3];
        assert_eq!(
            Euclidean.distance(&x, &c, &dims).to_bits(),
            cm.distance(&x, &c, &dims).to_bits()
        );
    }
}
