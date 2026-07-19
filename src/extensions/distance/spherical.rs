//! Standalone great-circle assignment geometry (the paper's model is
//! Euclidean and gives no spherical equation; the standard embedding
//! used here is validated against hand-derived closed forms) and
//! the shared hyperspherical embedding it and the compound metric use.

use crate::engine::mathsfn;
use crate::extensions::distance::PairwiseDistance;

/// Squared great-circle (geodesic) distance treating all active dimensions
/// as the hyperspherical coordinates of one point on the unit sphere (the last
/// active dimension being the azimuthal angle; see the `great_circle` helper
/// for the embedding). The paper's model is Euclidean and gives no spherical
/// equation; the embedding is the standard hyperspherical one.
///
/// For a dataset mixing angular and linear columns, use
/// [`ColumnMetrics`](crate::extensions::distance::ColumnMetrics), which applies this
/// geometry to the spherical column group only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Spherical;

impl PairwiseDistance for Spherical {
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        let gc = great_circle(x_row, centre_row, active_dims);
        gc * gc
    }
}

/// Great-circle arc length between the hyperspherical embeddings of the two
/// rows' spherical coordinates. With angles θ₁…θ_s the embedding is the
/// standard one:
///
/// ```text
/// u₁ = cos θ₁
/// u₂ = sin θ₁ cos θ₂
/// …
/// uₛ   = sin θ₁ ⋯ sin θ_{s−1} cos θ_s
/// uₛ₊₁ = sin θ₁ ⋯ sin θ_{s−1} sin θ_s      (Σ uᵢ² = 1 identically)
/// ```
///
/// The arc length is `acos(⟨u, v⟩)` with the dot product clamped to [−1, 1]
/// against rounding. For s = 1 this reduces to the wrapped absolute angle
/// difference on the circle: `acos(cos(θ − φ)) = |θ − φ| wrapped to [0, π]`.
/// Computes the dot product incrementally; no allocation in the hot path.
pub(crate) fn great_circle(a_row: &[f64], b_row: &[f64], spherical_cols: &[usize]) -> f64 {
    let s = spherical_cols.len();
    let mut dot = 0.0_f64;
    let mut prefix_a = 1.0_f64;
    let mut prefix_b = 1.0_f64;
    for (i, &col) in spherical_cols.iter().enumerate() {
        let (ta, tb) = (a_row[col], b_row[col]);
        if i + 1 == s {
            // Azimuthal (last) angle: contributes both cos and sin components.
            dot += prefix_a * mathsfn::cos(ta) * prefix_b * mathsfn::cos(tb);
            dot += prefix_a * mathsfn::sin(ta) * prefix_b * mathsfn::sin(tb);
        } else {
            dot += prefix_a * mathsfn::cos(ta) * prefix_b * mathsfn::cos(tb);
            prefix_a *= mathsfn::sin(ta);
            prefix_b *= mathsfn::sin(tb);
        }
    }
    // Rounding can push |dot| a few ULP beyond 1; acos would return NaN.
    mathsfn::acos(dot.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;
    use crate::test_support::assert_abs_eq;

    #[test]
    fn one_active_dim_is_wrapped_angle_difference() {
        // s = 1 reduces to |θ − φ| wrapped: π − 0.1 vs −π + 0.1 are 0.2 apart
        // across the wrap (NOT 2π − 0.2). Hand-derived key = 0.2² = 0.04.
        let x = [PI - 0.1];
        let c = [-PI + 0.1];
        assert_abs_eq(Spherical.distance(&x, &c, &[0]), 0.04, 1e-12);
        // Antipodal points: acos(cos π) = π ⇒ key = π².
        assert_abs_eq(Spherical.distance(&[0.0], &[PI], &[0]), PI * PI, 1e-12);
    }

    #[test]
    fn two_active_dims_match_hand_embedding() {
        // A = (π/2, 0)  → (cos π/2, sin π/2·cos 0, sin π/2·sin 0) = (0, 1, 0)
        // B = (π/2, π/2) → (0, 0, 1); ⟨A,B⟩ = 0 ⇒ great circle = π/2.
        let x = [FRAC_PI_2, 0.0];
        let c = [FRAC_PI_2, FRAC_PI_2];
        assert_abs_eq(
            Spherical.distance(&x, &c, &[0, 1]),
            FRAC_PI_2 * FRAC_PI_2,
            1e-12,
        );
    }

    #[test]
    fn near_boundary_dot_products_are_clamped() {
        // Identical coordinates: dot may land a few ULP above 1;
        // clamped, acos(1) = 0, key exactly 0.
        let x = [2.0, -3.0];
        assert_abs_eq(Spherical.distance(&x, &x, &[0, 1]), 0.0, 0.0);
    }
}
