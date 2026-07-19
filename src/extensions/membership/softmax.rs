//! The softmax membership kernel, as one shelf entry: the
//! SoftBART-style deterministic weights `φᵢₖ ∝ exp(−d²ᵢₖ/τ)` with fixed
//! temperature τ: the first kernel of the deterministic-weights family, not
//! an endorsed formulation.

use crate::engine::mathsfn;
use crate::extensions::membership::MembershipKernel;

/// `softmax(−d²/τ)` membership weights at fixed temperature τ
/// (scaled-space squared-distance units). Shift-stable by construction: the
/// row's best key is subtracted before exponentiating, so the nearest cell
/// always carries weight exactly 1 and a row can never underflow to
/// all-zero. As τ → 0 the weights approach the hard assignment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftmaxKernel {
    tau: f64,
}

impl SoftmaxKernel {
    /// A softmax kernel with fixed temperature `tau` (> 0, scaled-space
    /// squared-distance units).
    pub fn new(tau: f64) -> Self {
        debug_assert!(tau.is_finite() && tau > 0.0);
        Self { tau }
    }

    /// The fixed temperature τ (scaled-space squared-distance units).
    pub fn tau(&self) -> f64 {
        self.tau
    }
}

impl MembershipKernel for SoftmaxKernel {
    fn weights(&self, keys: &[f64], weights: &mut [f64]) {
        debug_assert_eq!(keys.len(), weights.len());
        let best = keys.iter().copied().fold(f64::INFINITY, f64::min);
        for (weight, &key) in weights.iter_mut().zip(keys) {
            *weight = mathsfn::exp(-(key - best) / self.tau);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_abs_eq, assert_rel_eq};

    #[test]
    fn softmax_weights_match_hand_values_and_are_shift_stable() {
        let kernel = SoftmaxKernel::new(0.5);
        let mut weights = [0.0_f64; 3];
        kernel.weights(&[0.2, 0.7, 0.2], &mut weights);
        // Best key 0.2 → weights exp(0), exp(−1), exp(0).
        assert_abs_eq(weights[0], 1.0, 0.0);
        assert_rel_eq(weights[1], crate::engine::mathsfn::exp(-1.0), 1e-15);
        assert_abs_eq(weights[2], 1.0, 0.0);
        // Shifting every key by a huge constant changes nothing (bitwise).
        let mut shifted = [0.0_f64; 3];
        kernel.weights(&[1e6 + 0.2, 1e6 + 0.7, 1e6 + 0.2], &mut shifted);
        for (a, b) in weights.iter().zip(&shifted) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // A far-away row still gives its best cell weight exactly 1.
        let mut far = [0.0_f64; 2];
        kernel.weights(&[1e9, 2e9], &mut far);
        assert_abs_eq(far[0], 1.0, 0.0);
    }

    #[test]
    fn small_tau_approaches_the_hard_assignment() {
        let kernel = SoftmaxKernel::new(1e-6);
        let mut weights = [0.0_f64; 3];
        kernel.weights(&[0.3, 0.1, 0.5], &mut weights);
        let total: f64 = weights.iter().sum();
        assert_rel_eq(weights[1] / total, 1.0, 1e-12);
    }
}
