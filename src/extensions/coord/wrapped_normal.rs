//! The spherical coordinate law: a wrapped normal on [−π, π].

use crate::engine::error::{Result, require_positive_finite};
use crate::engine::mathsfn;
use crate::extensions::coord::{CoordinateDistribution, wrap_to_pi};

/// The spherical default: a wrapped normal on [−π, π] with the true wrapped
/// log-density (not truncated; truncation would break the prior ≡ proposal
/// cancellation the point relies on).
#[derive(Debug, Clone, PartialEq)]
pub struct WrappedNormal {
    sigma_c: f64,
}

/// Wrapping terms for the wrapped-normal density sum: with σ_c ≤ ~2 the k = ±10
/// tails are < e⁻¹⁰⁰, far below f64 resolution; the count is fixed (not
/// adaptive) so the density is a pinned, deterministic function.
const WRAPPED_NORMAL_TERMS: i32 = 10;

impl WrappedNormal {
    /// Wrapped N(0, σ_c²) on [−π, π] with underlying standard deviation
    /// `sigma_c` (radians). Fails with
    /// [`AddiVortesError::InvalidHyperparameter`](crate::AddiVortesError::InvalidHyperparameter)
    /// unless `sigma_c` is finite and strictly positive.
    pub fn new(sigma_c: f64) -> Result<Self> {
        Ok(Self {
            sigma_c: require_positive_finite("sigma_c", sigma_c)?,
        })
    }
}

impl CoordinateDistribution for WrappedNormal {
    fn sample(&self, rng: &mut dyn rand_core::Rng) -> f64 {
        let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
        wrap_to_pi(self.sigma_c * z)
    }

    fn log_density(&self, x: f64) -> f64 {
        // f(x) = Σ_k φ(x + 2πk; 0, σ²), k = −K..K (K pinned above).
        let var = self.sigma_c * self.sigma_c;
        let norm = 1.0 / mathsfn::exp(0.5 * mathsfn::ln(2.0 * std::f64::consts::PI * var));
        let mut sum = 0.0_f64;
        for k in -WRAPPED_NORMAL_TERMS..=WRAPPED_NORMAL_TERMS {
            let shifted = x + 2.0 * std::f64::consts::PI * f64::from(k);
            sum += norm * mathsfn::exp(-shifted * shifted / (2.0 * var));
        }
        mathsfn::ln(sum)
    }
}
