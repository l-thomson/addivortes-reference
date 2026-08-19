//! The paper's Euclidean coordinate law: N(0, σ_c²) (paper §2.3.2).

use crate::engine::error::{Result, require_positive_finite};
use crate::engine::mathsfn;
use crate::extensions::coord::CoordinateDistribution;

/// The Euclidean default: N(0, σ_c²) (paper §2.3.2; default σ_c = 0.8).
#[derive(Debug, Clone, PartialEq)]
pub struct EuclideanNormal {
    sigma_c: f64,
}

impl EuclideanNormal {
    /// N(0, σ_c²) with standard deviation `sigma_c` (scaled space). Fails
    /// with [`AddiVortesError::InvalidHyperparameter`](crate::AddiVortesError::InvalidHyperparameter)
    /// unless `sigma_c` is finite and strictly positive.
    pub fn new(sigma_c: f64) -> Result<Self> {
        Ok(Self {
            sigma_c: require_positive_finite("sigma_c", sigma_c)?,
        })
    }
}

impl CoordinateDistribution for EuclideanNormal {
    fn sample(&self, rng: &mut dyn rand_core::Rng) -> f64 {
        let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
        self.sigma_c * z
    }

    fn log_density(&self, x: f64) -> f64 {
        let var = self.sigma_c * self.sigma_c;
        -0.5 * mathsfn::ln(2.0 * std::f64::consts::PI * var) - x * x / (2.0 * var)
    }
}
