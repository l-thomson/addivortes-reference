//! The paper's Euclidean coordinate law: N(0, σ_c²) (paper §2.3.2).

use crate::engine::mathsfn;
use crate::extensions::coord::CoordinateDistribution;

/// The Euclidean default: N(0, σ_c²) (paper §2.3.2; default σ_c = 0.8).
#[derive(Debug, Clone, PartialEq)]
pub struct EuclideanNormal {
    sigma_c: f64,
}

impl EuclideanNormal {
    /// N(0, σ_c²) with standard deviation `sigma_c` (must be positive;
    /// validated by the config's σ_c hyperparameter check).
    pub fn new(sigma_c: f64) -> Self {
        debug_assert!(sigma_c > 0.0);
        Self { sigma_c }
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
