//! Gaussian AddiVortes (Stone & Gosling 2025): the paper model.
//!
//! `y = Σⱼ g(x; Tⱼ) + ε`, `ε ~ N(0, σ²)`, conjugate Gaussian cell means,
//! one global σ² with the calibrated scaled-inverse-χ² draw. The
//! all-defaults path: this file adds nothing to the engine's defaults,
//! which is itself the claim worth pinning.

use crate::engine::config::AddiVortesConfig;
use crate::engine::data::Data;
use crate::engine::error::Result;
use crate::engine::model::{FittedAddiVortes, ResponseFamily};

/// Fit Gaussian AddiVortes: the standard burn-in/thinning loop over the
/// default sweep.
pub(crate) fn fit(config: AddiVortesConfig, x: &Data, y: &[f64]) -> Result<FittedAddiVortes> {
    // The menu decides the family; the knobs stay free.
    config
        .with_response_family(ResponseFamily::Gaussian)
        .fit(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> (Data, Vec<f64>) {
        let n = 40;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&v| 2.0 * v + 0.1 * (v * 7.0)).collect();
        (Data::new(xs, n, 1).unwrap(), ys)
    }

    /// The front is the family path: same seed, bit-identical posterior.
    #[test]
    fn front_is_bit_identical_to_the_config_path() {
        let (x, y) = toy();
        let cfg = || {
            AddiVortesConfig::new(11)
                .with_m(20)
                .with_burn_in(20)
                .with_draws(30)
                .with_omega(0.5)
        };
        let a = fit(cfg(), &x, &y).unwrap();
        let b = cfg().fit(&x, &y).unwrap();
        assert_eq!(a.posterior(), b.posterior());
        assert_eq!(a.in_sample_rmse(), b.in_sample_rmse());
    }
}
