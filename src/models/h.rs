//! H-AddiVortes: heteroscedastic Gaussian response.
//!
//! The mean is the ordinary additive ensemble; the noise is a conditional
//! variance `s²(x) = ∏ h(x; T′)` modelled by a second, multiplicative
//! ensemble of m′ inverse-χ² tessellations (H paper §3.2-3.3). Each sweep
//! the variance ensemble backfits on squared mean-residuals and hands
//! per-observation precisions to the mean side, whose cell draws become
//! precision-weighted (the engine pairs the weighted cell family
//! automatically). The reported global σ² is pinned at 1; all scale lives
//! in `s²(x)`.
//!
//! Model rule owned here: m′ ≥ 1.

use crate::engine::builder::SamplerBuilder;
use crate::engine::config::AddiVortesConfig;
use crate::engine::data::Data;
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::model::{FittedAddiVortes, ResponseFamily};
use crate::extensions::scale::HVariance;

/// Fit H-AddiVortes with a variance ensemble of `m_prime` tessellations.
pub(crate) fn fit(
    config: AddiVortesConfig,
    x: &Data,
    y: &[f64],
    m_prime: usize,
) -> Result<FittedAddiVortes> {
    if m_prime < 1 {
        return Err(AddiVortesError::InvalidHyperparameter {
            name: "m_prime".into(),
            reason: "the variance ensemble needs at least one tessellation".into(),
        });
    }
    SamplerBuilder::new(config.with_response_family(ResponseFamily::Gaussian))
        .with_scale_model(HVariance::new(m_prime)?)
        .fit(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ramp plus deterministic alternating noise. The noise is load-bearing:
    /// a zero-residual response makes the calibrated λ exactly 0, which the
    /// variance prior cannot accept (the degenerate-data boundary).
    fn toy() -> (Data, Vec<f64>) {
        let n = 50;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &v)| 2.0 * v + if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        (Data::new(xs, n, 1).unwrap(), ys)
    }

    fn cfg(seed: u64) -> AddiVortesConfig {
        AddiVortesConfig::new(seed)
            .with_m(20)
            .with_burn_in(40)
            .with_draws(40)
            .with_omega(0.5)
    }

    /// The front is the builder path: same seed, bit-identical posterior.
    #[test]
    fn front_is_bit_identical_to_the_builder_path() {
        let (x, y) = toy();
        let a = fit(cfg(7), &x, &y, 10).unwrap();
        let b = SamplerBuilder::new(cfg(7))
            .with_scale_model(HVariance::new(10).unwrap())
            .fit(&x, &y)
            .unwrap();
        assert_eq!(a.posterior(), b.posterior());
    }

    #[test]
    fn front_rejects_an_empty_variance_ensemble() {
        let (x, y) = toy();
        assert!(fit(cfg(7), &x, &y, 0).is_err());
    }

    /// The scale rule holds exactly: the reported global σ² is pinned at 1
    /// on every kept draw; all scale lives in s²(x).
    #[test]
    fn reported_global_sigma_is_pinned_at_one() {
        let (x, y) = toy();
        let fitted = fit(cfg(7), &x, &y, 10).unwrap();
        assert!(fitted.posterior().sigma_sq().iter().all(|&s| s == 1.0));
        assert!(fitted.in_sample_rmse().is_finite());
    }
}
