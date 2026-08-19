//! Soft AddiVortes: Gaussian response with softmax cell membership.
//!
//! Same additive ensemble and conjugate Gaussian cell values as
//! [`gaussian`](super::gaussian), but an observation's contribution is a
//! weighted blend `φᵢₖ ∝ exp(−d²ᵢₖ/τ)` over cells at fixed temperature τ
//! (τ → 0 recovers the hard assignment). σ² keeps the calibrated global
//! scaled-inverse-χ² draw.
//!
//! Model rules owned here: τ finite and positive; soft membership and a
//! linear cell basis do not compose (enforced by the engine at fit).
//! Membership is not reachable through the loop verbs (it rewires the
//! residual algebra, not the response), so this file wires it through the
//! crate-internal builder: the route reserved for model files.

use crate::engine::builder::SamplerBuilder;
use crate::engine::config::AddiVortesConfig;
use crate::engine::data::Data;
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::model::{FittedAddiVortes, ResponseFamily};
use crate::extensions::membership::SoftmaxKernel;

/// Fit Soft AddiVortes at fixed temperature `tau` (> 0, scaled-space
/// squared-distance units).
pub(crate) fn fit(
    config: AddiVortesConfig,
    x: &Data,
    y: &[f64],
    tau: f64,
) -> Result<FittedAddiVortes> {
    if !(tau.is_finite() && tau > 0.0) {
        return Err(AddiVortesError::InvalidHyperparameter {
            name: "tau".into(),
            reason: format!("softmax temperature must be finite and positive, got {tau}"),
        });
    }
    SamplerBuilder::new(config.with_response_family(ResponseFamily::Gaussian))
        .with_membership(SoftmaxKernel::new(tau)?)
        .fit(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::basis::LinearBasis;

    fn toy() -> (Data, Vec<f64>) {
        let n = 50;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&v| 3.0 * v).collect();
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
        let a = fit(cfg(3), &x, &y, 0.5).unwrap();
        let b = SamplerBuilder::new(cfg(3))
            .with_membership(SoftmaxKernel::new(0.5).unwrap())
            .fit(&x, &y)
            .unwrap();
        assert_eq!(a.posterior(), b.posterior());
    }

    #[test]
    fn front_rejects_a_bad_temperature() {
        let (x, y) = toy();
        assert!(fit(cfg(3), &x, &y, 0.0).is_err());
        assert!(fit(cfg(3), &x, &y, f64::NAN).is_err());
    }

    /// The stated incompatibility is mechanically enforced: soft membership
    /// plus a linear basis is an error at fit, not a wrong chain.
    #[test]
    fn soft_plus_basis_is_a_loud_error() {
        let (x, y) = toy();
        let err = SamplerBuilder::new(cfg(3).with_response_family(ResponseFamily::Gaussian))
            .with_membership(SoftmaxKernel::new(0.5).unwrap())
            .with_cell_basis(LinearBasis::new(vec![0]))
            .fit(&x, &y);
        assert!(err.is_err());
    }

    /// Behavioural sanity on smooth data: the soft fit is a working
    /// regression (finite, and not wildly worse than the hard fit on the
    /// same seed; both deterministic, so the bound is stable).
    #[test]
    fn soft_fit_tracks_the_signal() {
        let (x, y) = toy();
        let soft = fit(cfg(3), &x, &y, 0.5).unwrap();
        let hard = cfg(3).fit(&x, &y).unwrap();
        assert!(soft.in_sample_rmse().is_finite());
        assert!(soft.in_sample_rmse() < 3.0 * hard.in_sample_rmse() + 0.1);
    }
}
