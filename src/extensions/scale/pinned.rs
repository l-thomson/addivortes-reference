//! The pinned-σ² scale model, as one shelf entry: a
//! constant error variance, no draw, no RNG. Binary-AddiVortes (Albert–Chib
//! probit) pins the latent-scale variance to exactly 1, and pinned-σ² is the natural
//! pairing for any [`ResponseModel`](crate::extensions::response::ResponseModel) whose
//! augmentation already accounts for the noise level.

use crate::engine::error::{Result, require_positive_finite};
use crate::extensions::scale::{ScaleCtx, ScaleModel};

/// σ² pinned to a constant (scaled space): `update` is a no-op that
/// consumes no RNG, so attaching this model never shifts the chain's pinned
/// draw order. The probit/Binary configuration is [`PinnedSigma::unit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinnedSigma {
    sigma_sq: f64,
}

impl PinnedSigma {
    /// A scale model pinned to the given σ² (scaled space). Fails with
    /// [`AddiVortesError::InvalidHyperparameter`](crate::AddiVortesError::InvalidHyperparameter)
    /// unless `sigma_sq` is finite and strictly positive.
    pub fn new(sigma_sq: f64) -> Result<Self> {
        Ok(Self {
            sigma_sq: require_positive_finite("sigma_sq", sigma_sq)?,
        })
    }

    /// σ² ≡ 1, the Albert–Chib probit (Binary-AddiVortes) configuration.
    pub fn unit() -> Self {
        Self { sigma_sq: 1.0 }
    }
}

impl ScaleModel for PinnedSigma {
    type Error = std::convert::Infallible;

    fn update(
        &mut self,
        _ctx: &ScaleCtx<'_>,
        _rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        Ok(())
    }

    fn sigma_sq(&self) -> f64 {
        self.sigma_sq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_sigma_reports_its_constant_and_no_precisions() {
        let model = PinnedSigma::unit();
        assert_eq!(model.sigma_sq().to_bits(), 1.0_f64.to_bits());
        assert!(ScaleModel::precisions(&model).is_none());
        let model = PinnedSigma::new(0.25).unwrap();
        assert_eq!(model.sigma_sq().to_bits(), 0.25_f64.to_bits());
    }

    #[test]
    fn pinned_sigma_rejects_a_bad_variance() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                PinnedSigma::new(bad),
                Err(crate::AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "sigma_sq"
            ));
        }
    }
}
