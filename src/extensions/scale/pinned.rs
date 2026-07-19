//! The pinned-σ² scale model, as one shelf entry: a
//! constant error variance, no draw, no RNG. Binary-AddiVortes (Albert–Chib
//! probit) pins the latent-scale variance to exactly 1, and pinned-σ² is the natural
//! pairing for any [`ResponseModel`](crate::extensions::response::ResponseModel) whose
//! augmentation already accounts for the noise level.

use crate::extensions::scale::{ScaleCtx, ScaleModel};

/// σ² pinned to a constant (scaled space): `update` is a no-op that
/// consumes no RNG, so attaching this model never shifts the chain's pinned
/// draw order. The probit/Binary configuration is [`PinnedSigma::unit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinnedSigma {
    sigma_sq: f64,
}

impl PinnedSigma {
    /// A scale model pinned to the given σ² (scaled space; must be finite
    /// and strictly positive: debug-asserted here, release-asserted by the
    /// sampler after every update).
    pub fn new(sigma_sq: f64) -> Self {
        debug_assert!(sigma_sq.is_finite() && sigma_sq > 0.0);
        Self { sigma_sq }
    }

    /// σ² ≡ 1, the Albert–Chib probit (Binary-AddiVortes) configuration.
    pub fn unit() -> Self {
        Self::new(1.0)
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
        let model = PinnedSigma::new(0.25);
        assert_eq!(model.sigma_sq().to_bits(), 0.25_f64.to_bits());
    }
}
