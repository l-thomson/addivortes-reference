//! The precision-weighted global σ² Gibbs draw: the scale model that pairs
//! with any weight-producing response step (robust-t, and any custom
//! `ResponseModel` that fills `weights`).

use crate::engine::error::{Result, require_non_negative_finite, require_positive_finite};
use crate::extensions::scale::{ScaleCtx, ScaleModel, sigma_sq_gamma_params};

/// The precision-weighted global σ² Gibbs draw: identical to
/// [`GlobalSigma`](crate::extensions::scale::GlobalSigma) except the RSS is
/// weighted by this sweep's response-side augmentation weights,
/// `σ² | rest ~ IG((ν+n)/2, (νλ + Σᵢ wᵢeᵢ²)/2)`: the correct full
/// conditional whenever the working likelihood is
/// `yᵢ | μ ~ N(μ, σ²/wᵢ)`. With no kernel step attached the weights are
/// unit and the draw coincides with the unweighted one.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedGlobalSigma {
    nu: f64,
    lambda: f64,
    sigma_sq: f64,
}

impl WeightedGlobalSigma {
    /// A weighted global-σ² model with prior degrees of freedom ν and
    /// calibrated λ, exactly as
    /// [`GlobalSigma::new`](crate::extensions::scale::GlobalSigma::new). σ² starts
    /// at 1.0: never read before the first update. Fails with
    /// [`AddiVortesError::InvalidHyperparameter`](crate::AddiVortesError::InvalidHyperparameter)
    /// unless ν is finite and strictly positive and λ is finite and
    /// non-negative.
    pub fn new(nu: f64, lambda: f64) -> Result<Self> {
        Ok(Self {
            nu: require_positive_finite("nu", nu)?,
            lambda: require_non_negative_finite("lambda", lambda)?,
            sigma_sq: 1.0,
        })
    }
}

impl ScaleModel for WeightedGlobalSigma {
    type Error = std::convert::Infallible;

    fn update(
        &mut self,
        ctx: &ScaleCtx<'_>,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        let (y, fit) = (ctx.y(), ctx.fit());
        let mut rss = 0.0_f64;
        match ctx.response_weights() {
            Some(weights) => {
                for ((observed, fitted), weight) in y.iter().zip(fit).zip(weights) {
                    let residual = observed - fitted;
                    rss += weight * residual * residual;
                }
            }
            None => {
                for (observed, fitted) in y.iter().zip(fit) {
                    let residual = observed - fitted;
                    rss += residual * residual;
                }
            }
        }
        let (shape, scale) = sigma_sq_gamma_params(self.nu, self.lambda, rss, y.len());
        let gamma = rand_distr::Gamma::new(shape, scale)
            .expect("shape and scale are positive by construction");
        let precision: f64 = rand_distr::Distribution::sample(&gamma, rng);
        self.sigma_sq = 1.0 / precision;
        Ok(())
    }

    fn sigma_sq(&self) -> f64 {
        self.sigma_sq
    }
}
