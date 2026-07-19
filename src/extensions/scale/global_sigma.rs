//! The global σ² Gibbs draw: the paper's scale model, and the fit-time default.

use crate::extensions::scale::{ScaleCtx, ScaleModel, sigma_sq_gamma_params};

/// The built-in scale model: the global σ² Gibbs draw,
/// IG((ν+n)/2, (νλ+RSS)/2), with the RSS reduced in ascending index (pinned).
/// Bit-identical to the pre-seam sampler's inline draw.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalSigma {
    nu: f64,
    lambda: f64,
    sigma_sq: f64,
}

impl GlobalSigma {
    /// A global-σ² model with prior degrees of freedom ν and calibrated λ.
    /// σ² starts at 1.0: never read before the first update.
    pub fn new(nu: f64, lambda: f64) -> Self {
        Self {
            nu,
            lambda,
            sigma_sq: 1.0,
        }
    }
}

impl ScaleModel for GlobalSigma {
    type Error = std::convert::Infallible;

    fn update(
        &mut self,
        ctx: &ScaleCtx<'_>,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        // Reads exactly (y, fit) from the context: the same values, the same
        // ascending-index RSS reduction and the same single Gamma draw as the
        // pre-context trait, so the rebuild is chain-identical (the golden
        // chain is the proof).
        let (y, fit) = (ctx.y(), ctx.fit());
        let mut rss = 0.0_f64;
        for (observed, fitted) in y.iter().zip(fit) {
            let residual = observed - fitted;
            rss += residual * residual;
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
