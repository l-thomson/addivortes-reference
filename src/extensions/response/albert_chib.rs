//! The Albert–Chib probit augmentation, as one shelf entry (the
//! Binary-AddiVortes fit side): once per sweep, draw the latent
//! `zᵢ ~ N(Fᵢ, 1)` truncated to the side the label dictates, restoring the
//! Gaussian working form on the σ ≡ 1 latent scale. Attached automatically
//! by `ResponseFamily::BinaryProbit` (with [`PinnedSigma::unit`] as the
//! scale); pairs with the predict-side probit link (`FittedAddiVortes`
//! predictions on the probability scale).
//!
//! [`PinnedSigma::unit`]: crate::extensions::scale::PinnedSigma::unit

use crate::extensions::response::ResponseModel;

/// The stateless Albert–Chib (1993) step: labels are read from the observed
/// response each sweep (`label = yᵢ > 0`; binary {0, 1} responses scale to
/// ±0.5, so the threshold is exact), and the latents are drawn by rejection
/// (adequate at the |F| ranges the prior admits; a far-tail variant would
/// use an inverse-CDF draw). Being stateless it composes with
/// [`Sampler::set_response`](crate::Sampler::set_response): an outer driver
/// can regenerate labels between sweeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AlbertChibProbit;

impl ResponseModel for AlbertChibProbit {
    type Error = std::convert::Infallible;

    fn augment(
        &mut self,
        y: &[f64],
        fit: &[f64],
        _sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
        working: &mut [f64],
        weights: &mut [f64],
    ) -> std::result::Result<(), Self::Error> {
        for i in 0..working.len() {
            let label = y[i] > 0.0;
            let z = loop {
                let draw: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
                let candidate = fit[i] + draw;
                if (candidate > 0.0) == label {
                    break candidate;
                }
            };
            working[i] = z;
            weights[i] = 1.0;
        }
        Ok(())
    }
}
