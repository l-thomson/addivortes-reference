//! Binary AddiVortes (Albert–Chib probit): {0, 1} labels in, probabilities
//! out through the probit link.
//!
//! The model, top to bottom: latent `zᵢ = f(xᵢ) + εᵢ`, `ε ~ N(0, 1)` with
//! σ² pinned at 1 (the identifiability rule), `yᵢ = 1{zᵢ > 0}` on the
//! centred scale; each sweep redraws the latents truncated to the side the
//! label dictates (Albert & Chib 1993), and the cell-value prior widens to
//! the ±3 latent range (σ_μ = 3/(k√m)).
//!
//! Two fit paths live here on purpose:
//! - [`fit`]: the shipped path, delegating to the engine's family wiring.
//! - [`fit_via_loop`]: the same model written as an *outer driver* over the
//!   public verbs (`set_response` / `step`) plus the crate-internal keep
//!   verb ([`FittedAddiVortes::from_parts`]) and the cell-prior width dial
//!   (`with_cell_prior_sd`), with the Albert–Chib draw in this file. This
//!   keeps the seam honest: everything probit-specific must stay expressible
//!   outside the engine, and a regression here means the seam has closed.

use rand_distr::Distribution;
use rand_distr::StandardNormal;

use crate::engine::config::AddiVortesConfig;
use crate::engine::data::Data;
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::model::{FittedAddiVortes, ResponseFamily};
use crate::engine::sampler::Sampler;
use crate::extensions::scale::PinnedSigma;

/// Fit Binary AddiVortes through the engine's family path.
pub(crate) fn fit(config: AddiVortesConfig, x: &Data, y: &[f64]) -> Result<FittedAddiVortes> {
    check_labels(y)?;
    config
        .with_response_family(ResponseFamily::BinaryProbit)
        .fit(x, y)
}

/// The model's own data rule: labels are exactly {0, 1}.
fn check_labels(y: &[f64]) -> Result<()> {
    match y.iter().find(|v| **v != 0.0 && **v != 1.0) {
        Some(bad) => Err(AddiVortesError::InvalidHyperparameter {
            name: "response".into(),
            reason: format!("Binary AddiVortes needs {{0, 1}} labels; found {bad}"),
        }),
        None => Ok(()),
    }
}

/// Binary AddiVortes as an outer driver over the public verbs: the
/// Albert–Chib latent draw lives HERE, not in the engine.
///
/// Correspondences that make this the same model as [`fit`]:
/// - **σ ≡ 1:** labels {0, 1} span a response range of exactly 1, so the
///   scaler's affine map has unit slope and "sd 1 on the response scale" is
///   "sd 1 in scaled space"; `PinnedSigma::unit()` plus unit-sd latent
///   draws reproduce the family path's pinned scale.
/// - **Threshold:** scaled space centres the labels at ±0.5, so the latent
///   sign rule `z_scaled > 0` is `z > 0.5` on the response scale.
/// - **Prior widening:** σ_μ = 3/(k√m), said directly through the
///   cell-prior width dial, the same expression the family wiring uses.
/// - **Family:** pinned to Gaussian so the engine contributes no probit
///   wiring of its own (no second augmentation, no link); the packaged
///   model therefore speaks the latent scale, classifying by the
///   response-scale threshold above.
///
/// The kept draws are packaged as an ordinary [`FittedAddiVortes`] through
/// the keep verb; burn-in and thinning follow the config exactly as the
/// engine's own loop does.
///
/// The caller owns the latent stream: `caller_rng` drives the truncated
/// draws, the engine's pinned chain seed drives everything else, so
/// "engine seed + caller stream" names exactly one chain.
pub(crate) fn fit_via_loop(
    config: AddiVortesConfig,
    x: &Data,
    y: &[f64],
    caller_rng: &mut dyn rand_core::Rng,
) -> Result<FittedAddiVortes> {
    check_labels(y)?;
    let burn_in = config.burn_in;
    let n_draws = config.n_draws;
    let thinning = config.thinning;
    let sigma_mu = 3.0 / (config.k * (config.m as f64).sqrt());
    let mut sampler = Sampler::new(
        config
            .with_response_family(ResponseFamily::Gaussian)
            .with_cell_prior_sd(sigma_mu),
        x,
        y,
    )?
    .with_scale_model(PinnedSigma::unit());

    let mut sigma_sqs = Vec::with_capacity(n_draws);
    let mut draws = Vec::with_capacity(n_draws);
    for sweep in 0..burn_in + n_draws * thinning {
        // --- the model's maths: Albert–Chib truncated latents ---
        let f = sampler.fitted_values();
        let z: Vec<f64> = f
            .iter()
            .zip(y)
            .map(|(&mean, &label)| {
                let positive = label > 0.0;
                loop {
                    let eps: f64 = StandardNormal.sample(caller_rng);
                    let candidate = mean + eps;
                    if (candidate > 0.5) == positive {
                        break candidate;
                    }
                }
            })
            .collect();
        // --- the engine's sweep ---
        sampler.set_response(&z)?;
        let draw = sampler.step()?;
        // Keep the last sweep of each post-burn-in `thinning`-sized block.
        if sweep >= burn_in && (sweep - burn_in) % thinning == thinning - 1 {
            sigma_sqs.push(draw.sigma_sq);
            draws.push(draw.tessellations.to_vec());
        }
    }
    FittedAddiVortes::from_parts(sampler, x, y, sigma_sqs, draws)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    /// Two separated clusters: labels follow x with a clean margin, so any
    /// correct probit fit must separate them.
    fn toy() -> (Data, Vec<f64>) {
        let n = 60;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&v| if v > 0.5 { 1.0 } else { 0.0 })
            .collect();
        (Data::new(xs, n, 1).unwrap(), ys)
    }

    fn cfg(seed: u64) -> AddiVortesConfig {
        AddiVortesConfig::new(seed)
            .with_m(20)
            .with_burn_in(60)
            .with_draws(60)
            .with_omega(0.5)
    }

    #[test]
    fn front_is_bit_identical_to_the_config_path() {
        let (x, y) = toy();
        let a = fit(cfg(5), &x, &y).unwrap();
        let b = cfg(5)
            .with_response_family(ResponseFamily::BinaryProbit)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(a.posterior(), b.posterior());
    }

    #[test]
    fn front_rejects_non_labels() {
        let (x, _) = toy();
        let bad = vec![0.5; 60];
        assert!(fit(cfg(5), &x, &bad).is_err());
    }

    /// The seam regression: the loop-driven probit, with the augmentation in
    /// this file, agrees with the family path in distribution (never in
    /// bits; the two paths use different RNG streams by construction).
    #[test]
    fn loop_driven_probit_agrees_with_the_family_path() {
        let (x, y) = toy();

        // Family path: predictions are P(Y = 1 | x) through the link.
        let family = fit(cfg(5), &x, &y).unwrap();
        let p = family.predict(&x).unwrap();

        // Loop path: caller owns the latent stream; the result is an
        // ordinary fitted model on the latent scale.
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let loop_fit = fit_via_loop(cfg(5), &x, &y, &mut rng).unwrap();

        // The pinned rule held on every kept draw, exactly.
        assert!(loop_fit.posterior().sigma_sq().iter().all(|&s| s == 1.0));

        // Latent-scale posterior-mean predictions from the packaged model.
        let latent = loop_fit.predict(&x).unwrap();

        // Both paths separate the clusters...
        let mean =
            |rows: &[usize], v: &[f64]| rows.iter().map(|&i| v[i]).sum::<f64>() / rows.len() as f64;
        let ones: Vec<usize> = (0..60).filter(|&i| y[i] == 1.0).collect();
        let zeros: Vec<usize> = (0..60).filter(|&i| y[i] == 0.0).collect();
        assert!(mean(&ones, &p) > mean(&zeros, &p));
        assert!(mean(&ones, &latent) > mean(&zeros, &latent));

        // ...and classify alike: family threshold p > 0.5 vs loop latent
        // mean above the response-scale threshold 0.5.
        let agree = (0..60)
            .filter(|&i| (p[i] > 0.5) == (latent[i] > 0.5))
            .count();
        assert!(
            agree >= 54,
            "family and loop-driven probit classify {agree}/60 alike"
        );
    }
}
