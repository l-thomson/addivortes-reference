//! Binary AddiVortes (Albert–Chib probit): {0, 1} labels in, probabilities
//! out through the probit link.
//!
//! The model, top to bottom: latent `zᵢ = f(xᵢ) + εᵢ`, `ε ~ N(0, 1)` with
//! σ² pinned at 1 (the identifiability rule), `yᵢ = 1{zᵢ > 0}` on the
//! centred scale; each sweep redraws the latents truncated to the side the
//! label dictates (Albert & Chib 1993), and the cell-value prior widens to
//! the ±3 latent range (σ_μ = 3/(k√m)).
//!
//! Two fit paths live here on purpose (OSS-1 spike):
//! - [`fit`] — the shipped path, delegating to the engine's family wiring.
//! - [`fit_via_loop`] — the same model written as an *outer driver* over the
//!   public verbs (`set_response` / `step`), with the Albert–Chib draw in
//!   this file. This is the seam probe: everything probit-specific must be
//!   expressible with the verbs plus config, or the gap is a finding.

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

/// The loop-driven fit's output. A model file cannot yet package its draws
/// as a `FittedAddiVortes` (no crate-internal constructor — the missing
/// "keep" verb; recorded as a spike finding), so the probe returns the raw
/// pieces the distributional comparison needs.
pub(crate) struct LoopProbitFit {
    /// σ² per kept draw — must be exactly 1.0 throughout (the pinned rule).
    pub sigma_sqs: Vec<f64>,
    /// Ensemble fit per training row (response scale) at the final kept
    /// sweep: the latent mean, comparable across paths by sign/rank.
    pub fitted: Vec<f64>,
}

/// Binary AddiVortes as an outer driver over the public verbs — the
/// Albert–Chib latent draw lives HERE, not in the engine.
///
/// Correspondences that make this the same model as [`fit`]:
/// - **σ ≡ 1:** labels {0, 1} span a response range of exactly 1, so the
///   scaler's affine map has unit slope and "sd 1 on the response scale" is
///   "sd 1 in scaled space" — `PinnedSigma::unit()` plus unit-sd latent
///   draws reproduce the family path's pinned scale.
/// - **Threshold:** scaled space centres the labels at ±0.5, so the latent
///   sign rule `z_scaled > 0` is `z > 0.5` on the response scale.
/// - **Prior widening:** the family path sets σ_μ = 3/(k√m); the only
///   config knob reaching σ_μ is `k` (σ_μ = 0.5/(k√m)), so k → k/6
///   expresses the widening exactly. (Seam friction, recorded: the latent
///   σ_μ rule is reachable only through this k-rescale.)
pub(crate) fn fit_via_loop(
    config: AddiVortesConfig,
    x: &Data,
    y: &[f64],
    caller_rng: &mut dyn rand_core::Rng,
) -> Result<LoopProbitFit> {
    check_labels(y)?;
    let burn_in = config.burn_in;
    let n_draws = config.n_draws;
    let widened = config.k / 6.0;
    let mut sampler =
        Sampler::new(config.with_k(widened), x, y)?.with_scale_model(PinnedSigma::unit());

    let mut sigma_sqs = Vec::with_capacity(n_draws);
    let mut fitted = Vec::new();
    for sweep in 0..burn_in + n_draws {
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
        if sweep >= burn_in {
            sigma_sqs.push(draw.sigma_sq);
        }
    }
    fitted.extend(sampler.fitted_values());
    Ok(LoopProbitFit { sigma_sqs, fitted })
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

    /// The seam probe: the loop-driven probit, with the augmentation in this
    /// file, agrees with the family path in distribution (never in bits —
    /// different RNG streams by construction).
    #[test]
    fn loop_driven_probit_agrees_with_the_family_path() {
        let (x, y) = toy();

        // Family path: predictions are P(Y = 1 | x) through the link.
        let family = fit(cfg(5), &x, &y).unwrap();
        let p = family.predict(&x).unwrap();

        // Loop path: caller owns the latent stream.
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let loop_fit = fit_via_loop(cfg(5), &x, &y, &mut rng).unwrap();

        // The pinned rule held on every kept draw, exactly.
        assert!(loop_fit.sigma_sqs.iter().all(|&s| s == 1.0));

        // Both paths separate the clusters...
        let mean =
            |rows: &[usize], v: &[f64]| rows.iter().map(|&i| v[i]).sum::<f64>() / rows.len() as f64;
        let ones: Vec<usize> = (0..60).filter(|&i| y[i] == 1.0).collect();
        let zeros: Vec<usize> = (0..60).filter(|&i| y[i] == 0.0).collect();
        assert!(mean(&ones, &p) > mean(&zeros, &p));
        assert!(mean(&ones, &loop_fit.fitted) > mean(&zeros, &loop_fit.fitted));

        // ...and classify alike: family threshold p > 0.5 vs loop latent
        // mean above the response-scale threshold 0.5.
        let agree = (0..60)
            .filter(|&i| (p[i] > 0.5) == (loop_fit.fitted[i] > 0.5))
            .count();
        assert!(
            agree >= 54,
            "family and loop-driven probit classify {agree}/60 alike"
        );
    }
}
