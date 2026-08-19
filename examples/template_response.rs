//! Template for a custom response family via `ResponseModel`:
//! copy this file, rename the step, fill the marked blocks, and run:
//!
//! ```sh
//! cargo run --example template_response
//! ```
//!
//! A `ResponseModel` runs once per sweep, first in the hook order, and restores
//! the Gaussian working form the engine requires: it fills `working` (the
//! response the conjugate kernel sees this sweep) and `weights`
//! (per-observation accumulation weights), so that
//! `workingᵢ | μ ~ N(μ, σ²/weightsᵢ)`. **The scope rule:** a response family
//! fits this extension point iff a data-augmentation step restores that form:
//! Albert–Chib probit latents, Pólya–Gamma logit, Kozumi–Kobayashi quantile,
//! robust-t scale mixtures. No such augmentation ⇒ out of scope, on purpose.
//!
//! Two pairing rules the type system cannot enforce for you:
//!
//! - Weights ≠ 1 need a weight-aware mean family. The default
//!   `GaussianCellModel` assumes unit weights; pair any weight-producing step
//!   with `WeightedGaussianModel` (hard membership).
//! - Weights ≠ 1 need a consistent σ² treatment. The default
//!   `GlobalSigma` prices an unweighted RSS. Either pin σ²
//!   (`PinnedSigma`, as probit does) or supply a `ScaleModel` whose update
//!   matches your weights (see `examples/template_scale.rs`).
//!
//! `check_response_model` proves the mechanical contract (every working entry
//! written and finite, weights strictly positive, the augmentation
//! deterministic) and reports which pairing rule your weights put you under.
//! It cannot prove the step statistically valid: that is exactly what the
//! Geweke/SBC battery establishes (`calibration::getting_it_right`;
//! `tests/calibration_acceptance.rs` validates the probit assembly this way,
//! against the public surface alone).

use addivortes::cell_model::WeightedGaussianModel;
use addivortes::response::ResponseModel;
use addivortes::scale::PinnedSigma;
use addivortes::{AddiVortesConfig, Data, conformance};
use rand_core::SeedableRng;

/// A worked example with real content but no latent draws: each observed
/// response is the mean of a known number of replicates, so observation
/// i carries variance σ²/nᵢ: the working response is the response itself
/// and the weight is the replicate count. (A latent-variable family such as
/// probit or robust-t would draw its latents here from `rng` instead.)
#[derive(Debug, Clone)]
struct ReplicateMeans {
    /// Replicate counts nᵢ (dimensionless weights, ascending index).
    counts: Vec<f64>,
}

impl ResponseModel for ReplicateMeans {
    /// No failure mode here; a fallible step names a real error type and it
    /// surfaces as `AddiVortesError::Extension`.
    type Error = std::convert::Infallible;

    fn augment(
        &mut self,
        y: &[f64],
        _fit: &[f64],
        _sigma_sq: f64,
        _rng: &mut dyn rand_core::Rng,
        working: &mut [f64],
        weights: &mut [f64],
    ) -> std::result::Result<(), Self::Error> {
        // ----- your augmentation here ---------------------------------------
        // `y` is the scaled response (the engine's internal space); `_fit` is
        // the current ensemble fit and `_sigma_sq` the previous sweep's σ²:
        // a latent-variable step reads both (probit draws truncated normals
        // around `fit`; robust-t draws mixture scales from the residuals).
        // Every value written must be finite; weights strictly positive.
        // A deterministic step must consume no RNG (the reproducibility
        // contract); a stochastic step takes all its randomness from `rng`.
        working.copy_from_slice(y);
        weights.copy_from_slice(&self.counts);
        Ok(())
        // --------------------------------------------------------------------
    }
}

fn main() -> addivortes::Result<()> {
    // A design where the first half of the responses average 4 replicates
    // and the second half is single-shot: the fit should trust the first
    // half four times as much.
    let n = 40;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
    let counts: Vec<f64> = (0..n).map(|i| if i < n / 2 { 4.0 } else { 1.0 }).collect();
    let x = Data::new(xs, n, 1)?;

    // 1. The one-command check, on a small fixture. Watch `weights_valid`: it
    //    reports that this step's weights are not all 1, which is what puts it
    //    under both pairing rules below.
    let fixture_y = [0.4, -0.2, 0.5, 0.1, -0.3, 0.25];
    let fixture_fit = [0.1, -0.1, 0.2, 0.05, -0.2, 0.1];
    let results = conformance::check_response_model(
        || ReplicateMeans {
            counts: vec![4.0, 4.0, 4.0, 1.0, 1.0, 1.0],
        },
        &fixture_y,
        &fixture_fit,
        0.04,
        &mut rand_chacha::ChaCha8Rng::seed_from_u64(7),
        &mut rand_chacha::ChaCha8Rng::seed_from_u64(7),
    );
    if !conformance::report(&results) {
        std::process::exit(1);
    }

    // 2. Fit-time assembly: the step plus the two pairing rules from the module
    // docs: a weight-aware mean family, and a σ² treatment consistent with
    // the weights (pinned here: the per-replicate noise is taken as known,
    // expressed in the engine's scaled space).
    let m = 10;
    let k = 3.0;
    let sigma_mu = 0.5 / (k * (m as f64).sqrt());
    let model = AddiVortesConfig::new(42)
        .with_m(m)
        .with_burn_in(20)
        .with_draws(30)
        .with_response_model(ReplicateMeans { counts })
        .with_cell_model(WeightedGaussianModel::new(sigma_mu * sigma_mu)?)
        .with_scale_model(PinnedSigma::new(0.04)?)
        .fit(&x, &y)?;

    println!(
        "template_response: replicate-weighted fit RMSE {:.4}",
        model.in_sample_rmse()
    );
    println!(
        "next step for a real family: validate through the battery \
         (calibration::getting_it_right; see tests/calibration_acceptance.rs \
         for the worked probit assembly)"
    );
    Ok(())
}
