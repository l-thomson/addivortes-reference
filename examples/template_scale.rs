//! Template for a custom scale/precision model: copy this
//! file, rename the model, fill the marked blocks, and run:
//!
//! ```sh
//! cargo run --example template_scale
//! ```
//!
//! A `ScaleModel` owns the noise side of the sweep: one per-sweep `update`
//! given a [`ScaleCtx`] (the working response, the current fit, the scaled
//! design, and the shared backfit machinery, enough for a variance
//! ensemble like H-AddiVortes to run its own tessellations inside
//! `update`), the scalar `sigma_sq()` the kernel reads, and optional
//! per-observation precisions the conductor multiplies into the
//! response-side weights (`wᵢ = wᵢ^resp · wᵢ^scale`).
//!
//! The pairing rule: a precision-supplying model needs a weight-aware mean
//! family: pair it with `WeightedGaussianModel` (hard membership).
//!
//! [`ScaleCtx`]: addivortes::scale::ScaleCtx

use addivortes::cell_model::WeightedGaussianModel;
use addivortes::scale::{ScaleCtx, ScaleModel};
use addivortes::{AddiVortesConfig, Data, conformance};

/// A worked heteroscedastic example: the relative noise profile is known
/// (here, one stretch of the data is twice as noisy), and the overall σ² is
/// still learned via the standard Gibbs draw, with the RSS precision-weighted
/// so it stays consistent with the profile:
/// σ² | rest ~ IG((ν+n)/2, (νλ + Σᵢ wᵢeᵢ²)/2).
#[derive(Debug, Clone)]
struct KnownProfileSigma {
    /// Known relative precisions wᵢ (dimensionless weights, ascending index).
    precisions: Vec<f64>,
    /// The current σ² draw (**scaled space**); never read before the first
    /// `update`.
    sigma_sq: f64,
}

impl KnownProfileSigma {
    fn new(precisions: Vec<f64>) -> Self {
        Self {
            precisions,
            sigma_sq: 1.0,
        }
    }
}

impl ScaleModel for KnownProfileSigma {
    /// No failure mode here; a fallible update names a real error type and
    /// it surfaces as `AddiVortesError::Extension`.
    type Error = std::convert::Infallible;

    fn update(
        &mut self,
        ctx: &ScaleCtx<'_>,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        // ----- your per-sweep scale update here -----------------------------
        // Everything is scaled space, ascending index. The context also
        // carries the design (`ctx.x()`) and the full structural machinery
        // (`ctx.move_set()`, `ctx.assigner()`, `ctx.coord_dists()`,
        // `ctx.inclusion_weights()`, `ctx.omega()`, `ctx.lambda_c()`); a
        // variance ensemble runs its own backfit block with them, exactly as
        // the shelf `HVariance` does. Take all randomness from `rng`.
        let (y, fit) = (ctx.y(), ctx.fit());
        let mut rss = 0.0_f64;
        for ((observed, fitted), w) in y.iter().zip(fit).zip(&self.precisions) {
            let residual = observed - fitted;
            rss += w * residual * residual;
        }
        let shape = 0.5 * (ctx.nu() + y.len() as f64);
        let scale = 2.0 / (ctx.nu() * ctx.calibrated_lambda() + rss);
        let gamma = rand_distr::Gamma::new(shape, scale)
            .expect("shape and scale are positive by construction");
        let precision: f64 = rand_distr::Distribution::sample(&gamma, rng);
        self.sigma_sq = 1.0 / precision;
        Ok(())
        // --------------------------------------------------------------------
    }

    fn sigma_sq(&self) -> f64 {
        // Must be finite and strictly positive after every update
        // (release-asserted by the sampler).
        self.sigma_sq
    }

    fn precisions(&self) -> Option<&[f64]> {
        // `None` (the default) means homoscedastic; the conductor composes
        // nothing. Every supplied value must be finite and strictly positive.
        Some(&self.precisions)
    }
}

fn main() -> addivortes::Result<()> {
    let n = 40;
    let profile: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.5 }).collect();

    // 1. The one-command check: σ² validity, precision well-formedness,
    //    bit-exact update determinism, on a minimal fixture context.
    //
    //    Note what it does NOT check: whether σ² responds to the data at all. A
    //    model that ignores the residuals and returns prior draws passes every
    //    line of this. That is check 1b.
    use rand_core::SeedableRng;
    let make = || KnownProfileSigma::new(profile.clone());
    let y_fixture: Vec<f64> = (0..n).map(|i| 0.4 * (i as f64 / n as f64) - 0.2).collect();
    // The fit is deliberately NON-ZERO. At fit = 0 the residual y − fit is just
    // y, so a model that scores the raw response instead of the residual — a
    // classic slip — is indistinguishable from a correct one, and check 1b below
    // goes blind. Never write a scale fixture with a zero fit.
    let fit_fixture: Vec<f64> = (0..n)
        .map(|i| 0.15 * (i as f64 / n as f64) - 0.05)
        .collect();
    let mut seed_a = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mut seed_b = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let results =
        conformance::check_scale_model(make, &y_fixture, &fit_fixture, &mut seed_a, &mut seed_b);
    if !conformance::report(&results) {
        std::process::exit(1);
    }

    // 1b. Opt in, because this model is meant to LEARN σ² from the data.
    //
    //     Two properties, neither assuming a functional form: inflating the
    //     residuals must raise σ², and shifting y and fit together (which leaves
    //     the residuals alone) must not move σ² at all.
    //
    //     Do NOT call this if your model deliberately pins σ² — the shipped
    //     `PinnedSigma` holds σ² = 1 for probit and is *correct* to ignore the
    //     data. That is exactly why this is a separate, opt-in check rather than
    //     part of the one above: a mandatory probe would fail correct code.
    let results =
        conformance::check_scale_model_learns_from_data(make, &y_fixture, &fit_fixture, 7);
    if !conformance::report(&results) {
        std::process::exit(1);
    }

    // 2. Fit-time selection: the scale model plus the pairing rule (a
    //    weight-aware mean family on the other side of the seam).
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
    let x = Data::new(xs, n, 1)?;
    let m = 10;
    let k = 3.0;
    let sigma_mu = 0.5 / (k * (m as f64).sqrt());
    let model = AddiVortesConfig::new(42)
        .with_m(m)
        .with_burn_in(20)
        .with_draws(30)
        .with_scale_model(make())
        .with_cell_model(WeightedGaussianModel::new(sigma_mu * sigma_mu)?)
        .fit(&x, &y)?;
    println!(
        "template_scale: all checks passed; known-profile fit RMSE {:.4}",
        model.in_sample_rmse()
    );
    Ok(())
}
