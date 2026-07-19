//! The deep-seam extension template: a worked custom `CellModel`,
//! checked by the one-command conjugacy conformance check, then assembled into the
//! probit (Binary-AddiVortes-style) sampler via a `ResponseModel`.
//!
//! The researcher workflow (the same three steps on every extension point):
//!
//! 1. copy this file, rename it, and fill the marked blocks with your maths;
//! 2. run it: the conformance verdicts are printed in plain language;
//! 3. only then run the global gates (golden chain, SBC battery) if the model
//!    is destined for real inference.
//!
//! ```sh
//! cargo run --example template_cell_model
//! ```
//!
//! Scope rule: the seam accepts conditionally-conjugate
//! models only: the kernel must see a Gaussian working likelihood
//! `rᵢ | μ ~ N(μ, σ²/wᵢ)`, restored by a `ResponseModel` when the response
//! family is not Gaussian (probit below). A likelihood with no such
//! augmentation does not fit this seam, by design.

use addivortes::cell_model::{CellModel, CellStats};
use addivortes::conformance;
use addivortes::response::ResponseModel;
use addivortes::scale::PinnedSigma;
use addivortes::{AddiVortesConfig, Data, MoveSetBuilder, Sampler};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

// ---------------------------------------------------------------------------
// 1. Your sufficient statistic: the per-cell running tally.
// ---------------------------------------------------------------------------

/// `>>> YOUR STATS HERE`: whatever per-cell accumulations your marginal
/// likelihood and cell-value draw need. This template keeps the weighted
/// Gaussian pair (Σw, Σw·r): the right choice for any model that reduces to
/// a precision-weighted Gaussian on a working response.
#[derive(Debug, Default, Clone)]
struct TemplateStats {
    weight: f64,
    weighted_sum: f64,
}

impl CellStats for TemplateStats {
    fn record(&mut self, value: f64, weight: f64) {
        // >>> YOUR ACCUMULATION HERE (must be order-free and additive:
        // the conformance sufficiency checks verify both).
        self.weight += weight;
        self.weighted_sum += weight * value;
    }
    fn merge(&mut self, other: &Self) {
        self.weight += other.weight;
        self.weighted_sum += other.weighted_sum;
    }
    fn remove(&mut self, other: &Self) {
        self.weight -= other.weight;
        self.weighted_sum -= other.weighted_sum;
    }
    fn reset(&mut self) {
        *self = Self::default();
    }
    fn occupied(&self) -> bool {
        self.weight > 0.0
    }
}

// ---------------------------------------------------------------------------
// 2. Your cell model: the rule book turning tallies into (a) the integrated
//    marginal likelihood of a candidate cell set and (b) the cell-value draw.
// ---------------------------------------------------------------------------

/// A Gaussian conjugate cell model with cell-value prior N(0, σ_μ²): the
/// worked example. For your own model, replace the two formulas and keep the
/// structure: everything must flow from the accumulated stats alone.
#[derive(Debug)]
struct TemplateCellModel {
    sigma_mu_sq: f64,
}

impl CellModel for TemplateCellModel {
    type Stats = TemplateStats;
    /// Use a real error type if your maths can fail mid-chain; the sampler
    /// surfaces it as `AddiVortesError::Extension`.
    type Error = std::convert::Infallible;

    fn log_marginal_terms(&self, stats: &[Self::Stats], sigma_sq: f64) -> Result<f64, Self::Error> {
        // >>> YOUR MARGINAL LIKELIHOOD HERE, summed over cells in ascending
        // index. Factors identical for any two structures over the same
        // observations may be dropped (they cancel in the acceptance ratio);
        // the per-cell normalising term must not be (the conformance check's
        // Bayes-factor check will catch it if you drop it).
        let mut total = 0.0;
        for cell in stats {
            let denominator = cell.weight * self.sigma_mu_sq + sigma_sq;
            total += 0.5 * addivortes::mathsfn::ln(sigma_sq / denominator);
            total += self.sigma_mu_sq * cell.weighted_sum * cell.weighted_sum
                / (2.0 * sigma_sq * denominator);
        }
        Ok(total)
    }

    fn draw_cell_values(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<Vec<f64>, Self::Error> {
        // >>> YOUR CONJUGATE DRAW HERE, one value per cell in ascending
        // index. On an empty statistic this must sample the PRIOR (that is
        // what conjugacy means at zero data; the conformance check relies on it).
        let mut values = Vec::with_capacity(stats.len());
        for cell in stats {
            let denominator = cell.weight * self.sigma_mu_sq + sigma_sq;
            let mean = self.sigma_mu_sq * cell.weighted_sum / denominator;
            let variance = self.sigma_mu_sq * sigma_sq / denominator;
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
            values.push(mean + variance.sqrt() * z);
        }
        Ok(values)
    }
}

// ---------------------------------------------------------------------------
// 3. (When your response family is not Gaussian) the augmentation step that
//    restores conditional conjugacy: probit via Albert–Chib latents here.
// ---------------------------------------------------------------------------

/// Albert–Chib (1993): z_i ~ N(F_i, 1) truncated to the side the binary label
/// dictates; conditional on z the model is Gaussian on the latent scale with
/// σ² pinned to 1. (Rejection sampling keeps the template dependency-free;
/// an inverse-CDF draw is the production choice.)
#[derive(Debug)]
struct AlbertChibProbit {
    labels: Vec<bool>,
}

impl ResponseModel for AlbertChibProbit {
    type Error = std::convert::Infallible;
    fn augment(
        &mut self,
        _y: &[f64],
        fit: &[f64],
        _sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
        working: &mut [f64],
        weights: &mut [f64],
    ) -> Result<(), Self::Error> {
        for i in 0..working.len() {
            let z = loop {
                let draw: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
                let candidate = fit[i] + draw;
                if (candidate > 0.0) == self.labels[i] {
                    break candidate;
                }
            };
            working[i] = z;
            weights[i] = 1.0;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The walkthrough (CI-gated: must pass end-to-end, unedited)
// ---------------------------------------------------------------------------

fn main() {
    let mut rng = ChaCha8Rng::from_seed([42; 32]);

    // Step 1: the one-command conjugacy conformance check on your cell model. The
    // fixture is scaled-space observations + per-observation weights.
    let model = TemplateCellModel { sigma_mu_sq: 0.02 };
    let observations = [0.31, -0.12, 0.07, 0.22, -0.44, 0.18];
    let weights = [1.0, 0.5, 2.0, 1.0, 1.5, 0.8];
    println!("conjugacy conformance check on the template cell model");
    let results = conformance::check_cell_model(&model, 0.3, &observations, &weights, &mut rng);
    if !conformance::report(&results) {
        eprintln!("template cell model failed its conformance checks");
        std::process::exit(1);
    }

    // Step 2: assemble the probit sampler: the custom cell model, the
    // Albert–Chib augmentation, the pinned scale: with zero sampler edits,
    // and run a short chain on separable synthetic labels.
    let n = 60;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let labels: Vec<bool> = xs.iter().map(|&v| v > 0.5).collect();
    // The sampler still requires a numeric y for scaling; the ResponseModel
    // replaces it with latents on the first sweep, so pass the labels as
    // 0/1 (their scaled values are never used by the probit path).
    let y01: Vec<f64> = labels.iter().map(|&b| f64::from(u8::from(b))).collect();
    let x = Data::new(xs, n, 1).unwrap();

    let mut sampler = Sampler::with_cell_model(
        AddiVortesConfig::new(7).with_m(10),
        &x,
        &y01,
        MoveSetBuilder::stone_gosling().build().unwrap(),
        TemplateCellModel { sigma_mu_sq: 0.02 },
    )
    .unwrap()
    .with_response_model(AlbertChibProbit { labels })
    .with_scale_model(PinnedSigma::unit());

    // Burn, then keep sweeps and check the latent-scale ensemble separates
    // the classes: the summed cell values at x = 0 must sit below those at
    // x = 1 on average (the probit signal).
    for _ in 0..50 {
        sampler.step().unwrap();
    }
    let mut separation = 0.0_f64;
    let kept = 50;
    for _ in 0..kept {
        let draw = sampler.step().unwrap();
        assert_eq!(draw.sigma_sq.to_bits(), 1.0_f64.to_bits());
        let ensemble_at = |value: f64| -> f64 {
            draw.tessellations
                .iter()
                .map(|t| {
                    // One active dim (p = 1): nearest centre by |x − c|.
                    let mut best = 0usize;
                    let mut best_distance = f64::INFINITY;
                    for (cell, chunk) in t.centres().chunks(t.dims().len()).enumerate() {
                        let distance = (value - chunk[0]).abs();
                        if distance < best_distance {
                            best_distance = distance;
                            best = cell;
                        }
                    }
                    t.mus()[best]
                })
                .sum()
        };
        separation += ensemble_at(0.5) - ensemble_at(-0.5); // scaled-space ends
    }
    separation /= kept as f64;
    println!("probit assembly: latent separation (high − low end) = {separation:.2}");
    assert!(
        separation > 0.0,
        "the probit chain failed to separate the classes"
    );
    println!("template cell model: ALL CHECKS PASSED");
}
