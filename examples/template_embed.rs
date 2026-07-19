//! The embed pattern, worked end-to-end: run the
//! AddiVortes engine as one conditional inside your own outer Gibbs
//! sampler, keeping the novel block (here a right-censoring latent draw)
//! entirely outside the crate, with zero engine edits.
//!
//! ```sh
//! cargo run --example template_embed
//! ```
//!
//! The model is Tobit-style censored regression: `y* = f(x) + ε`,
//! `ε ~ N(0, σ²)`, but only `y = min(y*, c)` is observed. The outer Gibbs
//! scan alternates two conditionals:
//!
//! 1. **latents | model** (this file's block): for each censored row draw
//!    `y*ᵢ ~ N(Fᵢ, σ²)` truncated to `(c, ∞)`, from the caller's own RNG;
//! 2. **model | latents** (the engine): [`Sampler::set_response`] feeds the
//!    completed response in, [`Sampler::step`] runs one full engine sweep.
//!
//! The two rules that make the pattern reproducible:
//!
//! - **Scale contract**: `set_response` takes values on the caller's response
//!   scale (the scale of the `y` given to `Sampler::new`) and scales them
//!   internally; `fitted_values()` reads the current fit back on the same
//!   scale. The caller never sees the internal scaled space; latents above
//!   the construction-time response range are fine (never clamped).
//! - The caller owns its RNG: every latent draw below comes from
//!   `caller_rng`, a stream this crate never touches; the engine's own
//!   pinned stream is derived from the config seed alone. Together
//!   "engine seed + caller stream" names exactly one chain.

use addivortes::{AddiVortesConfig, Data, Result, Sampler};
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// One standard-normal draw from the caller's stream.
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    StandardNormal.sample(rng)
}

/// A 53-bit uniform in [0, 1) from the caller's stream.
fn uniform(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// `N(mean, sd²)` truncated to `(lower, ∞)` by rejection, fine here because
/// the fit hovers near the censoring threshold (acceptance ≳ 0.5); a real
/// extension would use an inverse-CDF draw.
fn truncated_normal_above(mean: f64, sd: f64, lower: f64, rng: &mut ChaCha8Rng) -> f64 {
    loop {
        let candidate = mean + sd * standard_normal(rng);
        if candidate > lower {
            return candidate;
        }
    }
}

fn main() -> Result<()> {
    // --- the censored data (truth f(x) = 3x, censoring threshold c = 2) ---
    let n = 60;
    let threshold = 2.0;
    let noise_sd = 0.3;
    let mut data_rng = ChaCha8Rng::seed_from_u64(7);
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let latent_truth: Vec<f64> = xs.iter().map(|&v| 3.0 * v).collect();
    let observed: Vec<f64> = latent_truth
        .iter()
        .map(|&f| (f + noise_sd * standard_normal(&mut data_rng)).min(threshold))
        .collect();
    let censored: Vec<bool> = observed.iter().map(|&y| y >= threshold).collect();
    let x = Data::new(xs.clone(), n, 1)?;
    println!(
        "censored regression: n = {n}, {} rows censored at c = {threshold}",
        censored.iter().filter(|&&c| c).count()
    );

    // --- the engine, constructed once on the completed-at-threshold response.
    // The response transform (and the σ² prior calibration) freeze here.
    let config = AddiVortesConfig::new(42).with_m(30).with_omega(0.5);
    let mut sampler = Sampler::new(config, &x, &observed)?;
    // The scale bridge for σ: scaled space → response scale.
    let y_range = sampler.scaler().y_max() - sampler.scaler().y_min();

    // The caller's stream, entirely separate from the engine's pinned RNG.
    let mut caller_rng = ChaCha8Rng::seed_from_u64(2027);

    // --- the outer Gibbs scan ---
    let burn_in = 150;
    let draws = 150;
    let mut y_completed = observed.clone();
    let mut sigma_sq_scaled = sampler.step()?.sigma_sq; // prime σ² with one sweep
    let mut posterior_mean_fit = vec![0.0_f64; n];
    for sweep in 0..(burn_in + draws) {
        // 1. latents | model: caller-side block, caller's RNG, caller's scale.
        let fit = sampler.fitted_values();
        let sigma = sigma_sq_scaled.sqrt() * y_range;
        for i in 0..n {
            if censored[i] {
                y_completed[i] = truncated_normal_above(fit[i], sigma, threshold, &mut caller_rng);
            }
        }
        // 2. model | latents: one engine sweep on the completed response.
        sampler.set_response(&y_completed)?;
        let draw = sampler.step()?;
        sigma_sq_scaled = draw.sigma_sq;
        if sweep >= burn_in {
            for (total, value) in posterior_mean_fit.iter_mut().zip(sampler.fitted_values()) {
                *total += value / draws as f64;
            }
        }
        // A stray uniform draw between sweeps: the caller's stream is free to
        // do anything; the engine chain depends only on what enters
        // set_response.
        let _ = uniform(&mut caller_rng);
    }

    // --- the censored tail is recovered above the threshold ---
    println!(" x      truth   observed  posterior fit");
    for i in (0..n).step_by(12).chain([n - 1]) {
        println!(
            " {:>4.2}   {:>5.2}   {:>6.2}    {:>6.2}{}",
            xs[i],
            latent_truth[i],
            observed[i],
            posterior_mean_fit[i],
            if censored[i] { "  (censored)" } else { "" }
        );
    }
    let fit_at_top = posterior_mean_fit[n - 1];
    assert!(
        fit_at_top > threshold,
        "the fitted latent surface must rise above the censoring threshold \
         at x = 1 (truth 3.0), got {fit_at_top}"
    );
    println!("fit at x = 1: {fit_at_top:.2} (truth 3.0, threshold {threshold}): ok");
    Ok(())
}
