//! Template for a custom centre-coordinate law: copy this file,
//! rename the law, fill the marked blocks, and run:
//!
//! ```sh
//! cargo run --example template_coord
//! ```
//!
//! A coordinate law serves as both the centre prior and the within-move
//! proposal. The two must be the same distribution: that coincidence makes
//! the densities cancel in every built-in ratio, exactly as in the paper, so
//! a custom law is valid by construction. The two methods must agree:
//! `sample()` draws from the density `exp(log_density)`. The check's KS
//! check verifies that agreement.
//!
//! Reproducibility rules: route transcendentals through `addivortes::mathsfn`
//! (never `f64::ln`; std float results are platform-dependent), and take all
//! randomness from the `rng` you are handed.

use std::sync::Arc;

use addivortes::{AddiVortesConfig, CoordinateDistribution, Data, conformance, mathsfn};

/// A Laplace (double-exponential) coordinate law: heavier tails than the
/// default normal, so Change/AddCentre proposals make occasional long jumps.
#[derive(Debug)]
struct Laplace {
    /// Scale b (the density is exp(−|x|/b) / 2b).
    b: f64,
}

impl CoordinateDistribution for Laplace {
    fn sample(&self, rng: &mut dyn rand_core::Rng) -> f64 {
        // ----- your sampler here --------------------------------------------
        // Inverse-CDF from one uniform: u < ½ ⇒ b·ln(2u); u ≥ ½ ⇒ −b·ln(2(1−u)).
        // The uniform comes from the top 53 bits of one next_u64 (the same
        // construction the crate pins); the max(tiny) guards ln(0).
        let u = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        if u < 0.5 {
            self.b * mathsfn::ln((2.0 * u).max(1e-300))
        } else {
            -self.b * mathsfn::ln((2.0 * (1.0 - u)).max(1e-300))
        }
        // --------------------------------------------------------------------
    }

    fn log_density(&self, x: f64) -> f64 {
        // ----- your log-density here (must integrate to 1) ------------------
        -x.abs() / self.b - mathsfn::ln(2.0 * self.b)
        // --------------------------------------------------------------------
    }
}

fn main() -> addivortes::Result<()> {
    // 1. The one-command conformance check: sample() vs exp(log_density) KS agreement.
    use rand_core::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let law = Laplace { b: 0.8 };
    let results = conformance::check_coordinate_distribution(&law, 4000, &mut rng);
    if !conformance::report(&results) {
        std::process::exit(1);
    }

    // 2. Fit-time selection: one law per caller-visible column (a one-hot
    //    group shares one law, exactly like `Vec<Metric>`).
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
    let x = Data::new(xs, n, 1)?;
    let model = AddiVortesConfig::new(42)
        .with_m(10)
        .with_burn_in(20)
        .with_draws(30)
        .with_coords(vec![Arc::new(Laplace { b: 0.8 })])
        .fit(&x, &y)?;
    println!(
        "template_coord: all checks passed; Laplace-law fit RMSE {:.4}",
        model.in_sample_rmse()
    );
    Ok(())
}
