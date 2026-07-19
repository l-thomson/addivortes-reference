//! Template for a custom variable-inclusion model: copy this
//! file, rename the model, fill the marked blocks, and run:
//!
//! ```sh
//! cargo run --example template_inclusion
//! ```
//!
//! An inclusion model supplies the per-covariate weights the built-in
//! AD/RD/Swap moves read when choosing which covariate enters or leaves a
//! tessellation, and may adapt them once per sweep. Weights index the
//! caller-visible pre-encoding columns, are relative (need not sum to 1),
//! and must stay finite and strictly positive.
//!
//! **The adaptive-update exactness warning (read before writing an adaptive
//! model):** AddiVortes dimension sets are distinct subsets, not
//! with-replacement draws, so DART's conjugate Dirichlet Gibbs update is not
//! the true full conditional here (the e_d(s) normalisers do not cancel). An
//! adaptive `update` must carry its own correction: the recommended pattern
//! is Metropolis–Hastings with proposal s′ ~ Dirichlet(α/p + u), or the
//! sampler it produces is invalid. The conformance checks below cover the mechanical
//! contracts only; statistical validity of an adaptive update is established
//! by the SBC battery (`addivortes::calibration`).

use addivortes::{AddiVortesConfig, Data, InclusionModel, InclusionUsage, conformance};

/// Fixed domain-knowledge weights: covariates believed more relevant a
/// priori get proportionally more of the moves' attention. Fixed weights
/// need no update and no correction.
#[derive(Debug, Clone)]
struct DomainPrior {
    weights: Vec<f64>,
}

impl InclusionModel for DomainPrior {
    /// No failure mode here; a fallible update names a real error type and
    /// it surfaces as `AddiVortesError::Extension`.
    type Error = std::convert::Infallible;

    fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn update(
        &mut self,
        _usage: &InclusionUsage,
        _rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        // ----- your per-sweep adaptation here (fixed weights: no-op) --------
        // `usage.counts()` holds how many active dimensions currently map to
        // each covariate. A no-op must consume no RNG (the reproducibility
        // contract); an adaptive update takes all randomness from `rng`.
        Ok(())
        // --------------------------------------------------------------------
    }
}

fn main() -> addivortes::Result<()> {
    let make = || DomainPrior {
        weights: vec![3.0, 1.0, 1.0],
    };

    // 1. The one-command conformance check: weight validity + update
    //    determinism. Build the usage the way the sampler does — one
    //    subset size per tessellation, one count per active dimension —
    //    so an adaptive `update` is exercised at a state it actually
    //    sees, not at the all-zero usage where adaptation is a no-op.
    use rand_core::SeedableRng;
    let mut seed_a = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mut seed_b = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mut usage = InclusionUsage::new(3);
    for dims in [&[0usize, 1][..], &[0], &[2, 0]] {
        usage.record_subset_size(dims.len());
        for &covariate in dims {
            usage.record(covariate);
        }
    }
    let results = conformance::check_inclusion_model(make, 3, &usage, &mut seed_a, &mut seed_b);
    if !conformance::report(&results) {
        std::process::exit(1);
    }

    // 2. Fit-time selection: one line on the config.
    let n = 40;
    let mut values = Vec::with_capacity(n * 3);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64 / (n - 1) as f64;
        let b = ((i * 3) % n) as f64 / n as f64;
        let c = ((i * 7) % n) as f64 / n as f64;
        values.extend_from_slice(&[a, b, c]);
        y.push(2.0 * a - 0.5 * b); // covariate 0 carries most of the signal
    }
    let x = Data::new(values, n, 3)?;
    let model = AddiVortesConfig::new(42)
        .with_m(10)
        .with_burn_in(20)
        .with_draws(30)
        .with_omega(1.5) // ω must satisfy ω < p (here p = 3)
        .with_inclusion(make())
        .fit(&x, &y)?;
    println!(
        "template_inclusion: all checks passed; weighted fit RMSE {:.4}",
        model.in_sample_rmse()
    );
    Ok(())
}
