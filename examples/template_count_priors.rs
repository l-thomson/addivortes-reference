//! Template for custom count priors (the `count_priors` point), copy this
//! file, swap the two ratio hooks, and run:
//!
//! ```sh
//! cargo run --example template_count_priors
//! ```
//!
//! The count priors are the cell-count prior P(b) and the dimension-count
//! prior P(d). They factor cleanly out of every move's acceptance ratio, so
//! they are supplied *once*, here, and every move prices structures through
//! them, **a custom count prior never touches a move**. Moves that leave a
//! count unchanged need no hook call at all: P(b)/P(b) = 1 under any prior.
//!
//! Contract: each hook returns the log prior ratio between two **adjacent**
//! counts, evaluated at the larger one. Both must be pure and deterministic,
//! and any transcendentals must route through `addivortes::mathsfn`, std
//! float results are platform-dependent and break bit-exact reproducibility.
//!
//! Select one on the config like any other point:
//! `AddiVortesConfig::new(seed).with_count_priors(my_prior)`, then `.fit()` as
//! normal (part 6 below). `ModelCtx::with_count_priors` overrides a single
//! context directly, which is what the checks below and the move oracles use.
//!
//! `ModelCtx`'s parameters are readable from outside the crate (`lambda_c`,
//! `omega`, `p`, `sigma_sq` and the rest are public fields; `#[non_exhaustive]`
//! blocks construction, not reads), so a prior may either take them from the
//! context or carry its own, as `Geometric` does here.

use addivortes::{
    AddiVortesConfig, CoordinateDistribution, CountPriors, Data, EuclideanNormal, ModelCtx,
    ShiftedPoissonBinomial, conformance, mathsfn,
};
use std::sync::Arc;

// ----- your count priors here ---------------------------------------------
/// A geometric cell-count prior: P(b) ∝ (1−q)^(b−1), so the ratio
/// P(b)/P(b−1) = (1−q) is *constant* in b: a flatter tail than the paper's
/// shifted Poisson, which penalises each extra cell more sharply as b grows.
/// The dimension prior is left at the paper's shifted Binomial by delegating.
#[derive(Debug)]
struct Geometric {
    q: f64,
    /// Delegate for the hook we are not changing.
    fallback: ShiftedPoissonBinomial,
}

impl CountPriors for Geometric {
    fn log_cell_count_ratio(&self, _b: usize, _ctx: &ModelCtx) -> f64 {
        // ln P(b) − ln P(b−1) = ln(1 − q), independent of b.
        // Route every transcendental through mathsfn: never f64::ln.
        mathsfn::ln(1.0 - self.q)
    }

    fn log_dim_count_ratio(&self, d: usize, ctx: &ModelCtx) -> f64 {
        self.fallback.log_dim_count_ratio(d, ctx)
    }
}
// ---------------------------------------------------------------------------

fn main() -> addivortes::Result<()> {
    let priors = Geometric {
        q: 0.3,
        fallback: ShiftedPoissonBinomial,
    };

    // A ModelCtx carries the priors the moves price through. The sampler builds
    // one per sweep; here we build one directly, to inspect the pricing before
    // committing it to a fit.
    let law = Arc::new(EuclideanNormal::new(0.8)?);
    let coord_dists: Vec<Arc<dyn CoordinateDistribution>> =
        (0..5).map(|_| Arc::clone(&law) as Arc<_>).collect();
    let weights = vec![1.0; 5];
    let ctx =
        ModelCtx::new(1.0, 2.0, 25.0, 0.01, 5, &coord_dists, &weights).with_count_priors(&priors);

    // 1. The one-command check: both hooks finite and pure over every adjacent
    //    count a sampler will ask about, plus a portability digest. It has no
    //    oracle: any finite, pure hook is the exact adjacent log-ratio of *some*
    //    positive sequence, so there is nothing here to contradict. An inverted
    //    ratio passes this check.
    if !conformance::report(&conformance::check_count_priors(&priors, &ctx, 40)) {
        std::process::exit(1);
    }

    // 1b. THE CHECK THAT ACTUALLY TESTS YOUR DERIVATION. Write the prior a second
    //     time, as a density, and let the crate confirm the hooks are its adjacent
    //     differences. You have already written this density down — it is the line
    //     in your paper above the derivation — and deriving the ratio from it is
    //     the step where the mistake happens.
    //
    //     Do not skip this because the check above is green. It is the only check
    //     on this extension point whose reference does not come from the code being tested.
    //     (What it cannot do: if the *density* is what you got wrong, both sides
    //     are wrong together and it passes. It checks your algebra, not your model.)
    let log_cell_pmf = |b: usize| {
        // Geometric: P(b) ∝ (1−q)^(b−1), unnormalised (the constant cancels).
        (b - 1) as f64 * mathsfn::ln(1.0 - priors.q)
    };
    let log_dim_pmf = |d: usize| {
        // Still the paper's shifted Binomial: d − 1 ~ Binomial(p−1, ω/p).
        let p = ctx.p;
        let theta = ctx.omega / p as f64;
        let ln_choose: f64 = (1..d).map(|k| mathsfn::ln((p - k) as f64)).sum::<f64>()
            - (1..d).map(|k| mathsfn::ln(k as f64)).sum::<f64>();
        ln_choose + (d - 1) as f64 * mathsfn::ln(theta) + (p - d) as f64 * mathsfn::ln(1.0 - theta)
    };
    if !conformance::report(&conformance::check_count_priors_against_density(
        &priors,
        &ctx,
        &log_cell_pmf,
        &log_dim_pmf,
        40,
    )) {
        std::process::exit(1);
    }

    // 2. The cell ratio is what we changed: constant ln(0.7) at every count.
    let expected = mathsfn::ln(0.7);
    for b in [2usize, 5, 40] {
        let got = ctx.log_cell_count_ratio(b);
        assert!(
            (got - expected).abs() < 1e-12,
            "cell ratio at b={b}: {got} != {expected}"
        );
    }

    // 3. The dimension ratio still comes from the delegated paper prior, and
    //    is strictly decreasing in d (each extra dimension costs more).
    let d2 = ctx.log_dim_count_ratio(2);
    let d4 = ctx.log_dim_count_ratio(4);
    assert!(d4 < d2, "shifted Binomial should penalise larger d");

    // 4. Purity/determinism: the same query must give the same answer.
    assert_eq!(ctx.log_cell_count_ratio(7), ctx.log_cell_count_ratio(7));

    // 5. Without an override you get the paper's prior, and the two have
    //    genuinely different shapes. The shifted Poisson's ratio is
    //    ln(λ_c / (b−1)): *decreasing* in b, so it favours growth while b is
    //    below λ_c and penalises it sharply above. The geometric ratio is flat.
    //    (The paper's λ_c = 25 is pinned explicitly here; at b = 5 its
    //    Poisson ratio is still positive.)
    let paper_ctx = ModelCtx::new(1.0, 2.0, 25.0, 0.01, 5, &coord_dists, &weights);
    let paper_small = paper_ctx.log_cell_count_ratio(5);
    let paper_large = paper_ctx.log_cell_count_ratio(40);
    assert!(
        paper_large < paper_small,
        "the shifted Poisson ratio must decrease in b"
    );
    assert!(
        paper_large < expected && expected < paper_small,
        "the flat geometric tail should cross the Poisson's between b=5 and b=40"
    );

    // 6. Fit-time selection: one call on the config, and every count-changing
    //    move prices through your hooks.
    //
    //    Read the printed cell counts against the ratios above, they explain
    //    each other. Both fits pin the paper's lambda_c = 25 (at the shipped
    //    default of 5 the two priors nearly coincide over the visited range,
    //    which would blunt the contrast). The geometric ratio is a flat
    //    ln(0.7) = -0.36: every extra cell costs the same, from the first
    //    one. The shifted-Poisson ratio is ln(lambda_c/(b-1)), which is
    //    *positive* while b is below lambda_c (+1.83 at b = 5), so it
    //    actively rewards growth until the tessellation approaches
    //    lambda_c = 25. The geometric prior is only the cheaper of the two
    //    beyond b = 40, and the chain never goes there. So the geometric fit
    //    comes out *smaller*, not larger: a flat penalty is a harsher regime
    //    than a Poisson one anywhere below its mean.
    let n = 60;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|v| 3.0 * v * v - v).collect();
    let x = Data::new(xs, n, 1)?;
    let config = || {
        AddiVortesConfig::new(42)
            .with_m(4)
            .with_burn_in(30)
            .with_draws(40)
            .with_lambda_c(25.0)
    };
    let paper_fit = config().fit(&x, &y)?;
    let geometric_fit = config()
        .with_count_priors(Geometric {
            q: 0.3,
            fallback: ShiftedPoissonBinomial,
        })
        .fit(&x, &y)?;

    let mean_cells = |model: &addivortes::FittedAddiVortes| {
        let posterior = model.posterior();
        (0..posterior.n_draws())
            .map(|d| {
                posterior
                    .tessellations(d)
                    .iter()
                    .map(|t| t.n_cells() as f64)
                    .sum::<f64>()
            })
            .sum::<f64>()
            / posterior.n_draws() as f64
    };

    println!(
        "template_count_priors: all checks passed; \
         geometric cell ratio {expected:.4} (flat) vs paper {paper_small:.4} (b=5) \
         -> {paper_large:.4} (b=40); dim ratio {d2:.4} -> {d4:.4}"
    );
    println!(
        "fitted: paper prior {:.2} cells/draw, geometric prior {:.2} cells/draw",
        mean_cells(&paper_fit),
        mean_cells(&geometric_fit)
    );
    Ok(())
}
