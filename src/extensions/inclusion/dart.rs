//! The Metropolis-corrected DART reference, as one shelf entry:
//! DART-style Dirichlet weights (Linero 2018) made exact for AddiVortes.
//!
//! The exactness warning this entry exists to demonstrate (on the trait docs): AddiVortes dimension sets are distinct subsets, so
//! DART's conjugate Gibbs update `s | rest ~ Dirichlet(α/p + u)` is not
//! the true full conditional here: the weighted subset prior
//! `P(D | d, s) ∝ ∏_{k∈D} s_k / e_d(s)` carries elementary-symmetric
//! normalisers that do not cancel across tessellations. The true full
//! conditional is
//!
//! ```text
//! π(s | D₁..D_m) ∝ Dir(s; α/p) · ∏_t ∏_{k∈D_t} s_k / e_{d_t}(s)
//! ```
//!
//! and the fix is the recommended pattern: keep the conjugate Dirichlet as
//! a proposal and correct with the exact Metropolis–Hastings ratio.
//! Under the pure proposal `Dirichlet(α/p + u)` everything except the
//! normalisers would cancel (`log α = Σ_t [ln e_{d_t}(s) − ln e_{d_t}(s′)]`),
//! but that sampler is only exact asymptotically in a way the battery can
//! see: the target's `1/e_d(s)` factors carry more simplex-boundary mass
//! than the conjugate proposal, the importance ratio `π/q` is unbounded
//! there, and the independence chain hangs in long boundary sojourns,
//! measurably so on the Geweke weight statistics at gate sizes. This entry
//! therefore proposes from the defensive mixture
//!
//! ```text
//! q(s′) = ½ Dir(s′; α/p + u) + ½ Dir(s′; α/p)
//! ```
//!
//! whose prior component dominates the target at every boundary face (each
//! subset contributes at least as many vanishing `s_k` factors to π's
//! numerator as its `e_{d_t}` normaliser removes), so `π/q` is bounded and
//! the sampler uniformly ergodic. The cancellation no longer telescopes;
//! the acceptance ratio is evaluated in full:
//!
//! ```text
//! log α = ln π(s′) − ln π(s) + ln q(s) − ln q(s′)
//! ```
//!
//! Only the SBC/Geweke battery can validate an adaptive update; this entry
//! passes it (`stat_gates::geweke_dart`, `stat_gates::sbc_ranks_dart`), and
//! the exact-integration oracle below pins the update's invariant
//! distribution against quadrature on the simplex.

use crate::engine::mathsfn;
use crate::extensions::inclusion::{InclusionModel, InclusionUsage, elementary_symmetric};
use crate::extensions::moves::uniform_f64;

/// The MH-corrected DART inclusion model: weights s ~ Dirichlet(α/p) a
/// priori, adapted once per sweep by an independence-sampler MH step with
/// the defensive-mixture proposal ½ Dir(α/p + u) + ½ Dir(α/p) (u = the
/// sweep's per-covariate usage counts) and the exact subset-prior
/// acceptance ratio above.
///
/// Exact for designs whose encoded columns equal the caller columns (no
/// categorical one-hot expansion): the subset prior operates on encoded
/// columns, and this reference prices its normalisers over the raw weights.
#[derive(Debug, Clone, PartialEq)]
pub struct DartInclusion {
    alpha: f64,
    weights: Vec<f64>,
}

impl DartInclusion {
    /// A DART model over `p` covariates with concentration `alpha` (the
    /// Dirichlet prior is s ~ Dirichlet(α/p, …, α/p); smaller α concentrates
    /// mass on fewer covariates). The chain starts at the uniform weights
    /// (the prior mean).
    pub fn new(alpha: f64, p: usize) -> Self {
        assert!(
            alpha.is_finite() && alpha > 0.0 && p >= 1,
            "alpha must be finite and positive over at least one covariate"
        );
        Self {
            alpha,
            weights: vec![1.0 / p as f64; p],
        }
    }

    /// The Dirichlet concentration α: the prior pseudo-count spread across
    /// the covariate weights (dimensionless).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl InclusionModel for DartInclusion {
    type Error = std::convert::Infallible;

    fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn update(
        &mut self,
        usage: &InclusionUsage,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        let p = self.weights.len();
        debug_assert_eq!(usage.counts().len(), p);
        let base = self.alpha / p as f64;
        let conjugate: Vec<f64> = usage
            .counts()
            .iter()
            .map(|&count| base + count as f64)
            .collect();
        let prior = vec![base; p];

        // Defensive-mixture proposal (module docs): half the sweeps propose
        // from the conjugate Dirichlet, half from the prior; the prior
        // component bounds the importance ratio at the simplex boundary.
        let shapes = if uniform_f64(rng) < 0.5 {
            &conjugate
        } else {
            &prior
        };
        let proposed = draw_dirichlet(shapes, rng);

        // The exact MH acceptance ratio, in full (the mixture proposal
        // breaks the conjugate cancellation): the target's usage exponents
        // are exactly the conjugate shapes, the subset-prior normalisers
        // e_{d_t}(s) enter inverted, and the proposal density is the
        // two-component mixture (the ½ weights cancel in the ratio).
        let max_d = usage.subset_sizes().iter().copied().max().unwrap_or(0);
        let current_e = elementary_symmetric(&self.weights, max_d);
        let proposed_e = elementary_symmetric(&proposed, max_d);
        let mut log_alpha = 0.0_f64;
        for k in 0..p {
            log_alpha +=
                (conjugate[k] - 1.0) * (mathsfn::ln(proposed[k]) - mathsfn::ln(self.weights[k]));
        }
        for &d in usage.subset_sizes() {
            log_alpha += mathsfn::ln(current_e[d]) - mathsfn::ln(proposed_e[d]);
        }
        log_alpha += log_mixture_density(&self.weights, &conjugate, &prior)
            - log_mixture_density(&proposed, &conjugate, &prior);
        if mathsfn::ln(uniform_f64(rng)) < log_alpha {
            self.weights = proposed;
        }
        Ok(())
    }
}

/// One Dirichlet(shapes) draw via normalised Gammas (ascending covariate
/// index, the pinned draw order). An exact-zero component (measure-zero
/// underflow) redraws the whole vector: the trait contract requires strictly
/// positive weights.
fn draw_dirichlet(shapes: &[f64], rng: &mut dyn rand_core::Rng) -> Vec<f64> {
    'draw: loop {
        let mut draws = Vec::with_capacity(shapes.len());
        let mut total = 0.0_f64;
        for &shape in shapes {
            let gamma =
                rand_distr::Gamma::new(shape, 1.0).expect("shape is positive by construction");
            let draw: f64 = rand_distr::Distribution::sample(&gamma, rng);
            total += draw;
            draws.push(draw);
        }
        for draw in &mut draws {
            *draw /= total;
            if !(draw.is_finite() && *draw > 0.0) {
                continue 'draw;
            }
        }
        break draws;
    }
}

/// ln Dir(x; shapes): the normalised log Dirichlet density on the simplex.
fn log_dirichlet_density(x: &[f64], shapes: &[f64]) -> f64 {
    let mut log_density = mathsfn::lgamma(shapes.iter().sum());
    for (&value, &shape) in x.iter().zip(shapes) {
        log_density += (shape - 1.0) * mathsfn::ln(value) - mathsfn::lgamma(shape);
    }
    log_density
}

/// The mixture proposal's log density up to the common ln ½ (which cancels
/// in the acceptance ratio): logsumexp of the two component log densities.
fn log_mixture_density(x: &[f64], conjugate: &[f64], prior: &[f64]) -> f64 {
    let a = log_dirichlet_density(x, conjugate);
    let b = log_dirichlet_density(x, prior);
    let max = a.max(b);
    max + mathsfn::ln(mathsfn::exp(a - max) + mathsfn::exp(b - max))
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;

    #[test]
    fn dart_starts_uniform_and_adapts_toward_used_covariates() {
        let mut model = DartInclusion::new(1.0, 4);
        assert!(model.weights().iter().all(|w| (w - 0.25).abs() < 1e-15));
        // Heavy usage of covariate 2, none elsewhere; d_t = 1 subsets keep
        // the correction mild.
        let mut usage = InclusionUsage::new(4);
        for _ in 0..30 {
            usage.record(2);
            usage.record_subset_size(1);
        }
        let mut rng = ChaCha8Rng::from_seed([3; 32]);
        let mut total_w2 = 0.0;
        let sweeps = 200;
        for _ in 0..sweeps {
            model.update(&usage, &mut rng).unwrap();
            total_w2 += model.weights()[2];
        }
        let mean_w2 = total_w2 / sweeps as f64;
        assert!(
            mean_w2 > 0.7,
            "the used covariate should dominate, mean weight {mean_w2:.3}"
        );
        // Weights stay a valid simplex point (strictly positive, finite).
        assert!(model.weights().iter().all(|w| w.is_finite() && *w > 0.0));
    }

    /// The exact-integration oracle: the isolated update chain's invariant
    /// distribution against midpoint quadrature of the true full conditional
    /// π(s | D) on the 2-simplex (p = 3, a fixed usage pattern with subset
    /// sizes 1–3). Pins the acceptance-ratio algebra: a wrong exponent, a
    /// dropped normaliser or a broken proposal density moves the means well
    /// past the tolerance.
    #[test]
    #[ignore = "long-running exact oracle: calibration/release CI legs only"]
    fn isolated_update_matches_exact_quadrature() {
        // Subsets over p=3: {0,1}, {0,1,2}, {1,2}, {0}.
        let mut usage = InclusionUsage::new(3);
        for &k in &[0, 1, 0, 1, 2, 1, 2, 0] {
            usage.record(k);
        }
        for &d in &[2, 3, 2, 1] {
            usage.record_subset_size(d);
        }
        let a = 1.5 / 3.0; // alpha/p
        // Exact target: prod s_k^{a-1} * s0^3 s1^3 s2^2 / (e2 * e3 * e2 * e1).
        let density = |s0: f64, s1: f64| -> f64 {
            let s2 = 1.0 - s0 - s1;
            if s2 <= 0.0 {
                return 0.0;
            }
            let e2 = s0 * s1 + s0 * s2 + s1 * s2;
            let e3 = s0 * s1 * s2;
            mathsfn::powf(s0, a - 1.0 + 3.0)
                * mathsfn::powf(s1, a - 1.0 + 3.0)
                * mathsfn::powf(s2, a - 1.0 + 2.0)
                / (e2 * e3 * e2)
        };
        let n_grid = 1200_usize;
        let h = 1.0 / n_grid as f64;
        let (mut mass, mut m0, mut m1, mut m2, mut mmax) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n_grid {
            let s0 = (i as f64 + 0.5) * h;
            for j in 0..n_grid {
                let s1 = (j as f64 + 0.5) * h;
                let f = density(s0, s1);
                if f > 0.0 {
                    let s2 = 1.0 - s0 - s1;
                    mass += f;
                    m0 += f * s0;
                    m1 += f * s1;
                    m2 += f * s2;
                    mmax += f * s0.max(s1).max(s2);
                }
            }
        }
        let exact = [m0 / mass, m1 / mass, m2 / mass, mmax / mass];

        let mut model = DartInclusion::new(1.5, 3);
        let mut rng = ChaCha8Rng::from_seed([7; 32]);
        let sweeps = 1_000_000_usize;
        let burn = 2_000_usize;
        let (mut c0, mut c1, mut c2, mut cmax) = (0.0, 0.0, 0.0, 0.0);
        for sweep in 0..(burn + sweeps) {
            model.update(&usage, &mut rng).unwrap();
            if sweep >= burn {
                let w = model.weights();
                c0 += w[0];
                c1 += w[1];
                c2 += w[2];
                cmax += w[0].max(w[1]).max(w[2]);
            }
        }
        let n = sweeps as f64;
        let chain = [c0 / n, c1 / n, c2 / n, cmax / n];
        println!("exact quadrature: {exact:?}");
        println!("MH chain:         {chain:?}");
        for (e, c) in exact.iter().zip(&chain) {
            assert!(
                (e - c).abs() < 0.005,
                "isolated update biased: exact {exact:?} vs chain {chain:?}"
            );
        }
    }

    /// The subset-prior correction is exactly zero when every subset size
    /// is 1 (e_1 of a normalised weight vector is 1, so the normaliser
    /// ratio vanishes): DART's conjugate case, in which the target is
    /// Dirichlet(α/p + u).
    #[test]
    fn singleton_subsets_make_the_correction_vanish() {
        let mut usage = InclusionUsage::new(3);
        usage.record(0);
        usage.record_subset_size(1);
        let model = DartInclusion::new(2.0, 3);
        let e = elementary_symmetric(model.weights(), 1);
        // e_1(normalised s) = 1 exactly.
        assert!((e[1] - 1.0).abs() < 1e-12);
    }
}
