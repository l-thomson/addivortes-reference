//! The statistical validation battery: the Geweke
//! joint-distribution cross-check and the SBC rank machinery as public
//! drivers accepting caller components, so component authors validate their
//! components mechanically from their own crates. This is the machinery
//! the CI battery leg runs, with the model under test fully
//! caller-supplied.
//!
//! This battery is the **calibration** rung of the crate's validation ladder,
//! which runs from the narrowest check to the broadest:
//!
//! 1. **oracles** — hand-derived closed-form values (the move ratios) and the
//!    detailed-balance telescoping, as ordinary unit tests;
//! 2. **conformance** — the local per-component
//!    [`conformance`](crate::conformance) checks, seconds;
//! 3. **calibration** — this battery: the joint-distribution gates (SBC and
//!    Geweke), run per-commit at CI sizes and at full N before a release;
//! 4. **interval coverage** — a formal frequentist coverage test on the
//!    Friedman benchmark. Coverage of the credible intervals, not code
//!    coverage;
//! 5. **reference comparison** — an RMSE neighbourhood check against the
//!    original authors' R package at a pinned commit. A comparison, never an
//!    oracle: the reference has known bugs;
//! 6. **golden chain** — the per-target bit-exact regression tests.
//!
//! Rungs 4 and 5 live in the repository's CI workflows, not on the crate
//! surface. `CONTRIBUTING.md` documents every rung and the command that runs
//! it.
//!
//! Division of labour. The caller owns the model: the prior/likelihood
//! generator (their components' own joint distribution) and the sampler under
//! test (built through the public surface; [`Sampler::pinned_prior`] is the
//! exact-prior constructor the batteries need). The battery owns the
//! simulator drivers, the sizing, the tie-aware KS machinery, the
//! Bonferroni-split verdict, and the SBC rank bookkeeping (CSV emission for
//! an external ECDF-band verdict, plus a coarse in-process χ² summary).
//!
//! What a pass means: these joint-distribution gates catch
//! wrong-but-self-consistent samplers, far stronger than the local checks,
//! but a Geweke/SBC pass at gate sizes is still evidence, not proof. The
//! battery ships with negative controls (in-crate: a deliberately mispriced
//! move must turn both gates red; the teeth tests); hold your own components
//! to the same standard.
//!
//! Both drivers are deterministic given the caller's closures: all randomness
//! lives in the caller's own RNG (the caller-owns-its-RNG rule).
//!
//! [`Sampler::pinned_prior`]: crate::Sampler::pinned_prior

use std::collections::BTreeMap;

use crate::diagnostics::{ks_critical_value, ks_two_sample, sbc_rank};
use crate::engine::error::Result;

// ---------------------------------------------------------------------------
// Geweke joint-distribution cross-check (Geweke 2004)
// ---------------------------------------------------------------------------

/// Sizing of one Geweke run. The defaults are the in-crate calibration-leg sizes
/// (sized from measured wall-clock): 4000 marginal-conditional
/// draws, 1500 successive-conditional keeps at thinning 20 after a
/// 100-transition discard, α = 0.01 per battery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GewekeSpec {
    /// Marginal-conditional (i.i.d. prior × likelihood) sample size.
    pub n_mc: usize,
    /// Successive-conditional kept sample size.
    pub n_sc: usize,
    /// Keep every `thin`-th successive-conditional transition (the KS
    /// critical value assumes rough independence; the SC chain is Markov).
    pub thin: usize,
    /// Transitions discarded before keeping (belt-and-braces: a
    /// prior-initialised SC chain already starts at stationarity).
    pub discard: usize,
    /// Per-battery significance, Bonferroni-split across the statistics.
    pub alpha: f64,
}

impl Default for GewekeSpec {
    fn default() -> Self {
        Self {
            n_mc: 4000,
            n_sc: 1500,
            thin: 20,
            discard: 100,
            alpha: 0.01,
        }
    }
}

/// One statistic's Geweke verdict: the tie-aware two-sample KS statistic
/// against its Bonferroni-split critical value (both dimensionless).
#[derive(Debug, Clone, PartialEq)]
pub struct GewekeOutcome {
    /// The test quantity's name (as emitted by the caller's closures).
    pub statistic: String,
    /// The KS statistic `D = sup |F_mc − F_sc|` (dimensionless).
    pub d: f64,
    /// The critical value at the Bonferroni-split significance
    /// (dimensionless).
    pub critical: f64,
}

impl GewekeOutcome {
    /// Whether this statistic's two simulators agree at the requested
    /// significance.
    pub fn passed(&self) -> bool {
        self.d <= self.critical
    }
}

/// The successive-conditional simulator the caller drives: one `transition`
/// is `y | θ` (regenerate the data from the current state, through
/// [`Sampler::set_response`]) followed by `θ | y` (one full sweep of the real
/// kernel under test, [`Sampler::step`]); `quantities` reads the test
/// quantities of the current state, matching the marginal-conditional
/// closure's names and definitions exactly. Any mismatch is a red herring the
/// battery cannot distinguish from a broken kernel.
///
/// [`Sampler::set_response`]: crate::Sampler::set_response
/// [`Sampler::step`]: crate::Sampler::step
pub trait SuccessiveConditional {
    /// One `y | θ` → `θ | y` transition of the chain.
    fn transition(&mut self) -> Result<()>;
    /// The test quantities of the current (state, data) pair, each on
    /// whatever scale the marginal-conditional closure uses for the same
    /// name (typically scaled space; the battery only compares like with
    /// like).
    fn quantities(&self) -> BTreeMap<String, f64>;
}

/// Run the Geweke cross-check: `spec.n_mc` marginal-conditional draws from
/// the caller's i.i.d. generator versus a thinned successive-conditional
/// chain, KS-compared per statistic at the Bonferroni-split significance.
/// The two simulators target the same joint distribution iff the kernel
/// under test targets its own prior × likelihood; any statistic's
/// disagreement is a defect (in the kernel, or in a generator that does not
/// match the kernel's model; the battery cannot tell those apart).
pub fn getting_it_right(
    spec: &GewekeSpec,
    mc_draw: &mut dyn FnMut() -> BTreeMap<String, f64>,
    sc: &mut dyn SuccessiveConditional,
) -> Result<Vec<GewekeOutcome>> {
    let mut mc: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for _ in 0..spec.n_mc {
        for (name, value) in mc_draw() {
            mc.entry(name).or_default().push(value);
        }
    }
    let mut sc_values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let total = spec.discard + spec.n_sc * spec.thin;
    for iteration in 0..total {
        sc.transition()?;
        if iteration >= spec.discard && (iteration - spec.discard + 1) % spec.thin == 0 {
            for (name, value) in sc.quantities() {
                sc_values.entry(name).or_default().push(value);
            }
        }
    }
    let alpha_per_stat = spec.alpha / mc.len() as f64;
    Ok(mc
        .iter()
        .map(|(name, mc_sample)| {
            let sc_sample = sc_values
                .get(name)
                .unwrap_or_else(|| panic!("SC never emitted statistic `{name}`"));
            GewekeOutcome {
                statistic: name.clone(),
                d: ks_two_sample(mc_sample, sc_sample),
                critical: ks_critical_value(alpha_per_stat, mc_sample.len(), sc_sample.len()),
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// SBC ranks (Talts et al. 2018)
// ---------------------------------------------------------------------------

/// Sizing of one SBC run. The defaults are the in-crate calibration-leg sizes:
/// 300 replications × 99 ranked draws (thin 10) after 300 burn-in sweeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SbcSpec {
    /// Prior-draw replications.
    pub replications: usize,
    /// Posterior draws ranked per replication (ranks lie in `0..=n_draws`).
    pub n_draws: usize,
    /// Keep every `thin`-th sweep when collecting the ranked draws.
    pub thin: usize,
    /// Burn-in sweeps discarded per replication.
    pub burn_in: usize,
}

impl Default for SbcSpec {
    fn default() -> Self {
        Self {
            replications: 300,
            n_draws: 99,
            thin: 10,
            burn_in: 300,
        }
    }
}

/// The collected SBC ranks: per quantity, one rank per replication, each in
/// `0..=n_draws` (counts, uniform under a correct sampler). Feed
/// [`write_csv`](SbcRanks::write_csv) to an external ECDF-band verdict (the
/// in-crate CI uses bayesplot's Säilynoja–Bürkner–Vehtari implementation),
/// or take the coarse in-process [`chi_squared`](SbcRanks::chi_squared)
/// summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbcRanks {
    ranks: BTreeMap<String, Vec<usize>>,
    n_draws: usize,
}

impl SbcRanks {
    /// An empty collector for ranks in `0..=n_draws`.
    pub fn new(n_draws: usize) -> Self {
        Self {
            ranks: BTreeMap::new(),
            n_draws,
        }
    }

    /// Rank `true_value` among `posterior_draws` (ties broken uniformly via
    /// `rng`, required for discrete quantities) and record it under
    /// `quantity`.
    pub fn record(
        &mut self,
        quantity: &str,
        true_value: f64,
        posterior_draws: &[f64],
        rng: &mut dyn rand_core::Rng,
    ) {
        debug_assert_eq!(posterior_draws.len(), self.n_draws);
        let rank = sbc_rank(true_value, posterior_draws, rng);
        self.ranks
            .entry(quantity.to_string())
            .or_default()
            .push(rank);
    }

    /// The collected ranks per quantity (counts in `0..=n_draws`).
    pub fn ranks(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.ranks
    }

    /// Write the `quantity,rank,n_draws` CSV the external ECDF-band verdict
    /// consumes (the same format the in-crate calibration leg emits).
    pub fn write_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write as _;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "quantity,rank,n_draws")?;
        for (name, ranks) in &self.ranks {
            for rank in ranks {
                writeln!(file, "{name},{rank},{}", self.n_draws)?;
            }
        }
        Ok(())
    }

    /// Coarse in-process uniformity summary: per quantity, the χ² statistic
    /// of the rank histogram over `bins` equal-width bins (dimensionless
    /// counts; compare against the χ²(bins − 1) critical value of your
    /// chosen significance). The ECDF-band verdict is stricter; prefer it
    /// where an R toolchain is available.
    pub fn chi_squared(&self, bins: usize) -> BTreeMap<String, f64> {
        debug_assert!(bins >= 2);
        self.ranks
            .iter()
            .map(|(name, ranks)| {
                let mut counts = vec![0usize; bins];
                for &rank in ranks {
                    let bin = (rank * bins / (self.n_draws + 1)).min(bins - 1);
                    counts[bin] += 1;
                }
                let expected = ranks.len() as f64 / bins as f64;
                let chi_sq: f64 = counts
                    .iter()
                    .map(|&count| {
                        let diff = count as f64 - expected;
                        diff * diff / expected
                    })
                    .sum();
                (name.clone(), chi_sq)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The structural-prior generator (the marginal-conditional side's tessellation
// draw, for models whose structure prior is the shifted Poisson/Binomial)
// ---------------------------------------------------------------------------

/// A draw-side description of the default structural prior (shifted
/// Poisson(λ_c) cells, shifted Binomial(p − 1, ω/p) dimension counts, the
/// weighted distinct-subset dimension prior, per-column coordinate laws):
/// what a marginal-conditional generator needs to draw tessellations that
/// match the kernel's own prior. Cell values and the support restriction are
/// caller closures, where the conjugate family and the empty-cell guard's
/// meaning live (hard: every cell owns an observation; soft: every cell
/// carries positive membership mass).
///
/// Dimension subsets are drawn by exact enumeration of all d-subsets
/// (sequential weighted draws would follow a different, successive-sampling
/// law): fine for the small p of battery fixtures, combinatorially explosive
/// beyond that.
pub struct StructuralPrior<'a> {
    /// Centre-count prior parameter λ_c (b − 1 ~ Poisson(λ_c); a count
    /// input).
    pub lambda_c: f64,
    /// Dimension-count prior parameter ω (d − 1 ~ Binomial(p − 1, ω/p); a
    /// count input).
    pub omega: f64,
    /// Per-encoded-column coordinate laws, the same laws
    /// the sampler under test runs.
    pub coord_dists: &'a [std::sync::Arc<dyn crate::CoordinateDistribution>],
    /// Per-encoded-column inclusion weights (strictly
    /// positive relative weights; all-1 is the uniform subset prior).
    pub weights: &'a [f64],
}

impl StructuralPrior<'_> {
    /// Draw one tessellation from the structural prior, conditioned on
    /// `support` (rejection): cell count, dimension count, the weighted
    /// dimension subset, per-(cell, dimension) centre coordinates from the
    /// coordinate laws, and one cell value per cell from `draw_value` (your
    /// conjugate family's prior, scaled space). Panics after 100 000
    /// rejections (a support predicate that tight is a fixture bug).
    pub fn draw(
        &self,
        draw_value: &mut dyn FnMut(&mut dyn rand_core::Rng) -> f64,
        support: &mut dyn FnMut(&crate::Tessellation) -> bool,
        rng: &mut dyn rand_core::Rng,
    ) -> crate::Tessellation {
        self.draw_payload(1, &mut |rng, cell| cell[0] = draw_value(rng), support, rng)
    }

    /// The basis-payload sibling of [`draw`](Self::draw): the same
    /// structural prior, but each cell carries `q` values, written into the
    /// slice handed to `draw_cell` (ascending coefficient index, the sampler's
    /// own draw order). `q = 1` is [`draw`](Self::draw).
    pub fn draw_payload(
        &self,
        q: usize,
        draw_cell: &mut dyn FnMut(&mut dyn rand_core::Rng, &mut [f64]),
        support: &mut dyn FnMut(&crate::Tessellation) -> bool,
        rng: &mut dyn rand_core::Rng,
    ) -> crate::Tessellation {
        let p = self.weights.len();
        debug_assert_eq!(self.coord_dists.len(), p);
        let poisson = rand_distr::Poisson::new(self.lambda_c).expect("λ_c is positive");
        let theta = self.omega / p as f64;
        let binomial =
            rand_distr::Binomial::new((p - 1) as u64, theta).expect("ω < p at the fit boundary");
        for _attempt in 0..100_000 {
            let b = 1 + rand_distr::Distribution::<f64>::sample(&poisson, rng) as usize;
            let d = 1 + rand_distr::Distribution::sample(&binomial, rng) as usize;
            let dims = draw_weighted_subset(p, d, self.weights, rng);
            let mut centres = Vec::with_capacity(b * d);
            for _cell in 0..b {
                for &dim in &dims {
                    centres.push(self.coord_dists[dim].sample(rng));
                }
            }
            let mut values = vec![0.0_f64; b * q];
            for cell in 0..b {
                draw_cell(rng, &mut values[cell * q..(cell + 1) * q]);
            }
            let tessellation = crate::Tessellation::new(centres, dims, values)
                .expect("structurally coherent by construction");
            if support(&tessellation) {
                return tessellation;
            }
        }
        panic!("structural-prior rejection sampling failed to satisfy the support predicate");
    }
}

/// All d-subsets of `0..p` (ascending; battery-fixture p only).
fn all_subsets(p: usize, d: usize) -> Vec<Vec<usize>> {
    let mut all = Vec::new();
    let mut current = Vec::with_capacity(d);
    fn recurse(
        start: usize,
        p: usize,
        d: usize,
        current: &mut Vec<usize>,
        all: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == d {
            all.push(current.clone());
            return;
        }
        for candidate in start..p {
            current.push(candidate);
            recurse(candidate + 1, p, d, current, all);
            current.pop();
        }
    }
    recurse(0, p, d, &mut current, &mut all);
    all
}

/// One dimension subset of size `d` from the weighted distinct-subset prior
/// `P(D | d, s) ∝ ∏_{k∈D} s_k`, by exact enumeration (see
/// [`StructuralPrior`]). Public so generators for adaptive inclusion models
/// (DART) can draw subsets given their weight state.
pub fn draw_weighted_subset(
    p: usize,
    d: usize,
    weights: &[f64],
    rng: &mut dyn rand_core::Rng,
) -> Vec<usize> {
    let candidates = all_subsets(p, d);
    let masses: Vec<f64> = candidates
        .iter()
        .map(|dims| dims.iter().map(|&k| weights[k]).product())
        .collect();
    let total: f64 = masses.iter().sum();
    let target = crate::extensions::moves::uniform_f64(rng) * total;
    let mut cumulative = 0.0;
    for (dims, mass) in candidates.iter().zip(&masses) {
        cumulative += mass;
        if target < cumulative {
            return dims.clone();
        }
    }
    candidates.last().expect("d ≤ p yields subsets").clone()
}
