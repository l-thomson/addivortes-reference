//! The calibration gate batteries (the calibration rung of the validation ladder;
//! CONTRIBUTING.md documents every rung and how to run it):
//!
//! - **SBC** (Talts et al. 2018): every test draws the model parameters from
//!   the prior, simulates data, runs the real sampler, and emits the posterior
//!   rank of each test quantity as CSV. The uniformity verdict is delegated
//!   to R (`ci/sbc-ecdf-check.R`, the Säilynoja–Bürkner–Vehtari ECDF
//!   simultaneous-confidence-band test via the authors' `bayesplot`
//!   implementation); the Rust side only emits ranks, so the validator is
//!   more trusted than the code it judges.
//! - **Geweke joint-distribution cross-check** (Geweke 2004), Rust-native: the
//!   marginal-conditional simulator (independent prior draws) and the
//!   successive-conditional simulator (alternate `y | θ` and one Gibbs sweep
//!   `θ | y`) target the same joint, so every statistic must agree. Judged by
//!   the two-sample KS test in `diagnostics`.
//! - **Injection negative controls**: an RD-only `p`-vs-`p−1` prior-ratio
//!   error (a plausible wrong-maths slip with no per-commit statistical
//!   oracle) is injected via a wrapped move and must turn the battery red,
//!   proving the gate has teeth rather than just a green checkmark.
//!
//! **CI leg: calibration / release.** Every test here is
//! `#[ignore]`d so the fast-PR leg never runs one. The calibration workflow runs
//! them under the determinism profile:
//!
//! ```sh
//! cargo nextest run --cargo-profile determinism --run-ignored ignored-only \
//!     -E 'test(/^stat_gates::/)'
//! ```
//!
//! Sizes default to the calibration leg (sized from measured per-fit
//! wall-clock) and are env-tunable
//! (`SBC_REPLICATIONS`, `SBC_DRAWS`, `SBC_THIN`, `SBC_BURN_IN`, `GEWEKE_MC`,
//! `GEWEKE_SC`, `GEWEKE_THIN`) so the release leg runs the same battery
//! full-N. Every test is deterministic given its pinned seeds.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use rand_distr::Distribution;

use crate::AddiVortesConfig;
use crate::diagnostics::{ks_critical_value, ks_two_sample, sbc_rank};
use crate::engine::data::{Data, Metric};
use crate::engine::sampler::{Sampler, expand_seed, splitmix64};
use crate::engine::tessellation::Tessellation;
use crate::extensions::coord::{CoordinateDistribution, EuclideanNormal, WrappedNormal};
use crate::extensions::distance::{CellAssigner, ColumnMetrics};
use crate::extensions::inclusion::WeightedInclusion;
use crate::extensions::moves::{
    AddCentre, AddDimension, Change, ModelCtx, MoveSet, MoveSetBuilder, Proposal, ProposalMove,
    RemoveCentre, RemoveDimension, Reverse, Swap, uniform_f64,
};

// ---------------------------------------------------------------------------
// Sizes (calibration-leg defaults; env-overridable for the release leg)
// ---------------------------------------------------------------------------

/// Read a size from the environment or fall back (release leg overrides).
fn env_size(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a positive integer, got `{value}`")),
        Err(_) => default,
    }
}

/// SBC replications per config (calibration leg: 300; release: 1000 via env).
fn sbc_replications() -> usize {
    env_size("SBC_REPLICATIONS", 300)
}
/// Posterior draws ranked per replication (ranks lie in `0..=draws`).
fn sbc_draws() -> usize {
    env_size("SBC_DRAWS", 99)
}
/// Keep every `thin`-th sweep (ESS ≈ number of ranked draws).
fn sbc_thin() -> usize {
    env_size("SBC_THIN", 10)
}
/// Burn-in sweeps discarded per replication.
fn sbc_burn_in() -> usize {
    env_size("SBC_BURN_IN", 300)
}
/// Geweke marginal-conditional (i.i.d. prior) sample size.
fn geweke_mc() -> usize {
    env_size("GEWEKE_MC", 4000)
}
/// Geweke successive-conditional kept sample size.
fn geweke_sc() -> usize {
    env_size("GEWEKE_SC", 1500)
}
/// Successive-conditional thinning (keeps the KS independence assumption
/// honest: the SC chain is a Markov chain, the KS critical value is not).
fn geweke_thin() -> usize {
    env_size("GEWEKE_THIN", 20)
}

/// Where the rank CSVs land (the workflow points R at the same directory).
fn out_dir() -> PathBuf {
    match std::env::var_os("STAT_GATES_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/stat-gates"),
    }
}

// ---------------------------------------------------------------------------
// Fixtures: the non-default extension-point instances the calibration gate covers
// ---------------------------------------------------------------------------

/// One calibration battery configuration: a fixed scaled-space design plus the pinned
/// prior. λ is pinned (not calibrated from data) because exact SBC/Geweke
/// requires the generating prior and the fitted prior to coincide; the
/// sampler enters through `Sampler::pinned_prior_for_tests`.
struct GateFixture {
    name: &'static str,
    x: Data,
    metrics: Vec<Metric>,
    m: usize,
    nu: f64,
    lambda: f64,
    lambda_c: f64,
    omega: f64,
    sigma_c: f64,
    k: f64,
    /// Inclusion weights over the p columns (all-1 = `UniformInclusion`).
    weights: Vec<f64>,
    /// Soft-membership temperature; `None` = hard membership.
    tau: Option<f64>,
    seed: u64,
}

/// Deterministic quasi-uniform design values via splitmix64.
fn uniform_column(state: &mut u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let u = (splitmix64(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            lo + u * (hi - lo)
        })
        .collect()
}

fn fixture_design(seed: u64, n: usize, metrics: &[Metric]) -> Data {
    let mut state = seed;
    let p = metrics.len();
    let columns: Vec<Vec<f64>> = metrics
        .iter()
        .map(|metric| match metric {
            Metric::Spherical => {
                uniform_column(&mut state, n, -std::f64::consts::PI, std::f64::consts::PI)
            }
            _ => uniform_column(&mut state, n, -0.5, 0.5),
        })
        .collect();
    let mut values = Vec::with_capacity(n * p);
    for row in 0..n {
        for column in &columns {
            values.push(column[row]);
        }
    }
    Data::new(values, n, p).unwrap()
}

/// Small-p config: p = 3 with ω pinned (default ω = 3 correctly errors on
/// p ≤ 3) and a small λ_c, both forcing the d = 1 / d = p / nC = 1 boundary
/// folds to be visited.
fn fixture_small_p() -> GateFixture {
    let metrics = vec![Metric::Euclidean; 3];
    GateFixture {
        name: "small-p",
        x: fixture_design(0xA11CE, 60, &metrics),
        metrics,
        m: 10,
        nu: 6.0,
        lambda: 0.02,
        lambda_c: 2.0,
        omega: 1.5,
        sigma_c: 0.8,
        k: 3.0,
        weights: vec![1.0; 3],
        tau: None,
        seed: 0x5BC0_0001,
    }
}

/// Spherical-active config: one wrapped-normal column (the coord point/3).
fn fixture_spherical() -> GateFixture {
    let metrics = vec![Metric::Spherical, Metric::Euclidean, Metric::Euclidean];
    GateFixture {
        name: "spherical",
        x: fixture_design(0x5F44E, 60, &metrics),
        metrics,
        m: 10,
        nu: 6.0,
        lambda: 0.02,
        lambda_c: 2.0,
        omega: 1.5,
        sigma_c: 0.8,
        k: 3.0,
        weights: vec![1.0; 3],
        tau: None,
        seed: 0x5BC0_0002,
    }
}

/// Soft-membership config: the dense path (joint payload draws,
/// membership-weighted fits) under the softmax kernel at fixed τ.
fn fixture_soft() -> GateFixture {
    let metrics = vec![Metric::Euclidean; 3];
    GateFixture {
        name: "soft-membership",
        x: fixture_design(0x50F7, 60, &metrics),
        metrics,
        m: 10,
        nu: 6.0,
        lambda: 0.02,
        lambda_c: 2.0,
        omega: 1.5,
        sigma_c: 0.8,
        k: 3.0,
        weights: vec![1.0; 3],
        tau: Some(0.1),
        seed: 0x5BC0_0005,
    }
}

/// Non-uniform `WeightedInclusion` config: validates the weighted
/// subset-prior ratios jointly.
fn fixture_weighted() -> GateFixture {
    let metrics = vec![Metric::Euclidean; 3];
    GateFixture {
        name: "weighted-inclusion",
        x: fixture_design(0x3E16_47ED, 60, &metrics),
        metrics,
        m: 10,
        nu: 6.0,
        lambda: 0.02,
        lambda_c: 2.0,
        omega: 1.5,
        sigma_c: 0.8,
        k: 3.0,
        weights: vec![1.0, 2.0, 4.0],
        tau: None,
        seed: 0x5BC0_0003,
    }
}

impl GateFixture {
    fn p(&self) -> usize {
        self.metrics.len()
    }

    fn n(&self) -> usize {
        self.x.n_rows()
    }

    fn sigma_mu_sq(&self) -> f64 {
        crate::engine::scaler::sigma_mu_sq(self.k, self.m)
    }

    fn uniform_weights(&self) -> bool {
        self.weights.iter().all(|w| *w == 1.0)
    }

    /// The same per-column coordinate laws the sampler builds.
    fn coord_dists(&self) -> Vec<Arc<dyn CoordinateDistribution>> {
        self.metrics
            .iter()
            .map(|metric| match metric {
                Metric::Spherical => {
                    Arc::new(WrappedNormal::new(self.sigma_c)) as Arc<dyn CoordinateDistribution>
                }
                _ => Arc::new(EuclideanNormal::new(self.sigma_c)) as Arc<_>,
            })
            .collect()
    }

    fn assigner(&self) -> ColumnMetrics {
        ColumnMetrics::new(self.metrics.clone())
    }

    fn sampler(&self, seed: u64, y: Vec<f64>, moves: MoveSetKind) -> Sampler {
        let mut config = AddiVortesConfig::new(seed)
            .with_m(self.m)
            .with_nu(self.nu)
            .with_omega(self.omega)
            .with_lambda_c(self.lambda_c)
            .with_sigma_c(self.sigma_c)
            .with_k(self.k);
        if !self.uniform_weights() {
            config.inclusion = Some(Arc::new(WeightedInclusion::new(self.weights.clone())));
        }
        if let Some(tau) = self.tau {
            config = config.with_membership(crate::extensions::membership::SoftmaxKernel::new(tau));
        }
        Sampler::pinned_prior_for_tests(
            config,
            self.x.clone(),
            self.metrics.clone(),
            y,
            self.lambda,
            build_move_set(moves),
        )
        .expect("pinned-prior sampler construction must succeed")
    }
}

// ---------------------------------------------------------------------------
// The H-AddiVortes configuration (two ensemble instances)
// ---------------------------------------------------------------------------

/// The H battery configuration: the mean fixture plus the variance-ensemble
/// size and its pinned (ν′, λ′), for the same reason as λ: exact SBC/Geweke
/// needs the generating and fitted priors to coincide, and the §3.3
/// calibration from data would break that.
struct HFixture {
    base: GateFixture,
    m_prime: usize,
    nu_prime: f64,
    lambda_prime: f64,
}

fn fixture_h() -> HFixture {
    let metrics = vec![Metric::Euclidean; 3];
    HFixture {
        base: GateFixture {
            name: "h-variance",
            x: fixture_design(0x8AD1, 60, &metrics),
            metrics,
            m: 10,
            nu: 6.0,
            lambda: 0.02, // unused by the H sampler (σ² ≡ 1); kept for assembly
            lambda_c: 2.0,
            omega: 1.5,
            sigma_c: 0.8,
            k: 3.0,
            weights: vec![1.0; 3],
            tau: None,
            seed: 0x5BC0_0004,
        },
        m_prime: 3,
        nu_prime: 8.0,
        lambda_prime: 0.3,
    }
}

/// One joint H prior draw: mean tessellations (Gaussian cell values) plus
/// variance tessellations (inverse-χ² cell values), both conditioned on the
/// empty-cell guard over the fixture design.
struct HPriorDraw {
    mean: PriorDraw,
    var_tessellations: Vec<Tessellation>,
    var_assignments: Vec<Vec<usize>>,
}

impl HPriorDraw {
    fn s_sq_at(&self, row: usize) -> f64 {
        self.var_tessellations
            .iter()
            .zip(&self.var_assignments)
            .map(|(tessellation, assignment)| tessellation.mus()[assignment[row]])
            .product()
    }
}

/// Draw one variance tessellation from the prior: the same structural law as
/// the mean side (shared count priors, coordinate laws, subset prior, by
/// construction), with cell values s² ~ χ⁻²(ν′, λ′), conditioned on no empty
/// cell over the design (the kernel's guard restricts this ensemble's prior
/// support identically).
fn draw_h_variance_tessellation(
    fixture: &HFixture,
    coord_dists: &[Arc<dyn CoordinateDistribution>],
    assigner: &dyn CellAssigner,
    rng: &mut ChaCha8Rng,
) -> (Tessellation, Vec<usize>) {
    let base = &fixture.base;
    let poisson = rand_distr::Poisson::new(base.lambda_c).unwrap();
    let theta = base.omega / base.p() as f64;
    let binomial = rand_distr::Binomial::new((base.p() - 1) as u64, theta).unwrap();
    let gamma = rand_distr::Gamma::new(
        0.5 * fixture.nu_prime,
        2.0 / (fixture.nu_prime * fixture.lambda_prime),
    )
    .unwrap();

    for _attempt in 0..100_000 {
        let b = 1 + Distribution::<f64>::sample(&poisson, rng) as usize;
        let d = 1 + Distribution::sample(&binomial, rng) as usize;
        let dims = draw_dims(base.p(), d, &base.weights, rng);
        let mut centres = Vec::with_capacity(b * d);
        for _cell in 0..b {
            for &dim in &dims {
                centres.push(coord_dists[dim].sample(rng));
            }
        }
        let mus: Vec<f64> = (0..b)
            .map(|_| {
                let precision: f64 = Distribution::sample(&gamma, rng);
                1.0 / precision
            })
            .collect();
        let tessellation = Tessellation::new(centres, dims, mus).unwrap();
        let assignment = assigner
            .assign_cells(&base.x, &tessellation)
            .expect("built-in metrics never yield non-finite distances");
        let mut occupied = vec![false; b];
        for &cell in &assignment {
            occupied[cell] = true;
        }
        if occupied.iter().all(|o| *o) {
            return (tessellation, assignment);
        }
    }
    panic!("prior rejection sampling failed to find a non-empty-cell variance tessellation");
}

fn draw_h_prior(fixture: &HFixture, rng: &mut ChaCha8Rng) -> HPriorDraw {
    let base = &fixture.base;
    let coord_dists = base.coord_dists();
    let assigner = base.assigner();
    // Mean tessellations exactly as the homoscedastic generator draws them;
    // σ² is unused in the H model (σ ≡ 1, the noise lives in s²(x)).
    let mut mean_tessellations = Vec::with_capacity(base.m);
    let mut mean_assignments = Vec::with_capacity(base.m);
    for _ in 0..base.m {
        let (tessellation, assignment, _) = draw_tessellation(base, &coord_dists, &assigner, rng);
        mean_tessellations.push(tessellation);
        mean_assignments.push(assignment);
    }
    let mut var_tessellations = Vec::with_capacity(fixture.m_prime);
    let mut var_assignments = Vec::with_capacity(fixture.m_prime);
    for _ in 0..fixture.m_prime {
        let (tessellation, assignment) =
            draw_h_variance_tessellation(fixture, &coord_dists, &assigner, rng);
        var_tessellations.push(tessellation);
        var_assignments.push(assignment);
    }
    HPriorDraw {
        mean: PriorDraw {
            tessellations: mean_tessellations,
            assignments: mean_assignments,
            memberships: None,
            sigma_sq: 1.0,
        },
        var_tessellations,
        var_assignments,
    }
}

/// y | θ under the H model: `y_i = F_i + s(x_i)·z_i`.
fn draw_h_response(prior: &HPriorDraw, n: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    (0..n)
        .map(|row| {
            let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
            prior.mean.fit_at(row) + prior.s_sq_at(row).sqrt() * z
        })
        .collect()
}

/// H test quantities: the mean side's, the variance side's, and the jointly
/// standardised residual sum `Σ e²_i / s²(x_i)` (χ²_n under the model, the
/// quantity most sensitive to joint mean/variance miscalibration).
fn h_state_quantities(
    mean_tessellations: &[Tessellation],
    var_tessellations: &[Tessellation],
    fit: impl Fn(usize) -> f64,
    s_sq: impl Fn(usize) -> f64,
    y: &[f64],
) -> BTreeMap<&'static str, f64> {
    let mut quantities = BTreeMap::new();
    let mean_mu: f64 = mean_tessellations
        .iter()
        .map(|t| t.mus().iter().sum::<f64>() / t.n_cells() as f64)
        .sum();
    quantities.insert("mean_mu", mean_mu);
    quantities.insert(
        "total_cells",
        mean_tessellations.iter().map(|t| t.n_cells() as f64).sum(),
    );
    quantities.insert(
        "var_total_cells",
        var_tessellations.iter().map(|t| t.n_cells() as f64).sum(),
    );
    quantities.insert("f_row_a", fit(F_ROWS[0]));
    quantities.insert("f_row_b", fit(F_ROWS[1]));
    quantities.insert("s_sq_row_a", s_sq(F_ROWS[0]));
    quantities.insert("s_sq_row_b", s_sq(F_ROWS[1]));
    let standardised_rss: f64 = y
        .iter()
        .enumerate()
        .map(|(row, &value)| {
            let residual = value - fit(row);
            residual * residual / s_sq(row)
        })
        .sum();
    quantities.insert("standardised_rss", standardised_rss);
    quantities
}

/// Assemble the H sampler: WeightedGaussian mean cells (per-observation
/// precisions demand the weighted statistic) + the shared HVariance handle,
/// through the pinned-prior constructor. Everything downstream is the real
/// two-ensemble kernel.
fn h_sampler(
    fixture: &HFixture,
    seed: u64,
    y: Vec<f64>,
) -> (
    Sampler,
    std::sync::Arc<std::sync::Mutex<crate::extensions::scale::HVariance>>,
) {
    let base = &fixture.base;
    let mut config = AddiVortesConfig::new(seed)
        .with_m(base.m)
        .with_nu(base.nu)
        .with_omega(base.omega)
        .with_lambda_c(base.lambda_c)
        .with_sigma_c(base.sigma_c)
        .with_k(base.k);
    config.cell_model = Some(Arc::new(
        crate::extensions::cell_model::WeightedGaussianModel::new(base.sigma_mu_sq()),
    ));
    let shared = std::sync::Arc::new(std::sync::Mutex::new(
        crate::extensions::scale::HVariance::new(fixture.m_prime)
            .with_prior(fixture.nu_prime, fixture.lambda_prime),
    ));
    let sampler = Sampler::pinned_prior_for_tests(
        config,
        base.x.clone(),
        base.metrics.clone(),
        y,
        base.lambda,
        build_move_set(MoveSetKind::Standard),
    )
    .expect("pinned-prior H sampler construction must succeed")
    .with_scale_model(crate::test_support::SharedHVariance {
        inner: std::sync::Arc::clone(&shared),
        cache: Vec::new(),
    });
    (sampler, shared)
}

// ---------------------------------------------------------------------------
// Move-set variants (the wrong-trials injection and its isolating positive control)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum MoveSetKind {
    /// The paper's six moves under the folded selection table.
    Standard,
    /// The standard six re-registered as custom moves (weighted-valid
    /// selection rule) with RD correct: the positive control isolating
    /// the injected ratio, not the selection rule, as the failure cause.
    WrappedCorrectRd,
    /// As above but RD carries the planted `p`-vs-`p−1` prior-ratio error.
    BuggyRd,
}

/// RemoveDimension with the wrong-trials error injected: the Binomial count
/// prior ratio computed over `p` trials instead of `p − 1`, i.e. the correct
/// ratio plus `ln((p−d+1)/(p−d+2))` at pre-move dimension count d (the same
/// derivation as `gate_tests::rd_p_vs_p_minus_one_injection_turns_oracles_red`).
#[derive(Debug)]
struct BuggyRemoveDimension;

impl ProposalMove for BuggyRemoveDimension {
    fn name(&self) -> &'static str {
        "RemoveDimension"
    }
    fn reverse(&self) -> Reverse {
        Reverse::Named("AddDimension")
    }
    fn is_valid(&self, tessellation: &Tessellation, ctx: &ModelCtx) -> bool {
        RemoveDimension.is_valid(tessellation, ctx)
    }
    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        RemoveDimension.propose(tessellation, ctx, rng)
    }
    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        let d = old.dims().len() as f64;
        let p = ctx.p as f64;
        RemoveDimension.log_structure_ratio(old, proposed, ctx)
            + crate::engine::mathsfn::ln((p - d + 1.0) / (p - d + 2.0))
    }
}

/// RemoveDimension delegated unchanged, registered as a custom move.
#[derive(Debug)]
struct WrappedRemoveDimension;

impl ProposalMove for WrappedRemoveDimension {
    fn name(&self) -> &'static str {
        "RemoveDimension"
    }
    fn reverse(&self) -> Reverse {
        Reverse::Named("AddDimension")
    }
    fn is_valid(&self, tessellation: &Tessellation, ctx: &ModelCtx) -> bool {
        RemoveDimension.is_valid(tessellation, ctx)
    }
    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        RemoveDimension.propose(tessellation, ctx, rng)
    }
    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        RemoveDimension.log_structure_ratio(old, proposed, ctx)
    }
}

fn build_move_set(kind: MoveSetKind) -> MoveSet {
    let rd: Box<dyn ProposalMove> = match kind {
        MoveSetKind::Standard => return MoveSetBuilder::stone_gosling().build().unwrap(),
        MoveSetKind::WrappedCorrectRd => Box::new(WrappedRemoveDimension),
        MoveSetKind::BuggyRd => Box::new(BuggyRemoveDimension),
    };
    MoveSetBuilder::empty()
        .with_move(Box::new(AddCentre), 0.2)
        .with_move(Box::new(RemoveCentre), 0.2)
        .with_move(Box::new(AddDimension), 0.2)
        .with_move(rd, 0.2)
        .with_move(Box::new(Change), 0.1)
        .with_move(Box::new(Swap), 0.1)
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// The prior/data generator (must match the sampler's model exactly)
// ---------------------------------------------------------------------------

/// One joint prior draw: the tessellations (with their cached assignments on
/// the fixture design) and σ².
struct PriorDraw {
    tessellations: Vec<Tessellation>,
    assignments: Vec<Vec<usize>>,
    /// Per-tessellation row-normalised membership matrices: `Some` on the
    /// soft configuration, `None` under hard membership.
    memberships: Option<Vec<Vec<f64>>>,
    sigma_sq: f64,
}

impl PriorDraw {
    /// Ensemble fit at observation `row`: `Σ_j μ_{j, cell_j(row)}` under hard
    /// membership, `Σ_j φ_j(row)·μ_j` under soft.
    fn fit_at(&self, row: usize) -> f64 {
        match &self.memberships {
            None => self
                .tessellations
                .iter()
                .zip(&self.assignments)
                .map(|(tessellation, assignment)| tessellation.mus()[assignment[row]])
                .sum(),
            Some(memberships) => self
                .tessellations
                .iter()
                .zip(memberships)
                .map(|(tessellation, membership)| {
                    let b = tessellation.n_cells();
                    membership[row * b..(row + 1) * b]
                        .iter()
                        .zip(tessellation.mus())
                        .map(|(phi, mu)| phi * mu)
                        .sum::<f64>()
                })
                .sum(),
        }
    }
}

/// All d-subsets of `0..p` (ascending; p is tiny here).
fn subsets(p: usize, d: usize) -> Vec<Vec<usize>> {
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

/// Sample a dimension subset of size `d` from the weighted subset prior
/// `P(D | d, s) ∝ ∏_{k∈D} s_k` by exact enumeration: sequential
/// weighted draws would give a different (successive-sampling) law, so this
/// must enumerate. Uniform weights reduce to the paper's uniform subset.
fn draw_dims(p: usize, d: usize, weights: &[f64], rng: &mut ChaCha8Rng) -> Vec<usize> {
    let candidates = subsets(p, d);
    let masses: Vec<f64> = candidates
        .iter()
        .map(|dims| dims.iter().map(|&k| weights[k]).product())
        .collect();
    let total: f64 = masses.iter().sum();
    let target = uniform_f64(rng) * total;
    let mut cumulative = 0.0;
    for (dims, mass) in candidates.iter().zip(&masses) {
        cumulative += mass;
        if target < cumulative {
            return dims.clone();
        }
    }
    candidates.last().unwrap().clone()
}

/// Draw one tessellation from the prior, conditioned on no empty cell over
/// the fixture design (rejection): the sampler's empty-cell guard restricts
/// the prior support per tessellation, and the
/// generator must condition on exactly the same event or SBC is testing the
/// wrong prior.
fn draw_tessellation(
    fixture: &GateFixture,
    coord_dists: &[Arc<dyn CoordinateDistribution>],
    assigner: &dyn CellAssigner,
    rng: &mut ChaCha8Rng,
) -> (Tessellation, Vec<usize>, Option<Vec<f64>>) {
    let sigma_mu = fixture.sigma_mu_sq().sqrt();
    let poisson = rand_distr::Poisson::new(fixture.lambda_c).unwrap();
    let theta = fixture.omega / fixture.p() as f64;
    let binomial = rand_distr::Binomial::new((fixture.p() - 1) as u64, theta).unwrap();

    for _attempt in 0..100_000 {
        let b = 1 + Distribution::<f64>::sample(&poisson, rng) as usize;
        let d = 1 + Distribution::sample(&binomial, rng) as usize;
        let dims = draw_dims(fixture.p(), d, &fixture.weights, rng);
        let mut centres = Vec::with_capacity(b * d);
        for _cell in 0..b {
            for &dim in &dims {
                centres.push(coord_dists[dim].sample(rng));
            }
        }
        let mus: Vec<f64> = (0..b)
            .map(|_| {
                let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
                sigma_mu * z
            })
            .collect();
        let tessellation = Tessellation::new(centres, dims, mus).unwrap();
        let assignment = assigner
            .assign_cells(&fixture.x, &tessellation)
            .expect("built-in metrics never yield non-finite distances");
        match fixture.tau {
            // Hard guard: every cell owns at least one observation.
            None => {
                let mut occupied = vec![false; b];
                for &cell in &assignment {
                    occupied[cell] = true;
                }
                if occupied.iter().all(|o| *o) {
                    return (tessellation, assignment, None);
                }
            }
            // Soft guard (the same code path as the kernel's): every cell
            // carries strictly positive total membership mass.
            Some(tau) => {
                let kernel = crate::extensions::membership::SoftmaxKernel::new(tau);
                let membership = crate::engine::backfit::compute_memberships(
                    assigner,
                    &kernel,
                    &fixture.x,
                    &tessellation,
                )
                .expect("built-in keys and the softmax kernel are well-formed");
                if crate::engine::backfit::occupied_under_membership(&membership, b) {
                    return (tessellation, assignment, Some(membership));
                }
            }
        }
    }
    panic!("prior rejection sampling failed to find a guard-passing tessellation");
}

fn draw_prior(fixture: &GateFixture, rng: &mut ChaCha8Rng) -> PriorDraw {
    let coord_dists = fixture.coord_dists();
    let assigner = fixture.assigner();
    // σ² ~ inverse-χ²(ν, λ) = 1/Gamma(ν/2, scale = 2/(νλ)): the same
    // parameterisation as the sampler's Gibbs draw at RSS = 0, n = 0.
    let gamma =
        rand_distr::Gamma::new(0.5 * fixture.nu, 2.0 / (fixture.nu * fixture.lambda)).unwrap();
    let precision: f64 = Distribution::sample(&gamma, rng);
    let sigma_sq = 1.0 / precision;

    let mut tessellations = Vec::with_capacity(fixture.m);
    let mut assignments = Vec::with_capacity(fixture.m);
    let mut memberships = fixture.tau.map(|_| Vec::with_capacity(fixture.m));
    for _ in 0..fixture.m {
        let (tessellation, assignment, membership) =
            draw_tessellation(fixture, &coord_dists, &assigner, rng);
        tessellations.push(tessellation);
        assignments.push(assignment);
        if let (Some(all), Some(one)) = (memberships.as_mut(), membership) {
            all.push(one);
        }
    }
    PriorDraw {
        tessellations,
        assignments,
        memberships,
        sigma_sq,
    }
}

/// y | θ (scaled space): `y_i = F_i + σ z_i`.
fn draw_response(prior: &PriorDraw, n: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let sigma = prior.sigma_sq.sqrt();
    (0..n)
        .map(|row| {
            let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
            prior.fit_at(row) + sigma * z
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test quantities (shared by SBC and Geweke)
// ---------------------------------------------------------------------------

/// The two fixed design rows whose ensemble fit is tracked.
const F_ROWS: [usize; 2] = [0, 29];

/// Continuous and (tie-broken) discrete test quantities of the state
/// `(tessellations, σ²)` and, for the data-dependent one, the response.
fn state_quantities(
    tessellations: &[Tessellation],
    sigma_sq: f64,
    fit: impl Fn(usize) -> f64,
    y: &[f64],
    full_fit: impl Fn(usize) -> f64,
) -> BTreeMap<&'static str, f64> {
    let mut quantities = BTreeMap::new();
    quantities.insert("sigma_sq", sigma_sq);
    let mean_mu: f64 = tessellations
        .iter()
        .map(|t| t.mus().iter().sum::<f64>() / t.n_cells() as f64)
        .sum();
    quantities.insert("mean_mu", mean_mu);
    quantities.insert(
        "total_cells",
        tessellations.iter().map(|t| t.n_cells() as f64).sum(),
    );
    quantities.insert(
        "total_dims",
        tessellations.iter().map(|t| t.dims().len() as f64).sum(),
    );
    quantities.insert("f_row_a", fit(F_ROWS[0]));
    quantities.insert("f_row_b", fit(F_ROWS[1]));
    // Data-dependent quantity: standardised residual sum of squares,
    // sensitive to joint miscalibration of F and σ² that the marginal
    // quantities can miss.
    let rss: f64 = y
        .iter()
        .enumerate()
        .map(|(row, &value)| {
            let residual = value - full_fit(row);
            residual * residual
        })
        .sum();
    quantities.insert("rss_over_sigma_sq", rss / sigma_sq);
    quantities
}

// ---------------------------------------------------------------------------
// SBC: run replications, emit ranks as CSV for the R ECDF-band verdict
// ---------------------------------------------------------------------------

fn run_sbc_and_emit(fixture: &GateFixture, moves: MoveSetKind, file_stem: &str) {
    let replications = sbc_replications();
    let n_draws = sbc_draws();
    let thin = sbc_thin();
    let burn_in = sbc_burn_in();
    let n = fixture.n();

    // quantity → ranks across replications.
    let mut ranks: BTreeMap<&'static str, Vec<usize>> = BTreeMap::new();

    for replication in 0..replications {
        let mut generator_seed = fixture.seed ^ ((replication as u64) << 20);
        let mut rng = ChaCha8Rng::from_seed(expand_seed(splitmix64(&mut generator_seed)));

        let prior = draw_prior(fixture, &mut rng);
        let y = draw_response(&prior, n, &mut rng);
        let truth = state_quantities(
            &prior.tessellations,
            prior.sigma_sq,
            |row| prior.fit_at(row),
            &y,
            |row| prior.fit_at(row),
        );

        let sampler_seed = fixture.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ replication as u64;
        let mut sampler = fixture.sampler(sampler_seed, y.clone(), moves);
        for _ in 0..burn_in {
            sampler.step().expect("standard-path sweeps cannot fail");
        }
        let mut posterior: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        for _ in 0..n_draws {
            for _ in 0..thin {
                sampler.step().expect("standard-path sweeps cannot fail");
            }
            let sigma_sq;
            let tessellations;
            {
                let draw = sampler.step().expect("standard-path sweeps cannot fail");
                sigma_sq = draw.sigma_sq;
                tessellations = draw.tessellations.to_vec();
            }
            let fit = sampler.fit_values().to_vec();
            let quantities =
                state_quantities(&tessellations, sigma_sq, |row| fit[row], &y, |row| fit[row]);
            for (name, value) in quantities {
                posterior.entry(name).or_default().push(value);
            }
        }

        for (name, true_value) in truth {
            let rank = sbc_rank(true_value, &posterior[name], &mut rng);
            ranks.entry(name).or_default().push(rank);
        }
    }

    // Emit: one CSV per battery config; the R job renders the verdict.
    let dir = out_dir();
    std::fs::create_dir_all(&dir).expect("stat-gates output directory must be creatable");
    let path = dir.join(format!("{file_stem}.csv"));
    let mut file = std::fs::File::create(&path).expect("rank CSV must be writable");
    writeln!(file, "quantity,rank,n_draws").unwrap();
    for (name, quantity_ranks) in &ranks {
        for rank in quantity_ranks {
            writeln!(file, "{name},{rank},{n_draws}").unwrap();
        }
    }
    println!(
        "SBC[{}]: wrote {} ranks/quantity x {} quantities to {}",
        fixture.name,
        replications,
        ranks.len(),
        path.display()
    );
}

// ---------------------------------------------------------------------------
// Geweke: marginal-conditional vs successive-conditional, KS-judged
// ---------------------------------------------------------------------------

struct GewekeOutcome {
    stat: &'static str,
    d: f64,
    critical: f64,
}

impl GewekeOutcome {
    fn passed(&self) -> bool {
        self.d <= self.critical
    }
}

/// Run both simulators and KS-compare every statistic. `alpha` is the
/// per-battery significance, Bonferroni-split across statistics (the KS
/// critical value is conservative for the discrete count statistics, a
/// deliberate choice for fewer false reds).
fn run_geweke(fixture: &GateFixture, moves: MoveSetKind, alpha: f64) -> Vec<GewekeOutcome> {
    let n_mc = geweke_mc();
    let n_sc = geweke_sc();
    let thin = geweke_thin();
    let discard = 100; // the SC chain starts at stationarity (prior init);
    // the short discard is a safety margin only.
    let n = fixture.n();

    // --- marginal-conditional: i.i.d. draws from prior × likelihood ---
    let mut mc_rng = ChaCha8Rng::from_seed(expand_seed(fixture.seed ^ 0x6E77));
    let mut mc: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    for _ in 0..n_mc {
        let prior = draw_prior(fixture, &mut mc_rng);
        let y = draw_response(&prior, n, &mut mc_rng);
        let quantities = state_quantities(
            &prior.tessellations,
            prior.sigma_sq,
            |row| prior.fit_at(row),
            &y,
            |row| prior.fit_at(row),
        );
        for (name, value) in quantities {
            mc.entry(name).or_default().push(value);
        }
    }

    // --- successive-conditional: alternate y | θ and one Gibbs sweep θ | y ---
    let mut sc_rng = ChaCha8Rng::from_seed(expand_seed(fixture.seed ^ 0x5C5C));
    let init = draw_prior(fixture, &mut sc_rng);
    let init_sigma_sq = init.sigma_sq;
    let y0 = draw_response(&init, n, &mut sc_rng);
    let mut sampler = fixture.sampler(fixture.seed ^ 0xC4A1, y0, moves);
    sampler
        .set_state_for_tests(init.tessellations)
        .expect("prior state is assignable");
    let mut current_sigma_sq = init_sigma_sq;

    let mut sc: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let total_iterations = discard + n_sc * thin;
    for iteration in 0..total_iterations {
        // y | θ (generator RNG: a separate stream from the kernel's is fine;
        // the two updates are distinct Gibbs blocks).
        let sigma = current_sigma_sq.sqrt();
        let y: Vec<f64> = (0..n)
            .map(|row| {
                let z: f64 = Distribution::sample(&rand_distr::StandardNormal, &mut sc_rng);
                sampler.fit_values()[row] + sigma * z
            })
            .collect();
        sampler.replace_scaled_response(y.clone());
        // θ | y: one full Gibbs sweep of the real kernel under test.
        let (sigma_sq, tessellations) = {
            let draw = sampler.step().expect("standard-path sweeps cannot fail");
            (draw.sigma_sq, draw.tessellations.to_vec())
        };
        current_sigma_sq = sigma_sq;
        if iteration >= discard && (iteration - discard + 1) % thin == 0 {
            let fit = sampler.fit_values().to_vec();
            let quantities =
                state_quantities(&tessellations, sigma_sq, |row| fit[row], &y, |row| fit[row]);
            for (name, value) in quantities {
                sc.entry(name).or_default().push(value);
            }
        }
    }

    let alpha_per_stat = alpha / mc.len() as f64;
    mc.iter()
        .map(|(name, mc_values)| {
            let sc_values = &sc[name];
            GewekeOutcome {
                stat: name,
                d: ks_two_sample(mc_values, sc_values),
                critical: ks_critical_value(alpha_per_stat, mc_values.len(), sc_values.len()),
            }
        })
        .collect()
}

fn assert_geweke_passes(fixture: &GateFixture, moves: MoveSetKind) {
    let outcomes = run_geweke(fixture, moves, 0.01);
    let mut failures = Vec::new();
    for outcome in &outcomes {
        println!(
            "geweke[{}] {}: D = {:.4} vs critical {:.4}: {}",
            fixture.name,
            outcome.stat,
            outcome.d,
            outcome.critical,
            if outcome.passed() { "ok" } else { "FAIL" }
        );
        if !outcome.passed() {
            failures.push(outcome.stat);
        }
    }
    assert!(
        failures.is_empty(),
        "Geweke joint-distribution cross-check failed for `{}` on {:?}: the \
         marginal-conditional and successive-conditional simulators disagree: \
         the Gibbs kernel does not target its own prior × likelihood",
        fixture.name,
        failures
    );
}

// ---------------------------------------------------------------------------
// The battery tests (CI leg: calibration / release; never fast-PR)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn sbc_ranks_small_p() {
    run_sbc_and_emit(&fixture_small_p(), MoveSetKind::Standard, "sbc-small-p");
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn sbc_ranks_spherical() {
    run_sbc_and_emit(&fixture_spherical(), MoveSetKind::Standard, "sbc-spherical");
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn sbc_ranks_weighted_inclusion() {
    run_sbc_and_emit(
        &fixture_weighted(),
        MoveSetKind::Standard,
        "sbc-weighted-inclusion",
    );
}

/// Emits ranks from the injected sampler; the workflow passes this
/// file to the R job with `--expect-fail`, and the band test must reject it
/// (the "gate has teeth" acceptance bullet).
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn sbc_ranks_injection_rd() {
    run_sbc_and_emit(&fixture_small_p(), MoveSetKind::BuggyRd, "sbc-injection-rd");
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_small_p() {
    assert_geweke_passes(&fixture_small_p(), MoveSetKind::Standard);
}

/// Pure-prior control: with σ_μ² ≈ 0 the marginal-likelihood ratio is exactly
/// zero, so the kernel is a pure sampler of the (empty-cell-restricted)
/// structural prior, and its cell-count histogram must match the i.i.d.
/// generator's. This is the check that isolated the 2026-07-02 AC/RC
/// pick-factor bug to the ratio itself (χ² band, deterministic seeds).
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn pure_prior_kernel_cell_count_marginal_matches_generator() {
    // σ_μ² ≈ 0 ⇒ the marginal-likelihood ratio is exactly 0 ⇒ the kernel is a
    // pure sampler of the empty-cell-restricted structural prior. Any cell-count
    // gap against the generator is then a prior-kernel disagreement, no data.
    let mut fixture = fixture_small_p();
    fixture.k = 1e12; // σ_μ² = 0.25/(k²m) ~ 2.5e-26
    fixture.omega = 1e-6;
    fixture.m = 1;
    let n_mc = 20_000;
    let mut rng = ChaCha8Rng::from_seed(expand_seed(fixture.seed ^ 0xDEB6));
    let mut mc_hist = BTreeMap::new();
    for _ in 0..n_mc {
        let prior = draw_prior(&fixture, &mut rng);
        *mc_hist
            .entry(prior.tessellations[0].n_cells())
            .or_insert(0usize) += 1;
    }
    let y0 = vec![0.0; fixture.n()];
    let mut sampler = fixture.sampler(1234, y0, MoveSetKind::Standard);
    let mut sc_hist = BTreeMap::new();
    for _ in 0..500 {
        sampler.step().unwrap();
    }
    for _ in 0..n_mc {
        for _ in 0..3 {
            sampler.step().unwrap();
        }
        let draw = sampler.step().unwrap();
        *sc_hist
            .entry(draw.tessellations[0].n_cells())
            .or_insert(0usize) += 1;
    }
    // Judge with the same KS machinery (counts → per-draw values); the
    // discrete-tie conservatism of `ks_two_sample` applies.
    let expand = |hist: &BTreeMap<usize, usize>| -> Vec<f64> {
        hist.iter()
            .flat_map(|(&b, &count)| std::iter::repeat_n(b as f64, count))
            .collect()
    };
    let (mc_values, sc_values) = (expand(&mc_hist), expand(&sc_hist));
    let d = ks_two_sample(&mc_values, &sc_values);
    let critical = ks_critical_value(0.01, mc_values.len(), sc_values.len());
    println!("b   generator   kernel");
    for b in 1..12usize {
        println!(
            "{b}: {:>8} {:>8}",
            mc_hist.get(&b).copied().unwrap_or(0),
            sc_hist.get(&b).copied().unwrap_or(0)
        );
    }
    println!("pure-prior cell-count KS: D = {d:.4} vs critical {critical:.4}");
    assert!(
        d <= critical,
        "the kernel's pure-prior cell-count marginal does not match the \
         generator (D = {d:.4} > {critical:.4}): an AC/RC-ratio-class bug"
    );
}

// ---------------------------------------------------------------------------
// The H-AddiVortes battery (two shared-machinery ensemble instances)
// ---------------------------------------------------------------------------

/// H Geweke: the marginal-conditional simulator draws (mean tessellations,
/// variance tessellations) i.i.d. from the joint prior and y from the
/// heteroscedastic likelihood; the successive-conditional simulator
/// alternates `y | θ` with one full two-ensemble Gibbs sweep of the real
/// kernel (variance backfit inside `ScaleModel::update`, then the weighted
/// mean backfit). Every statistic must agree.
fn run_h_geweke(fixture: &HFixture, alpha: f64) -> Vec<GewekeOutcome> {
    let n_mc = geweke_mc();
    let n_sc = geweke_sc();
    let thin = geweke_thin();
    let discard = 100;
    let base = &fixture.base;
    let n = base.n();

    // --- marginal-conditional: i.i.d. draws from prior × likelihood ---
    let mut mc_rng = ChaCha8Rng::from_seed(expand_seed(base.seed ^ 0x6E77));
    let mut mc: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    for _ in 0..n_mc {
        let prior = draw_h_prior(fixture, &mut mc_rng);
        let y = draw_h_response(&prior, n, &mut mc_rng);
        let quantities = h_state_quantities(
            &prior.mean.tessellations,
            &prior.var_tessellations,
            |row| prior.mean.fit_at(row),
            |row| prior.s_sq_at(row),
            &y,
        );
        for (name, value) in quantities {
            mc.entry(name).or_default().push(value);
        }
    }

    // --- successive-conditional: y | θ then one two-ensemble sweep ---
    let mut sc_rng = ChaCha8Rng::from_seed(expand_seed(base.seed ^ 0x5C5C));
    let init = draw_h_prior(fixture, &mut sc_rng);
    let y0 = draw_h_response(&init, n, &mut sc_rng);
    let (mut sampler, shared) = h_sampler(fixture, base.seed ^ 0xC4A1, y0);
    sampler
        .set_state_for_tests(init.mean.tessellations)
        .expect("prior mean state is assignable");
    shared
        .lock()
        .expect("test lock")
        .set_state_for_tests(init.var_tessellations, &base.x, &base.assigner())
        .expect("prior variance state is assignable");

    let mut sc: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let total_iterations = discard + n_sc * thin;
    for iteration in 0..total_iterations {
        // y | θ from the current (F, s²) state, generator stream.
        let fit = sampler.fit_values().to_vec();
        let s_sq: Vec<f64> = shared
            .lock()
            .expect("test lock")
            .s_sq_values()
            .expect("variance state injected above")
            .to_vec();
        let y: Vec<f64> = (0..n)
            .map(|row| {
                let z: f64 = Distribution::sample(&rand_distr::StandardNormal, &mut sc_rng);
                fit[row] + s_sq[row].sqrt() * z
            })
            .collect();
        sampler.replace_scaled_response(y.clone());
        // θ | y: one full two-ensemble Gibbs sweep of the real kernel.
        let mean_tessellations = {
            let draw = sampler.step().expect("H-path sweeps cannot fail");
            draw.tessellations.to_vec()
        };
        if iteration >= discard && (iteration - discard + 1) % thin == 0 {
            let fit = sampler.fit_values().to_vec();
            let handle = shared.lock().expect("test lock");
            let var_tessellations = handle
                .tessellations()
                .expect("variance state present")
                .to_vec();
            let s_sq = handle.s_sq_values().expect("variance state present");
            let quantities = h_state_quantities(
                &mean_tessellations,
                &var_tessellations,
                |row| fit[row],
                |row| s_sq[row],
                &y,
            );
            drop(handle);
            for (name, value) in quantities {
                sc.entry(name).or_default().push(value);
            }
        }
    }

    let alpha_per_stat = alpha / mc.len() as f64;
    mc.iter()
        .map(|(name, mc_values)| {
            let sc_values = &sc[name];
            GewekeOutcome {
                stat: name,
                d: ks_two_sample(mc_values, sc_values),
                critical: ks_critical_value(alpha_per_stat, mc_values.len(), sc_values.len()),
            }
        })
        .collect()
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_h_variance() {
    let fixture = fixture_h();
    let outcomes = run_h_geweke(&fixture, 0.01);
    let mut failures = Vec::new();
    for outcome in &outcomes {
        println!(
            "geweke[{}] {}: D = {:.4} vs critical {:.4}: {}",
            fixture.base.name,
            outcome.stat,
            outcome.d,
            outcome.critical,
            if outcome.passed() { "ok" } else { "FAIL" }
        );
        if !outcome.passed() {
            failures.push(outcome.stat);
        }
    }
    assert!(
        failures.is_empty(),
        "H Geweke joint-distribution cross-check failed on {failures:?}: the two-ensemble \
         kernel does not target its own prior × likelihood"
    );
}

/// H SBC: rank CSVs for the R ECDF-band verdict, exactly like the
/// single-ensemble configs.
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn sbc_ranks_h_variance() {
    let fixture = fixture_h();
    let base = &fixture.base;
    let replications = sbc_replications();
    let n_draws = sbc_draws();
    let thin = sbc_thin();
    let burn_in = sbc_burn_in();
    let n = base.n();

    let mut ranks: BTreeMap<&'static str, Vec<usize>> = BTreeMap::new();
    for replication in 0..replications {
        let mut generator_seed = base.seed ^ ((replication as u64) << 20);
        let mut rng = ChaCha8Rng::from_seed(expand_seed(splitmix64(&mut generator_seed)));

        let prior = draw_h_prior(&fixture, &mut rng);
        let y = draw_h_response(&prior, n, &mut rng);
        let truth = h_state_quantities(
            &prior.mean.tessellations,
            &prior.var_tessellations,
            |row| prior.mean.fit_at(row),
            |row| prior.s_sq_at(row),
            &y,
        );

        let sampler_seed = base.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ replication as u64;
        let (mut sampler, shared) = h_sampler(&fixture, sampler_seed, y.clone());
        for _ in 0..burn_in {
            sampler.step().expect("H-path sweeps cannot fail");
        }
        let mut posterior: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        for _ in 0..n_draws {
            for _ in 0..thin {
                sampler.step().expect("H-path sweeps cannot fail");
            }
            let mean_tessellations = {
                let draw = sampler.step().expect("H-path sweeps cannot fail");
                draw.tessellations.to_vec()
            };
            let fit = sampler.fit_values().to_vec();
            let handle = shared.lock().expect("test lock");
            let var_tessellations = handle
                .tessellations()
                .expect("variance state present after the first sweep")
                .to_vec();
            let s_sq = handle.s_sq_values().expect("variance state present");
            let quantities = h_state_quantities(
                &mean_tessellations,
                &var_tessellations,
                |row| fit[row],
                |row| s_sq[row],
                &y,
            );
            drop(handle);
            for (name, value) in quantities {
                posterior.entry(name).or_default().push(value);
            }
        }
        for (name, true_value) in truth {
            let rank = sbc_rank(true_value, &posterior[name], &mut rng);
            ranks.entry(name).or_default().push(rank);
        }
    }

    let dir = out_dir();
    std::fs::create_dir_all(&dir).expect("stat-gates output directory must be creatable");
    let path = dir.join("sbc-h-variance.csv");
    let mut file = std::fs::File::create(&path).expect("rank CSV must be writable");
    writeln!(file, "quantity,rank,n_draws").unwrap();
    for (name, quantity_ranks) in &ranks {
        for rank in quantity_ranks {
            writeln!(file, "{name},{rank},{n_draws}").unwrap();
        }
    }
    println!(
        "SBC[h-variance]: wrote {} ranks/quantity x {} quantities to {}",
        replications,
        ranks.len(),
        path.display()
    );
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_spherical() {
    assert_geweke_passes(&fixture_spherical(), MoveSetKind::Standard);
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn sbc_ranks_soft_membership() {
    run_sbc_and_emit(
        &fixture_soft(),
        MoveSetKind::Standard,
        "sbc-soft-membership",
    );
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_soft_membership() {
    assert_geweke_passes(&fixture_soft(), MoveSetKind::Standard);
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_weighted_inclusion() {
    assert_geweke_passes(&fixture_weighted(), MoveSetKind::Standard);
}

/// Positive control isolating the injection: the same custom-registered move
/// set with a correct RemoveDimension (weighted-valid selection rule, like the
/// injection run) passes, so a red `geweke_injection_rd_goes_red` is
/// attributable to the injected ratio, not to the selection-rule switch.
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_wrapped_correct_rd_positive_control() {
    assert_geweke_passes(&fixture_small_p(), MoveSetKind::WrappedCorrectRd);
}

/// Negative control (suite self-validation): the planted
/// RD-only injection must make the Geweke cross-check go red.
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_injection_rd_goes_red() {
    let outcomes = run_geweke(&fixture_small_p(), MoveSetKind::BuggyRd, 0.01);
    for outcome in &outcomes {
        println!(
            "geweke[injection-rd] {}: D = {:.4} vs critical {:.4}",
            outcome.stat, outcome.d, outcome.critical
        );
    }
    assert!(
        outcomes.iter().any(|outcome| !outcome.passed()),
        "the RD p-vs-p−1 injection was NOT detected: the Geweke battery has no teeth"
    );
}

// ---------------------------------------------------------------------------
// Interval coverage: Friedman (release leg only). Interval coverage, not code coverage.
// ---------------------------------------------------------------------------

/// Friedman (1991) benchmark data in raw space: X ~ U(0,1)^p,
/// f(x) = 10·sin(π x₁x₂) + 20(x₃ − 0.5)² + 10x₄ + 5x₅, y = f + σ·ε.
/// Returns `(x, y, f_true)`.
fn friedman(n: usize, p: usize, noise_sd: f64, seed: u64) -> (Data, Vec<f64>, Vec<f64>) {
    debug_assert!(p >= 5);
    let mut rng = ChaCha8Rng::from_seed(expand_seed(seed));
    let mut values = Vec::with_capacity(n * p);
    let mut f_true = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..p).map(|_| uniform_f64(&mut rng)).collect();
        let f = 10.0 * crate::engine::mathsfn::sin(std::f64::consts::PI * row[0] * row[1])
            + 20.0 * (row[2] - 0.5) * (row[2] - 0.5)
            + 10.0 * row[3]
            + 5.0 * row[4];
        let z: f64 = Distribution::sample(&rand_distr::StandardNormal, &mut rng);
        f_true.push(f);
        y.push(f + noise_sd * z);
        values.extend_from_slice(&row);
    }
    (Data::new(values, n, p).unwrap(), y, f_true)
}

/// Interval coverage: Friedman n=150 p=10, the paper's default configuration,
/// averaged over a fixed seed set. Formal two-sided score test of
/// H₀: coverage = 0.90 at α = 0.01 on the low side; the acceptance region is
/// capped at 0.95 (+ the same z-slack) on the high side to catch
/// over-inflated intervals. The intervals are the naive f-intervals
/// (`credible_interval`, no noise term), matching the paper's §5.1 usage.
/// Minimum detectable bias at these sizes (N = 150·10 trials, α = 0.01,
/// power 0.5): ≈ ±2.0 percentage points around 0.90.
#[test]
#[ignore = "stochastic battery: release CI leg only"]
fn interval_coverage_friedman() {
    let n = 150;
    let p = 10;
    let n_fits = env_size("INTERVAL_COVERAGE_FITS", 10);
    let level = 0.9;

    let mut covered = 0usize;
    let mut trials = 0usize;
    let mut sigma_means = Vec::with_capacity(n_fits);
    for fit_index in 0..n_fits {
        let seed = 20_260_702 + fit_index as u64;
        let (x, y, f_true) = friedman(n, p, 1.0, seed ^ 0xF00D);
        let model = AddiVortesConfig::new(seed)
            .fit(&x, &y)
            .expect("Friedman fit at paper defaults must succeed");
        let intervals = model
            .credible_interval(&x, level)
            .expect("interval computation must succeed");
        for (interval, &truth) in intervals.iter().zip(&f_true) {
            trials += 1;
            if interval.lower <= truth && truth <= interval.upper {
                covered += 1;
            }
        }
        let sigma = model.sigma();
        sigma_means.push(sigma.iter().sum::<f64>() / sigma.len() as f64);
    }

    let p_hat = covered as f64 / trials as f64;
    let se = (0.9 * 0.1 / trials as f64).sqrt();
    let z_low = (p_hat - 0.90) / se;
    let z_cap = (p_hat - 0.95) / se;
    println!(
        "interval coverage: {covered}/{trials} = {p_hat:.4} (z vs 0.90 = {z_low:+.2}, \
         z vs 0.95 cap = {z_cap:+.2})"
    );
    // σ chain mixes through truth: the response-scale posterior
    // mean error SD must bracket the generating σ = 1 within a lenient band.
    let sigma_grand = sigma_means.iter().sum::<f64>() / sigma_means.len() as f64;
    println!("interval-coverage posterior mean sigma (truth 1.0): {sigma_grand:.3}");
    assert!(
        z_low > -2.576,
        "interval coverage {p_hat:.4} is significantly BELOW the nominal 0.90 \
         (z = {z_low:.2}): the posterior intervals are too narrow"
    );
    assert!(
        z_cap < 2.576,
        "interval coverage {p_hat:.4} is significantly above the 0.95 over-inflation \
         cap (z = {z_cap:.2}): the posterior intervals are too wide"
    );
    assert!(
        (0.5..2.0).contains(&sigma_grand),
        "posterior sigma {sigma_grand} does not bracket the generating value 1.0"
    );
}

// ---------------------------------------------------------------------------
// Reference comparison: fixture and this crate's RMSE emission (release leg; the R comparison
// verdict is ci/reference-compare.R against the authors' R package at a pinned commit)
// ---------------------------------------------------------------------------

/// Emit the shared reference-comparison fixture (train/test CSVs) and this crate's f-recovery
/// RMSE: fit on n=150 p=10 Friedman train data at the default configuration,
/// evaluate RMSE(f̂, f_true) on 800 independent test points.
/// `ci/reference-compare.R` fits the authors' R package (pinned) on the same
/// CSVs and asserts |RMSE_rust − RMSE_R| ≤ 0.05.
///
/// The 800-point evaluation follows the paper's Figure 9, but the rest of that
/// protocol deliberately does not: Figure 9 simulates 100 datasets at n=200
/// with 5-fold cross-validated hyperparameters (Table 2), where this fits one
/// n=150 dataset at the defaults. Nothing here reproduces the paper's figure —
/// the comparison is Rust against R on byte-identical CSVs, so the fixture only
/// has to be shared, not paper-faithful.
#[test]
#[ignore = "stochastic battery: release CI leg only"]
fn reference_comparison_emit_fixture_and_rmse() {
    let (x_train, y_train, _) = friedman(150, 10, 1.0, 0x14AC);
    let (x_test, _, f_test) = friedman(800, 10, 1.0, 0x14AD);

    let model = AddiVortesConfig::new(20_260_702)
        .fit(&x_train, &y_train)
        .expect("Friedman fit at paper defaults must succeed");
    let predictions = model.predict(&x_test).expect("prediction must succeed");
    let mut sum_sq = 0.0_f64;
    for (prediction, truth) in predictions.iter().zip(&f_test) {
        let residual = prediction - truth;
        sum_sq += residual * residual;
    }
    let rmse = (sum_sq / f_test.len() as f64).sqrt();

    let dir = out_dir();
    std::fs::create_dir_all(&dir).expect("stat-gates output directory must be creatable");
    let write_csv = |name: &str, x: &Data, extra: Option<(&str, &[f64])>| {
        let mut file = std::fs::File::create(dir.join(name)).expect("CSV must be writable");
        let p = x.n_cols();
        let mut header: Vec<String> = (1..=p).map(|c| format!("x{c}")).collect();
        if let Some((label, _)) = extra {
            header.push(label.into());
        }
        writeln!(file, "{}", header.join(",")).unwrap();
        for row in 0..x.n_rows() {
            let mut fields: Vec<String> = x.row(row).iter().map(|v| format!("{v:.17e}")).collect();
            if let Some((_, values)) = extra {
                fields.push(format!("{:.17e}", values[row]));
            }
            writeln!(file, "{}", fields.join(",")).unwrap();
        }
    };
    write_csv("reference-train.csv", &x_train, Some(("y", &y_train)));
    write_csv("reference-test.csv", &x_test, Some(("f_true", &f_test)));
    std::fs::write(dir.join("reference-rust-rmse.txt"), format!("{rmse:.6}\n"))
        .expect("RMSE file must be writable");
    println!(
        "reference comparison: rust f-recovery RMSE {rmse:.4} written to {} (R verdict: ci/reference-compare.R)",
        dir.display()
    );
    // Standalone sanity only (measured 2026-07-02: ≈ 1.67 at n_train = 150,
    // paper defaults, out of sample): a gross regression fails here without
    // needing the R side; the real ±0.05 verdict is ci/reference-compare.R.
    assert!(
        rmse < 3.0,
        "rust f-recovery RMSE {rmse:.3} is far outside the measured \
         neighbourhood (≈ 1.7): grossly miscalibrated"
    );
}

// ---------------------------------------------------------------------------
// The public battery drivers must reproduce the gate verdicts: green on the
// correct kernel, red on the wrong-trials injection.
// ---------------------------------------------------------------------------

/// A successive-conditional simulator over the public surface only (the shape
/// an external component author writes): y | θ through `fitted_values` +
/// `set_response` (identity scaler on the pinned-prior path), θ | y through
/// `step`, quantities captured per transition.
struct PublicSc {
    sampler: Sampler,
    rng: ChaCha8Rng,
    sigma_sq: f64,
    y: Vec<f64>,
    quantities: BTreeMap<String, f64>,
}

impl crate::calibration::SuccessiveConditional for PublicSc {
    fn transition(&mut self) -> crate::engine::error::Result<()> {
        let fit = self.sampler.fitted_values();
        let sigma = self.sigma_sq.sqrt();
        for (slot, fitted) in self.y.iter_mut().zip(&fit) {
            let z: f64 = Distribution::sample(&rand_distr::StandardNormal, &mut self.rng);
            *slot = fitted + sigma * z;
        }
        let y = self.y.clone();
        self.sampler.set_response(&y)?;
        let (sigma_sq, tessellations) = {
            let draw = self.sampler.step()?;
            (draw.sigma_sq, draw.tessellations.to_vec())
        };
        self.sigma_sq = sigma_sq;
        let fit = self.sampler.fitted_values();
        self.quantities = public_quantities(&tessellations, sigma_sq, &fit, &self.y);
        Ok(())
    }

    fn quantities(&self) -> BTreeMap<String, f64> {
        self.quantities.clone()
    }
}

fn public_quantities(
    tessellations: &[Tessellation],
    sigma_sq: f64,
    fit: &[f64],
    y: &[f64],
) -> BTreeMap<String, f64> {
    let quantities = state_quantities(tessellations, sigma_sq, |row| fit[row], y, |row| fit[row]);
    quantities
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect()
}

fn run_public_battery(kind: MoveSetKind) -> Vec<crate::calibration::GewekeOutcome> {
    let fixture = fixture_small_p();
    let n = fixture.n();
    // Marginal-conditional generator (the existing in-crate one, wrapped).
    let mut mc_rng = ChaCha8Rng::from_seed(expand_seed(fixture.seed ^ 0xBA77));
    let mut mc_draw = || {
        let prior = draw_prior(&fixture, &mut mc_rng);
        let y = draw_response(&prior, n, &mut mc_rng);
        let fit: Vec<f64> = (0..n).map(|row| prior.fit_at(row)).collect();
        public_quantities(&prior.tessellations, prior.sigma_sq, &fit, &y)
    };
    // Successive-conditional over the public surface, prior-initialised.
    let mut sc_rng = ChaCha8Rng::from_seed(expand_seed(fixture.seed ^ 0xBA55));
    let init = draw_prior(&fixture, &mut sc_rng);
    let y0 = draw_response(&init, n, &mut sc_rng);
    let mut sampler = fixture.sampler(fixture.seed ^ 0xBAC4, y0.clone(), kind);
    sampler
        .set_state_for_tests(init.tessellations)
        .expect("prior state is assignable");
    let mut sc = PublicSc {
        sampler,
        rng: sc_rng,
        sigma_sq: init.sigma_sq,
        y: y0,
        quantities: BTreeMap::new(),
    };
    let spec = crate::calibration::GewekeSpec {
        n_mc: geweke_mc(),
        n_sc: geweke_sc(),
        thin: geweke_thin(),
        ..crate::calibration::GewekeSpec::default()
    };
    crate::calibration::getting_it_right(&spec, &mut mc_draw, &mut sc)
        .expect("standard-path sweeps cannot fail")
}

/// The externalised driver reproduces the gate: green on the correct kernel.
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn battery_public_driver_passes_on_the_correct_kernel() {
    let outcomes = run_public_battery(MoveSetKind::Standard);
    for outcome in &outcomes {
        println!(
            "battery[public] {}: D = {:.4} vs critical {:.4}",
            outcome.statistic, outcome.d, outcome.critical
        );
    }
    assert!(
        outcomes
            .iter()
            .all(crate::calibration::GewekeOutcome::passed),
        "the public battery driver disagrees with the in-crate gate on the correct kernel"
    );
}

/// The public battery's negative control: the planted RD injection must
/// turn the public driver red; the battery has teeth outside the crate too.
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn battery_public_driver_detects_the_rd_injection() {
    let outcomes = run_public_battery(MoveSetKind::BuggyRd);
    for outcome in &outcomes {
        println!(
            "battery[public-injection] {}: D = {:.4} vs critical {:.4}",
            outcome.statistic, outcome.d, outcome.critical
        );
    }
    assert!(
        outcomes.iter().any(|outcome| !outcome.passed()),
        "the RD p-vs-p−1 injection was NOT detected by the public battery driver"
    );
}

// ---------------------------------------------------------------------------
// The DART reference: the adaptive inclusion skeleton, validated
// through the externalised battery (only the battery can: the local checks
// proves mechanics, not statistical validity).
// ---------------------------------------------------------------------------

const DART_ALPHA: f64 = 1.5;

fn fixture_dart() -> GateFixture {
    let metrics = vec![Metric::Euclidean; 3];
    GateFixture {
        name: "dart",
        x: fixture_design(0xDA47, 60, &metrics),
        metrics,
        m: 6,
        nu: 6.0,
        lambda: 0.02,
        lambda_c: 2.0,
        omega: 1.5,
        sigma_c: 0.8,
        k: 3.0,
        weights: vec![1.0; 3], // the *initial* state; DART adapts per sweep
        tau: None,
        seed: 0x5BC0_0006,
    }
}

/// s ~ Dirichlet(α/p) via normalised Gamma draws (the prior of the DART
/// weights).
fn draw_dirichlet_weights(p: usize, alpha: f64, rng: &mut ChaCha8Rng) -> Vec<f64> {
    loop {
        let mut draws = Vec::with_capacity(p);
        let mut total = 0.0_f64;
        for _ in 0..p {
            let gamma = rand_distr::Gamma::new(alpha / p as f64, 1.0).unwrap();
            let draw: f64 = Distribution::sample(&gamma, rng);
            total += draw;
            draws.push(draw);
        }
        for draw in &mut draws {
            *draw /= total;
        }
        if draws.iter().all(|w| w.is_finite() && *w > 0.0) {
            return draws;
        }
    }
}

/// One joint DART prior draw: s from its Dirichlet prior, then the
/// tessellations from the structural prior under s (weighted subsets),
/// conditioned on the hard guard; σ² from the pinned inverse-χ².
fn draw_dart_prior(fixture: &GateFixture, rng: &mut ChaCha8Rng) -> (Vec<f64>, PriorDraw) {
    let s = draw_dirichlet_weights(fixture.p(), DART_ALPHA, rng);
    let mut weighted = GateFixture {
        x: fixture.x.clone(),
        metrics: fixture.metrics.clone(),
        weights: s.clone(),
        name: fixture.name,
        ..*fixture
    };
    weighted.weights = s.clone();
    let prior = draw_prior(&weighted, rng);
    (s, prior)
}

fn dart_quantities(
    s: &[f64],
    tessellations: &[Tessellation],
    sigma_sq: f64,
    fit: impl Fn(usize) -> f64,
    y: &[f64],
) -> BTreeMap<&'static str, f64> {
    let mut quantities = state_quantities(tessellations, sigma_sq, &fit, y, &fit);
    quantities.insert("s_first", s[0]);
    quantities.insert("s_max", s.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    quantities
}

fn dart_sampler(fixture: &GateFixture, seed: u64, y: Vec<f64>) -> Sampler {
    let mut config = AddiVortesConfig::new(seed)
        .with_m(fixture.m)
        .with_nu(fixture.nu)
        .with_omega(fixture.omega)
        .with_lambda_c(fixture.lambda_c)
        .with_sigma_c(fixture.sigma_c)
        .with_k(fixture.k);
    config.inclusion = Some(Arc::new(crate::extensions::inclusion::DartInclusion::new(
        DART_ALPHA,
        fixture.p(),
    )));
    Sampler::pinned_prior_for_tests(
        config,
        fixture.x.clone(),
        fixture.metrics.clone(),
        y,
        fixture.lambda,
        build_move_set(MoveSetKind::Standard),
    )
    .expect("pinned-prior DART sampler construction must succeed")
}

/// DART Geweke through the externalised battery drivers: the adaptive
/// weights are part of θ (s_first/s_max are ranked quantities), so a wrong
/// update (e.g. the uncorrected conjugate Gibbs) shows up here.
fn run_dart_geweke(correction: bool) -> Vec<crate::calibration::GewekeOutcome> {
    let fixture = fixture_dart();
    let n = fixture.n();

    let mut mc_rng = ChaCha8Rng::from_seed(expand_seed(fixture.seed ^ 0x6E77));
    let mut mc_draw = || {
        let (s, prior) = draw_dart_prior(&fixture, &mut mc_rng);
        let y = draw_response(&prior, n, &mut mc_rng);
        dart_quantities(
            &s,
            &prior.tessellations,
            prior.sigma_sq,
            |row| prior.fit_at(row),
            &y,
        )
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect()
    };

    struct DartSc {
        sampler: Sampler,
        rng: ChaCha8Rng,
        sigma_sq: f64,
        y: Vec<f64>,
        latest: BTreeMap<String, f64>,
    }
    impl crate::calibration::SuccessiveConditional for DartSc {
        fn transition(&mut self) -> crate::engine::error::Result<()> {
            let fit = self.sampler.fitted_values();
            let sigma = self.sigma_sq.sqrt();
            for (slot, fitted) in self.y.iter_mut().zip(&fit) {
                let z: f64 = Distribution::sample(&rand_distr::StandardNormal, &mut self.rng);
                *slot = fitted + sigma * z;
            }
            let y = self.y.clone();
            self.sampler.set_response(&y)?;
            let (sigma_sq, tessellations) = {
                let draw = self.sampler.step()?;
                (draw.sigma_sq, draw.tessellations.to_vec())
            };
            self.sigma_sq = sigma_sq;
            let fit = self.sampler.fitted_values();
            let s = self.sampler.inclusion_weights().to_vec();
            self.latest = dart_quantities(&s, &tessellations, sigma_sq, |row| fit[row], &self.y)
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect();
            Ok(())
        }
        fn quantities(&self) -> BTreeMap<String, f64> {
            self.latest.clone()
        }
    }

    let mut sc_rng = ChaCha8Rng::from_seed(expand_seed(fixture.seed ^ 0x5C5C));
    let init = {
        let (_, prior) = draw_dart_prior(&fixture, &mut sc_rng);
        prior
    };
    let y0 = draw_response(&init, n, &mut sc_rng);
    let mut sampler = if correction {
        dart_sampler(&fixture, fixture.seed ^ 0xC4A1, y0.clone())
    } else {
        // The negative control: the uncorrected conjugate Gibbs update (the
        // naive DART port the exactness warning is about).
        let mut config = AddiVortesConfig::new(fixture.seed ^ 0xC4A1)
            .with_m(fixture.m)
            .with_nu(fixture.nu)
            .with_omega(fixture.omega)
            .with_lambda_c(fixture.lambda_c)
            .with_sigma_c(fixture.sigma_c)
            .with_k(fixture.k);
        config.inclusion = Some(Arc::new(UncorrectedDart::new(DART_ALPHA, fixture.p())));
        Sampler::pinned_prior_for_tests(
            config,
            fixture.x.clone(),
            fixture.metrics.clone(),
            y0.clone(),
            fixture.lambda,
            build_move_set(MoveSetKind::Standard),
        )
        .expect("pinned-prior sampler construction must succeed")
    };
    sampler
        .set_state_for_tests(init.tessellations)
        .expect("prior state is assignable");
    let mut sc = DartSc {
        sampler,
        rng: sc_rng,
        sigma_sq: init.sigma_sq,
        y: y0,
        latest: BTreeMap::new(),
    };
    let spec = crate::calibration::GewekeSpec {
        n_mc: geweke_mc(),
        n_sc: geweke_sc(),
        thin: geweke_thin(),
        discard: 300, // the s chain starts at the uniform state, not the prior
        ..crate::calibration::GewekeSpec::default()
    };
    crate::calibration::getting_it_right(&spec, &mut mc_draw, &mut sc)
        .expect("DART sweeps cannot fail")
}

/// The naive DART port: conjugate Dirichlet Gibbs with no subset-prior
/// correction, the invalid sampler the exactness warning describes.
#[derive(Debug, Clone)]
struct UncorrectedDart {
    alpha: f64,
    weights: Vec<f64>,
}

impl UncorrectedDart {
    fn new(alpha: f64, p: usize) -> Self {
        Self {
            alpha,
            weights: vec![1.0 / p as f64; p],
        }
    }
}

impl crate::extensions::inclusion::InclusionModel for UncorrectedDart {
    type Error = std::convert::Infallible;
    fn weights(&self) -> &[f64] {
        &self.weights
    }
    fn update(
        &mut self,
        usage: &crate::extensions::inclusion::InclusionUsage,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        let p = self.weights.len();
        loop {
            let mut draws = Vec::with_capacity(p);
            let mut total = 0.0_f64;
            for &count in usage.counts() {
                let gamma =
                    rand_distr::Gamma::new(self.alpha / p as f64 + count as f64, 1.0).unwrap();
                let draw: f64 = Distribution::sample(&gamma, rng);
                total += draw;
                draws.push(draw);
            }
            for draw in &mut draws {
                *draw /= total;
            }
            if draws.iter().all(|w| w.is_finite() && *w > 0.0) {
                self.weights = draws;
                return Ok(());
            }
        }
    }
}

#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_dart() {
    let outcomes = run_dart_geweke(true);
    let mut failures = Vec::new();
    for outcome in &outcomes {
        println!(
            "geweke[dart] {}: D = {:.4} vs critical {:.4}: {}",
            outcome.statistic,
            outcome.d,
            outcome.critical,
            if outcome.passed() { "ok" } else { "FAIL" }
        );
        if !outcome.passed() {
            failures.push(outcome.statistic.clone());
        }
    }
    assert!(
        failures.is_empty(),
        "the MH-corrected DART failed Geweke on {failures:?}"
    );
}

/// The exactness warning, demonstrated: the naive conjugate-Gibbs DART port
/// (no subset-prior correction) must turn the battery red.
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn geweke_uncorrected_dart_goes_red() {
    let outcomes = run_dart_geweke(false);
    for outcome in &outcomes {
        println!(
            "geweke[dart-uncorrected] {}: D = {:.4} vs critical {:.4}",
            outcome.statistic, outcome.d, outcome.critical
        );
    }
    assert!(
        outcomes.iter().any(|outcome| !outcome.passed()),
        "the uncorrected DART port was NOT detected: the exactness warning has no teeth"
    );
}

/// DART SBC through the public battery drivers: rank CSV for
/// the R ECDF-band verdict, the weights ranked alongside the standard
/// quantities.
#[test]
#[ignore = "stochastic battery: calibration/release CI legs only"]
fn sbc_ranks_dart() {
    let fixture = fixture_dart();
    let n = fixture.n();
    let spec = crate::calibration::SbcSpec::default();
    let mut ranks = crate::calibration::SbcRanks::new(spec.n_draws);

    for replication in 0..spec.replications {
        let mut generator_seed = fixture.seed ^ ((replication as u64) << 20);
        let mut rng = ChaCha8Rng::from_seed(expand_seed(splitmix64(&mut generator_seed)));
        let (s, prior) = draw_dart_prior(&fixture, &mut rng);
        let y = draw_response(&prior, n, &mut rng);
        let truth = dart_quantities(
            &s,
            &prior.tessellations,
            prior.sigma_sq,
            |row| prior.fit_at(row),
            &y,
        );

        let sampler_seed = fixture.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ replication as u64;
        let mut sampler = dart_sampler(&fixture, sampler_seed, y.clone());
        for _ in 0..spec.burn_in {
            sampler.step().expect("DART sweeps cannot fail");
        }
        let mut posterior: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        for _ in 0..spec.n_draws {
            for _ in 0..spec.thin {
                sampler.step().expect("DART sweeps cannot fail");
            }
            let (sigma_sq, tessellations) = {
                let draw = sampler.step().expect("DART sweeps cannot fail");
                (draw.sigma_sq, draw.tessellations.to_vec())
            };
            let fit = sampler.fit_values().to_vec();
            let s = sampler.inclusion_weights().to_vec();
            let quantities = dart_quantities(&s, &tessellations, sigma_sq, |row| fit[row], &y);
            for (name, value) in quantities {
                posterior.entry(name).or_default().push(value);
            }
        }
        for (name, true_value) in truth {
            ranks.record(name, true_value, &posterior[name], &mut rng);
        }
    }

    let dir = out_dir();
    std::fs::create_dir_all(&dir).expect("stat-gates output directory must be creatable");
    let path = dir.join("sbc-dart.csv");
    ranks.write_csv(&path).expect("rank CSV must be writable");
    println!(
        "SBC[dart]: wrote {} ranks/quantity x {} quantities to {} (chi2: {:?})",
        spec.replications,
        ranks.ranks().len(),
        path.display(),
        ranks.chi_squared(20)
    );
}
