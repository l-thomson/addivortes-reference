//! Every model skeleton passes the Geweke battery as acceptance tests. Each
//! composes through the crate-internal wiring surface (`calibration::*`,
//! `SamplerBuilder`, `pinned_prior`) with zero engine edits. The in-crate
//! `stat_gates` batteries remain the fine-grained per-point gates; this
//! file proves the same verdicts are reachable from the model-file seam.
//!
//! The skeletons: the paper Gaussian
//! (the default model), the Binary-probit response family, H-AddiVortes
//! (mean + variance ensembles), linear cells at q = 1 (the intercept basis,
//! the block algebra run where its law is known exactly), soft membership
//! (the dense path under the softmax kernel), and the Metropolis-corrected
//! DART reference, plus robust-t, validated the same way
//! (with a wrong-σ²-pairing negative control).
//!
//! CI leg: calibration gate (stochastic; `--ignored`), like the in-crate batteries.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use crate::calibration::{GewekeOutcome, GewekeSpec, StructuralPrior, SuccessiveConditional};
use crate::engine::builder::SamplerBuilder;
use crate::extensions::basis::{CellBasis, LinearBasis, LinearGaussianModel};
use crate::extensions::cell_model::WeightedGaussianModel;
use crate::extensions::coord::{CoordinateDistribution, EuclideanNormal};
use crate::extensions::distance::{CellAssigner, ColumnMetrics};
use crate::extensions::inclusion::DartInclusion;
use crate::extensions::membership::{MembershipKernel, SoftmaxKernel};
use crate::extensions::moves::MoveSetBuilder;
use crate::extensions::scale::{GlobalSigma, HVariance, ScaleCtx, ScaleModel};
use crate::{AddiVortesConfig, Data, Metric, ResponseFamily, Sampler, Tessellation, mathsfn};
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};
use rand_distr::Distribution;

// The shared acceptance fixture (small-p, battery-sized: the in-crate
// gates' shape, rebuilt from documented constants only).
const N: usize = 50;
const P: usize = 3;
const M: usize = 6;
const NU: f64 = 6.0;
const LAMBDA: f64 = 0.02;
const LAMBDA_C: f64 = 2.0;
const OMEGA: f64 = 1.5;
const SIGMA_C: f64 = 0.8;
const K: f64 = 3.0;
const TAU: f64 = 0.1;
const DART_ALPHA: f64 = 1.5;
/// Robust-t error degrees of freedom for skeleton 7 (well inside the
/// heavy-tailed regime).
const T_DF: f64 = 4.0;
// The H variance ensemble's pinned shape (pinned for the same reason as λ:
// the generating and fitted priors must coincide).
const M_PRIME: usize = 3;
const NU_PRIME: f64 = 8.0;
const LAMBDA_PRIME: f64 = 0.3;
/// The two fixed design rows whose ensemble fit is tracked.
const ROWS: [usize; 2] = [0, N / 2];

/// σ_μ = 0.5/(k√m): the paper's scaled-space cell-value prior SD,
/// the same documented constant the sampler derives.
fn sigma_mu_sq() -> f64 {
    let sigma_mu = 0.5 / (K * (M as f64).sqrt());
    sigma_mu * sigma_mu
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * mathsfn::erfc(-z / std::f64::consts::SQRT_2)
}

fn uniform(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    Distribution::sample(&rand_distr::StandardNormal, rng)
}

/// A standard Student-t draw with ν′ = `df`: the robust-t observation
/// noise (the λ mixture marginalised out).
fn student_t(df: f64, rng: &mut ChaCha8Rng) -> f64 {
    Distribution::sample(&rand_distr::StudentT::new(df).expect("df > 0"), rng)
}

/// σ² ~ inverse-χ²(ν, λ) = 1/Gamma(ν/2, scale 2/(νλ)): the paper's scale
/// prior, in the sampler's own parameterisation.
fn draw_inv_chi_sq(nu: f64, lambda: f64, rng: &mut ChaCha8Rng) -> f64 {
    let gamma = rand_distr::Gamma::new(0.5 * nu, 2.0 / (nu * lambda)).expect("positive shape");
    let precision: f64 = Distribution::sample(&gamma, rng);
    1.0 / precision
}

/// A deterministic pre-scaled design (columns already in [−0.5, 0.5]: the
/// pinned-prior coordinate system).
fn design(seed: u64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let values: Vec<f64> = (0..N * P).map(|_| uniform(&mut rng) - 0.5).collect();
    Data::new(values, N, P).expect("coherent shape")
}

/// The basis leg's design: column 0 (the basis covariate) spans [-2, 2] so the
/// slope term carries more of the fit than the intercept; the rest is the
/// standard [-0.5, 0.5] fixture.
fn basis_design(seed: u64) -> Data {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(N * P);
    for _ in 0..N {
        values.push(4.0 * (uniform(&mut rng) - 0.5));
        for _ in 1..P {
            values.push(uniform(&mut rng) - 0.5);
        }
    }
    Data::new(values, N, P).expect("coherent shape")
}

fn coord_laws() -> Vec<Arc<dyn CoordinateDistribution>> {
    (0..P)
        .map(|_| {
            Arc::new(EuclideanNormal::new(SIGMA_C).unwrap()) as Arc<dyn CoordinateDistribution>
        })
        .collect()
}

fn structural_prior<'a>(
    coord_dists: &'a [Arc<dyn CoordinateDistribution>],
    weights: &'a [f64],
) -> StructuralPrior<'a> {
    StructuralPrior {
        lambda_c: LAMBDA_C,
        omega: OMEGA,
        coord_dists,
        weights,
    }
}

/// One hard-guard ensemble draw of `m` tessellations from the structural
/// prior, with cell values from `draw_value`, conditioned on every cell
/// owning at least one observation of the design.
fn draw_hard_ensemble(
    x: &Data,
    assigner: &ColumnMetrics,
    coord_dists: &[Arc<dyn CoordinateDistribution>],
    weights: &[f64],
    m: usize,
    draw_value: &mut dyn FnMut(&mut dyn Rng) -> f64,
    rng: &mut ChaCha8Rng,
) -> (Vec<Tessellation>, Vec<Vec<usize>>) {
    let prior = structural_prior(coord_dists, weights);
    let mut tessellations = Vec::with_capacity(m);
    let mut assignments = Vec::with_capacity(m);
    for _ in 0..m {
        let mut assignment = Vec::new();
        let tessellation = prior.draw(
            draw_value,
            &mut |candidate| {
                let cells = assigner
                    .assign_cells(x, candidate)
                    .expect("built-in keys are finite");
                let mut occupied = vec![false; candidate.n_cells()];
                for &cell in &cells {
                    occupied[cell] = true;
                }
                let all = occupied.iter().all(|o| *o);
                if all {
                    assignment = cells;
                }
                all
            },
            rng,
        );
        tessellations.push(tessellation);
        assignments.push(assignment);
    }
    (tessellations, assignments)
}

/// Hard-membership ensemble fit at `row`.
fn fit_at(tessellations: &[Tessellation], assignments: &[Vec<usize>], row: usize) -> f64 {
    tessellations
        .iter()
        .zip(assignments)
        .map(|(t, a)| t.mus()[a[row]])
        .sum()
}

/// The Gaussian-family test quantities (identical definitions on both
/// simulators; the same set as the in-crate gates).
fn gaussian_quantities(
    tessellations: &[Tessellation],
    sigma_sq: f64,
    fit: &[f64],
    y: &[f64],
) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    out.insert("sigma_sq".into(), sigma_sq);
    let mean_mu: f64 = tessellations
        .iter()
        .map(|t| t.mus().iter().sum::<f64>() / t.n_cells() as f64)
        .sum();
    out.insert("mean_mu".into(), mean_mu);
    out.insert(
        "total_cells".into(),
        tessellations.iter().map(|t| t.n_cells() as f64).sum(),
    );
    out.insert(
        "total_dims".into(),
        tessellations.iter().map(|t| t.dims().len() as f64).sum(),
    );
    out.insert("f_row_a".into(), fit[ROWS[0]]);
    out.insert("f_row_b".into(), fit[ROWS[1]]);
    let rss: f64 = y
        .iter()
        .zip(fit)
        .map(|(&value, &f)| (value - f) * (value - f))
        .sum();
    out.insert("rss_over_sigma_sq".into(), rss / sigma_sq);
    out
}

/// The base pinned-prior config shared by every skeleton.
fn base_config(seed: u64) -> AddiVortesConfig {
    AddiVortesConfig::new(seed)
        .with_m(M)
        .with_nu(NU)
        .with_omega(OMEGA)
        .with_lambda_c(LAMBDA_C)
        .with_sigma_c(SIGMA_C)
        .with_k(K)
}

fn pinned_sampler(builder: SamplerBuilder, x: &Data, y0: Vec<f64>) -> Sampler {
    builder
        .pinned_prior(
            x.clone(),
            vec![Metric::Euclidean; P],
            y0,
            LAMBDA,
            MoveSetBuilder::stone_gosling()
                .build()
                .expect("the paper set builds"),
        )
        .expect("pinned-prior construction succeeds")
}

/// `base_config` wrapped for component wiring.
fn base_builder(seed: u64) -> SamplerBuilder {
    SamplerBuilder::new(base_config(seed))
}

/// An arbitrary (non-stationary) starting response; the SC chain earns
/// stationarity through the discard: no state injection exists publicly.
fn y_init() -> Vec<f64> {
    (0..N)
        .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
        .collect()
}

/// The Gaussian-likelihood successive-conditional simulator: `y | θ` from
/// the current fit and σ², one full sweep of the real kernel through
/// `step`, quantities off the public draw. Serves the paper-Gaussian,
/// linear-cells, soft-membership and DART skeletons (`track_inclusion`
/// adds the DART weight statistics through `inclusion_weights`).
struct GaussianSc {
    sampler: Sampler,
    rng: ChaCha8Rng,
    sigma_sq: f64,
    y: Vec<f64>,
    track_inclusion: bool,
    /// `Some(ν′)` switches the observation law to fit + σ·t_ν′: the
    /// robust-t likelihood with its λ mixture marginalised out.
    t_df: Option<f64>,
    latest: BTreeMap<String, f64>,
}

impl SuccessiveConditional for GaussianSc {
    fn transition(&mut self) -> crate::Result<()> {
        let fit = self.sampler.fitted_values();
        let sigma = self.sigma_sq.sqrt();
        self.y = fit
            .iter()
            .map(|&f| {
                let noise = match self.t_df {
                    Some(df) => student_t(df, &mut self.rng),
                    None => standard_normal(&mut self.rng),
                };
                f + sigma * noise
            })
            .collect();
        let y = self.y.clone();
        self.sampler.set_response(&y)?;
        let (sigma_sq, tessellations) = {
            let draw = self.sampler.step()?;
            (draw.sigma_sq, draw.tessellations.to_vec())
        };
        self.sigma_sq = sigma_sq;
        let fit = self.sampler.fitted_values();
        let mut quantities = gaussian_quantities(&tessellations, sigma_sq, &fit, &self.y);
        if self.track_inclusion {
            let weights = self.sampler.inclusion_weights();
            quantities.insert("s_first".into(), weights[0]);
            quantities.insert(
                "s_max".into(),
                weights.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
        }
        self.latest = quantities;
        Ok(())
    }

    fn quantities(&self) -> BTreeMap<String, f64> {
        self.latest.clone()
    }
}

fn spec() -> GewekeSpec {
    GewekeSpec {
        n_mc: 3000,
        n_sc: 1000,
        thin: 20,
        discard: 500,
        alpha: 0.01,
    }
}

fn assert_green(name: &str, outcomes: &[GewekeOutcome]) {
    let mut failures = Vec::new();
    for outcome in outcomes {
        println!(
            "battery[{name}] {}: D = {:.4} vs critical {:.4}: {}",
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
        "the {name} skeleton failed the externalised Geweke battery on {failures:?}"
    );
}

/// The shared scalar battery run: the marginal-conditional generator is the
/// paper prior (weights and cell-value law fixed) with Gaussian observation
/// noise: or fit + σ·t_ν′ when `t_df` is set (the robust-t likelihood);
/// the sampler under test is `configure`'s: the paper default, or the same
/// model through a differently-shaped seam (linear q = 1, robust-t).
fn run_scalar_gaussian_battery(
    seed: u64,
    t_df: Option<f64>,
    configure: impl FnOnce(SamplerBuilder) -> SamplerBuilder,
) -> Vec<GewekeOutcome> {
    let x = design(seed);
    let assigner = ColumnMetrics::new(vec![Metric::Euclidean; P]);
    let coord_dists = coord_laws();
    let weights = [1.0_f64; P];
    let sigma_mu = sigma_mu_sq().sqrt();

    let mut mc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6E77);
    let mut mc_draw = || {
        let sigma_sq = draw_inv_chi_sq(NU, LAMBDA, &mut mc_rng);
        let (tessellations, assignments) = draw_hard_ensemble(
            &x,
            &assigner,
            &coord_dists,
            &weights,
            M,
            &mut |rng| {
                sigma_mu * {
                    let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
                    z
                }
            },
            &mut mc_rng,
        );
        let fit: Vec<f64> = (0..N)
            .map(|row| fit_at(&tessellations, &assignments, row))
            .collect();
        let sigma = sigma_sq.sqrt();
        let y: Vec<f64> = fit
            .iter()
            .map(|&f| {
                let noise = match t_df {
                    Some(df) => student_t(df, &mut mc_rng),
                    None => standard_normal(&mut mc_rng),
                };
                f + sigma * noise
            })
            .collect();
        gaussian_quantities(&tessellations, sigma_sq, &fit, &y)
    };

    let mut sc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5C5C);
    let sigma_sq0 = draw_inv_chi_sq(NU, LAMBDA, &mut sc_rng);
    let sampler = pinned_sampler(configure(base_builder(seed ^ 0xC4A1)), &x, y_init());
    let mut sc = GaussianSc {
        sampler,
        rng: sc_rng,
        sigma_sq: sigma_sq0,
        y: y_init(),
        track_inclusion: false,
        t_df,
        latest: BTreeMap::new(),
    };
    crate::calibration::getting_it_right(&spec(), &mut mc_draw, &mut sc)
        .expect("standard-path sweeps cannot fail")
}

/// Skeleton 1/7: the paper Gaussian: the default model through the public
/// pinned-prior constructor, nothing configured.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn paper_gaussian_passes_the_externalised_battery() {
    let outcomes = run_scalar_gaussian_battery(0xACC0_0001, None, |config| config);
    assert_green("paper-gaussian", &outcomes);
}

/// Skeleton 4/7: linear cells at q = 1: the intercept-only
/// basis is *exactly* the scalar Gaussian family (the in-crate oracle pins
/// the algebra to ≤ 1e-12), so the marginal-conditional generator is the
/// paper prior while every score/redraw runs the q×q block path.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn linear_cells_q1_pass_the_externalised_battery() {
    let outcomes = run_scalar_gaussian_battery(0xACC0_0004, None, |config| {
        config.with_cell_model(LinearGaussianModel::new(sigma_mu_sq(), 1).unwrap())
    });
    assert_green("linear-q1", &outcomes);
}

/// Skeleton 4b/7: linear cells at q = 2: the basis path proper.
/// The generator draws β ~ N(0, σ_β² I_q) per cell and forms the ensemble fit
/// as `z(xᵢ)·β_k`; the sampler under test runs the engine's basis-row supply,
/// the q×q block accumulate, and the vector payload redraw. This is the leg
/// that pins the *fit-time* basis path, which the q = 1 leg cannot reach (at
/// q = 1 the linear family collapses to the scalar one and takes no basis).
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn linear_cells_q2_pass_the_externalised_battery() {
    let outcomes = run_basis_battery(0xACC0_0009);
    assert_green("linear-q2", &outcomes);
}

/// The q = 2 basis battery run: the marginal-conditional generator is the
/// structural prior with a q-wide coefficient payload, and the fit is the
/// basis inner product rather than a cell constant.
fn run_basis_battery(seed: u64) -> Vec<GewekeOutcome> {
    run_basis_battery_inner(seed, true)
}

/// `use_basis = false` makes the *generator* form its fit from the intercept
/// alone, i.e. simulates a sampler whose basis path is broken. The battery must
/// go red; that is what `the_basis_leg_detects_a_broken_basis_path` pins.
fn run_basis_battery_inner(seed: u64, use_basis: bool) -> Vec<GewekeOutcome> {
    // The basis leg needs its own design. Under the battery's identity scaler
    // the standard fixture spans [-0.5, 0.5], so a slope coefficient
    // contributes at most x₀² = ¼ of the intercept's variance to the fit, and
    // a sampler that ignored the basis entirely would still pass. Column 0
    // therefore spans [-2, 2]: the slope term now dominates the intercept, and
    // the leg has real power against a broken basis path (pinned by
    // `the_basis_leg_detects_a_broken_basis_path`).
    let x = basis_design(seed);
    let assigner = ColumnMetrics::new(vec![Metric::Euclidean; P]);
    let coord_dists = coord_laws();
    let weights = [1.0_f64; P];
    let sigma_beta = sigma_mu_sq().sqrt();
    let basis = LinearBasis::new(vec![0]);
    let q = basis.q();
    assert_eq!(q, 2);

    // The ensemble fit under a basis payload: z(xᵢ)·β_k, summed over the
    // ensemble. The same quantity the engine's `update_fit` computes.
    let fit_at_basis =
        |tessellations: &[Tessellation], assignments: &[Vec<usize>], x: &Data, row: usize| {
            let mut z = vec![0.0_f64; q];
            basis.row(x.row(row), &mut z);
            tessellations
                .iter()
                .zip(assignments)
                .map(|(t, a)| {
                    let beta = t.cell_payload(a[row]);
                    if use_basis {
                        beta.iter().zip(&z).map(|(b, zi)| b * zi).sum::<f64>()
                    } else {
                        beta[0]
                    }
                })
                .sum::<f64>()
        };

    let mut mc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6E77);
    let mut mc_draw = || {
        let sigma_sq = draw_inv_chi_sq(NU, LAMBDA, &mut mc_rng);
        let prior = structural_prior(&coord_dists, &weights);
        let mut tessellations = Vec::with_capacity(M);
        let mut assignments = Vec::with_capacity(M);
        for _ in 0..M {
            let mut assignment = Vec::new();
            let tessellation = prior.draw_payload(
                q,
                &mut |rng, cell| {
                    for slot in cell.iter_mut() {
                        let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
                        *slot = sigma_beta * z;
                    }
                },
                &mut |candidate| {
                    let cells = assigner
                        .assign_cells(&x, candidate)
                        .expect("built-in keys are finite");
                    let mut occupied = vec![false; candidate.n_cells()];
                    for &cell in &cells {
                        occupied[cell] = true;
                    }
                    let all = occupied.iter().all(|o| *o);
                    if all {
                        assignment = cells;
                    }
                    all
                },
                &mut mc_rng,
            );
            tessellations.push(tessellation);
            assignments.push(assignment);
        }
        let fit: Vec<f64> = (0..N)
            .map(|row| fit_at_basis(&tessellations, &assignments, &x, row))
            .collect();
        let sigma = sigma_sq.sqrt();
        let y: Vec<f64> = fit
            .iter()
            .map(|&f| f + sigma * standard_normal(&mut mc_rng))
            .collect();
        gaussian_quantities(&tessellations, sigma_sq, &fit, &y)
    };

    let mut sc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5C5C);
    let sigma_sq0 = draw_inv_chi_sq(NU, LAMBDA, &mut sc_rng);
    let builder = base_builder(seed ^ 0xC4A1)
        .with_cell_model(LinearGaussianModel::new(sigma_mu_sq(), q).unwrap())
        .with_cell_basis(LinearBasis::new(vec![0]));
    let sampler = pinned_sampler(builder, &x, y_init());
    let mut sc = GaussianSc {
        sampler,
        rng: sc_rng,
        sigma_sq: sigma_sq0,
        y: y_init(),
        track_inclusion: false,
        t_df: None,
        latest: BTreeMap::new(),
    };
    crate::calibration::getting_it_right(&spec(), &mut mc_draw, &mut sc)
        .expect("basis-path sweeps cannot fail")
}

/// Skeleton 7/7: the robust-t family: the generator draws
/// y = F + σ·t_ν′ (the scale mixture marginalised), the sampler under test
/// is the full family assembly (`RobustTStep` λ redraws, weighted cells,
/// the precision-weighted σ² draw) selected by one config call.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn robust_t_family_passes_the_externalised_battery() {
    let outcomes = run_scalar_gaussian_battery(0xACC0_0007, Some(T_DF), |config| {
        config.with_response_family(ResponseFamily::RobustT { df: T_DF })
    });
    assert_green("robust-t", &outcomes);
}

/// The robust-t negative control: break the deep-seam pairing rule on
/// purpose: the t kernel step with the unweighted σ² draw (the
/// default `GlobalSigma` instead of the precision-weighted one). The σ²
/// full conditional is then wrong and the successive-conditional chain is
/// not stationary; in practice the mis-pairing feeds back (heavy-tailed
/// residuals inflate the unweighted σ², which inflates the next t draw)
/// until the chain diverges numerically. Either verdict: Geweke red or
/// divergence: is the battery saying so loudly; a silent green is the
/// failure. This is the teeth check for the pairing rules the templates
/// document.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn robust_t_with_unweighted_sigma_goes_red() {
    let run = || {
        run_scalar_gaussian_battery(0xACC0_0008, Some(T_DF), |config| {
            config
                .with_response_family(ResponseFamily::RobustT { df: T_DF })
                .with_scale_model(GlobalSigma::new(NU, LAMBDA).unwrap())
        })
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Err(_) => {
            println!("battery[robust-t-negative] chain diverged (σ² overflow): loudly red");
        }
        Ok(outcomes) => {
            let failed: Vec<&str> = outcomes
                .iter()
                .filter(|o| !o.passed())
                .map(|o| o.statistic.as_str())
                .collect();
            for outcome in &outcomes {
                println!(
                    "battery[robust-t-negative] {}: D = {:.4} vs critical {:.4}",
                    outcome.statistic, outcome.d, outcome.critical
                );
            }
            assert!(
                !failed.is_empty(),
                "the wrong σ² pairing must fail the battery: a silent pass means \
                 the gate lost its teeth"
            );
        }
    }
}

/// Skeleton 6/7: the Metropolis-corrected DART reference: the
/// inclusion weights join θ (s ~ Dirichlet(α/p) in the generator; the
/// weight statistics are ranked), so the wrong update is visible.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn dart_mh_correction_passes_the_externalised_battery() {
    let seed = 0xACC0_0006_u64;
    let x = design(seed);
    let assigner = ColumnMetrics::new(vec![Metric::Euclidean; P]);
    let coord_dists = coord_laws();
    let sigma_mu = sigma_mu_sq().sqrt();

    let mut mc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6E77);
    let mut mc_draw = || {
        // s ~ Dirichlet(α/p) via normalised Gammas, redrawing the
        // measure-zero underflow (weights must stay strictly positive).
        let s: Vec<f64> = loop {
            let gamma = rand_distr::Gamma::new(DART_ALPHA / P as f64, 1.0).expect("positive");
            let draws: Vec<f64> = (0..P)
                .map(|_| Distribution::sample(&gamma, &mut mc_rng))
                .collect();
            let total: f64 = draws.iter().sum();
            let normalised: Vec<f64> = draws.iter().map(|d| d / total).collect();
            if normalised.iter().all(|w| w.is_finite() && *w > 0.0) {
                break normalised;
            }
        };
        let sigma_sq = draw_inv_chi_sq(NU, LAMBDA, &mut mc_rng);
        let (tessellations, assignments) = draw_hard_ensemble(
            &x,
            &assigner,
            &coord_dists,
            &s,
            M,
            &mut |rng| {
                sigma_mu * {
                    let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
                    z
                }
            },
            &mut mc_rng,
        );
        let fit: Vec<f64> = (0..N)
            .map(|row| fit_at(&tessellations, &assignments, row))
            .collect();
        let sigma = sigma_sq.sqrt();
        let y: Vec<f64> = fit
            .iter()
            .map(|&f| f + sigma * standard_normal(&mut mc_rng))
            .collect();
        let mut quantities = gaussian_quantities(&tessellations, sigma_sq, &fit, &y);
        quantities.insert("s_first".into(), s[0]);
        quantities.insert(
            "s_max".into(),
            s.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        quantities
    };

    let mut sc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5C5C);
    let sigma_sq0 = draw_inv_chi_sq(NU, LAMBDA, &mut sc_rng);
    let builder =
        base_builder(seed ^ 0xC4A1).with_inclusion(DartInclusion::new(DART_ALPHA, P).unwrap());
    let sampler = pinned_sampler(builder, &x, y_init());
    let mut sc = GaussianSc {
        sampler,
        rng: sc_rng,
        sigma_sq: sigma_sq0,
        y: y_init(),
        track_inclusion: true,
        t_df: None,
        latest: BTreeMap::new(),
    };
    let outcomes = crate::calibration::getting_it_right(&spec(), &mut mc_draw, &mut sc)
        .expect("DART sweeps cannot fail");
    assert_green("dart-mh", &outcomes);
}

// ---------------------------------------------------------------------------
// Soft membership (the dense path)
// ---------------------------------------------------------------------------

/// Row-normalised membership matrix of `tessellation` over the design,
/// through the public key/kernel surface: or `None` when the soft
/// empty-cell guard rejects (some cell with no strictly positive mass).
fn soft_memberships(
    assigner: &ColumnMetrics,
    kernel: &SoftmaxKernel,
    x: &Data,
    tessellation: &Tessellation,
) -> Option<Vec<f64>> {
    let keys = assigner
        .membership_keys(x, tessellation)
        .expect("pairwise assigners provide dense keys");
    let b = tessellation.n_cells();
    let mut memberships = vec![0.0_f64; N * b];
    for i in 0..N {
        let row_keys = &keys[i * b..(i + 1) * b];
        let row = &mut memberships[i * b..(i + 1) * b];
        kernel.weights(row_keys, row);
        let total: f64 = row.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return None;
        }
        for weight in row.iter_mut() {
            *weight /= total;
        }
    }
    // The soft guard: every cell carries strictly positive total mass
    // (partial_cmp so a NaN mass also rejects).
    for cell in 0..b {
        let mass: f64 = (0..N).map(|i| memberships[i * b + cell]).sum();
        if mass.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
    }
    Some(memberships)
}

/// Skeleton 5/7: soft membership: the dense path under the
/// softmax kernel at fixed τ, guard and fits both computed through the
/// public key/kernel surface on the generator side.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn soft_membership_passes_the_externalised_battery() {
    let seed = 0xACC0_0005_u64;
    let x = design(seed);
    let assigner = ColumnMetrics::new(vec![Metric::Euclidean; P]);
    let coord_dists = coord_laws();
    let weights = [1.0_f64; P];
    let sigma_mu = sigma_mu_sq().sqrt();
    let kernel = SoftmaxKernel::new(TAU).unwrap();

    let mut mc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6E77);
    let mut mc_draw = || {
        let sigma_sq = draw_inv_chi_sq(NU, LAMBDA, &mut mc_rng);
        let prior = structural_prior(&coord_dists, &weights);
        let mut tessellations = Vec::with_capacity(M);
        let mut memberships = Vec::with_capacity(M);
        for _ in 0..M {
            let mut membership = Vec::new();
            let tessellation = prior.draw(
                &mut |rng| {
                    sigma_mu * {
                        let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
                        z
                    }
                },
                &mut |candidate| match soft_memberships(&assigner, &kernel, &x, candidate) {
                    Some(matrix) => {
                        membership = matrix;
                        true
                    }
                    None => false,
                },
                &mut mc_rng,
            );
            tessellations.push(tessellation);
            memberships.push(membership);
        }
        let fit: Vec<f64> = (0..N)
            .map(|row| {
                tessellations
                    .iter()
                    .zip(&memberships)
                    .map(|(t, membership)| {
                        let b = t.n_cells();
                        membership[row * b..(row + 1) * b]
                            .iter()
                            .zip(t.mus())
                            .map(|(phi, mu)| phi * mu)
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        let sigma = sigma_sq.sqrt();
        let y: Vec<f64> = fit
            .iter()
            .map(|&f| f + sigma * standard_normal(&mut mc_rng))
            .collect();
        gaussian_quantities(&tessellations, sigma_sq, &fit, &y)
    };

    let mut sc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5C5C);
    let sigma_sq0 = draw_inv_chi_sq(NU, LAMBDA, &mut sc_rng);
    let builder = base_builder(seed ^ 0xC4A1).with_membership(SoftmaxKernel::new(TAU).unwrap());
    let sampler = pinned_sampler(builder, &x, y_init());
    let mut sc = GaussianSc {
        sampler,
        rng: sc_rng,
        sigma_sq: sigma_sq0,
        y: y_init(),
        track_inclusion: false,
        t_df: None,
        latest: BTreeMap::new(),
    };
    let outcomes = crate::calibration::getting_it_right(&spec(), &mut mc_draw, &mut sc)
        .expect("dense-path sweeps cannot fail");
    assert_green("soft-membership", &outcomes);
}

// ---------------------------------------------------------------------------
// Binary probit (the built-in response family)
// ---------------------------------------------------------------------------

/// The probit successive-conditional simulator: {0, 1} labels from
/// Bernoulli(Φ(F)) on the current latent fit, one family sweep via `step`.
struct ProbitSc {
    sampler: Sampler,
    rng: ChaCha8Rng,
    y: Vec<f64>,
    latest: BTreeMap<String, f64>,
}

fn probit_quantities(
    tessellations: &[Tessellation],
    fit: &[f64],
    y: &[f64],
) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    let mean_mu: f64 = tessellations
        .iter()
        .map(|t| t.mus().iter().sum::<f64>() / t.n_cells() as f64)
        .sum();
    out.insert("mean_mu".into(), mean_mu);
    out.insert(
        "total_cells".into(),
        tessellations.iter().map(|t| t.n_cells() as f64).sum(),
    );
    out.insert(
        "total_dims".into(),
        tessellations.iter().map(|t| t.dims().len() as f64).sum(),
    );
    out.insert("f_row_a".into(), fit[ROWS[0]]);
    out.insert("f_row_b".into(), fit[ROWS[1]]);
    out.insert("ones".into(), y.iter().filter(|&&v| v > 0.0).count() as f64);
    out
}

impl SuccessiveConditional for ProbitSc {
    fn transition(&mut self) -> crate::Result<()> {
        let fit = self.sampler.fitted_values();
        self.y = fit
            .iter()
            .map(|&f| {
                if uniform(&mut self.rng) < normal_cdf(f) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let y = self.y.clone();
        self.sampler.set_response(&y)?;
        let tessellations = {
            let draw = self.sampler.step()?;
            draw.tessellations.to_vec()
        };
        let fit = self.sampler.fitted_values();
        self.latest = probit_quantities(&tessellations, &fit, &self.y);
        Ok(())
    }

    fn quantities(&self) -> BTreeMap<String, f64> {
        self.latest.clone()
    }
}

/// Skeleton 2/7: the built-in Binary-probit response family.
/// `with_response_family` assembles the Albert–Chib augmentation, the
/// pinned unit scale and the widened latent cell prior; the generator uses
/// the family's documented latent σ_μ = 3/(k√m).
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn binary_probit_family_passes_the_externalised_battery() {
    let seed = 0xACC0_0002_u64;
    let x = design(seed);
    let assigner = ColumnMetrics::new(vec![Metric::Euclidean; P]);
    let coord_dists = coord_laws();
    let weights = [1.0_f64; P];
    let latent_sigma_mu = 3.0 / (K * (M as f64).sqrt());

    let mut mc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6E77);
    let mut mc_draw = || {
        let (tessellations, assignments) = draw_hard_ensemble(
            &x,
            &assigner,
            &coord_dists,
            &weights,
            M,
            &mut |rng| {
                latent_sigma_mu * {
                    let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
                    z
                }
            },
            &mut mc_rng,
        );
        let fit: Vec<f64> = (0..N)
            .map(|row| fit_at(&tessellations, &assignments, row))
            .collect();
        let y: Vec<f64> = fit
            .iter()
            .map(|&f| {
                if uniform(&mut mc_rng) < normal_cdf(f) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        probit_quantities(&tessellations, &fit, &y)
    };

    let builder = SamplerBuilder::new(
        base_config(seed ^ 0xC4A1).with_response_family(ResponseFamily::BinaryProbit),
    );
    let y0: Vec<f64> = (0..N).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let sampler = pinned_sampler(builder, &x, y0.clone());
    let mut sc = ProbitSc {
        sampler,
        rng: ChaCha8Rng::seed_from_u64(seed ^ 0x5C5C),
        y: y0,
        latest: BTreeMap::new(),
    };
    let outcomes = crate::calibration::getting_it_right(&spec(), &mut mc_draw, &mut sc)
        .expect("probit sweeps cannot fail");
    assert_green("binary-probit-family", &outcomes);
}

// ---------------------------------------------------------------------------
// H-AddiVortes (mean + variance ensembles)
// ---------------------------------------------------------------------------

/// The test-side [`ScaleModel`] wrapper sharing an [`HVariance`] with the
/// simulator (which reads `s²(xᵢ)` and the variance tessellations between
/// sweeps): built here from the public trait alone, the same shape an
/// external consumer would write.
#[derive(Debug)]
struct SharedHVariance {
    inner: Arc<Mutex<HVariance>>,
    cache: Vec<f64>,
}

impl ScaleModel for SharedHVariance {
    type Error = crate::AddiVortesError;

    fn update(
        &mut self,
        ctx: &ScaleCtx<'_>,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<(), Self::Error> {
        let mut inner = self.inner.lock().expect("test lock");
        ScaleModel::update(&mut *inner, ctx, rng)?;
        self.cache.clear();
        self.cache
            .extend_from_slice(ScaleModel::precisions(&*inner).expect("updated above"));
        Ok(())
    }

    fn sigma_sq(&self) -> f64 {
        1.0
    }

    fn precisions(&self) -> Option<&[f64]> {
        if self.cache.is_empty() {
            None
        } else {
            Some(&self.cache)
        }
    }
}

/// H test quantities: the mean side's, the variance side's, and the jointly
/// standardised residual sum `Σ e²ᵢ/s²(xᵢ)` (χ²_n under the model).
fn h_quantities(
    mean_tessellations: &[Tessellation],
    var_tessellations: &[Tessellation],
    fit: &[f64],
    s_sq: &[f64],
    y: &[f64],
) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    let mean_mu: f64 = mean_tessellations
        .iter()
        .map(|t| t.mus().iter().sum::<f64>() / t.n_cells() as f64)
        .sum();
    out.insert("mean_mu".into(), mean_mu);
    out.insert(
        "total_cells".into(),
        mean_tessellations.iter().map(|t| t.n_cells() as f64).sum(),
    );
    out.insert(
        "var_total_cells".into(),
        var_tessellations.iter().map(|t| t.n_cells() as f64).sum(),
    );
    out.insert("f_row_a".into(), fit[ROWS[0]]);
    out.insert("f_row_b".into(), fit[ROWS[1]]);
    out.insert("s_sq_row_a".into(), s_sq[ROWS[0]]);
    out.insert("s_sq_row_b".into(), s_sq[ROWS[1]]);
    let standardised_rss: f64 = y
        .iter()
        .enumerate()
        .map(|(row, &value)| {
            let residual = value - fit[row];
            residual * residual / s_sq[row]
        })
        .sum();
    out.insert("standardised_rss".into(), standardised_rss);
    out
}

/// The H successive-conditional simulator: `y | θ` from the current fit and
/// s²(x), one two-ensemble sweep via `step`; the variance side is read
/// through the shared handle.
struct HSc {
    sampler: Sampler,
    shared: Arc<Mutex<HVariance>>,
    rng: ChaCha8Rng,
    y: Vec<f64>,
    latest: BTreeMap<String, f64>,
}

impl SuccessiveConditional for HSc {
    fn transition(&mut self) -> crate::Result<()> {
        let fit = self.sampler.fitted_values();
        {
            let inner = self.shared.lock().expect("test lock");
            let s_sq = inner.s_sq_values().expect("primed before the battery");
            self.y = fit
                .iter()
                .zip(s_sq)
                .map(|(&f, &s2)| f + s2.sqrt() * standard_normal(&mut self.rng))
                .collect();
        }
        let y = self.y.clone();
        self.sampler.set_response(&y)?;
        let mean_tessellations = {
            let draw = self.sampler.step()?;
            draw.tessellations.to_vec()
        };
        let fit = self.sampler.fitted_values();
        let inner = self.shared.lock().expect("test lock");
        let s_sq = inner.s_sq_values().expect("updated by the sweep").to_vec();
        let var_tessellations = inner.tessellations().expect("updated by the sweep");
        self.latest = h_quantities(&mean_tessellations, var_tessellations, &fit, &s_sq, &self.y);
        Ok(())
    }

    fn quantities(&self) -> BTreeMap<String, f64> {
        self.latest.clone()
    }
}

/// Skeleton 3/7: H-AddiVortes (the scale and cell-model points): the mean
/// ensemble under
/// per-observation precisions and the multiplicative variance ensemble,
/// plugged through the public `ScaleModel` seam; the generator draws both
/// ensembles from the shared structural prior.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn h_addivortes_passes_the_externalised_battery() {
    let seed = 0xACC0_0003_u64;
    let x = design(seed);
    let assigner = ColumnMetrics::new(vec![Metric::Euclidean; P]);
    let coord_dists = coord_laws();
    let weights = [1.0_f64; P];
    let sigma_mu = sigma_mu_sq().sqrt();

    let mut mc_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6E77);
    let mut mc_draw = || {
        let (mean_tessellations, mean_assignments) = draw_hard_ensemble(
            &x,
            &assigner,
            &coord_dists,
            &weights,
            M,
            &mut |rng| {
                sigma_mu * {
                    let z: f64 = Distribution::sample(&rand_distr::StandardNormal, rng);
                    z
                }
            },
            &mut mc_rng,
        );
        // Variance tessellations: the same structural law, cell values
        // s² ~ inverse-χ²(ν′, λ′).
        let gamma = rand_distr::Gamma::new(0.5 * NU_PRIME, 2.0 / (NU_PRIME * LAMBDA_PRIME))
            .expect("positive shape");
        let (var_tessellations, var_assignments) = draw_hard_ensemble(
            &x,
            &assigner,
            &coord_dists,
            &weights,
            M_PRIME,
            &mut |rng| {
                let precision: f64 = Distribution::sample(&gamma, rng);
                1.0 / precision
            },
            &mut mc_rng,
        );
        let fit: Vec<f64> = (0..N)
            .map(|row| fit_at(&mean_tessellations, &mean_assignments, row))
            .collect();
        let s_sq: Vec<f64> = (0..N)
            .map(|row| {
                var_tessellations
                    .iter()
                    .zip(&var_assignments)
                    .map(|(t, a)| t.mus()[a[row]])
                    .product()
            })
            .collect();
        let y: Vec<f64> = fit
            .iter()
            .zip(&s_sq)
            .map(|(&f, &s2)| f + s2.sqrt() * standard_normal(&mut mc_rng))
            .collect();
        h_quantities(&mean_tessellations, &var_tessellations, &fit, &s_sq, &y)
    };

    // The H sampler: weighted-Gaussian mean cells (the per-observation
    // precisions demand the weighted statistic) + the shared variance
    // ensemble on the ScaleModel seam.
    let builder = base_builder(seed ^ 0xC4A1)
        .with_cell_model(WeightedGaussianModel::new(sigma_mu_sq()).unwrap());
    let shared = Arc::new(Mutex::new(
        HVariance::new(M_PRIME)
            .unwrap()
            .with_prior(NU_PRIME, LAMBDA_PRIME)
            .unwrap(),
    ));
    let mut sampler = pinned_sampler(builder, &x, y_init()).with_scale_model(SharedHVariance {
        inner: Arc::clone(&shared),
        cache: Vec::new(),
    });
    // Prime one sweep so the lazy variance ensemble exists before the first
    // `y | θ` regeneration (the discard absorbs the arbitrary start).
    sampler.step().expect("H sweeps cannot fail");
    let mut sc = HSc {
        sampler,
        shared,
        rng: ChaCha8Rng::seed_from_u64(seed ^ 0x5C5C),
        y: y_init(),
        latest: BTreeMap::new(),
    };
    let outcomes = crate::calibration::getting_it_right(&spec(), &mut mc_draw, &mut sc)
        .expect("H sweeps cannot fail");
    assert_green("h-addivortes", &outcomes);
}

/// Guard for the q = 2 leg itself: `pinned_prior` must really carry the basis
/// payload, or the battery above would be validating the scalar path.
#[test]
fn the_basis_leg_sampler_actually_carries_a_q2_payload() {
    let x = basis_design(0xACC0_0009);
    let builder = base_builder(1)
        .with_cell_model(LinearGaussianModel::new(sigma_mu_sq(), 2).unwrap())
        .with_cell_basis(LinearBasis::new(vec![0]));
    let mut sampler = pinned_sampler(builder, &x, y_init());
    let draw = sampler.step().expect("a basis sweep runs");
    for tessellation in draw.tessellations {
        assert_eq!(
            tessellation.q(),
            2,
            "the battery leg is not on the basis path"
        );
    }
}

/// The negative control for the q = 2 leg: a generator that ignores the basis
/// (fit = the intercept coefficient alone) must be *caught*. Without this, a
/// leg whose design gave the slope no leverage would pass whatever the engine
/// did — which is exactly what the first draft of this leg did.
#[test]
#[ignore = "stochastic battery: calibration gate CI leg (run with --ignored under the determinism profile)"]
fn the_basis_leg_detects_a_broken_basis_path() {
    let outcomes = run_basis_battery_inner(0xACC0_0009, false);
    assert!(
        outcomes.iter().any(|o| !o.passed()),
        "the q = 2 battery leg has no power: it passed a generator that ignores the basis"
    );
}
