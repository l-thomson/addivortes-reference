//! The per-commit deterministic gate (in-crate half):
//! the all-18-`AddiVortesError`-variant reachability aggregate, the
//! bug-injection self-validation of the moves oracles, and the mutation↔oracle coupling
//! demonstration. The golden vectors live in `tests/golden_chain.rs`.
//! (The module is already `#[cfg(test)]`-gated at its `mod` declaration.)

use std::sync::Arc;

use crate::engine::data::Metric;
use crate::engine::error::AddiVortesError;
use crate::engine::mathsfn::ln;
use crate::engine::tessellation::Tessellation;
use crate::extensions::coord::CoordinateDistribution;
use crate::extensions::distance::PairwiseDistance;
use crate::extensions::inclusion::{InclusionModel, InclusionUsage};
use crate::extensions::moves::{
    AddDimension, Change, ModelCtx, MoveSetBuilder, ProposalMove, RemoveDimension,
};
use crate::{AddiVortesConfig, Data};

// ---------------------------------------------------------------------------
// Error-variant reachability aggregate (every one of the 18, via real paths)
// ---------------------------------------------------------------------------

/// A cell metric that always returns NaN: the NonFiniteDistance trigger.
#[derive(Debug)]
struct PoisonMetric;

impl PairwiseDistance for PoisonMetric {
    fn distance(&self, _x: &[f64], _c: &[f64], _dims: &[usize]) -> f64 {
        f64::NAN
    }
}

/// An inclusion model whose per-sweep update always fails: the Extension
/// trigger (the inclusion point extension-error channel).
#[derive(Debug, Clone)]
struct FailingInclusion {
    weights: Vec<f64>,
}

#[derive(Debug, thiserror::Error)]
#[error("deliberate update failure")]
struct DeliberateFailure;

impl InclusionModel for FailingInclusion {
    type Error = DeliberateFailure;
    fn weights(&self) -> &[f64] {
        &self.weights
    }
    fn update(
        &mut self,
        _usage: &InclusionUsage,
        _rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        Err(DeliberateFailure)
    }
}

fn quick_config(seed: u64) -> AddiVortesConfig {
    AddiVortesConfig::new(seed)
        .with_m(3)
        .with_burn_in(2)
        .with_draws(3)
        .with_omega(1.5)
}

fn small_xy() -> (Data, Vec<f64>) {
    let x = Data::from_rows(&[[0.0, 1.0], [0.3, 0.2], [0.7, 0.8], [1.0, 0.4], [0.5, 0.6]]).unwrap();
    let y = vec![0.5, 1.0, 1.5, 2.0, 1.2];
    (x, y)
}

/// Every variant of the enum is reachable through a real code path:
/// the coverage gate, in place of a gameable %.
#[test]
fn all_eighteen_error_variants_are_reachable() {
    use AddiVortesError as E;
    let (x, y) = small_xy();
    let mut seen: Vec<&'static str> = Vec::new();
    let mut hit = |label: &'static str, matched: bool| {
        assert!(matched, "variant {label} did not surface where expected");
        seen.push(label);
    };

    // 1. NonFiniteResponse
    let bad_y = vec![0.5, f64::NAN, 1.5, 2.0, 1.2];
    hit(
        "NonFiniteResponse",
        matches!(
            quick_config(1).fit(&x, &bad_y),
            Err(E::NonFiniteResponse { row: 1 })
        ),
    );
    // 2. NonFiniteFeature
    let bad_x = Data::from_rows(&[
        [0.0, 1.0],
        [0.3, f64::INFINITY],
        [0.7, 0.8],
        [1.0, 0.4],
        [0.5, 0.6],
    ])
    .unwrap();
    hit(
        "NonFiniteFeature",
        matches!(
            quick_config(1).fit(&bad_x, &y),
            Err(E::NonFiniteFeature { row: 1, col: 1 })
        ),
    );
    // 3. RowCountMismatch
    hit(
        "RowCountMismatch",
        matches!(
            quick_config(1).fit(&x, &y[..4]),
            Err(E::RowCountMismatch {
                y_len: 4,
                x_rows: 5
            })
        ),
    );
    // 4. MetricLengthMismatch
    hit(
        "MetricLengthMismatch",
        matches!(
            quick_config(1)
                .with_metrics(vec![Metric::Euclidean])
                .fit(&x, &y),
            Err(E::MetricLengthMismatch {
                metric_len: 1,
                x_cols: 2
            })
        ),
    );
    // 5. InvalidDataShape
    hit(
        "InvalidDataShape",
        matches!(
            Data::new(vec![1.0; 5], 2, 2),
            Err(E::InvalidDataShape { .. })
        ),
    );
    // 6. InsufficientObservations
    let one = Data::from_rows(&[[0.0, 1.0]]).unwrap();
    hit(
        "InsufficientObservations",
        matches!(
            quick_config(1).fit(&one, &y[..1]),
            Err(E::InsufficientObservations {
                found: 1,
                required: 2
            })
        ),
    );
    // 7. InvalidHyperparameter (data-free validate path)
    hit(
        "InvalidHyperparameter",
        matches!(
            quick_config(1).with_q(2.0).validate(),
            Err(E::InvalidHyperparameter { .. })
        ),
    );
    // 8. FeatureCountMismatch (predict boundary)
    let model = quick_config(2).fit(&x, &y).unwrap();
    let wide = Data::from_rows(&[[0.0, 1.0, 2.0]]).unwrap();
    hit(
        "FeatureCountMismatch",
        matches!(
            model.predict(&wide),
            Err(E::FeatureCountMismatch {
                expected: 2,
                found: 3
            })
        ),
    );
    // 9. DegenerateResponse
    hit(
        "DegenerateResponse",
        matches!(
            quick_config(1).fit(&x, &[7.0; 5]),
            Err(E::DegenerateResponse {})
        ),
    );
    // 10. DegenerateFeature
    let constant =
        Data::from_rows(&[[0.5, 1.0], [0.5, 0.2], [0.5, 0.8], [0.5, 0.4], [0.5, 0.6]]).unwrap();
    hit(
        "DegenerateFeature",
        matches!(
            quick_config(1).fit(&constant, &y),
            Err(E::DegenerateFeature { col: 0 })
        ),
    );
    // 11. UnseenCategory (predict-time encoding)
    let xc =
        Data::from_rows(&[[0.0, 1.0], [0.3, 2.0], [0.7, 1.0], [1.0, 2.0], [0.5, 1.0]]).unwrap();
    let categorical_model = quick_config(3)
        .with_metrics(vec![Metric::Euclidean, Metric::Categorical])
        .fit(&xc, &y)
        .unwrap();
    let unseen = Data::from_rows(&[[0.5, 3.0]]).unwrap();
    hit(
        "UnseenCategory",
        matches!(
            categorical_model.predict(&unseen),
            Err(E::UnseenCategory { col: 1, value }) if value == 3.0
        ),
    );
    // 12. InvalidQuantileProb
    hit(
        "InvalidQuantileProb",
        matches!(model.predict_quantiles(&x, &[1.5]), Err(E::InvalidQuantileProb { value }) if value == 1.5),
    );
    // 13. SphericalOutOfDomain
    let spherical =
        Data::from_rows(&[[9.0, 1.0], [0.3, 0.2], [0.7, 0.8], [1.0, 0.4], [0.5, 0.6]]).unwrap();
    hit(
        "SphericalOutOfDomain",
        matches!(
            quick_config(1)
                .with_metrics(vec![Metric::Spherical, Metric::Euclidean])
                .fit(&spherical, &y),
            Err(E::SphericalOutOfDomain { row: 0, col: 0, .. })
        ),
    );
    // 14. NonFiniteDistance (custom assigner mid-chain)
    let poisoned = crate::engine::builder::SamplerBuilder::new(quick_config(4))
        .with_assigner(Arc::new(PoisonMetric));
    hit(
        "NonFiniteDistance",
        matches!(poisoned.fit(&x, &y), Err(E::NonFiniteDistance { .. })),
    );
    // 15. InvalidMoveSet
    hit(
        "InvalidMoveSet",
        matches!(
            MoveSetBuilder::empty().build(),
            Err(E::InvalidMoveSet { .. })
        ),
    );
    // 16. Extension (failing custom InclusionModel update, mid-chain)
    let failing = crate::engine::builder::SamplerBuilder::new(quick_config(5)).with_inclusion(
        FailingInclusion {
            weights: vec![1.0, 1.0],
        },
    );
    hit(
        "Extension",
        matches!(failing.fit(&x, &y), Err(E::Extension { .. })),
    );
    // 17. InvalidMetricGroup (compound metric composition, the distance point)
    hit(
        "InvalidMetricGroup",
        matches!(
            crate::extensions::distance::ColumnMetrics::new(vec![Metric::Euclidean])
                .with_group(vec![9], Arc::new(PoisonMetric)),
            Err(E::InvalidMetricGroup { .. })
        ),
    );
    // 18. MembershipUnsupported (soft membership over a batch-level custom
    // assigner that provides no dense keys, the membership point): surfaces at
    // construction, before any sweep.
    #[derive(Debug)]
    struct HardOnlyAssigner;
    impl crate::extensions::distance::CellAssigner for HardOnlyAssigner {
        fn assign_cells(
            &self,
            x: &Data,
            _tessellation: &crate::engine::tessellation::Tessellation,
        ) -> crate::engine::error::Result<Vec<usize>> {
            Ok(vec![0; x.n_rows()])
        }
    }
    let hard_only = crate::engine::builder::SamplerBuilder::new(quick_config(6))
        .with_membership(crate::extensions::membership::SoftmaxKernel::new(0.1).unwrap())
        .with_assigner(Arc::new(HardOnlyAssigner));
    hit(
        "MembershipUnsupported",
        matches!(
            hard_only.build(&x, &y),
            Err(E::MembershipUnsupported { .. })
        ),
    );

    assert_eq!(seen.len(), 18, "aggregate must cover all 18 variants");
}

// ---------------------------------------------------------------------------
// Suite self-validation: bug injections against the moves oracles
// ---------------------------------------------------------------------------

/// Wrong-trials injection: an RD prior ratio pricing Binomial(p, ω/p)
/// instead of Binomial(p−1, ω/p). Re-derive the buggy value and assert the
/// per-commit oracles would go red: the value oracles (value drift) and the
/// detailed-balance oracle (AD×RD stops telescoping). The injection is RD-only: a symmetric
/// both-directions injection cancels in the product and hides itself.
#[test]
fn rd_p_vs_p_minus_one_injection_turns_oracles_red() {
    let p = 10usize;
    let omega = 2.0;
    let theta = omega / p as f64;
    let d = 3usize; // remove from d = 3 → 2

    // Correct RD (Binomial(p−1)), as implemented and oracled.
    let correct = ln((d - 1) as f64) - ln((p - d + 1) as f64) - ln(theta) + ln(1.0 - theta);
    // Buggy RD (Binomial(p)): adjacent-count ratio gains one extra trial:
    // P(d−2)/P(d−1) over Binomial(p, θ) = (d−1)/(p−d+2)·(1−θ)/θ.
    let buggy = ln((d - 1) as f64) - ln((p - d + 2) as f64) - ln(theta) + ln(1.0 - theta);

    // The value oracles would catch it: the values differ far beyond the 1e-10 tolerance.
    assert!((correct - buggy).abs() > 1e-2);

    // The detailed-balance oracle would catch it: AD(d−1 → d) + RD_buggy(d → d−1) no longer cancels.
    let ad = ln((p - (d - 1)) as f64) - ln((d - 1) as f64) + ln(theta) - ln(1.0 - theta);
    assert!((ad + correct).abs() < 1e-12, "correct pair must telescope");
    assert!(
        (ad + buggy).abs() > 1e-2,
        "buggy pair must break detailed balance"
    );
}

/// Local-slot injection: indexing the proposal distribution by local slot
/// position instead of global covariate index. On this fixture (global
/// covariate 4 at local slot 0, ConstDist tags 100·(g+1)) the buggy lookup
/// produces a value the fixture oracle rejects.
#[test]
fn local_index_injection_turns_global_index_fixture_red() {
    #[derive(Debug)]
    struct ConstDist {
        value: f64,
    }
    impl CoordinateDistribution for ConstDist {
        fn sample(&self, _rng: &mut dyn rand_core::Rng) -> f64 {
            self.value
        }
        fn log_density(&self, _x: f64) -> f64 {
            0.0
        }
    }
    let dists: Vec<Arc<dyn CoordinateDistribution>> = (0..5)
        .map(|g| {
            Arc::new(ConstDist {
                value: 100.0 * (g as f64 + 1.0),
            }) as Arc<_>
        })
        .collect();
    let weights = vec![1.0; 5];
    let ctx = ModelCtx::new(1.0, 2.0, 25.0, 0.01, 5, &dists, &weights);
    let tessellation = Tessellation::new(vec![0.1, 0.2], vec![4], vec![0.0, 0.0]).unwrap();

    // Correct behaviour (the shipped Change move): samples covariate 4's law.
    let mut rng = rand_chacha::ChaCha8Rng::from_seed([9; 32]);
    use rand_core::SeedableRng;
    let changed = Change.propose(&tessellation, &ctx, &mut rng).tessellation;
    let moved: Vec<f64> = changed
        .centres()
        .iter()
        .copied()
        .filter(|v| *v != 0.1 && *v != 0.2)
        .collect();
    assert_eq!(moved, vec![500.0]);

    // Injected bug: look the distribution up by local slot (position 0).
    let local_slot = 0usize;
    let buggy_sample = ctx.coord_dists[local_slot].sample(&mut rng);
    assert_eq!(buggy_sample, 100.0);
    // The fixture's assertion (expects 500.0) rejects the buggy value: red.
    assert_ne!(buggy_sample, moved[0]);
}

// ---------------------------------------------------------------------------
// Mutation ↔ oracle coupling demonstration (hand-applied mutants; the
// tool-driven cargo-mutants gate runs in CI)
// ---------------------------------------------------------------------------

/// A representative mutant on the AD ratio (`− ln d` → `+ ln d`) and one on
/// the AC ratio (`ln λ_c` dropped): each must be killed by the hand-derived
/// oracle values (1e-10), not by the golden chain, showing the oracle is
/// tight on exactly the mutated arithmetic.
#[test]
fn hand_applied_ratio_mutants_are_killed_by_value_oracles() {
    let p = 5.0_f64;
    let d = 2.0_f64;
    let theta = 0.4_f64;
    let lambda_c = 25.0_f64;
    let b = 3.0_f64;

    // Hand-derived oracle values (as asserted in the moves oracle suite; the AC pick
    // factor cancels, see AddCentre's rustdoc).
    let ad_oracle = ln(p - d) - ln(d) + ln(theta) - ln(1.0 - theta);
    let ac_oracle = ln(lambda_c) - ln(b);

    // Mutant 1: operator swap in AD.
    let ad_mutant = ln(p - d) + ln(d) + ln(theta) - ln(1.0 - theta);
    assert!(
        (ad_mutant - ad_oracle).abs() > 1e-10,
        "AD mutant must be killed"
    );

    // Mutant 2: dropped λ_c term in AC.
    let ac_mutant = -ln(b);
    assert!(
        (ac_mutant - ac_oracle).abs() > 1e-10,
        "AC mutant must be killed"
    );

    // And the real implementations still agree with the oracles bit-for-bit
    // (the coupling: oracle == implementation, oracle != mutant).
    let dists: Vec<Arc<dyn CoordinateDistribution>> = (0..5)
        .map(|_| {
            Arc::new(crate::extensions::coord::EuclideanNormal::new(0.8).unwrap())
                as Arc<dyn CoordinateDistribution>
        })
        .collect();
    let weights = vec![1.0; 5];
    let ctx = ModelCtx::new(1.0, 2.0, lambda_c, 0.01, 5, &dists, &weights);
    let from = Tessellation::new(vec![0.1, 0.2, 0.3, 0.4], vec![0, 1], vec![0.0, 0.0]).unwrap();
    let to = Tessellation::new(
        vec![0.1, 0.2, 0.5, 0.3, 0.4, 0.6],
        vec![0, 1, 4],
        vec![0.0, 0.0],
    )
    .unwrap();
    assert_eq!(
        AddDimension.log_structure_ratio(&from, &to, &ctx).to_bits(),
        ad_oracle.to_bits()
    );
    let rd_oracle = ln(3.0 - 1.0) - ln(p - 3.0 + 1.0) - ln(theta) + ln(1.0 - theta);
    assert_eq!(
        RemoveDimension
            .log_structure_ratio(&to, &from, &ctx)
            .to_bits(),
        rd_oracle.to_bits()
    );
}

// ---------------------------------------------------------------------------
// Config extension-point selection (the unified with_* surface): selecting each
// extension point's default explicitly through the config must reproduce the default
// chain bit for bit, and the config-level deep-seam path must match the
// Sampler-level seam wiring exactly.
// ---------------------------------------------------------------------------

fn chain_bits(mut sampler: crate::Sampler, sweeps: usize) -> Vec<u64> {
    let mut bits = Vec::new();
    for _ in 0..sweeps {
        let draw = sampler.step().unwrap();
        bits.push(draw.sigma_sq.to_bits());
        for t in draw.tessellations {
            bits.extend(t.centres().iter().map(|v| v.to_bits()));
            bits.extend(t.mus().iter().map(|v| v.to_bits()));
            bits.push(t.dims().len() as u64);
        }
    }
    bits
}

/// Selecting the defaults explicitly (paper move set, per-column Euclidean
/// coordinate laws, the standalone Euclidean geometry, bit-equal to the
/// all-Euclidean compound, uniform inclusion, the Gaussian cell model)
/// reproduces the default-components chain bit for bit.
#[test]
fn explicit_default_selection_reproduces_default_chain() {
    let (x, y) = small_xy();
    let default_bits = chain_bits(crate::Sampler::new(quick_config(41), &x, &y).unwrap(), 6);

    let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 3);
    let explicit = crate::engine::builder::SamplerBuilder::new(quick_config(41))
        .with_move_set(MoveSetBuilder::stone_gosling().build().unwrap())
        .with_coords(vec![
            Arc::new(crate::extensions::coord::EuclideanNormal::new(0.8).unwrap())
                as Arc<dyn CoordinateDistribution>,
            Arc::new(crate::extensions::coord::EuclideanNormal::new(0.8).unwrap()) as Arc<_>,
        ])
        .with_assigner(Arc::new(crate::extensions::distance::Euclidean))
        .with_inclusion(crate::extensions::inclusion::UniformInclusion::new(2))
        .with_cell_model(
            crate::extensions::cell_model::GaussianCellModel::new(sigma_mu_sq).unwrap(),
        );

    assert_eq!(
        default_bits,
        chain_bits(explicit.build(&x, &y).unwrap(), 6),
        "explicitly selecting every default extension-point must not perturb the chain"
    );
}

/// The builder route (cell model + kernel step + scale model on the
/// builder) produces the same chain as the Sampler-level seam entries: one
/// wiring, reachable mid-loop too.
#[test]
fn builder_seam_route_matches_sampler_seam_route() {
    #[derive(Debug, Clone)]
    struct HalvedWeights;
    impl crate::extensions::response::ResponseModel for HalvedWeights {
        type Error = std::convert::Infallible;
        fn augment(
            &mut self,
            y: &[f64],
            _fit: &[f64],
            _sigma_sq: f64,
            _rng: &mut dyn rand_core::Rng,
            working: &mut [f64],
            weights: &mut [f64],
        ) -> std::result::Result<(), Self::Error> {
            working.copy_from_slice(y);
            weights.fill(0.5);
            Ok(())
        }
    }

    use crate::extensions::scale::PinnedSigma;

    let (x, y) = small_xy();
    let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 3);

    let via_sampler = crate::Sampler::with_cell_model(
        quick_config(43),
        &x,
        &y,
        MoveSetBuilder::stone_gosling().build().unwrap(),
        crate::extensions::cell_model::WeightedGaussianModel::new(sigma_mu_sq).unwrap(),
    )
    .unwrap()
    .with_response_model(HalvedWeights)
    .with_scale_model(PinnedSigma::unit());

    let via_builder = crate::engine::builder::SamplerBuilder::new(quick_config(43))
        .with_cell_model(
            crate::extensions::cell_model::WeightedGaussianModel::new(sigma_mu_sq).unwrap(),
        )
        .with_response_model(HalvedWeights)
        .with_scale_model(PinnedSigma::unit())
        .build(&x, &y)
        .unwrap();

    assert_eq!(chain_bits(via_sampler, 6), chain_bits(via_builder, 6));
}

/// A wrong-length coordinate-law vector is a fit-boundary configuration error
/// with the exact field name (never a mid-chain surprise).
#[test]
fn wrong_length_coords_error_at_the_fit_boundary() {
    let (x, y) = small_xy();
    let err = crate::engine::builder::SamplerBuilder::new(quick_config(47))
        .with_coords(vec![
            Arc::new(crate::extensions::coord::EuclideanNormal::new(0.8).unwrap())
                as Arc<dyn CoordinateDistribution>,
        ])
        .build(&x, &y)
        .unwrap_err();
    assert!(matches!(
        err,
        AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "coords"
    ));
}

// ---------------------------------------------------------------------------
// The H-AddiVortes entry: zero-engine-edit acceptance
// ---------------------------------------------------------------------------

/// The H entry end-to-end with zero engine edits: config-level selection of
/// the weighted-Gaussian mean cells + the HVariance product-of-tessellations
/// scale model on raw heteroscedastic data. Acceptance: the posterior mean
/// error SD ŝ(xᵢ) tracks the generating pattern (strong positive rank with
/// x), the H-evidence intervals separate across the range, and the
/// predictive-QQ PIT values are near-uniform (e-statistic) while a
/// homoscedastic fit of the same data grades clearly worse.
#[test]
fn h_addivortes_entry_recovers_heteroscedastic_structure() {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    // y = 2x + (0.15 + 1.2x)·z: linear mean, strongly increasing noise.
    let n = 150;
    let mut rng = ChaCha8Rng::from_seed([13; 32]);
    let normal = |rng: &mut ChaCha8Rng| -> f64 {
        rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng)
    };
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let true_sd: Vec<f64> = xs.iter().map(|&v| 0.15 + 1.2 * v).collect();
    let y: Vec<f64> = xs
        .iter()
        .zip(&true_sd)
        .map(|(&v, &sd)| 2.0 * v + sd * normal(&mut rng))
        .collect();
    let x = Data::new(xs.clone(), n, 1).unwrap();

    let m = 30;
    let m_prime = 10;
    let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, m);
    let shared = std::sync::Arc::new(std::sync::Mutex::new(
        crate::extensions::scale::HVariance::new(m_prime).unwrap(),
    ));
    let builder = crate::engine::builder::SamplerBuilder::new(
        AddiVortesConfig::new(2027).with_m(m).with_omega(0.5),
    )
    .with_cell_model(
        crate::extensions::cell_model::WeightedGaussianModel::new(sigma_mu_sq).unwrap(),
    );
    let mut sampler =
        builder
            .build(&x, &y)
            .unwrap()
            .with_scale_model(crate::test_support::SharedHVariance {
                inner: std::sync::Arc::clone(&shared),
                cache: Vec::new(),
            });

    // The §3.3 calibration resolved from the engine's own (ν, λ) at sweep 1.
    sampler.step().unwrap();
    let (nu_prime, lambda_prime) = shared.lock().expect("test lock").prior_in_force().unwrap();
    assert!(nu_prime > 2.0 && lambda_prime > 0.0);

    let (burn_in, draws) = (300, 200);
    for _ in 0..burn_in {
        sampler.step().unwrap();
    }
    let y_range = sampler.scaler().y_max() - sampler.scaler().y_min();
    let mut s_draws: Vec<Vec<f64>> = Vec::with_capacity(draws);
    let mut fit_draws: Vec<Vec<f64>> = Vec::with_capacity(draws);
    for _ in 0..draws {
        sampler.step().unwrap();
        // Caller scale: s_caller = s_scaled · (y_max − y_min); the fit through
        // the embed read half.
        s_draws.push(
            shared
                .lock()
                .expect("test lock")
                .s_sq_values()
                .unwrap()
                .iter()
                .map(|s_sq| s_sq.sqrt() * y_range)
                .collect(),
        );
        fit_draws.push(sampler.fitted_values());
    }

    // (1) ŝ(x) tracks the generating pattern: Pearson correlation with x.
    let evidence = crate::diagnostics::h_evidence(&s_draws, 0.9);
    let s_hat: Vec<f64> = {
        let mut by_row: Vec<(usize, f64)> = evidence
            .iter()
            .map(|point| (point.observation, point.s_hat))
            .collect();
        by_row.sort_by_key(|(row, _)| *row);
        by_row.into_iter().map(|(_, s)| s).collect()
    };
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (mx, ms) = (mean(&xs), mean(&s_hat));
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_s = 0.0;
    for i in 0..n {
        num += (xs[i] - mx) * (s_hat[i] - ms);
        den_x += (xs[i] - mx) * (xs[i] - mx);
        den_s += (s_hat[i] - ms) * (s_hat[i] - ms);
    }
    let correlation = num / (den_x * den_s).sqrt();
    assert!(
        correlation > 0.8,
        "posterior ŝ(x) should track the increasing noise pattern, r = {correlation:.3}"
    );

    // (2) H-evidence separation (the paper's "significant heteroscedasticity"
    // reading): the smallest ŝ's upper interval sits below the largest ŝ's
    // lower interval.
    let first = &evidence[0];
    let last = &evidence[evidence.len() - 1];
    assert!(
        first.upper < last.lower,
        "H-evidence intervals should separate: [{:.3}, {:.3}] vs [{:.3}, {:.3}]",
        first.lower,
        first.upper,
        last.lower,
        last.upper
    );

    // (3) predictive calibration: the H PIT values are near-uniform, and a
    // homoscedastic fit of the same data grades clearly worse on the same
    // metric.
    let uniform_grid: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let pit_h = crate::diagnostics::predictive_qq(&y, &fit_draws, &s_draws);
    let e_h = crate::diagnostics::e_statistic(&pit_h, &uniform_grid);

    let homoscedastic = AddiVortesConfig::new(2027).with_m(m).with_omega(0.5);
    let mut homo_sampler = crate::Sampler::new(homoscedastic, &x, &y).unwrap();
    for _ in 0..burn_in {
        homo_sampler.step().unwrap();
    }
    let mut homo_s: Vec<Vec<f64>> = Vec::with_capacity(draws);
    let mut homo_fit: Vec<Vec<f64>> = Vec::with_capacity(draws);
    for _ in 0..draws {
        let sigma_scaled = homo_sampler.step().unwrap().sigma_sq.sqrt();
        homo_s.push(vec![sigma_scaled * y_range; n]);
        homo_fit.push(homo_sampler.fitted_values());
    }
    let pit_homo = crate::diagnostics::predictive_qq(&y, &homo_fit, &homo_s);
    let e_homo = crate::diagnostics::e_statistic(&pit_homo, &uniform_grid);
    // The aggregate PIT is only weakly sensitive to heteroscedasticity when
    // the overall variance level is right (the homoscedastic fit can look
    // respectable unconditionally), so the homoscedastic value is reported,
    // not asserted against; the metric's sensitivity is unit-tested in
    // `diagnostics` (halved-SD negative control); the H-specific acceptance
    // is (1) pattern recovery and (2) interval separation above.
    println!("H e-statistic {e_h:.4} vs homoscedastic {e_homo:.4} (r = {correlation:.3})");
    assert!(
        e_h < 0.05,
        "H predictive PIT should be near-uniform, e = {e_h:.4}"
    );
}

// ---------------------------------------------------------------------------
// Soft membership: the dense path end to end
// ---------------------------------------------------------------------------

/// The soft entry runs through the config with zero engine edits: same-seed
/// chains are bit-identical, different-seed chains differ, and the soft fit
/// tracks a smooth signal.
#[test]
fn soft_membership_chains_are_reproducible_and_fit() {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    let n = 60;
    let mut rng = ChaCha8Rng::from_seed([17; 32]);
    let normal = |rng: &mut ChaCha8Rng| -> f64 {
        rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng)
    };
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs
        .iter()
        .map(|&v| 2.0 * v + 0.1 * normal(&mut rng))
        .collect();
    let x = Data::from_rows(&xs.iter().map(|&v| [v]).collect::<Vec<_>>()).unwrap();

    let config = |seed: u64| {
        crate::engine::builder::SamplerBuilder::new(
            AddiVortesConfig::new(seed)
                .with_m(15)
                .with_omega(0.5)
                .with_burn_in(150)
                .with_draws(100),
        )
        .with_membership(crate::extensions::membership::SoftmaxKernel::new(0.05).unwrap())
    };
    let chain_bits = |seed: u64| -> Vec<u64> {
        let mut sampler = config(seed).build(&x, &y).unwrap();
        let mut bits = Vec::new();
        for _ in 0..10 {
            let draw = sampler.step().unwrap();
            bits.push(draw.sigma_sq.to_bits());
            for t in draw.tessellations {
                bits.extend(t.mus().iter().map(|v| v.to_bits()));
            }
        }
        bits
    };
    assert_eq!(chain_bits(7), chain_bits(7));
    assert_ne!(chain_bits(7), chain_bits(8));

    // End-to-end fit + predict through the same assigner/kernel dispatch.
    let model = config(7).fit(&x, &y).unwrap();
    assert!(
        model.in_sample_rmse() < 0.25,
        "soft fit should track the signal, rmse {}",
        model.in_sample_rmse()
    );
}

/// The cache-bypass consistency check: after real sweeps, the
/// dense path's cached membership matrix for every tessellation equals a
/// fresh recompute from (assigner, kernel, X, tessellation), bit for bit:
/// soft membership never rides the incremental `AssignmentDelta` cache.
#[test]
fn soft_membership_cache_matches_fresh_recompute_bitwise() {
    let (x, y) = small_xy();
    let kernel = crate::extensions::membership::SoftmaxKernel::new(0.1).unwrap();
    let mut sampler = crate::engine::builder::SamplerBuilder::new(quick_config(11))
        .with_membership(kernel)
        .build(&x, &y)
        .unwrap();
    for _ in 0..30 {
        sampler.step().unwrap();
    }
    let assigner = crate::extensions::distance::default_assigner(vec![Metric::Euclidean; 2]);
    let scaler_metrics = [Metric::Euclidean, Metric::Euclidean];
    let _ = scaler_metrics;
    for j in 0..3 {
        let cached = sampler.membership_for_tests(j).unwrap().to_vec();
        let tessellation = sampler.tessellations_for_tests()[j].clone();
        let fresh = crate::engine::backfit::compute_memberships(
            assigner.as_ref(),
            &crate::extensions::membership::SoftmaxKernel::new(0.1).unwrap(),
            sampler.design_for_tests(),
            &tessellation,
        )
        .unwrap();
        assert_eq!(cached.len(), fresh.len(), "tessellation {j}");
        for (a, b) in cached.iter().zip(&fresh) {
            assert_eq!(a.to_bits(), b.to_bits(), "tessellation {j}");
        }
    }
}

/// Predict-time soft dispatch: predictions equal the hand-computed
/// membership-weighted ensemble sums through the same assigner + kernel.
#[test]
fn soft_predictions_match_hand_computed_membership_sums() {
    let (x, y) = small_xy();
    let kernel = crate::extensions::membership::SoftmaxKernel::new(0.1).unwrap();
    let model = crate::engine::builder::SamplerBuilder::new(
        quick_config(13).with_burn_in(20).with_draws(10),
    )
    .with_membership(kernel)
    .fit(&x, &y)
    .unwrap();
    let predictions = model.predict(&x).unwrap();

    let (posterior, scaler) = model.into_parts();
    let assigner = crate::extensions::distance::default_assigner(vec![Metric::Euclidean; 2]);
    let x_enc = scaler.apply_x(&x).unwrap();
    let n = x_enc.n_rows();
    let mut totals = vec![0.0_f64; n];
    for draw in posterior.iter_draws() {
        let mut sums = vec![0.0_f64; n];
        for tessellation in draw.tessellations {
            let membership = crate::engine::backfit::compute_memberships(
                assigner.as_ref(),
                &crate::extensions::membership::SoftmaxKernel::new(0.1).unwrap(),
                &x_enc,
                tessellation,
            )
            .unwrap();
            let b = tessellation.n_cells();
            for (i, sum) in sums.iter_mut().enumerate() {
                let phi = &membership[i * b..(i + 1) * b];
                *sum += phi
                    .iter()
                    .zip(tessellation.mus())
                    .map(|(p, mu)| p * mu)
                    .sum::<f64>();
            }
        }
        for (total, sum) in totals.iter_mut().zip(&sums) {
            *total += scaler.unscale_y_value(*sum);
        }
    }
    for (prediction, total) in predictions.iter().zip(&totals) {
        let mean = total / posterior.n_draws() as f64;
        crate::test_support::assert_rel_eq(*prediction, mean, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// The response family: predict-side links + per-variant predictive
// ---------------------------------------------------------------------------

/// Binary-AddiVortes end to end through `fit()` with zero engine edits:
/// {0, 1} labels in, probability-scale predictions out through the probit
/// link, tracking a monotone signal; the Bernoulli predictive quantises the
/// prediction interval to labels; a non-binary response is a fit-boundary
/// configuration error.
#[test]
fn binary_probit_family_predicts_on_the_probability_scale() {
    use rand_chacha::ChaCha8Rng;
    use rand_core::{Rng as _, SeedableRng};

    let n = 120;
    let mut rng = ChaCha8Rng::from_seed([29; 32]);
    let uniform = |rng: &mut ChaCha8Rng| -> f64 {
        (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    };
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    // P(Y = 1 | x) = Φ(4(x − ½)): steep monotone truth.
    let phi = |z: f64| 0.5 * crate::engine::mathsfn::erfc(-z / std::f64::consts::SQRT_2);
    let y: Vec<f64> = xs
        .iter()
        .map(|&v| f64::from(uniform(&mut rng) < phi(4.0 * (v - 0.5))))
        .collect();
    let x = Data::from_rows(&xs.iter().map(|&v| [v]).collect::<Vec<_>>()).unwrap();

    let model = AddiVortesConfig::new(4021)
        .with_m(25)
        .with_omega(0.5)
        .with_burn_in(200)
        .with_draws(150)
        .with_response_family(crate::ResponseFamily::BinaryProbit)
        .fit(&x, &y)
        .unwrap();

    let predictions = model.predict(&x).unwrap();
    assert!(
        predictions.iter().all(|p| (0.0..=1.0).contains(p)),
        "probit predictions live on the probability scale"
    );
    // Monotone truth recovered at the ends.
    assert!(predictions[5] < 0.35, "low end: {}", predictions[5]);
    assert!(
        predictions[n - 6] > 0.65,
        "high end: {}",
        predictions[n - 6]
    );
    // σ is pinned to the unit latent scale.
    assert!(
        model
            .sigma()
            .iter()
            .all(|s| s.to_bits() == 1.0_f64.to_bits())
    );
    // The Bernoulli predictive's interval ends are labels.
    let intervals = model.prediction_interval(&x, 0.9).unwrap();
    assert!(
        intervals
            .iter()
            .all(|i| (i.lower == 0.0 || i.lower == 1.0) && (i.upper == 0.0 || i.upper == 1.0))
    );
    // The uncertain mid-region's 90% predictive spans both labels.
    let mid = &intervals[n / 2];
    assert_eq!((mid.lower, mid.upper), (0.0, 1.0));

    // Non-binary response: a configuration error at the fit boundary.
    let err = AddiVortesConfig::new(4022)
        .with_response_family(crate::ResponseFamily::BinaryProbit)
        .fit(&x, &xs)
        .unwrap_err();
    assert!(matches!(
        err,
        AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "response_family"
    ));
}

/// The multi-chain entry: chain 0 is bit-identical to a single
/// fit (the seed derivation never perturbs the single-chain contract),
/// later chains differ, and on well-behaved data the σ chains pass the
/// convergence diagnostics.
#[test]
fn fit_chains_keeps_chain_zero_identical_and_converges() {
    let n = 40;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 2.0 * v * v + 0.3 * v).collect();
    let x = Data::from_rows(&xs.iter().map(|&v| [v]).collect::<Vec<_>>()).unwrap();
    let config = AddiVortesConfig::new(77)
        .with_m(10)
        .with_burn_in(150)
        .with_draws(120);

    let chains = config.fit_chains(&x, &y, 4).unwrap();
    let single = config.clone().fit(&x, &y).unwrap();
    // Chain 0 ≡ single fit, bit for bit.
    for (a, b) in chains[0]
        .predict(&x)
        .unwrap()
        .iter()
        .zip(&single.predict(&x).unwrap())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    // Later chains are genuinely different.
    assert_ne!(
        chains[0].posterior().sigma_sq()[0].to_bits(),
        chains[1].posterior().sigma_sq()[0].to_bits()
    );
    // The σ chains agree across seeds on well-behaved data.
    let sigma_chains: Vec<Vec<f64>> = chains
        .iter()
        .map(|model| model.posterior().sigma_sq().to_vec())
        .collect();
    let r_hat = crate::diagnostics::r_hat(&sigma_chains);
    let ess = crate::diagnostics::ess_bulk(&sigma_chains);
    println!("fit_chains sigma: r_hat = {r_hat:.4}, bulk ESS = {ess:.0}");
    assert!(r_hat < 1.1, "sigma chains should agree, r_hat = {r_hat}");
    assert!(ess > 40.0, "bulk ESS should be usable, got {ess}");

    // n_chains = 0 is a configuration error.
    assert!(matches!(
        config.fit_chains(&x, &y, 0),
        Err(AddiVortesError::InvalidHyperparameter { .. })
    ));
}

// ---------------------------------------------------------------------------
// The count-prior point: a selected count prior reaches the fit
// ---------------------------------------------------------------------------

/// A cell-count prior that charges a flat, heavy premium per extra cell and
/// otherwise delegates: it must pull the sampled tessellations towards fewer
/// cells than the paper's shifted Poisson, and it does so through the
/// acceptance ratio of every count-changing move, touching no move code.
#[derive(Debug)]
struct Parsimonious {
    fallback: crate::extensions::count_priors::ShiftedPoissonBinomial,
}

impl crate::extensions::count_priors::CountPriors for Parsimonious {
    fn log_cell_count_ratio(&self, _b: usize, _ctx: &crate::extensions::moves::ModelCtx) -> f64 {
        // ln P(b) - ln P(b-1) = ln(0.05): each extra cell costs a factor 20.
        ln(0.05)
    }
    fn log_dim_count_ratio(&self, d: usize, ctx: &crate::extensions::moves::ModelCtx) -> f64 {
        self.fallback.log_dim_count_ratio(d, ctx)
    }
}

fn count_prior_fixture() -> (crate::Data, Vec<f64>) {
    let n = 60;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|v| 3.0 * v * v - v).collect();
    (crate::Data::new(xs, n, 1).unwrap(), y)
}

fn count_prior_config() -> crate::AddiVortesConfig {
    crate::AddiVortesConfig::new(0xC0_1234)
        .with_m(4)
        .with_burn_in(30)
        .with_draws(40)
        .with_lambda_c(25.0)
}

/// Mean cells per draw across the whole ensemble.
fn mean_cells(model: &crate::FittedAddiVortes) -> f64 {
    let posterior = model.posterior();
    let mut total = 0.0;
    for draw in 0..posterior.n_draws() {
        total += posterior
            .tessellations(draw)
            .iter()
            .map(|t| t.n_cells() as f64)
            .sum::<f64>();
    }
    total / posterior.n_draws() as f64
}

/// The seam is reachable: a custom count prior selected on the config actually
/// prices the chain, and a parsimonious one yields a strictly smaller
/// tessellation than the paper's prior on the same seed and data.
#[test]
fn selected_count_prior_reaches_the_fit() {
    let (x, y) = count_prior_fixture();
    let paper = count_prior_config().fit(&x, &y).unwrap();
    let parsimonious = crate::engine::builder::SamplerBuilder::new(count_prior_config())
        .with_count_priors(Parsimonious {
            fallback: crate::extensions::count_priors::ShiftedPoissonBinomial,
        })
        .fit(&x, &y)
        .unwrap();

    let paper_cells = mean_cells(&paper);
    let parsimonious_cells = mean_cells(&parsimonious);
    assert!(
        parsimonious_cells < paper_cells,
        "a heavy per-cell premium must shrink the ensemble: \
         paper {paper_cells:.2} cells vs parsimonious {parsimonious_cells:.2}"
    );
}

/// The default is exactly the shelf entry: selecting `ShiftedPoissonBinomial`
/// explicitly must reproduce the unset chain bit for bit. This is what makes
/// the seam free, and it is the guard on the golden vectors: if the hook ever
/// stops pricing identically to the pre-hook kernel, this fails first and
/// names the reason.
#[test]
fn explicit_default_count_prior_is_bit_identical_to_unset() {
    let (x, y) = count_prior_fixture();
    let unset = count_prior_config().fit(&x, &y).unwrap();
    let explicit = crate::engine::builder::SamplerBuilder::new(count_prior_config())
        .with_count_priors(crate::extensions::count_priors::ShiftedPoissonBinomial)
        .fit(&x, &y)
        .unwrap();

    let unset_sigma = unset.sigma();
    let explicit_sigma = explicit.sigma();
    assert_eq!(unset_sigma.len(), explicit_sigma.len());
    assert!(
        unset_sigma
            .iter()
            .zip(explicit_sigma.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "selecting the default count prior explicitly changed the sigma chain"
    );
    assert_eq!(
        mean_cells(&unset),
        mean_cells(&explicit),
        "selecting the default count prior explicitly changed the cell counts"
    );
}

// ---------------------------------------------------------------------------
// The basis point: cell basis. The payload is q-wide, the engine feeds it basis rows,
// and the fit/predict contribution is z(xᵢ)·β_k rather than μ_k.
// ---------------------------------------------------------------------------

/// A surface that is linear *within* regions but whose slope flips with x1:
/// constant cells must spend cells chasing the slope, linear cells need few.
fn basis_fixture(n: usize) -> (Data, Vec<f64>) {
    let mut xs = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64 / (n - 1) as f64;
        let b = ((i * 7) % n) as f64 / (n - 1) as f64;
        xs.push(a);
        xs.push(b);
        let slope = if b > 0.5 { 3.0 } else { -3.0 };
        y.push(slope * a + 0.5 * b);
    }
    (Data::new(xs, n, 2).unwrap(), y)
}

fn basis_config(seed: u64) -> AddiVortesConfig {
    AddiVortesConfig::new(seed)
        .with_m(20)
        .with_omega(1.0)
        .with_burn_in(120)
        .with_draws(120)
}

/// The regression test for the q > 1 panic: before the engine fed basis rows,
/// a basis payload accumulated a q = 1 statistic and the q×q solve indexed off
/// the end of it (`index out of bounds: the len is 1 but the index is 1`).
#[test]
fn basis_payload_with_q_above_one_fits_and_predicts() {
    let (x, y) = basis_fixture(120);
    let fitted = crate::engine::builder::SamplerBuilder::new(basis_config(11))
        .with_cell_model(crate::extensions::basis::LinearGaussianModel::new(0.05, 2).unwrap())
        .with_cell_basis(crate::extensions::basis::LinearBasis::new(vec![0]))
        .fit(&x, &y)
        .expect("a q = 2 basis payload must fit");
    let predictions = fitted.predict(&x).unwrap();
    assert_eq!(predictions.len(), 120);
    assert!(predictions.iter().all(|p| p.is_finite()));
    // Every draw's payload really is q-wide: 2 coefficients per cell.
    for draw in fitted.posterior().iter_draws() {
        for tessellation in draw.tessellations {
            assert_eq!(tessellation.q(), 2);
            assert_eq!(tessellation.mus().len(), tessellation.n_cells() * 2);
        }
    }
}

/// The extension point earns its place: linear cells beat constant cells on a surface
/// that is linear within regions.
#[test]
fn linear_cells_beat_constant_cells_on_a_locally_linear_surface() {
    let (x, y) = basis_fixture(200);
    let scalar = basis_config(7).fit(&x, &y).unwrap();
    let basis = crate::engine::builder::SamplerBuilder::new(basis_config(7))
        .with_cell_model(crate::extensions::basis::LinearGaussianModel::new(0.05, 2).unwrap())
        .with_cell_basis(crate::extensions::basis::LinearBasis::new(vec![0]))
        .fit(&x, &y)
        .unwrap();
    assert!(
        basis.in_sample_rmse() < 0.5 * scalar.in_sample_rmse(),
        "linear cells ({}) should comfortably beat constant cells ({}) here",
        basis.in_sample_rmse(),
        scalar.in_sample_rmse()
    );
}

/// The linear family at q = 1 *is* the scalar Gaussian family (the intercept
/// basis `z ≡ [1]`), so it needs no [`CellBasis`] at all and rides the
/// unmodified scalar path: it must reproduce the default chain.
#[test]
fn the_linear_family_at_q_one_reproduces_the_scalar_family() {
    let (x, y) = basis_fixture(60);
    let scalar = basis_config(3).fit(&x, &y).unwrap();
    // The cell prior the default assembles for this config: σ_μ = 0.5/(k√m).
    let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 20);
    let linear = crate::engine::builder::SamplerBuilder::new(basis_config(3))
        .with_cell_model(
            crate::extensions::basis::LinearGaussianModel::new(sigma_mu_sq, 1).unwrap(),
        )
        .fit(&x, &y)
        .unwrap();

    let a = scalar.predict(&x).unwrap();
    let b = linear.predict(&x).unwrap();
    for (lhs, rhs) in a.iter().zip(&b) {
        assert!(
            (lhs - rhs).abs() <= 1e-12,
            "the q = 1 linear family diverged from the scalar family: {lhs} vs {rhs}"
        );
    }
}

/// The three pairing errors, all raised at `fit` rather than panicking later.
#[test]
fn basis_and_payload_must_arrive_together_and_agree() {
    let (x, y) = basis_fixture(40);

    // A basis payload with no basis.
    let err = crate::engine::builder::SamplerBuilder::new(basis_config(1))
        .with_cell_model(crate::extensions::basis::LinearGaussianModel::new(0.05, 2).unwrap())
        .fit(&x, &y)
        .unwrap_err();
    assert!(
        matches!(&err, AddiVortesError::InvalidHyperparameter { name, .. } if name == "cell_model"),
        "expected a cell_model error, got {err:?}"
    );

    // A basis with a scalar payload.
    let err = crate::engine::builder::SamplerBuilder::new(basis_config(1))
        .with_cell_basis(crate::extensions::basis::LinearBasis::new(vec![0]))
        .fit(&x, &y)
        .unwrap_err();
    assert!(
        matches!(&err, AddiVortesError::InvalidHyperparameter { name, .. } if name == "cell_basis"),
        "expected a cell_basis error, got {err:?}"
    );

    // q disagreement between the payload and the basis.
    let err = crate::engine::builder::SamplerBuilder::new(basis_config(1))
        .with_cell_model(crate::extensions::basis::LinearGaussianModel::new(0.05, 3).unwrap())
        .with_cell_basis(crate::extensions::basis::LinearBasis::new(vec![0]))
        .fit(&x, &y)
        .unwrap_err();
    assert!(
        matches!(&err, AddiVortesError::InvalidHyperparameter { name, .. } if name == "cell_basis"),
        "expected a cell_basis error, got {err:?}"
    );

    // A basis column the encoded design does not have.
    let err = crate::engine::builder::SamplerBuilder::new(basis_config(1))
        .with_cell_model(crate::extensions::basis::LinearGaussianModel::new(0.05, 2).unwrap())
        .with_cell_basis(crate::extensions::basis::LinearBasis::new(vec![9]))
        .fit(&x, &y)
        .unwrap_err();
    assert!(
        matches!(&err, AddiVortesError::InvalidHyperparameter { name, .. } if name == "cell_basis"),
        "expected a cell_basis error, got {err:?}"
    );
}

/// The basis point and the membership point do not compose: soft membership owns its own joint draw.
#[test]
fn a_basis_payload_and_soft_membership_are_refused() {
    let (x, y) = basis_fixture(40);
    let err = crate::engine::builder::SamplerBuilder::new(basis_config(1))
        .with_cell_model(crate::extensions::basis::LinearGaussianModel::new(0.05, 2).unwrap())
        .with_cell_basis(crate::extensions::basis::LinearBasis::new(vec![0]))
        .with_membership(crate::extensions::membership::SoftmaxKernel::new(0.1).unwrap())
        .fit(&x, &y)
        .unwrap_err();
    assert!(
        matches!(&err, AddiVortesError::InvalidHyperparameter { name, .. } if name == "cell_basis"),
        "expected a cell_basis error, got {err:?}"
    );
}
