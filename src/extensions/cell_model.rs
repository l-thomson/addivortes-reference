//! **Cell payload family**, *"cells should hold something other than a Gaussian mean."*
//!
//! Implement [`CellModel`] + [`CellStats`]: your sufficient statistic
//! (`record`/`merge`/`remove`, order-free and additive), the integrated
//! log-marginal terms, and the conjugate payload redraw. That triple is what
//! the per-tessellation update reduces to in the paper's model and both
//! published variants (accumulate per-cell statistics, score a candidate
//! structure by its integrated marginal likelihood, redraw the per-cell
//! payloads from their conjugate posterior); what varies between them is the
//! conjugate family, not the shape of the update.
//!
//! Rules that keep it sound: an empty statistic draws the prior; never drop the
//! per-cell normalising term; route transcendentals through `mathsfn`.
//!
//! Shelf: [`GaussianCellModel`] (the paper's leaf model, and the fit-time
//! default), [`WeightedGaussianModel`] (precision-weighted, correct under
//! **hard** membership only), [`InvChiSqCellModel`] (the variance family).
//! Template: `examples/template_cell_model.rs`. Conformance checks:
//! `conformance::check_cell_model` (sufficiency, a marginal-vs-Monte-Carlo
//! Bayes factor, cell-local SBC of the value draw), with
//! `check_variance_cell_model` / `check_basis_cell_model` as the
//! variance-family and basis-payload siblings (the base check hardcodes the
//! mean-family working likelihood).
//!
//! Scope rule: **conditionally-conjugate models only**. A response family
//! belongs here iff conditional conjugacy can be restored, usually via a
//! `ResponseModel` on the response point (`crate::extensions::response`);
//! genuinely non-conjugate likelihoods are out of scope by design.
//!
//! The pinned per-sweep hook order (part of the reproducibility contract):
//! `ResponseModel::augment` → `ScaleModel::update`
//! → `InclusionModel::update` → the sequential j-loop.
//!
//! Sources: the conjugate Gaussian cell (and its σ_μ = 0.5/(k√m) calibration
//! shape) is BART's leaf model (Chipman, George & McCulloch 2010) as
//! transplanted by the paper (Stone & Gosling 2025); the weighted variant is
//! textbook precision-weighted conjugacy (Gelman et al. 2013, *Bayesian Data
//! Analysis*, 3rd edn, §2.5); [`InvChiSqCellModel`] is the H-AddiVortes
//! variance cell (Stone & Gosling 2025b, §3.2).

mod gaussian;
mod inv_chi_sq;
mod weighted_gaussian;

#[allow(unused_imports)]
pub use gaussian::{GaussianCellModel, GaussianCellStats};
#[allow(unused_imports)]
pub use inv_chi_sq::{InvChiSqCellModel, InvChiSqStats};
#[allow(unused_imports)]
pub use weighted_gaussian::{WeightedGaussianModel, WeightedGaussianStats};

use crate::engine::mathsfn;

// ---------------------------------------------------------------------------
// CellStats
// ---------------------------------------------------------------------------

/// A per-cell accumulated sufficient statistic (stochtree's `SuffStat`
/// pattern under cell-native names). Observations arrive with an explicit
/// **weight**, `1.0` under hard assignment; fractional under soft assignment
/// or precision weighting (H-AddiVortes Eq. 6). What the statistic *contains*
/// is the paired [`CellModel`]'s business (scalar sums here; a linear-in-cell
/// model may carry matrices).
pub trait CellStats: Clone + Default + std::fmt::Debug {
    /// Absorb one observation's working value with the given weight.
    fn record(&mut self, value: f64, weight: f64);

    /// Absorb one observation against its basis row `z`. Scalar
    /// families ignore `z` (it is the intercept `[1.0]`) and fall through to
    /// [`record`](Self::record); a basis statistic overrides this to accumulate
    /// its (ZᵀWZ, ZᵀWr) blocks.
    fn record_basis(&mut self, z: &[f64], value: f64, weight: f64) {
        debug_assert_eq!(z.len(), 1, "a scalar statistic only takes the q = 1 basis");
        self.record(value, weight);
    }
    /// Add another accumulator of the same type (stochtree `Add`).
    fn merge(&mut self, other: &Self);
    /// Subtract another accumulator of the same type (stochtree `Subtract`).
    fn remove(&mut self, other: &Self);
    /// Clear to the empty state.
    fn reset(&mut self);
    /// Whether the cell holds any observation mass: the question the
    /// sampler's empty-cell guard asks.
    fn occupied(&self) -> bool;
}

// ---------------------------------------------------------------------------
// CellModel
// ---------------------------------------------------------------------------

/// The swappable cell-value rule book (stochtree's `LeafModel` pattern):
/// turns accumulated [`CellStats`] into (a) the structure-dependent
/// integrated log-marginal-likelihood of a candidate cell set and (b) the
/// conjugate cell-value draws. `log_marginal_terms` plays both of
/// stochtree's split/no-split roles, AddiVortes proposes whole structures,
/// so the sampler evaluates the proposed and the current cell set and takes
/// the difference.
pub trait CellModel: std::fmt::Debug + Send + Sync {
    /// The paired per-cell sufficient statistic.
    type Stats: CellStats + 'static;
    /// The extension-error channel: surfaced by the sampler
    /// as [`AddiVortesError::Extension`](crate::AddiVortesError::Extension).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Structure-dependent integrated log-marginal-likelihood terms of one
    /// candidate cell set, summed in **ascending cell index** (pinned
    /// evaluation order). Constant
    /// factors shared by any two structures over the same observations may be
    /// omitted (they cancel in every acceptance ratio).
    fn log_marginal_terms(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
    ) -> std::result::Result<f64, Self::Error>;

    /// Draw one cell value per cell in **ascending cell index** (pinned
    /// evaluation order), one value per statistic. Values are in **scaled
    /// space**.
    fn draw_cell_values(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<Vec<f64>, Self::Error>;

    /// Whether cell outputs are linear in a covariate basis rather than
    /// scalar (stochtree's `RequiresBasis`, cell-native name). Scalar models
    /// keep the default `false`; a `true` model requires a
    /// [`CellBasis`](crate::extensions::basis::CellBasis) on the config, and
    /// the engine records through [`CellStats::record_basis`] and draws through
    /// [`draw_cell_payload`](Self::draw_cell_payload).
    fn cell_basis(&self) -> bool {
        false
    }

    /// The payload width q: values this model puts in each cell. Scalar
    /// families keep the default `1`; a basis payload returns its q, which must
    /// match the configured [`CellBasis`](crate::extensions::basis::CellBasis)
    /// (validated at `fit`).
    fn payload_width(&self) -> usize {
        1
    }

    /// Draw the full per-cell payload in **ascending cell index**, flattened
    /// row-major: `q` values per cell, matching
    /// [`Tessellation::mus`](crate::Tessellation::mus). Scalar families inherit
    /// this from [`draw_cell_values`](Self::draw_cell_values); a basis payload
    /// overrides it to emit its coefficient vectors.
    fn draw_cell_payload(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<Vec<f64>, Self::Error> {
        self.draw_cell_values(stats, sigma_sq, rng)
    }
}

/// Conjugate posterior of one cell's μ given its (possibly weighted) count
/// and sum: mean = σ_μ²·S / (n_k·σ_μ² + σ²), var = σ_μ²·σ² / (n_k·σ_μ² + σ²).
pub(crate) fn mu_posterior(count: f64, sum: f64, sigma_sq: f64, sigma_mu_sq: f64) -> (f64, f64) {
    let denominator = count * sigma_mu_sq + sigma_sq;
    (
        sigma_mu_sq * sum / denominator,
        sigma_mu_sq * sigma_sq / denominator,
    )
}

/// The structure-dependent part of the integrated (marginal) log-likelihood
/// under one tessellation, from (count, sum) pairs in ascending cell index:
///
/// ```text
/// Σ_k [ 0.5·ln(σ² / (n_k σ_μ² + σ²))  +  σ_μ² S_k² / (2σ²(n_k σ_μ² + σ²)) ]
/// ```
///
/// The (2πσ²)^{−n/2} and exp(−Σr²/2σ²) factors cancel in every acceptance
/// ratio and are omitted. The **complete per-cell** first term is what makes
/// the ±0.5·ln σ² constants of cell-count-changing moves emerge automatically.
pub(crate) fn gaussian_marginal_terms(
    counts_and_sums: impl Iterator<Item = (f64, f64)>,
    sigma_sq: f64,
    sigma_mu_sq: f64,
) -> f64 {
    let mut total = 0.0_f64;
    for (count, sum) in counts_and_sums {
        let denominator = count * sigma_mu_sq + sigma_sq;
        total += 0.5 * mathsfn::ln(sigma_sq / denominator);
        total += sigma_mu_sq * sum * sum / (2.0 * sigma_sq * denominator);
    }
    total
}
#[cfg(test)]
mod tests {
    use crate::engine::error::AddiVortesError;
    use crate::extensions::basis::LinearGaussianModel;
    use crate::extensions::erasure::{ErasedCellKernel, KernelOf};
    use crate::extensions::response::ResponseModel;
    use crate::extensions::scale::PinnedSigma;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::test_support::{assert_abs_eq, assert_rel_eq};

    fn rejects_as(result: Result<impl std::fmt::Debug, AddiVortesError>, argument: &str) {
        assert!(
            matches!(
                &result,
                Err(AddiVortesError::InvalidHyperparameter { name, .. }) if name == argument
            ),
            "expected `{argument}` to be rejected, got {result:?}"
        );
    }

    /// Every shelf cell model checks its prior arguments in every build profile.
    #[test]
    fn cell_model_constructors_reject_out_of_domain_arguments() {
        for bad in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            rejects_as(GaussianCellModel::new(bad), "sigma_mu_sq");
            rejects_as(WeightedGaussianModel::new(bad), "sigma_mu_sq");
            rejects_as(InvChiSqCellModel::new(bad, 0.5), "nu");
            rejects_as(InvChiSqCellModel::new(6.0, bad), "lambda");
            rejects_as(LinearGaussianModel::new(bad, 2), "sigma_beta_sq");
        }
        rejects_as(LinearGaussianModel::new(0.5, 0), "q");
    }

    // CI leg: fast-PR (deterministic).

    #[test]
    fn mu_posterior_matches_hand_values() {
        // n_k = 4, S = 2, σ² = 0.5, σ_μ² = 0.25: denom = 1.5,
        // mean = 0.25·2/1.5 = 1/3, var = 0.25·0.5/1.5 = 1/12.
        let (mean, variance) = mu_posterior(4.0, 2.0, 0.5, 0.25);
        assert_rel_eq(mean, 1.0 / 3.0, 1e-15);
        assert_rel_eq(variance, 1.0 / 12.0, 1e-15);
        // Empty-cell limit (defensive; the sampler never draws for one):
        // mean → 0, var → σ_μ²: the prior.
        let (mean, variance) = mu_posterior(0.0, 0.0, 0.5, 0.25);
        assert_abs_eq(mean, 0.0, 0.0);
        assert_abs_eq(variance, 0.25, 0.0);
    }

    #[test]
    fn complete_per_cell_term_makes_sigma_constant_emerge() {
        // Cell-count-changing comparison (2 → 3 cells over the same 4
        // residuals): the marginal-likelihood difference must contain the
        // −0.5·ln σ² of the extra normalising constant AUTOMATICALLY,
        // hand-derived expected values below, unchanged by the seam.
        let residuals = [1.0, 2.0, 3.0, 4.0];
        let sigma_sq = 2.0;
        let sigma_mu_sq = 0.5;
        let model = GaussianCellModel::new(sigma_mu_sq).unwrap();
        let kernel = KernelOf(model);

        let two = kernel.accumulate(&[0, 0, 1, 1], &residuals, None, 2, None);
        let three = kernel.accumulate(&[0, 1, 2, 2], &residuals, None, 3, None);
        let ln = crate::engine::mathsfn::ln;

        let expected_two = ln(2.0 / 3.0) + 0.5 * (9.0 + 49.0) * 0.5 / (2.0 * 3.0);
        assert_abs_eq(
            kernel.log_marginal(&two, sigma_sq).unwrap(),
            expected_two,
            1e-12,
        );
        let expected_three = ln(2.0 / 2.5)
            + 0.5 * ln(2.0 / 3.0)
            + 0.5 * (0.5 * 1.0 / (2.0 * 2.5) + 0.5 * 4.0 / (2.0 * 2.5) + 0.5 * 49.0 / (2.0 * 3.0));
        assert_abs_eq(
            kernel.log_marginal(&three, sigma_sq).unwrap(),
            expected_three,
            1e-12,
        );
    }

    /// The trivial second `CellStats` (weights ≡ 1) produces bit-identical
    /// marginal terms and μ draws through the same call sites (the
    /// seam-boundary guarantee, via the real trait pair; the full-chain
    /// bit-identity proof lives in the sampler tests).
    #[test]
    fn weighted_stats_at_unit_weights_are_bit_identical() {
        let residuals = [0.3, -1.2, 0.7, 2.2, -0.4];
        let assignment = [0usize, 1, 0, 2, 1];
        let gaussian = KernelOf(GaussianCellModel::new(0.02).unwrap());
        let weighted = KernelOf(WeightedGaussianModel::new(0.02).unwrap());

        let g_stats = gaussian.accumulate(&assignment, &residuals, None, 3, None);
        let w_stats = weighted.accumulate(&assignment, &residuals, None, 3, None);
        let a = gaussian.log_marginal(&g_stats, 1.7).unwrap();
        let b = weighted.log_marginal(&w_stats, 1.7).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());

        let mut rng_a = ChaCha8Rng::from_seed([7; 32]);
        let mut rng_b = ChaCha8Rng::from_seed([7; 32]);
        let mus_a = gaussian
            .draw_cell_values(&g_stats, 1.7, &mut rng_a)
            .unwrap();
        let mus_b = weighted
            .draw_cell_values(&w_stats, 1.7, &mut rng_b)
            .unwrap();
        assert_eq!(
            mus_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            mus_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    // ---- full-chain seam proofs -------------------------------------------

    fn seam_data() -> (crate::Data, Vec<f64>) {
        let x = crate::Data::from_rows(&[
            [0.0, 5.0],
            [0.25, 4.0],
            [0.5, 3.0],
            [0.75, 2.0],
            [1.0, 1.0],
            [1.25, 0.0],
        ])
        .unwrap();
        let y = vec![0.1, 0.4, 0.7, 1.4, 1.8, 2.3];
        (x, y)
    }

    fn seam_config(seed: u64) -> crate::AddiVortesConfig {
        crate::AddiVortesConfig::new(seed).with_m(4).with_omega(1.5)
    }

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

    /// Swapping in the built-in Gaussian `CellModel` through the public seam
    /// entry reproduces the default chain bit for bit: the seam is a true
    /// refactor, not a behaviour change (the golden-chain test pins the same
    /// fact against the checked-in vector).
    #[test]
    fn explicit_gaussian_cell_model_reproduces_default_chain() {
        let (x, y) = seam_data();
        let default_bits = chain_bits(crate::Sampler::new(seam_config(7), &x, &y).unwrap(), 8);
        let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 4);
        let via_seam = crate::Sampler::with_cell_model(
            seam_config(7),
            &x,
            &y,
            crate::extensions::moves::MoveSetBuilder::stone_gosling()
                .build()
                .unwrap(),
            GaussianCellModel::new(sigma_mu_sq).unwrap(),
        )
        .unwrap();
        assert_eq!(default_bits, chain_bits(via_seam, 8));
    }

    /// The trivial second `CellStats` (weights ≡ 1) plugs into BOTH the
    /// accept/reject and the μ-draw call sites unchanged and the whole chain
    /// stays bit-identical.
    #[test]
    fn weighted_model_at_unit_weights_reproduces_default_chain() {
        let (x, y) = seam_data();
        let default_bits = chain_bits(crate::Sampler::new(seam_config(11), &x, &y).unwrap(), 8);
        let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 4);
        let via_weighted = crate::Sampler::with_cell_model(
            seam_config(11),
            &x,
            &y,
            crate::extensions::moves::MoveSetBuilder::stone_gosling()
                .build()
                .unwrap(),
            WeightedGaussianModel::new(sigma_mu_sq).unwrap(),
        )
        .unwrap();
        assert_eq!(default_bits, chain_bits(via_weighted, 8));
    }

    // ---- binding tests: the two sibling papers as skeletons ---------------

    /// Albert–Chib probit skeleton (Binary-AddiVortes): latent z_i drawn from
    /// a normal truncated to the side y_i indicates; conditional on z the
    /// model is Gaussian on the latent scale with σ² pinned to 1. Rejection
    /// sampling suffices for a skeleton: the real extension would use an
    /// inverse-CDF draw.
    #[derive(Debug)]
    struct AlbertChibProbit {
        /// The binary labels (true = 1), fixed at fit time.
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
        ) -> std::result::Result<(), Self::Error> {
            for i in 0..working.len() {
                // z_i ~ N(F_i, 1) truncated to (0,∞) if labels[i], else (−∞,0].
                let z = loop {
                    let draw: f64 =
                        rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
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

    /// Binary-AddiVortes skeleton (binding test): probit via the
    /// augmentation `ResponseModel` + pinned σ², running through the REAL
    /// sampler with zero sampler edits.
    #[test]
    fn binding_probit_skeleton_runs_through_the_unmodified_sampler() {
        let (x, y) = seam_data();
        let labels: Vec<bool> = y.iter().map(|v| *v > 1.0).collect();
        let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 4);
        let mut sampler = crate::Sampler::with_cell_model(
            seam_config(23),
            &x,
            &y, // the scaled response is immediately replaced by the latents
            crate::extensions::moves::MoveSetBuilder::stone_gosling()
                .build()
                .unwrap(),
            GaussianCellModel::new(sigma_mu_sq).unwrap(),
        )
        .unwrap()
        .with_response_model(AlbertChibProbit { labels })
        .with_scale_model(PinnedSigma::unit());
        for _ in 0..5 {
            let draw = sampler.step().unwrap();
            assert_eq!(draw.sigma_sq.to_bits(), 1.0_f64.to_bits());
        }
    }

    /// H-AddiVortes skeleton (binding test): a per-observation
    /// variance model entering as precision weights on the stats (paper
    /// Eq. 6) with σ² ≡ 1: the weighted Gaussian cell model + a
    /// weight-producing `ResponseModel`, zero sampler edits. (The real extension
    /// replaces the fixed s² with its own multiplicative ensemble updated in
    /// `ScaleModel::update`.)
    #[derive(Debug)]
    struct FixedPrecisionWeights {
        s_sq: Vec<f64>,
    }

    impl ResponseModel for FixedPrecisionWeights {
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
            for (weight, s_sq) in weights.iter_mut().zip(&self.s_sq) {
                *weight = 1.0 / s_sq;
            }
            Ok(())
        }
    }

    #[test]
    fn binding_heteroscedastic_skeleton_runs_through_the_unmodified_sampler() {
        let (x, y) = seam_data();
        let n = y.len();
        let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 4);
        let s_sq: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * i as f64).collect();
        let mut sampler = crate::Sampler::with_cell_model(
            seam_config(29),
            &x,
            &y,
            crate::extensions::moves::MoveSetBuilder::stone_gosling()
                .build()
                .unwrap(),
            WeightedGaussianModel::new(sigma_mu_sq).unwrap(),
        )
        .unwrap()
        .with_response_model(FixedPrecisionWeights { s_sq })
        .with_scale_model(PinnedSigma::unit());
        for _ in 0..5 {
            let draw = sampler.step().unwrap();
            assert!(draw.tessellations.iter().all(|t| t.n_cells() >= 1));
        }
    }

    /// The linear family's q = 1 intercept-only configuration runs through
    /// the UNMODIFIED sampler ("runs with zero engine edits"): the
    /// q = 1 case rides the scalar-payload path. Same-seed chains are
    /// reproducible and finite throughout.
    #[test]
    fn linear_q1_cells_run_through_the_unmodified_sampler() {
        let (x, y) = seam_data();
        let sigma_mu_sq = crate::engine::scaler::sigma_mu_sq(3.0, 4);
        let run = || {
            crate::Sampler::with_cell_model(
                seam_config(41),
                &x,
                &y,
                crate::extensions::moves::MoveSetBuilder::stone_gosling()
                    .build()
                    .unwrap(),
                LinearGaussianModel::new(sigma_mu_sq, 1).unwrap(),
            )
            .unwrap()
        };
        let bits_a = chain_bits(run(), 8);
        let bits_b = chain_bits(run(), 8);
        assert_eq!(bits_a, bits_b);
    }

    /// The extension-error channel: a deliberately-failing `CellModel`
    /// surfaces as `AddiVortesError::Extension` mid-chain.
    #[test]
    fn failing_cell_model_surfaces_extension_error() {
        #[derive(Debug, thiserror::Error)]
        #[error("deliberate cell-model failure")]
        struct DeliberateFailure;

        #[derive(Debug)]
        struct FailingModel;
        impl CellModel for FailingModel {
            type Stats = GaussianCellStats;
            type Error = DeliberateFailure;
            fn log_marginal_terms(
                &self,
                _stats: &[Self::Stats],
                _sigma_sq: f64,
            ) -> std::result::Result<f64, Self::Error> {
                Err(DeliberateFailure)
            }
            fn draw_cell_values(
                &self,
                _stats: &[Self::Stats],
                _sigma_sq: f64,
                _rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<Vec<f64>, Self::Error> {
                Err(DeliberateFailure)
            }
        }

        let (x, y) = seam_data();
        let mut sampler = crate::Sampler::with_cell_model(
            seam_config(31),
            &x,
            &y,
            crate::extensions::moves::MoveSetBuilder::stone_gosling()
                .build()
                .unwrap(),
            FailingModel,
        )
        .unwrap();
        let err = sampler.step().unwrap_err();
        assert!(matches!(err, AddiVortesError::Extension { .. }));
    }

    #[test]
    fn stats_algebra_round_trips() {
        let mut a = GaussianCellStats::default();
        a.record(1.5, 1.0);
        a.record(-0.5, 1.0);
        let mut b = GaussianCellStats::default();
        b.record(2.0, 1.0);
        let mut merged = a.clone();
        merged.merge(&b);
        assert_eq!(merged.count(), 3.0);
        assert_eq!(merged.sum(), 3.0);
        merged.remove(&b);
        assert_eq!(merged.count(), a.count());
        assert_eq!(merged.sum().to_bits(), a.sum().to_bits());
        merged.reset();
        assert!(!merged.occupied());
        assert!(a.occupied());
    }
}
