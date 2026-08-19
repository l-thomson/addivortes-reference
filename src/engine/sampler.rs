//! The MCMC sampler (`Sampler`): the backfitting Gauss–Seidel kernel and the
//! owned RNG.
//!
//! ## Pinned RNG-consumption order (part of the reproducibility contract)
//!
//! Chain initialisation consumes no RNG (the initial single-cell
//! tessellations are deterministic). Each sweep then consumes, in order:
//!
//! 1. the `ResponseModel::augment` hook (deep seam): absent on the
//!    default path (no copies, no RNG);
//! 2. the scale draw: one Gamma sample for the built-in global σ²;
//! 3. the per-sweep `InclusionModel::update` (zero draws for the built-in
//!    models, the no-op guarantee, the inclusion point);
//! 4. per tessellation j (ascending): the move-selection uniform, the move's
//!    own proposal draws, then (only if the empty-cell guard passes) the
//!    acceptance uniform, and finally the cell-value draws (one Normal per
//!    cell for the built-in Gaussian `CellModel`).

use std::sync::Arc;

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

use crate::Metric;
use crate::engine::backfit::{Composition, EnsembleUnit};
use crate::engine::builder::Components;
use crate::engine::column::semantics_for;
use crate::engine::config::AddiVortesConfig;
use crate::engine::data::{self, Data, Warning};
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::scaler::{self, FittedScaler};
use crate::engine::tessellation::Tessellation;
use crate::extensions::basis::CellBasis;
use crate::extensions::cell_model::CellModel;
use crate::extensions::coord::CoordinateDistribution;
use crate::extensions::count_priors::CountPriors;
use crate::extensions::distance::{AssignmentCache, CellAssigner};
use crate::extensions::erasure::{
    ErasedCellKernel, ErasedResponseModel, ErasedScaleModel, KernelOf,
};
use crate::extensions::inclusion::{ErasedInclusionModel, InclusionUsage, InvalidInclusionWeights};
use crate::extensions::membership::MembershipKernel;
use crate::extensions::moves::{ModelCtx, MoveSet};
use crate::extensions::response::ResponseModel;
use crate::extensions::scale::{InvalidScalePrecisions, ScaleCtx, ScaleModel};

// ---------------------------------------------------------------------------
// Seed expansion (determinism infrastructure, pinned)
// ---------------------------------------------------------------------------

/// One step of the splitmix64 generator (Steele, Lea & Flood 2014; Vigna's
/// reference implementation). Advances `state` in place and returns the next
/// output.
///
/// The constants, the two xor-shift/multiply rounds, and the finalising shift are
/// pinned: changing any of them changes every chain the crate ever samples, so
/// they are frozen by the `expand_seed` key-bytes test below.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Expand the user's mandatory `u64` seed into the 32-byte ChaCha8 key:
/// four consecutive splitmix64 outputs, each packed little-endian.
///
/// This is deliberately not `SeedableRng::seed_from_u64`: that helper's
/// expansion is an implementation detail of `rand_core` and could change under a
/// dependency bump. This expansion is ours and is frozen by test vectors.
pub(crate) fn expand_seed(seed: u64) -> [u8; 32] {
    let mut state = seed;
    let mut key = [0u8; 32];
    for chunk in key.chunks_exact_mut(8) {
        chunk.copy_from_slice(&splitmix64(&mut state).to_le_bytes());
    }
    key
}

// ---------------------------------------------------------------------------
// Draws
// ---------------------------------------------------------------------------

/// One posterior draw, borrowing the sampler's state (passive record).
/// All values are in scaled space.
#[derive(Debug, Clone, PartialEq)]
pub struct Draw<'a> {
    /// Error variance σ² of this sweep (scaled space).
    pub sigma_sq: f64,
    /// The m tessellations after this sweep (scaled space).
    pub tessellations: &'a [Tessellation],
}

/// An owned posterior draw (the iterator item). All values are in scaled
/// space.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedDraw {
    /// Error variance σ² of this sweep (scaled space).
    pub sigma_sq: f64,
    /// The m tessellations after this sweep (scaled space).
    pub tessellations: Vec<Tessellation>,
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// The raw Gibbs/backfitting sampler: yields every sweep; burn-in and thinning
/// are `fit()`'s concern. Owns the ChaCha8 RNG built from the
/// mandatory seed via the pinned splitmix64 expansion (`expand_seed`).
#[derive(Debug)]
pub struct Sampler {
    rng: ChaCha8Rng,
    /// Scaled, encoded design matrix.
    x: Data,
    /// The working response (scaled space): the scaled caller response on the
    /// default path; rewritten once per sweep by a configured [`ResponseModel`]
    /// (deep seam).
    y: Vec<f64>,
    scaler: FittedScaler,
    warnings: Vec<Warning>,
    config: AddiVortesConfig,

    /// The mean ensemble: tessellations + cached assignments + the cell
    /// kernel (deep seam; default: Gaussian conjugate, batch-erased,
    /// bit-identical to the pre-seam kernel), composed additively.
    ensemble: EnsembleUnit,
    /// Running total fit F_i = Σ_j μ_{j, cell_j(i)} (scaled space).
    fit: Vec<f64>,

    /// Cell-value prior variance σ_μ² (scaled space; ModelCtx input).
    sigma_mu_sq: f64,
    /// Calibrated λ of the σ² ~ χ⁻²(ν, λ) prior (scaled space;
    /// exposed to scale models through the ScaleCtx).
    lambda: f64,

    // The rest of the deep seam: batch-erased, defaults
    // bit-identical to the pre-seam kernel.
    /// The scale draw (default: the global σ² Gibbs draw).
    scale: Box<dyn ErasedScaleModel>,
    /// The augmentation hook; `None` = run directly on the scaled response
    /// (zero copies, zero RNG; the default chain is seam-independent).
    response_model: Option<Box<dyn ErasedResponseModel>>,
    /// The caller's scaled response, retained only when a `ResponseModel` is
    /// configured (it reads y while writing the working response).
    y_observed: Option<Vec<f64>>,
    /// Per-observation accumulation weights, allocated only with a `ResponseModel`.
    obs_weights: Option<Vec<f64>>,
    /// Scratch for the conductor's weight composition (`wᵢ = wᵢ^resp ·
    /// wᵢ^scale`), allocated only when a `ResponseModel` and a
    /// precision-supplying `ScaleModel` are both present.
    composed_weights: Vec<f64>,

    p_enc: usize,
    coord_dists: Vec<Arc<dyn CoordinateDistribution>>,
    inclusion: Box<dyn ErasedInclusionModel>,
    /// Inclusion weights expanded to encoded columns (rebuilt after `update`).
    weights_enc: Vec<f64>,
    assigner: Arc<dyn CellAssigner>,
    move_set: Arc<MoveSet>,
    /// Soft-membership kernel; retained for the fitted model's predict path.
    membership: Option<Arc<dyn MembershipKernel>>,
    /// Cell basis; retained for the fitted model's predict path.
    basis: Option<Arc<dyn CellBasis>>,
    /// Count priors; `None` prices through the paper pair.
    count_priors: Option<Arc<dyn CountPriors>>,
    /// True when any component axis was set explicitly.
    custom_components: bool,
}

impl Sampler {
    /// A sampler over raw caller data with the default (paper) components.
    /// Component wiring is crate-internal: `SamplerBuilder`.
    pub fn new(config: AddiVortesConfig, x: &Data, y: &[f64]) -> Result<Self> {
        Self::with_components(config, Components::default(), x, y)
    }

    /// A sampler with a caller-supplied move set (crate-internal;
    /// overrides any set on the components).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_move_set(
        config: AddiVortesConfig,
        x: &Data,
        y: &[f64],
        move_set: MoveSet,
    ) -> Result<Self> {
        Self::with_components(
            config,
            Components {
                // Single-threaded sharing; Send + Sync is never required
                // of move sets.
                #[allow(clippy::arc_with_non_send_sync)]
                move_set: Some(Arc::new(move_set)),
                ..Components::default()
            },
            x,
            y,
        )
    }

    /// The single construction path: plain config + component wiring.
    pub(crate) fn with_components(
        config: AddiVortesConfig,
        components: Components,
        x: &Data,
        y: &[f64],
    ) -> Result<Self> {
        // The move set resolves first; everything else in `assemble`.
        #[allow(clippy::arc_with_non_send_sync)]
        let move_set = match &components.move_set {
            Some(move_set) => Arc::clone(move_set),
            None => Arc::new(crate::extensions::moves::default_move_set()?),
        };
        Self::checked(config, components, x, y, move_set)
    }

    fn checked(
        config: AddiVortesConfig,
        components: Components,
        x: &Data,
        y: &[f64],
        move_set: Arc<MoveSet>,
    ) -> Result<Self> {
        // Boundary validation on raw input, in the canonical order.
        let metrics: Vec<Metric> = config
            .metrics
            .clone()
            .unwrap_or_else(|| vec![Metric::Euclidean; x.n_cols()]);
        data::validate_fit(x, y, &metrics, config.omega)?;
        // The BinaryProbit family is a {0, 1}-label model: any
        // other response value is a configuration error at the fit boundary.
        if config.family == crate::engine::model::ResponseFamily::BinaryProbit {
            if let Some(bad) = y.iter().find(|v| **v != 0.0 && **v != 1.0) {
                return Err(AddiVortesError::InvalidHyperparameter {
                    name: "response_family".into(),
                    reason: format!("BinaryProbit needs a {{0, 1}} response; found {bad}"),
                });
            }
        }
        // The RobustT family's one parameter is validated at the same
        // boundary: ν′ must be a usable degrees-of-freedom count.
        if let crate::engine::model::ResponseFamily::RobustT { df } = config.family {
            if !(df.is_finite() && df > 0.0) {
                return Err(AddiVortesError::InvalidHyperparameter {
                    name: "response_family".into(),
                    reason: format!("RobustT needs finite df > 0; found {df}"),
                });
            }
        }
        let warnings = data::fit_warnings(x);

        // Scaling + encoding; everything below is in scaled space.
        let (scaler, x_enc, y_scaled) = FittedScaler::fit(x, y, &metrics)?;

        // One-time prior calibration. A response that is an exact linear
        // function of the features has σ̂ = 0 and so λ = 0; the global σ²
        // draw stays well-defined under that prior (σ² = RSS/χ²), so the fit
        // proceeds. A scale model that cannot accept λ = 0 (the H variance
        // calibration, whose λ′ = λ^(1/m′) would pin every cell at 0)
        // refuses with `DegenerateResidual` when it reads
        // `ScaleCtx::calibrated_lambda`.
        let sigma_hat = scaler::sigma_hat(&x_enc, &y_scaled);
        let lambda = scaler::calibrate_lambda(config.nu, config.q, sigma_hat);

        Self::assemble(
            config, components, x_enc, y_scaled, scaler, warnings, lambda, move_set,
        )
    }

    /// Assemble a sampler over already-scaled, already-encoded state with a
    /// caller-supplied λ. The single constructor body shared by the public
    /// boundary path above and the pinned-prior statistical-gate path (the
    /// SBC/Geweke batteries pin λ instead of calibrating it from data: exact
    /// SBC requires the generating prior and the fitted prior to be the same
    /// distribution, which a data-dependent λ is not).
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        config: AddiVortesConfig,
        components: Components,
        x_enc: Data,
        y_scaled: Vec<f64>,
        scaler: FittedScaler,
        warnings: Vec<Warning>,
        lambda: f64,
        move_set: Arc<MoveSet>,
    ) -> Result<Self> {
        let p_enc = x_enc.n_cols();
        let n = x_enc.n_rows();
        // Recorded before wiring: a fit with any explicit component refuses
        // to serialise (trait objects have no portable form).
        let custom_components = components.any_custom();
        // The cell-prior width dial: a directly-set σ_μ wins over the k-rule
        // (and, below, over the family rule); unset, both expressions are
        // exactly the pre-dial ones, so the default chain is bit-identical.
        let sigma_mu_sq = match config.cell_prior_sd {
            Some(sigma_mu) => sigma_mu * sigma_mu,
            None => scaler::sigma_mu_sq(config.k, config.m),
        };
        // (For BinaryProbit the latent-scale σ_μ² below overrides what the
        // ModelCtx carries; the built-in moves never read it.)

        // Per-encoded-column coordinate distributions: the
        // config's per-raw-column laws expanded alongside one-hot encoding
        // (a group shares one law), or the defaults per metric.
        let coord_dists: Vec<Arc<dyn CoordinateDistribution>> = match &components.coords {
            Some(coords) => {
                if coords.len() != scaler.n_raw_cols() {
                    return Err(AddiVortesError::InvalidHyperparameter {
                        name: "coords".into(),
                        reason: format!(
                            "{} coordinate laws for {} caller-visible columns",
                            coords.len(),
                            scaler.n_raw_cols()
                        ),
                    });
                }
                scaler
                    .col_map()
                    .iter()
                    .map(|&raw| Arc::clone(&coords[raw]))
                    .collect()
            }
            None => scaler
                .metrics()
                .iter()
                .map(|metric| semantics_for(metric).default_coord_law(config.sigma_c))
                .collect::<Result<_>>()?,
        };

        // Inclusion model: the config's, or the point's default over
        // raw columns (the extension point's module owns which concrete model that is).
        let inclusion: Box<dyn ErasedInclusionModel> = match &components.inclusion {
            Some(model) => model.clone_erased(),
            None => crate::extensions::inclusion::default_erased(scaler.n_raw_cols()),
        };
        // A wrong-length weight vector at the fit boundary is a configuration
        // error (InvalidHyperparameter); only mid-chain weight corruption from
        // a custom update is the Extension channel.
        if inclusion.weights().len() != scaler.n_raw_cols() {
            return Err(AddiVortesError::InvalidHyperparameter {
                name: "inclusion_weights".into(),
                reason: format!(
                    "{} weights for {} caller-visible columns",
                    inclusion.weights().len(),
                    scaler.n_raw_cols()
                ),
            });
        }
        // The weight values are a boundary check too (the plain config no
        // longer sees the model, so this is the first place they exist).
        if let Some(w) = inclusion
            .weights()
            .iter()
            .find(|w| !w.is_finite() || **w <= 0.0)
        {
            return Err(AddiVortesError::InvalidHyperparameter {
                name: "inclusion_weights".into(),
                reason: format!("every weight must be finite and positive, got {w}"),
            });
        }
        let weights_enc =
            expand_weights(inclusion.weights(), scaler.col_map(), scaler.n_raw_cols())?;

        // Assigner: the config's, or the point's default over the
        // encoded metrics (the extension point's module owns which concrete assigner that is).
        let assigner: Arc<dyn CellAssigner> = match &components.assigner {
            Some(assigner) => Arc::clone(assigner),
            None => crate::extensions::distance::default_assigner(scaler.metrics().to_vec()),
        };

        // Chain initialisation (paper Algorithm 1): m single-centre,
        // one-dimensional tessellations, each single cell's μ = mean(y_scaled).
        // A single-cell tessellation's output is independent of its centre
        // coordinate and active dimension, so both are pinned deterministically
        // (dimension 0, coordinate 0.0) and consume no RNG, required for the
        // bit-exact golden chain.
        let mean_y = y_scaled.iter().sum::<f64>() / n as f64;
        let init = Tessellation {
            centres: vec![0.0],
            dims: vec![0],
            mus: vec![mean_y],
        };
        let tessellations = vec![init; config.m];
        // Single-centre tessellations assign every observation to cell 0. The
        // key cache starts cold (empty): the first reassign per tessellation
        // falls back to a full recompute, which seeds it.
        let assignments = vec![AssignmentCache::new(vec![0usize; n], Vec::new()); config.m];
        let fit = vec![mean_y * config.m as f64; n];

        let rng = ChaCha8Rng::from_seed(expand_seed(config.seed));
        // The deep seam: the config's choices, or defaults that are
        // bit-identical to the pre-seam kernel. The Sampler-level `with_*`
        // entry points still override after construction (last wins).
        // The response family's fit-side defaults: BinaryProbit
        // attaches the Albert–Chib augmentation and the pinned unit scale,
        // and widens the cell-value prior to the ±3 latent range
        // (σ_μ = 3/(k√m), the Binary paper's shape); explicit seam choices
        // override per component, as everywhere.
        let family = config.family;
        let family_sigma_mu_sq = match family {
            crate::engine::model::ResponseFamily::Gaussian
            | crate::engine::model::ResponseFamily::RobustT { .. } => sigma_mu_sq,
            // A directly-set σ_μ wins over the family rule too: the dial is
            // the model file saying what the prior width is.
            crate::engine::model::ResponseFamily::BinaryProbit
                if config.cell_prior_sd.is_some() =>
            {
                sigma_mu_sq
            }
            crate::engine::model::ResponseFamily::BinaryProbit => {
                let sigma_mu = 3.0 / (config.k * (config.m as f64).sqrt());
                sigma_mu * sigma_mu
            }
        };
        // RobustT's augmentation weights need the weight-aware mean family
        // and the precision-weighted σ² draw (the deep-seam pairing rules,
        // assembled here so the caller never has to know them).
        //
        // A heteroscedastic scale model needs the same pairing for the
        // same reason: its per-observation precisions make the accumulation
        // weights fractional, which the hard-assignment `GaussianCellStats`
        // refuses. Without this arm, `with_scale_model(HVariance::new(m)?)` on
        // its own panicked partway through the first sweep — the crate's own H
        // sampler paired the weighted family by hand, and nothing said a caller
        // had to.
        let heteroscedastic_scale = components
            .scale_model
            .as_ref()
            .is_some_and(|factory| factory.heteroscedastic());
        let cell_kernel: Box<dyn ErasedCellKernel> = match (&components.cell_model, family) {
            (Some(factory), _) => factory.kernel(),
            (None, crate::engine::model::ResponseFamily::RobustT { .. }) => {
                crate::extensions::erasure::weighted_cell_kernel(family_sigma_mu_sq)?
            }
            (None, _) if heteroscedastic_scale => {
                crate::extensions::erasure::weighted_cell_kernel(family_sigma_mu_sq)?
            }
            (None, _) => crate::extensions::erasure::default_cell_kernel(family_sigma_mu_sq)?,
        };
        // The cell basis. A basis payload and a basis must arrive
        // together and agree on q; a scalar payload takes no basis. Checked
        // here, where the payload is first known, so the failure is a clean
        // error at `fit` rather than a mis-shaped statistic later.
        let basis_rows = match (&components.basis, cell_kernel.cell_basis()) {
            (Some(basis), true) => {
                let q = cell_kernel.payload_width();
                if basis.q() != q {
                    return Err(AddiVortesError::InvalidHyperparameter {
                        name: "cell_basis".into(),
                        reason: format!(
                            "basis dimension q = {} does not match the cell model's payload width {q}",
                            basis.q()
                        ),
                    });
                }
                if components.membership.is_some() {
                    return Err(AddiVortesError::InvalidHyperparameter {
                        name: "cell_basis".into(),
                        reason: "a basis payload and soft membership do not \
                                 compose: soft membership owns its own joint payload draw"
                            .into(),
                    });
                }
                // Bounds-check the basis against the encoded design once, so an
                // out-of-range column is an error here and not a panic in the
                // hot loop.
                let p_enc = x_enc.n_cols();
                if basis.max_column().is_some_and(|max| max >= p_enc) {
                    let max = basis.max_column().unwrap_or_default();
                    return Err(AddiVortesError::InvalidHyperparameter {
                        name: "cell_basis".into(),
                        reason: format!(
                            "basis column {max} is out of range for the {p_enc}-column \
                             encoded design"
                        ),
                    });
                }
                // z(xᵢ) over the scaled design, once: the basis is a fixed set
                // of columns, so it never moves with the tessellation.
                let mut values = vec![0.0_f64; n * q];
                for i in 0..n {
                    basis.row(x_enc.row(i), &mut values[i * q..(i + 1) * q]);
                }
                Some(crate::extensions::erasure::BasisRows { values, q })
            }
            (Some(_), false) => {
                return Err(AddiVortesError::InvalidHyperparameter {
                    name: "cell_basis".into(),
                    reason: "a cell basis needs a basis payload: this cell model holds a scalar \
                             per cell (see `basis::LinearGaussianModel`)"
                        .into(),
                });
            }
            (None, true) => {
                return Err(AddiVortesError::InvalidHyperparameter {
                    name: "cell_model".into(),
                    reason: "a basis payload needs a cell basis: set one with \
                             `with_cell_basis` (see `basis::LinearBasis`)"
                        .into(),
                });
            }
            (None, false) => None,
        };
        // A basis cell carries q coefficients, so the initial single-cell
        // payload is q wide. β = [mean_y, 0, …]: with the intercept first in
        // every shelf basis, z·β = mean_y, so the initial fit is unchanged.
        let tessellations = match &basis_rows {
            None => tessellations,
            Some(basis) => {
                let mut payload = vec![0.0_f64; basis.q];
                payload[0] = mean_y;
                vec![
                    Tessellation {
                        centres: vec![0.0],
                        dims: vec![0],
                        mus: payload,
                    };
                    config.m
                ]
            }
        };

        let scale_model: Box<dyn ErasedScaleModel> = match (&components.scale_model, family) {
            (Some(factory), _) => factory.scale(),
            (None, crate::engine::model::ResponseFamily::Gaussian) => {
                crate::extensions::erasure::default_scale_model(config.nu, lambda)?
            }
            (None, crate::engine::model::ResponseFamily::BinaryProbit) => {
                Box::new(crate::extensions::scale::PinnedSigma::unit())
            }
            (None, crate::engine::model::ResponseFamily::RobustT { .. }) => Box::new(
                crate::extensions::scale::WeightedGlobalSigma::new(config.nu, lambda)?,
            ),
        };
        let response_model: Option<Box<dyn ErasedResponseModel>> =
            match (&components.response_model, family) {
                (Some(factory), _) => Some(factory.step()),
                (None, crate::engine::model::ResponseFamily::Gaussian) => None,
                (None, crate::engine::model::ResponseFamily::BinaryProbit) => {
                    Box::new(crate::extensions::response::AlbertChibProbit)
                        as Box<dyn ErasedResponseModel>
                }
                .into(),
                (None, crate::engine::model::ResponseFamily::RobustT { df }) => {
                    Box::new(crate::extensions::response::RobustTStep::new(df)?)
                        as Box<dyn ErasedResponseModel>
                }
                .into(),
            };
        // A kernel step reads the observed response while writing the working
        // one (see `with_response_model`).
        let (y_observed, obs_weights) = if response_model.is_some() {
            (Some(y_scaled.clone()), Some(vec![1.0; n]))
        } else {
            (None, None)
        };

        // The path selection: soft membership switches the mean
        // ensemble onto the dense path. Initial memberships are computed
        // through the real key/kernel path (an assigner without dense keys
        // errors here, at construction). Hard membership keeps the
        // golden-pinned diagonal path, with no runtime branch on the default.
        let ensemble = match &components.membership {
            None => {
                let unit = EnsembleUnit::new(
                    tessellations,
                    assignments,
                    cell_kernel,
                    Composition::Additive,
                );
                match basis_rows {
                    Some(basis) => unit.with_basis(basis),
                    None => unit,
                }
            }
            Some(kernel) => {
                let memberships: Vec<Vec<f64>> = tessellations
                    .iter()
                    .map(|tessellation| {
                        crate::engine::backfit::compute_memberships(
                            assigner.as_ref(),
                            kernel.as_ref(),
                            &x_enc,
                            tessellation,
                        )
                    })
                    .collect::<Result<_>>()?;
                EnsembleUnit::new_dense(
                    tessellations,
                    assignments,
                    cell_kernel,
                    Arc::clone(kernel),
                    memberships,
                )
            }
        };

        Ok(Self {
            rng,
            x: x_enc,
            y: y_scaled,
            scaler,
            warnings,
            config,
            ensemble,
            fit,
            sigma_mu_sq,
            lambda,
            scale: scale_model,
            response_model,
            y_observed,
            obs_weights,
            composed_weights: Vec::new(),
            p_enc,
            coord_dists,
            inclusion,
            weights_enc,
            assigner,
            move_set,
            membership: components.membership,
            basis: components.basis,
            count_priors: components.count_priors,
            custom_components,
        })
    }

    /// A sampler with a caller-supplied [`CellModel`] (deep seam). The
    /// model owns its own prior parameters (the built-in Gaussian's
    /// σ_μ² = (0.5/(k√m))² is available as
    /// [`GaussianCellModel::new`](crate::extensions::cell_model::GaussianCellModel)).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_cell_model<M: CellModel + 'static>(
        config: AddiVortesConfig,
        x: &Data,
        y: &[f64],
        move_set: MoveSet,
        model: M,
    ) -> Result<Self> {
        let mut sampler = Self::with_move_set(config, x, y, move_set)?;
        sampler.ensemble.kernel = Box::new(KernelOf(model));
        Ok(sampler)
    }

    /// Attach a data-augmentation [`ResponseModel`] (deep seam): consuming
    /// setter, call before the first sweep. Allocates the working-response
    /// and weight buffers; the step runs first in every sweep's pinned hook
    /// order (augment → scale → inclusion → j-loop).
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_response_model<K: ResponseModel + 'static>(mut self, step: K) -> Self {
        let n = self.x.n_rows();
        self.y_observed = Some(self.y.clone());
        self.obs_weights = Some(vec![1.0; n]);
        self.response_model = Some(Box::new(step));
        self
    }

    /// Replace the scale draw with a caller-supplied [`ScaleModel`]
    /// (deep seam): consuming setter, call before the first sweep.
    #[must_use]
    pub(crate) fn with_scale_model<S: ScaleModel + 'static>(mut self, scale: S) -> Self {
        self.scale = Box::new(scale);
        self
    }

    /// One full Gibbs sweep (augment → scale/σ² → inclusion update →
    /// sequential j-loop, the pinned hook order), returning this sweep's
    /// draw. Mid-chain errors are `NonFiniteDistance` (custom assigner) or
    /// `Extension` (custom inclusion/cell/scale/augmentation models); the
    /// standard path raises neither.
    pub fn step(&mut self) -> Result<Draw<'_>> {
        // --- 0. augmentation hook (deep seam; None = zero copies/RNG) ---
        if let Some(step) = &mut self.response_model {
            let y_observed = self
                .y_observed
                .as_ref()
                .expect("y_observed is set with the kernel step");
            let weights = self
                .obs_weights
                .as_mut()
                .expect("obs_weights is set with the kernel step");
            step.augment(
                y_observed,
                &self.fit,
                self.scale.sigma_sq(),
                &mut self.rng,
                &mut self.y,
                weights,
            )?;
        }

        // --- 1. scale update (the global σ² Gibbs draw by default). The
        // context carries the scaled design and the shared backfit machinery
        // so a variance ensemble is an ordinary supplier;
        // the inclusion weights are the previous sweep's (the pinned hook
        // order runs scale before inclusion).
        let scale_ctx = ScaleCtx {
            y: &self.y,
            fit: &self.fit,
            x: &self.x,
            move_set: self.move_set.as_ref(),
            assigner: self.assigner.as_ref(),
            omega: self.config.omega,
            lambda_c: self.config.lambda_c,
            nu: self.config.nu,
            lambda: self.lambda,
            p_enc: self.p_enc,
            coord_dists: &self.coord_dists,
            weights_enc: &self.weights_enc,
            response_weights: self.obs_weights.as_deref(),
        };
        self.scale.update(&scale_ctx, &mut self.rng)?;
        // Release-mode posterior-critical invariant.
        let sigma_sq = self.scale.sigma_sq();
        assert!(
            sigma_sq.is_finite() && sigma_sq > 0.0,
            "sigma_sq must be finite and positive after every draw"
        );
        // Mid-chain precision validation (the expand_weights
        // precedent: custom-model corruption is the Extension channel).
        if let Some(precisions) = self.scale.precisions() {
            validate_precisions(precisions, self.x.n_rows())?;
        }

        // --- 2. per-sweep inclusion update at the pinned point ---
        let mut usage = InclusionUsage::new(self.scaler.n_raw_cols());
        let col_map = self.scaler.col_map().to_vec();
        for tessellation in &self.ensemble.tessellations {
            usage.record_subset_size(tessellation.dims().len());
            for &dim in tessellation.dims() {
                usage.record(col_map[dim]);
            }
        }
        self.inclusion.update(&usage, &mut self.rng)?;
        self.weights_enc =
            expand_weights(self.inclusion.weights(), &col_map, self.scaler.n_raw_cols())?;

        // --- 3. sequential j-loop (never parallelise): the backfit block
        // per tessellation. σ², the weights and the
        // coordinate laws are constant across the loop, so one context serves
        // every j.
        let ctx = ModelCtx::new(
            self.scale.sigma_sq(),
            self.config.omega,
            self.config.lambda_c,
            self.sigma_mu_sq,
            self.p_enc,
            &self.coord_dists,
            &self.weights_enc,
        );
        // The count-prior point: a selected count prior replaces the paper's pair for every
        // move that changes a count. Left unset, the context keeps its own
        // ShiftedPoissonBinomial, which prices identically to the pre-hook
        // kernel: the default chain is provably untouched by this seam.
        let ctx = match &self.count_priors {
            Some(priors) => ctx.with_count_priors(priors.as_ref()),
            None => ctx,
        };
        // Conductor composition: the per-observation
        // precision entering the conjugate accumulation is the product
        // wᵢ = wᵢ^response · wᵢ^scale of the augmentation weights and the
        // scale model's precisions. The default path (neither present) stays
        // None: zero copies, bit-identical to the pre-rebuild conductor.
        let obs_weights: Option<&[f64]> =
            match (self.obs_weights.as_deref(), self.scale.precisions()) {
                (None, None) => None,
                (Some(response), None) => Some(response),
                (None, Some(scale)) => Some(scale),
                (Some(response), Some(scale)) => {
                    self.composed_weights.clear();
                    self.composed_weights
                        .extend(response.iter().zip(scale).map(|(r, s)| r * s));
                    Some(&self.composed_weights)
                }
            };
        for j in 0..self.ensemble.tessellations.len() {
            self.ensemble.backfit_one(
                j,
                &self.x,
                &self.y,
                &mut self.fit,
                obs_weights,
                &ctx,
                &self.move_set,
                self.assigner.as_ref(),
                &mut self.rng,
            )?;
        }

        Ok(Draw {
            sigma_sq: self.scale.sigma_sq(),
            tessellations: &self.ensemble.tessellations,
        })
    }

    /// Replace the response for subsequent sweeps: the embed entry.
    /// Together with [`step`](Sampler::step) this lets a
    /// researcher run the engine as one conditional inside their own outer
    /// Gibbs sampler, keeping novel blocks (cutpoints, censoring, propensity
    /// updates) entirely in their crate. Worked, runnable example:
    /// `examples/template_embed.rs`.
    ///
    /// **Scale contract:** `y` is on the caller's response scale (the
    /// same scale as the response handed to [`Sampler::new`]) and is mapped
    /// into the sampler's internal scaled space through the affine transform
    /// frozen at construction (an outer-Gibbs author holds the latent/working
    /// response, never the internal scaled space). Values outside the
    /// construction-time response range are legitimate (latents routinely
    /// are): they scale affinely, never clamped. The σ² prior calibration
    /// (σ̂, λ) was fitted against the construction-time response and is
    /// deliberately not recomputed here.
    ///
    /// With a configured augmentation step this replaces the observed
    /// response that `augment` reads each sweep; otherwise it replaces the
    /// working response directly.
    ///
    /// The caller owns its RNG: whatever randomness
    /// produced `y` must come from the caller's own generator. This call
    /// consumes no RNG and never touches the sampler's pinned stream, so
    /// "engine seed + caller stream" still names exactly one chain.
    ///
    /// Errors: `RowCountMismatch` if `y`'s length differs from the training
    /// response; `NonFiniteResponse` on any NaN/±∞ (nothing non-finite is
    /// ever silently dropped or repaired).
    pub fn set_response(&mut self, y: &[f64]) -> Result<()> {
        if y.len() != self.y.len() {
            return Err(AddiVortesError::RowCountMismatch {
                y_len: y.len(),
                x_rows: self.y.len(),
            });
        }
        if let Some(row) = y.iter().position(|v| !v.is_finite()) {
            return Err(AddiVortesError::NonFiniteResponse { row });
        }
        let scaler = &self.scaler;
        let destination = match self.y_observed.as_mut() {
            Some(observed) => observed,
            None => &mut self.y,
        };
        for (slot, &value) in destination.iter_mut().zip(y) {
            *slot = scaler.scale_y_value(value);
        }
        Ok(())
    }

    /// The current ensemble fit at the training observations, on the
    /// caller's response scale (raw space, the inverse of the frozen
    /// response transform), ascending row index. The embed entry's read half:
    /// an outer-Gibbs conditional typically centres its latent draws on the
    /// engine's current fit. Consumes no RNG.
    pub fn fitted_values(&self) -> Vec<f64> {
        self.fit
            .iter()
            .map(|&f| self.scaler.unscale_y_value(f))
            .collect()
    }

    /// The current inclusion weights, one per caller-visible
    /// (pre-encoding) column (dimensionless relative weights: the state
    /// an adaptive model like `DartInclusion` carries between sweeps).
    /// Consumes no RNG.
    pub fn inclusion_weights(&self) -> &[f64] {
        self.inclusion.weights()
    }

    /// The scaling/encoding state frozen at construction (see
    /// [`FittedScaler`]). The embed entry's scale bridge:
    /// [`y_min`](FittedScaler::y_min)/[`y_max`](FittedScaler::y_max) give the
    /// affine response transform, so an outer-Gibbs author can move scale
    /// quantities between the scaled space of [`Draw`] and the caller's
    /// response scale (σ_caller = σ_scaled · (y_max − y_min)).
    pub fn scaler(&self) -> &FittedScaler {
        &self.scaler
    }

    /// Fit-time warnings (for the fitted-model assembly).
    #[allow(dead_code)] // superseded by into_fitted_parts; kept for diagnostics
    pub(crate) fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// The stored configuration (for the fitted-model assembly).
    pub(crate) fn config(&self) -> &AddiVortesConfig {
        &self.config
    }

    /// The pinned-prior battery constructor (`battery` module): a
    /// sampler over already-scaled, already-encoded `x_enc` (columns in
    /// the sampler's own coordinate system: Euclidean in [−0.5, 0.5],
    /// spherical in [−π, π]) and `y_scaled` (in [−0.5, 0.5]-scale space),
    /// with a pinned λ for the σ² ~ χ⁻²(ν, λ) prior. This bypasses the
    /// data-dependent σ̂ calibration and the response transform, both of
    /// which would make the generating prior differ from the fitted prior
    /// and so break exact SBC/Geweke (the batteries require the two to
    /// coincide). Everything downstream is the real sampler, unchanged: the
    /// scaler is the identity, so [`set_response`](Sampler::set_response)
    /// and [`fitted_values`](Sampler::fitted_values) speak scaled space
    /// directly, exactly what a successive-conditional simulator needs.
    ///
    /// The paper move set is `MoveSetBuilder::stone_gosling().build()`; all
    /// other components arrive explicitly.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn pinned_prior(
        config: AddiVortesConfig,
        components: Components,
        x_enc: Data,
        metrics_enc: Vec<Metric>,
        y_scaled: Vec<f64>,
        lambda: f64,
        move_set: MoveSet,
    ) -> Result<Self> {
        let scaler = FittedScaler::identity(x_enc.n_cols(), metrics_enc);
        Self::assemble(
            config,
            components,
            x_enc,
            y_scaled,
            scaler,
            Vec::new(),
            lambda,
            {
                // Single-threaded sharing; see `Sampler::new`.
                #[allow(clippy::arc_with_non_send_sync)]
                Arc::new(move_set)
            },
        )
    }

    /// The statistical gates' internal alias for [`Sampler::pinned_prior`].
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pinned_prior_for_tests(
        config: AddiVortesConfig,
        components: Components,
        x_enc: Data,
        metrics_enc: Vec<Metric>,
        y_scaled: Vec<f64>,
        lambda: f64,
        move_set: MoveSet,
    ) -> Result<Self> {
        Self::pinned_prior(
            config,
            components,
            x_enc,
            metrics_enc,
            y_scaled,
            lambda,
            move_set,
        )
    }

    /// Test-only (Geweke successive-conditional simulator): replace the scaled
    /// response in place. The cached fit vector depends on tessellations and X
    /// only, so it stays valid.
    #[cfg(test)]
    pub(crate) fn replace_scaled_response(&mut self, y: Vec<f64>) {
        debug_assert_eq!(y.len(), self.y.len());
        self.y = y;
    }

    /// Test-only: the running ensemble fit `F_i = Σ_j μ_{j, cell_j(i)}`
    /// (scaled space).
    #[cfg(test)]
    pub(crate) fn fit_values(&self) -> &[f64] {
        &self.fit
    }

    /// Test-only (Geweke): set the tessellation state to an externally-drawn
    /// value, recomputing assignments and the fit cache. Consumes no RNG. (σ²
    /// needs no injection: the next sweep's scale update redraws it from
    /// (y, fit) before anything reads it; the injected σ² enters the
    /// successive-conditional simulator only through the y the test draws.)
    #[cfg(test)]
    pub(crate) fn set_state_for_tests(&mut self, tessellations: Vec<Tessellation>) -> Result<()> {
        let n = self.x.n_rows();
        let mut assignments = Vec::with_capacity(tessellations.len());
        for tessellation in &tessellations {
            let assignment = self.assigner.assign_cells(&self.x, tessellation)?;
            assignments.push(AssignmentCache::new(assignment, Vec::new()));
        }
        let mut fit = vec![0.0_f64; n];
        match &mut self.ensemble.path {
            crate::engine::backfit::Path::Diagonal => {
                let q = self.ensemble.kernel.payload_width();
                let basis = self.ensemble.basis.as_ref();
                for (tessellation, cache) in tessellations.iter().zip(&assignments) {
                    let assignment = cache.assignment();
                    for i in 0..n {
                        fit[i] += crate::engine::backfit::cell_contribution(
                            &tessellation.mus,
                            q,
                            assignment[i],
                            basis,
                            i,
                        );
                    }
                }
            }
            crate::engine::backfit::Path::Dense(dense) => {
                // The dense path's state is the membership matrices; the fit
                // is the sum of per-tessellation dot products.
                let mut memberships = Vec::with_capacity(tessellations.len());
                for tessellation in &tessellations {
                    let membership = crate::engine::backfit::compute_memberships(
                        self.assigner.as_ref(),
                        dense.kernel.as_ref(),
                        &self.x,
                        tessellation,
                    )?;
                    let b = tessellation.n_cells();
                    for i in 0..n {
                        let phi = &membership[i * b..(i + 1) * b];
                        fit[i] += phi
                            .iter()
                            .zip(&tessellation.mus)
                            .map(|(p, mu)| p * mu)
                            .sum::<f64>();
                    }
                    memberships.push(membership);
                }
                dense.memberships = memberships;
            }
        }
        self.ensemble.tessellations = tessellations;
        self.ensemble.assignments = assignments;
        self.fit = fit;
        Ok(())
    }

    /// Test-only: the dense path's cached membership matrix for tessellation
    /// `j` (row-major n×b, scaled space): the reassign-consistency check
    /// compares it against a fresh recompute.
    #[cfg(test)]
    pub(crate) fn membership_for_tests(&self, j: usize) -> Option<&[f64]> {
        match &self.ensemble.path {
            crate::engine::backfit::Path::Diagonal => None,
            crate::engine::backfit::Path::Dense(dense) => Some(&dense.memberships[j]),
        }
    }

    /// Test-only: the current tessellations (scaled space).
    #[cfg(test)]
    pub(crate) fn tessellations_for_tests(&self) -> &[Tessellation] {
        &self.ensemble.tessellations
    }

    /// Test-only: the scaled, encoded design (the assigner's coordinate
    /// system).
    #[cfg(test)]
    pub(crate) fn design_for_tests(&self) -> &Data {
        &self.x
    }

    /// Tear down into the pieces the fitted model owns.
    pub(crate) fn into_fitted_parts(self) -> FittedParts {
        FittedParts {
            config: self.config,
            scaler: self.scaler,
            warnings: self.warnings,
            assigner: self.assigner,
            membership: self.membership,
            basis: self.basis,
            custom_components: self.custom_components,
        }
    }
}

/// What a finished sampler hands the fitted model (crate-internal).
pub(crate) struct FittedParts {
    pub(crate) config: AddiVortesConfig,
    pub(crate) scaler: FittedScaler,
    pub(crate) warnings: Vec<Warning>,
    pub(crate) assigner: Arc<dyn CellAssigner>,
    pub(crate) membership: Option<Arc<dyn MembershipKernel>>,
    pub(crate) basis: Option<Arc<dyn CellBasis>>,
    pub(crate) custom_components: bool,
}

/// Validate a custom scale model's per-observation precisions:
/// length n, every value finite and strictly positive. Mid-chain corruption
/// surfaces as the Extension channel, mirroring `expand_weights`.
fn validate_precisions(precisions: &[f64], n: usize) -> Result<()> {
    if precisions.len() != n {
        return Err(AddiVortesError::Extension {
            source: Arc::new(InvalidScalePrecisions {
                detail: format!("{} precisions for {} observations", precisions.len(), n),
            }),
        });
    }
    if let Some(bad) = precisions.iter().find(|w| !w.is_finite() || **w <= 0.0) {
        return Err(AddiVortesError::Extension {
            source: Arc::new(InvalidScalePrecisions {
                detail: format!("precision {bad} is not finite and positive"),
            }),
        });
    }
    Ok(())
}

/// Expand raw-column inclusion weights to encoded columns (one-hot groups
/// share their source column's weight, the inclusion point), validating what a
/// custom model returned.
fn expand_weights(raw: &[f64], col_map: &[usize], n_raw: usize) -> Result<Vec<f64>> {
    if raw.len() != n_raw {
        return Err(AddiVortesError::Extension {
            source: Arc::new(InvalidInclusionWeights {
                detail: format!("{} weights for {} covariates", raw.len(), n_raw),
            }),
        });
    }
    if let Some(bad) = raw.iter().find(|w| !w.is_finite() || **w <= 0.0) {
        return Err(AddiVortesError::Extension {
            source: Arc::new(InvalidInclusionWeights {
                detail: format!("weight {bad} is not finite and positive"),
            }),
        });
    }
    Ok(col_map.iter().map(|&raw_col| raw[raw_col]).collect())
}

/// The sampler yields every sweep, forever; burn-in/thinning are `fit()`'s
/// concern. Items are owned copies of the sweep state.
impl Iterator for Sampler {
    type Item = Result<OwnedDraw>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step().map(|draw| OwnedDraw {
            sigma_sq: draw.sigma_sq,
            tessellations: draw.tessellations.to_vec(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use rand_core::Rng;

    use super::*;
    use crate::extensions::coord::EuclideanNormal;
    use crate::extensions::distance::ColumnMetrics;
    use crate::extensions::moves::MoveSetBuilder;

    /// splitmix64 against the published reference vector for seed 0
    /// (first outputs 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, …).
    #[test]
    fn splitmix64_matches_reference_vector() {
        let mut state = 0u64;
        assert_eq!(splitmix64(&mut state), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(&mut state), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(splitmix64(&mut state), 0x06C4_5D18_8009_454F);
        assert_eq!(splitmix64(&mut state), 0xF88B_B8A8_724C_81EC);
    }

    /// A response that is an exact linear function of the features has zero
    /// OLS residual, so the σ² prior calibrates λ to 0. The global σ² draw
    /// is well-defined under that prior, so the Gaussian, robust-t and
    /// probit families fit; the H variance calibration cannot accept it and
    /// refuses with `DegenerateResidual`.
    #[test]
    fn zero_residual_response_fits_unless_a_scale_model_refuses_lambda_zero() {
        let n = 50;
        let column: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let x = Data::new(column.clone(), n, 1).unwrap();
        let linear: Vec<f64> = column.iter().map(|&v| 2.0 * v).collect();
        let quick = || {
            AddiVortesConfig::new(3)
                .with_m(3)
                .with_omega(0.5)
                .with_burn_in(2)
                .with_draws(2)
        };
        let gaussian = quick().fit(&x, &linear).expect("the Gaussian family fits");
        assert!(
            gaussian
                .posterior()
                .sigma_sq()
                .iter()
                .all(|s| s.is_finite() && *s > 0.0)
        );
        quick()
            .with_response_family(crate::engine::model::ResponseFamily::RobustT { df: 5.0 })
            .fit(&x, &linear)
            .expect("the robust-t family fits");
        let flag: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i >= n / 2))).collect();
        let x_flag = Data::new(flag.clone(), n, 1).unwrap();
        quick()
            .with_response_family(crate::engine::model::ResponseFamily::BinaryProbit)
            .fit(&x_flag, &flag)
            .expect("the label family fits separable data");
        assert_eq!(
            crate::engine::builder::SamplerBuilder::new(quick())
                .with_scale_model(crate::extensions::scale::HVariance::new(4).unwrap())
                .fit(&x, &linear)
                .unwrap_err(),
            AddiVortesError::DegenerateResidual {}
        );
    }

    /// The full 32-byte key for representative seeds, byte for byte. Pure integer
    /// arithmetic, bit-identical on every target, so this runs on every CI leg.
    #[test]
    fn expand_seed_key_bytes_are_pinned() {
        #[rustfmt::skip]
        let expected_seed_0: [u8; 32] = [
            0xAF, 0xCD, 0x1D, 0x7B, 0x39, 0xA8, 0x20, 0xE2,
            0xF4, 0x65, 0xB9, 0xA1, 0x6A, 0x9E, 0x78, 0x6E,
            0x4F, 0x45, 0x09, 0x80, 0x18, 0x5D, 0xC4, 0x06,
            0xEC, 0x81, 0x4C, 0x72, 0xA8, 0xB8, 0x8B, 0xF8,
        ];
        #[rustfmt::skip]
        let expected_seed_42: [u8; 32] = [
            0x95, 0x6E, 0xEB, 0x2F, 0x26, 0x32, 0xD7, 0xBD,
            0x03, 0xF1, 0x66, 0xB2, 0x33, 0xE3, 0xEF, 0x28,
            0x52, 0x9F, 0x0F, 0x13, 0x57, 0x67, 0x52, 0x47,
            0x94, 0xE3, 0x4A, 0x0E, 0xFF, 0xE1, 0x1C, 0x58,
        ];
        #[rustfmt::skip]
        let expected_seed_deadbeef: [u8; 32] = [
            0x9B, 0xEB, 0xC9, 0x68, 0x0F, 0xB9, 0xDF, 0x4A,
            0x22, 0x09, 0xA1, 0x41, 0x31, 0x6A, 0x58, 0xDE,
            0x1D, 0xFC, 0x1C, 0x8E, 0x2F, 0xBC, 0x1F, 0x02,
            0x90, 0x67, 0xE1, 0x7B, 0x73, 0xCE, 0x66, 0x74,
        ];
        assert_eq!(expand_seed(0), expected_seed_0);
        assert_eq!(expand_seed(42), expected_seed_42);
        assert_eq!(expand_seed(0xDEAD_BEEF), expected_seed_deadbeef);
    }

    /// The raw ChaCha8 integer stream from an expanded seed. Also pure integer
    /// arithmetic, so target-independent; a `rand_chacha` bump that changes the
    /// stream (chain-altering by definition) turns this red.
    #[test]
    fn chacha8_integer_stream_is_pinned() {
        let mut rng = ChaCha8Rng::from_seed(expand_seed(42));
        let observed: [u64; 4] = std::array::from_fn(|_| rng.next_u64());
        let expected: [u64; 4] = [
            0x3115_9EF9_87C9_1AFC,
            0x1755_9844_B416_9001,
            0xF7D0_AFBF_9AD9_A69F,
            0xB920_7AD5_FD37_495A,
        ];
        assert_eq!(observed, expected);
    }

    // The conjugate-piece oracles (σ²-Gamma params, μ posterior, the
    // complete per-cell term) moved to `cell_model::tests` with the seam;
    // the arithmetic they pin now lives there.

    // ---- sampler behaviour ----

    fn toy_data() -> (Data, Vec<f64>) {
        let x = Data::from_rows(&[
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

    fn small_config(seed: u64) -> AddiVortesConfig {
        let mut config = AddiVortesConfig::new(seed);
        config.m = 4;
        config.omega = 1.5; // p = 2 data: default ω = 3 would (correctly) error
        config.burn_in = 0;
        config.n_draws = 5;
        config
    }

    /// A custom assigner that does not override `reassign` (so every
    /// proposal takes the default full-recompute path) samples exactly the
    /// chain the built-in incremental paths produce: the incremental update
    /// is a pure refactor of when distances are computed, not what.
    #[test]
    fn default_reassign_chain_matches_incremental_chain() {
        #[derive(Debug)]
        struct FullRecomputeOnly(ColumnMetrics);
        impl CellAssigner for FullRecomputeOnly {
            fn assign_cells(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<usize>> {
                self.0.assign_cells(x, tessellation)
            }
        }

        let (x, y) = toy_data();
        let chain_bits = |assigner: Option<Arc<dyn CellAssigner>>| -> Vec<u64> {
            let components = Components {
                assigner,
                ..Components::default()
            };
            let sampler = Sampler::with_components(small_config(11), components, &x, &y).unwrap();
            let mut bits = Vec::new();
            for draw in sampler.take(12) {
                let draw = draw.unwrap();
                bits.push(draw.sigma_sq.to_bits());
                for t in &draw.tessellations {
                    bits.extend(t.dims().iter().map(|&d| d as u64));
                    bits.extend(t.centres().iter().map(|v| v.to_bits()));
                    bits.extend(t.mus().iter().map(|v| v.to_bits()));
                }
            }
            bits
        };

        let full_only = FullRecomputeOnly(ColumnMetrics::new(vec![Metric::Euclidean; 2]));
        assert_eq!(chain_bits(None), chain_bits(Some(Arc::new(full_only))));
    }

    #[test]
    fn same_seed_is_bit_identical_different_seed_differs() {
        let (x, y) = toy_data();
        let run = |seed: u64| -> Vec<u64> {
            let sampler = Sampler::new(small_config(seed), &x, &y).unwrap();
            let mut bits = Vec::new();
            for draw in sampler.take(5) {
                let draw = draw.unwrap();
                bits.push(draw.sigma_sq.to_bits());
                for t in &draw.tessellations {
                    for &c in t.centres() {
                        bits.push(c.to_bits());
                    }
                    for &mu in t.mus() {
                        bits.push(mu.to_bits());
                    }
                    bits.push(t.dims().len() as u64);
                }
            }
            bits
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn uniform_and_all_ones_weighted_inclusion_chains_are_identical() {
        // UniformInclusion and WeightedInclusion(vec![1.0; p]) are both no-op
        // updates with bit-equal weights, so the chains must be bit-identical:
        // the inclusion machinery is provably inert on the default path.
        let (x, y) = toy_data();
        let run = |weighted: bool| -> Vec<u64> {
            let mut components = Components::default();
            if weighted {
                components.inclusion = Some(Arc::new(
                    crate::extensions::inclusion::WeightedInclusion::new(vec![1.0, 1.0]),
                ));
            }
            let sampler = Sampler::with_components(small_config(11), components, &x, &y).unwrap();
            sampler
                .take(4)
                .map(|d| d.unwrap().sigma_sq.to_bits())
                .collect()
        };
        assert_eq!(run(false), run(true));
    }

    // ---- the embed entry ----

    #[test]
    fn set_response_validates_length_and_finiteness() {
        let (x, y) = toy_data();
        let mut sampler = Sampler::new(small_config(3), &x, &y).unwrap();
        let err = sampler.set_response(&[1.0]).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::RowCountMismatch {
                y_len: 1,
                x_rows: 6
            }
        );
        let mut bad = y.clone();
        bad[4] = f64::NAN;
        let err = sampler.set_response(&bad).unwrap_err();
        assert_eq!(err, AddiVortesError::NonFiniteResponse { row: 4 });
    }

    /// The scale contract, half one: `set_response` takes the caller's
    /// response scale, so re-feeding the construction response is a bit-exact
    /// no-op: the chain is unchanged (which also proves the call consumes no
    /// RNG).
    #[test]
    fn set_response_on_the_construction_response_is_chain_neutral() {
        let (x, y) = toy_data();
        let chain_bits = |replace_each_sweep: bool| -> Vec<u64> {
            let mut sampler = Sampler::new(small_config(17), &x, &y).unwrap();
            let mut bits = Vec::new();
            for _ in 0..6 {
                if replace_each_sweep {
                    sampler.set_response(&y).unwrap();
                }
                let draw = sampler.step().unwrap();
                bits.push(draw.sigma_sq.to_bits());
                for t in draw.tessellations {
                    bits.extend(t.centres().iter().map(|v| v.to_bits()));
                    bits.extend(t.mus().iter().map(|v| v.to_bits()));
                }
            }
            bits
        };
        assert_eq!(chain_bits(false), chain_bits(true));
    }

    /// The scale contract, half two: `fitted_values` is the running fit
    /// mapped back to the caller's response scale. At initialisation the
    /// scaled fit is m·mean(y_scaled) in every row, so the caller-scale
    /// value is its exact affine unscaling.
    #[test]
    fn fitted_values_are_on_the_caller_scale() {
        let (x, y) = toy_data();
        let sampler = Sampler::new(small_config(5), &x, &y).unwrap();
        // toy_data's y spans [0.1, 2.3]; scale, average, sum over m = 4, unscale.
        let (y_min, y_max) = (0.1, 2.3);
        let y_scaled: Vec<f64> = y
            .iter()
            .map(|&v| (v - y_min) / (y_max - y_min) - 0.5)
            .collect();
        let mean_y = y_scaled.iter().sum::<f64>() / y_scaled.len() as f64;
        let expected = (mean_y * 4.0 + 0.5) * (y_max - y_min) + y_min;
        for value in sampler.fitted_values() {
            assert_eq!(value.to_bits(), expected.to_bits());
        }
    }

    /// With a `ResponseModel` configured, `set_response` replaces the observed
    /// response the step's `augment` reads (scaled through the frozen
    /// transform), not the working response it writes.
    #[test]
    fn set_response_feeds_the_response_models_observed_response() {
        use std::sync::{Arc, Mutex};

        #[derive(Debug)]
        struct Probe {
            seen: Arc<Mutex<Vec<f64>>>,
        }
        impl crate::extensions::response::ResponseModel for Probe {
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
                *self.seen.lock().expect("test lock") = y.to_vec();
                working.copy_from_slice(y);
                weights.fill(1.0);
                Ok(())
            }
        }

        let (x, y) = toy_data();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut sampler = Sampler::new(small_config(19), &x, &y)
            .unwrap()
            .with_response_model(Probe {
                seen: Arc::clone(&seen),
            });
        let replacement: Vec<f64> = y.iter().map(|v| v + 0.75).collect();
        sampler.set_response(&replacement).unwrap();
        sampler.step().unwrap();

        // What augment saw must be the replacement mapped through the frozen
        // construction-time transform (y range [0.1, 2.3]).
        let expected: Vec<u64> = replacement
            .iter()
            .map(|&v| ((v - 0.1) / (2.3 - 0.1) - 0.5).to_bits())
            .collect();
        let observed: Vec<u64> = seen
            .lock()
            .expect("test lock")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(observed, expected);
    }

    // ---- the scale rebuild (ScaleCtx + per-observation precisions) ----

    /// A kernel step that copies the response through and writes constant
    /// accumulation weights; no RNG.
    #[derive(Debug, Clone)]
    struct ConstWeights(f64);
    impl crate::extensions::response::ResponseModel for ConstWeights {
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
            weights.fill(self.0);
            Ok(())
        }
    }

    /// A scale model pinned to σ² = 1 that supplies constant per-observation
    /// precisions; no RNG.
    #[derive(Debug, Clone)]
    struct ConstPrecisions(Vec<f64>);
    impl crate::extensions::scale::ScaleModel for ConstPrecisions {
        type Error = std::convert::Infallible;
        fn update(
            &mut self,
            _ctx: &crate::extensions::scale::ScaleCtx<'_>,
            _rng: &mut dyn rand_core::Rng,
        ) -> std::result::Result<(), Self::Error> {
            Ok(())
        }
        fn sigma_sq(&self) -> f64 {
            1.0
        }
        fn precisions(&self) -> Option<&[f64]> {
            Some(&self.0)
        }
    }

    fn weighted_chain_bits(mut sampler: Sampler, sweeps: usize) -> Vec<u64> {
        let mut bits = Vec::new();
        for _ in 0..sweeps {
            let draw = sampler.step().unwrap();
            bits.push(draw.sigma_sq.to_bits());
            for t in draw.tessellations {
                bits.extend(t.centres().iter().map(|v| v.to_bits()));
                bits.extend(t.mus().iter().map(|v| v.to_bits()));
            }
        }
        bits
    }

    /// The conductor's composition rule wᵢ = wᵢ^resp · wᵢ^scale, proven bit
    /// for bit: response weights 0.5 composed with scale precisions 2.0 must
    /// sample exactly the chain of response weights 1.0 with no precisions:
    /// the products are bit-equal (0.5 · 2.0 == 1.0 exactly) and neither
    /// side's hooks consume RNG.
    #[test]
    fn composed_weights_match_their_bit_equal_product_chain() {
        let (x, y) = toy_data();
        let n = y.len();
        let sigma_mu_sq = scaler::sigma_mu_sq(3.0, 4);
        let weighted_model =
            || crate::extensions::cell_model::WeightedGaussianModel::new(sigma_mu_sq).unwrap();
        let paper_moves = || MoveSetBuilder::stone_gosling().build().unwrap();
        let composed =
            Sampler::with_cell_model(small_config(23), &x, &y, paper_moves(), weighted_model())
                .unwrap()
                .with_response_model(ConstWeights(0.5))
                .with_scale_model(ConstPrecisions(vec![2.0; n]));
        let product =
            Sampler::with_cell_model(small_config(23), &x, &y, paper_moves(), weighted_model())
                .unwrap()
                .with_response_model(ConstWeights(1.0))
                .with_scale_model(crate::extensions::scale::PinnedSigma::unit());
        assert_eq!(
            weighted_chain_bits(composed, 8),
            weighted_chain_bits(product, 8)
        );
    }

    /// The cell-prior width dial, proven bit for bit: a directly-set σ_μ
    /// equal to the k-rule's own value must sample exactly the k-rule chain.
    /// The dial threads the same variance to the same places (the cell
    /// kernel and the ModelCtx) and changes nothing else.
    #[test]
    fn cell_prior_dial_matches_its_bit_equal_k_rule_chain() {
        let (x, y) = toy_data();
        // The k-rule value for k = 1.5 at m = 4, written as the k-rule's own
        // expression so both sides carry identical bits.
        let sigma_mu = 0.5 / (1.5 * (4.0_f64).sqrt());
        let dialled = Sampler::new(small_config(29).with_cell_prior_sd(sigma_mu), &x, &y).unwrap();
        let ruled = Sampler::new(small_config(29).with_k(1.5), &x, &y).unwrap();
        assert_eq!(
            weighted_chain_bits(dialled, 8),
            weighted_chain_bits(ruled, 8)
        );
    }

    /// A precision-only scale model (no kernel step) enters the accumulation
    /// exactly like a weight-only kernel step (σ² pinned either way): the
    /// (None, Some) and (Some, None) composition arms are the same chain.
    #[test]
    fn scale_precisions_alone_match_response_model_weights_alone() {
        let (x, y) = toy_data();
        let n = y.len();
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * i as f64).collect();
        let sigma_mu_sq = scaler::sigma_mu_sq(3.0, 4);
        let weighted_model =
            || crate::extensions::cell_model::WeightedGaussianModel::new(sigma_mu_sq).unwrap();
        let paper_moves = || MoveSetBuilder::stone_gosling().build().unwrap();
        let via_scale =
            Sampler::with_cell_model(small_config(29), &x, &y, paper_moves(), weighted_model())
                .unwrap()
                .with_scale_model(ConstPrecisions(weights.clone()));
        // The kernel-step arm writes working = y each sweep, identical to
        // the untouched working response, and consumes no RNG.
        let via_step = {
            #[derive(Debug)]
            struct VaryingWeights(Vec<f64>);
            impl crate::extensions::response::ResponseModel for VaryingWeights {
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
                    weights.copy_from_slice(&self.0);
                    Ok(())
                }
            }
            Sampler::with_cell_model(small_config(29), &x, &y, paper_moves(), weighted_model())
                .unwrap()
                .with_response_model(VaryingWeights(weights))
                .with_scale_model(crate::extensions::scale::PinnedSigma::unit())
        };
        assert_eq!(
            weighted_chain_bits(via_scale, 8),
            weighted_chain_bits(via_step, 8)
        );
    }

    /// Mid-chain precision corruption surfaces as the Extension channel,
    /// mirroring the inclusion point's weight validation.
    #[test]
    fn invalid_scale_precisions_surface_extension_error() {
        let (x, y) = toy_data();
        let mut wrong_length = Sampler::new(small_config(31), &x, &y)
            .unwrap()
            .with_scale_model(ConstPrecisions(vec![1.0; 2]));
        assert!(matches!(
            wrong_length.step().unwrap_err(),
            AddiVortesError::Extension { .. }
        ));
        let mut non_positive = Sampler::new(small_config(31), &x, &y)
            .unwrap()
            .with_scale_model(ConstPrecisions(vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]));
        assert!(matches!(
            non_positive.step().unwrap_err(),
            AddiVortesError::Extension { .. }
        ));
    }

    /// The ScaleCtx handed to a scale model carries the sweep's real state:
    /// the working response, the current fit, the scaled design and the
    /// shared machinery.
    #[test]
    fn scale_ctx_exposes_the_sweep_state() {
        use std::sync::{Arc, Mutex};

        type SeenState = (Vec<f64>, Vec<f64>, usize, usize, usize);
        #[derive(Debug)]
        struct Probe {
            seen: Arc<Mutex<SeenState>>,
        }
        impl crate::extensions::scale::ScaleModel for Probe {
            type Error = std::convert::Infallible;
            fn update(
                &mut self,
                ctx: &crate::extensions::scale::ScaleCtx<'_>,
                _rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<(), Self::Error> {
                *self.seen.lock().expect("test lock") = (
                    ctx.y().to_vec(),
                    ctx.fit().to_vec(),
                    ctx.x().n_rows(),
                    ctx.p_enc(),
                    ctx.move_set().len(),
                );
                assert_eq!(ctx.inclusion_weights().len(), ctx.p_enc());
                assert_eq!(ctx.coord_dists().len(), ctx.p_enc());
                Ok(())
            }
            fn sigma_sq(&self) -> f64 {
                1.0
            }
        }

        let (x, y) = toy_data();
        let seen = Arc::new(Mutex::new((Vec::new(), Vec::new(), 0, 0, 0)));
        let mut sampler = Sampler::new(small_config(37), &x, &y)
            .unwrap()
            .with_scale_model(Probe {
                seen: Arc::clone(&seen),
            });
        let expected_y: Vec<u64> = {
            let (lo, hi) = (0.1, 2.3);
            y.iter()
                .map(|&v| ((v - lo) / (hi - lo) - 0.5).to_bits())
                .collect()
        };
        let expected_fit: Vec<u64> = sampler.fit_values().iter().map(|v| v.to_bits()).collect();
        sampler.step().unwrap();
        let (seen_y, seen_fit, n_rows, p_enc, n_moves) = seen.lock().expect("test lock").clone();
        assert_eq!(
            seen_y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected_y
        );
        // The fit handed to the scale update is the pre-sweep fit (the pinned
        // hook order: scale runs before the j-loop).
        assert_eq!(
            seen_fit.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected_fit
        );
        assert_eq!((n_rows, p_enc, n_moves), (6, 2, 6));
    }

    /// Hand accumulation of (count, sum) pairs per cell in ascending
    /// observation order: the replay's independent version of the kernel's
    /// statistics.
    fn hand_stats(assignment: &[usize], residuals: &[f64], n_cells: usize) -> Vec<(f64, f64)> {
        let mut stats = vec![(0.0_f64, 0.0_f64); n_cells];
        for (i, &cell) in assignment.iter().enumerate() {
            stats[cell].0 += 1.0;
            stats[cell].1 += residuals[i];
        }
        stats
    }

    #[test]
    fn hand_replayed_first_sweep_matches_step() {
        // m = 1, n = 3, p = 1: replay the documented RNG-consumption order and
        // the conjugate formulas by hand against step()'s first sweep.
        let x = Data::from_rows(&[[0.0], [0.5], [1.0]]).unwrap();
        let y = vec![0.0, 1.0, 3.0];
        let mut config = AddiVortesConfig::new(123);
        config.m = 1;
        let mut sampler = Sampler::new(config.clone(), &x, &y).unwrap();

        // --- replay ---
        let mut rng = ChaCha8Rng::from_seed(expand_seed(123));
        // y scaled to [−0.5, 0.5]: min 0, max 3 → [−0.5, −1/6, 0.5]; mean = −1/18.
        let y_scaled = [-0.5, -0.5 + 1.0 / 3.0, 0.5];
        let mean_y = y_scaled.iter().sum::<f64>() / 3.0;
        // Initial fit (m = 1): F_i = mean_y; RSS = Σ (y_i − mean)².
        let rss: f64 = y_scaled.iter().map(|v| (v - mean_y) * (v - mean_y)).sum();
        // σ̂: n = 3 ≤ p_enc + 1 + 1? n > p+1 ⇔ 3 > 2 → OLS path on a 1-column design.
        let sigma_hat =
            scaler::ols_residual_sd(&Data::new(vec![-0.5, 0.0, 0.5], 3, 1).unwrap(), &y_scaled)
                .unwrap();
        let lambda = scaler::calibrate_lambda(6.0, 0.85, sigma_hat);
        let (shape, scale_param) =
            crate::extensions::scale::sigma_sq_gamma_params(6.0, lambda, rss, 3);
        let gamma = rand_distr::Gamma::new(shape, scale_param).unwrap();
        let precision: f64 = rand_distr::Distribution::sample(&gamma, &mut rng);
        let expected_sigma_sq = 1.0 / precision;

        // Inclusion update: UniformInclusion consumes no RNG.
        // j-loop, j = 0. State: b = 1, d = 1 = p. Valid: AC, Change. Folds:
        // RC→AC (AC: .2+.2 = .4); AD/RD both invalid (their masses drop);
        // Swap→Change (Change: .1+.1 = .2). Normalised: [2/3, ~, ~, ~, 1/3, ~].
        // Selection probabilities are replicated through the MoveSet itself
        // (they carry their own oracles); the replay pins the draw order
        // and the conjugate formulas by hand.
        let sigma_mu_sq = scaler::sigma_mu_sq(3.0, 1);
        let move_set = MoveSetBuilder::stone_gosling().build().unwrap();
        let dists: Vec<Arc<dyn CoordinateDistribution>> =
            vec![Arc::new(EuclideanNormal::new(0.8).unwrap())];
        let weights = [1.0_f64];
        let ctx = ModelCtx::new(
            expected_sigma_sq,
            config.omega,
            config.lambda_c,
            sigma_mu_sq,
            1,
            &dists,
            &weights,
        );
        let u_select = crate::extensions::moves::uniform_f64(&mut rng);
        let init_state = Tessellation {
            centres: vec![0.0],
            dims: vec![0],
            mus: vec![mean_y],
        };
        let probs = move_set.selection_probs(&init_state, &ctx);
        let selected_ac = u_select < probs[0];

        let x_scaled = [-0.5, 0.0, 0.5];
        let mut current_assignment = vec![0usize, 0, 0];
        let mut current_centres = vec![0.0_f64];
        let residuals = y_scaled; // F = g_j for m = 1, so R = y − F + g = y.

        if selected_ac {
            // AC: sample one coordinate (EuclideanNormal σ_c = 0.8).
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
            let new_centre = 0.8 * z;
            // Proposed 2 cells: assignment by nearest centre {0.0, new_centre}.
            let proposed_centres = [0.0, new_centre];
            let proposed_assignment: Vec<usize> = x_scaled
                .iter()
                .map(|&xi| {
                    let d0 = (xi - proposed_centres[0]) * (xi - proposed_centres[0]);
                    let d1 = (xi - proposed_centres[1]) * (xi - proposed_centres[1]);
                    usize::from(d1.total_cmp(&d0) == std::cmp::Ordering::Less)
                })
                .collect();
            let empty = (0..2).any(|c| !proposed_assignment.contains(&c));
            if !empty {
                let current = hand_stats(&current_assignment, &residuals, 1);
                let proposed = hand_stats(&proposed_assignment, &residuals, 2);
                let terms = |pairs: &[(f64, f64)]| {
                    crate::extensions::cell_model::gaussian_marginal_terms(
                        pairs.iter().copied(),
                        expected_sigma_sq,
                        sigma_mu_sq,
                    )
                };
                let log_lik = terms(&proposed) - terms(&current);
                // structure(AC, b=1): ln λ_c − ln 1 (hand-derived; the RC
                // pick factor cancels against the ordering multiplicity,
                // see AddCentre); the selection correction is replicated
                // through the MoveSet (its boundary values carry their own
                // oracles).
                let log_structure =
                    crate::engine::mathsfn::ln(5.0) - crate::engine::mathsfn::ln(1.0);
                let proposed_state = Tessellation {
                    centres: proposed_centres.to_vec(),
                    dims: vec![0],
                    mus: vec![mean_y, 0.0],
                };
                let log_selection =
                    move_set.log_selection_ratio(0, &init_state, &proposed_state, &ctx);
                let log_alpha = log_lik + log_structure + log_selection;
                let u = crate::extensions::moves::uniform_f64(&mut rng);
                if crate::engine::mathsfn::ln(u) < log_alpha {
                    current_centres = proposed_centres.to_vec();
                    current_assignment = proposed_assignment;
                }
            }
        } else {
            // Change: pick centre (single draw over 1 centre), resample coord.
            let _pick = crate::extensions::moves::uniform_index(1, &mut rng);
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
            let moved = 0.8 * z;
            // One cell: assignment unchanged; likelihood identical (single cell
            // output independent of centre) → log_lik = 0, structure 0,
            // selection ln(q(Change|1 cell)/q(Change|1 cell)) = 0 → always accept
            // iff ln(u) < 0 (true unless u rounds to ≥ 1, impossible).
            let u = crate::extensions::moves::uniform_f64(&mut rng);
            if crate::engine::mathsfn::ln(u) < 0.0 {
                current_centres = vec![moved];
            }
        }

        // μ redraw per cell (ascending).
        let stats = hand_stats(&current_assignment, &residuals, current_centres.len());
        let mut expected_mus = Vec::new();
        for &(count, sum) in &stats {
            let (mean, variance) = crate::extensions::cell_model::mu_posterior(
                count,
                sum,
                expected_sigma_sq,
                sigma_mu_sq,
            );
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
            expected_mus.push(mean + variance.sqrt() * z);
        }

        // --- compare with the real sweep ---
        let draw = sampler.step().unwrap();
        assert_eq!(draw.sigma_sq.to_bits(), expected_sigma_sq.to_bits());
        assert_eq!(draw.tessellations.len(), 1);
        let t = &draw.tessellations[0];
        assert_eq!(t.centres().len(), current_centres.len());
        for (observed, expected) in t.centres().iter().zip(&current_centres) {
            assert_eq!(observed.to_bits(), expected.to_bits());
        }
        for (observed, expected) in t.mus().iter().zip(&expected_mus) {
            assert_eq!(observed.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn advisory_posterior_sanity_smoke() {
        // Advisory calibration smoke (the full SBC battery lives in
        // `stat_gates`): a linear signal with known noise; the chain must land the
        // posterior σ² in a wide band and track the signal in-sample.
        let n = 40;
        let mut rng = ChaCha8Rng::from_seed(expand_seed(99));
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let noise_sd = 0.1;
        let y: Vec<f64> = xs
            .iter()
            .map(|&v| {
                let z: f64 =
                    rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
                2.0 * v + noise_sd * z
            })
            .collect();
        let x = Data::new(xs, n, 1).unwrap();

        let mut config = AddiVortesConfig::new(2024);
        config.m = 20;
        let mut sampler = Sampler::new(config, &x, &y).unwrap();
        // Burn then average σ² over draws.
        for _ in 0..100 {
            sampler.step().unwrap();
        }
        let mut sigma_sqs = Vec::new();
        for _ in 0..100 {
            sigma_sqs.push(sampler.step().unwrap().sigma_sq);
        }
        let mean_sigma_sq = sigma_sqs.iter().sum::<f64>() / sigma_sqs.len() as f64;
        // Wide band: true scaled σ = 0.1/range(y); range ≈ 2 + noise ⇒ scaled
        // σ² ≈ (0.1/2.2)² ≈ 2e-3. Accept anything within a decade.
        assert!(
            mean_sigma_sq > 2e-4 && mean_sigma_sq < 2e-2,
            "posterior sigma_sq {mean_sigma_sq} far from truth ≈ 2e-3"
        );
    }
}
