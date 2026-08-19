//! Variable inclusion: the [`InclusionModel`] trait and
//! the weighted subset-prior machinery the AD/RD/Swap moves consume. Each
//! shelf approach is one file in `inclusion/`: [`UniformInclusion`] (`uniform`:
//! exactly the paper's model, and the fit-time default),
//! [`WeightedInclusion`] (`weighted`), and [`DartInclusion`] (`dart`: the
//! Metropolis-corrected DART reference).
//!
//! Weights index caller-visible pre-encoding columns and are relative
//! (they need not sum to 1). The engine provides the weighted subset-prior
//! and selection corrections inside every AD/RD/Swap acceptance ratio (the
//! `e_d(s)` machinery, detailed balance by construction, exactly zero at
//! uniform weights) and one-hot weight expansion. Custom models are checked
//! by `conformance::check_inclusion_model`; the worked starting point is
//! `examples/template_inclusion.rs`, and the adaptive-update exactness
//! warning is on [`InclusionModel::update`]. Read-only reporting needs no
//! code at all: `FittedAddiVortes::variable_inclusion_proportions`
//! summarises posterior usage per caller-visible column.
//!
//! Sources: the uniform prior is the paper's (Stone & Gosling 2025); the
//! Dirichlet sparsity prior behind [`DartInclusion`] is DART (Linero 2018).
//! The exact MH correction and the defensive-mixture proposal are this
//! crate's adaptation (the mixture device is Hesterberg 1995, with
//! independence-sampler ergodicity per Mengersen & Tweedie 1996; the full
//! exactness argument is on the `dart` module).
//! `variable_inclusion_proportions` mirrors BART's variable-importance
//! counts (Chipman, George & McCulloch 2010).

mod dart;
mod uniform;
mod weighted;

pub use dart::DartInclusion;
pub use uniform::UniformInclusion;
pub use weighted::WeightedInclusion;

use crate::engine::error::{Result, from_extension};
use crate::engine::mathsfn;
use crate::extensions::moves::uniform_f64;

/// This extension point's fit-time default inclusion model, type-erased so the sampler
/// resolves the default without naming a concrete approach.
/// `UniformInclusion` is exactly the paper's model.
pub(crate) fn default_erased(n_raw_cols: usize) -> Box<dyn ErasedInclusionModel> {
    Box::new(UniformInclusion::new(n_raw_cols))
}

/// Per-covariate usage counts for one sweep, indexed by pre-encoding
/// (caller-visible) covariate: one-hot groups aggregate back onto their source
/// column. Filled by the sampler in ascending covariate order (pinned);
/// read by `InclusionModel::update`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InclusionUsage {
    counts: Vec<u64>,
    subset_sizes: Vec<usize>,
}

impl InclusionUsage {
    /// A zeroed container for `n_raw_covariates` counts.
    pub fn new(n_raw_covariates: usize) -> Self {
        Self {
            counts: vec![0; n_raw_covariates],
            subset_sizes: Vec::new(),
        }
    }

    /// How many active tessellation dimensions currently map to each
    /// pre-encoding covariate.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// The per-tessellation dimension-subset sizes d_t (counts, one per
    /// tessellation, sampler order). An exact adaptive update needs these:
    /// the distinct-subset prior's e_{d_t}(s) normalisers enter the true
    /// full conditional per tessellation (see [`DartInclusion`]).
    pub fn subset_sizes(&self) -> &[usize] {
        &self.subset_sizes
    }

    /// Record one active dimension for `covariate` (pre-encoding index).
    ///
    /// The sampler fills a usage once per sweep, in ascending tessellation
    /// order: one [`record_subset_size`](Self::record_subset_size)`(d_t)` per
    /// tessellation, then one `record(col)` per active dimension of that
    /// tessellation. A realistic usage therefore satisfies
    /// `counts.iter().sum::<u64>() == subset_sizes.iter().sum::<usize>()`.
    /// Public so an out-of-crate `InclusionModel` can be exercised (e.g. via
    /// [`check_inclusion_model`](crate::conformance::check_inclusion_model))
    /// at the non-zero usages its adaptive `update` actually sees.
    ///
    /// # Panics
    ///
    /// Panics if `covariate` is out of range for the `n_raw_covariates` the
    /// usage was created with.
    pub fn record(&mut self, covariate: usize) {
        self.counts[covariate] += 1;
    }

    /// Record one tessellation's dimension-subset size (see
    /// [`record`](Self::record) for the fill pattern).
    pub fn record_subset_size(&mut self, d: usize) {
        self.subset_sizes.push(d);
    }
}

/// The variable-inclusion point: supplies the per-covariate weights
/// the built-in AD/RD/Swap moves read, and may adapt them once per sweep.
///
/// Weights are relative (they need not sum to 1), must be strictly positive
/// and finite, and index the caller-visible pre-encoding columns; the
/// library expands them alongside one-hot encoding exactly like `Vec<Metric>`.
///
/// Exactness warning (adaptive updates, see the extension guide):
/// AddiVortes dimension sets are distinct subsets, not with-replacement
/// draws, so DART's conjugate Dirichlet Gibbs update is not the true full
/// conditional here (the e_d(s) normalisers do not cancel). An adaptive
/// `update` must carry its own correction (e.g. Metropolis–Hastings with
/// proposal s′ ~ Dirichlet(α/p + u)) or the sampler it produces is invalid.
pub trait InclusionModel: std::fmt::Debug + Send + Sync {
    /// The extension-error channel: surfaced as
    /// [`AddiVortesError::Extension`] by the sampler.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The current weights, one per pre-encoding covariate.
    fn weights(&self) -> &[f64];

    /// Once-per-sweep adaptation at the pinned point (σ² draw → this →
    /// j-loop). A no-op implementation must consume no RNG.
    fn update(
        &mut self,
        usage: &InclusionUsage,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error>;
}

/// Object-safe erasure of [`InclusionModel`] for storage in the sampler: folds
/// the associated `Error` into [`AddiVortesError::Extension`].
pub(crate) trait ErasedInclusionModel: std::fmt::Debug + Send + Sync {
    fn weights(&self) -> &[f64];
    fn update(&mut self, usage: &InclusionUsage, rng: &mut dyn rand_core::Rng) -> Result<()>;
    /// The config holds an `Arc`; the sampler needs its own mutable copy, so
    /// custom inclusion models must be `Clone`.
    fn clone_erased(&self) -> Box<dyn ErasedInclusionModel>;
}

/// Raised (as `AddiVortesError::Extension`) when a custom inclusion model
/// returns unusable weights mid-chain: wrong length, non-finite, or
/// non-positive.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("inclusion model returned invalid weights: {detail}")]
pub(crate) struct InvalidInclusionWeights {
    pub(crate) detail: String,
}

impl<M: InclusionModel + Clone + 'static> ErasedInclusionModel for M {
    fn weights(&self) -> &[f64] {
        InclusionModel::weights(self)
    }

    fn update(&mut self, usage: &InclusionUsage, rng: &mut dyn rand_core::Rng) -> Result<()> {
        InclusionModel::update(self, usage, rng).map_err(from_extension)
    }

    fn clone_erased(&self) -> Box<dyn ErasedInclusionModel> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Weighted subset-prior machinery
// ---------------------------------------------------------------------------

/// The single-draw weighted-choice helper: every dimension
/// selection (AD incoming, RD outgoing, Swap incoming) routes through here.
/// Draws one uniform, walks the candidates in ascending order, and returns the
/// position in `candidates` of the chosen entry. Weights are looked up by
/// global covariate index and must be strictly positive.
pub(crate) fn weighted_choice(
    candidates: &[usize],
    weights: &[f64],
    rng: &mut dyn rand_core::Rng,
) -> usize {
    debug_assert!(!candidates.is_empty());
    let total: f64 = candidates.iter().map(|&c| weights[c]).sum();
    let target = uniform_f64(rng) * total;
    let mut cumulative = 0.0_f64;
    for (position, &candidate) in candidates.iter().enumerate() {
        cumulative += weights[candidate];
        if target < cumulative {
            return position;
        }
    }
    candidates.len() - 1 // rounding guard: the draw landed on the top edge
}

/// True when every weight is bit-identical: the uniform case, in which every
/// subset-prior correction is exactly zero. Detecting it keeps the default
/// (`UniformInclusion`) chain bit-identical to the paper's constants at any p
/// (floating-point e_d ratios would otherwise reintroduce ULP noise), and
/// costs O(p) instead of O(p·d).
fn weights_are_uniform(weights: &[f64]) -> bool {
    weights.windows(2).all(|w| w[0].to_bits() == w[1].to_bits())
}

/// Elementary symmetric polynomials e_0..e_max over `weights`, by the standard
/// recursion (ascending index, pinned): after absorbing weight s,
/// `e[j] ← e[j] + s·e[j−1]` for j = max..1.
pub(crate) fn elementary_symmetric(weights: &[f64], max: usize) -> Vec<f64> {
    let mut e = vec![0.0_f64; max + 1];
    e[0] = 1.0;
    for &s in weights {
        for j in (1..=max).rev() {
            e[j] += s * e[j - 1];
        }
    }
    e
}

/// Sum of weights over the active dims and over the rest: (W_in, W_out).
fn weight_split(dims: &[usize], weights: &[f64]) -> (f64, f64) {
    let mut w_in = 0.0_f64;
    for &d in dims {
        w_in += weights[d];
    }
    let total: f64 = weights.iter().sum();
    (w_in, total - w_in)
}

/// The AD subset/selection correction for adding global covariate `j` to
/// `dims` (derived from the weighted subset prior P(D|d,s) ∝ ∏s_k / e_d(s) and
/// the weighted incoming/outgoing selections):
///
/// ```text
/// ln[ s_j · e_d · W_out / ( e_{d+1} · (W_in + s_j) ) ]
/// ```
///
/// Exactly 0 under uniform weights (all-equal shortcut above; mathematically
/// C(p,d)(p−d) = C(p,d+1)(d+1)).
pub(crate) fn log_add_dimension_correction(dims: &[usize], j: usize, weights: &[f64]) -> f64 {
    if weights_are_uniform(weights) {
        return 0.0;
    }
    let d = dims.len();
    let e = elementary_symmetric(weights, d + 1);
    let (w_in, w_out) = weight_split(dims, weights);
    mathsfn::ln(weights[j] * e[d] * w_out / (e[d + 1] * (w_in + weights[j])))
}

/// The RD counterpart for removing global covariate `j` from `dims`:
///
/// ```text
/// ln[ e_d · W_in / ( e_{d−1} · s_j · (W_out + s_j) ) ]
/// ```
///
/// (the exact reciprocal of the AD correction from the smaller set, so detailed
/// balance holds by construction). Exactly 0 under uniform weights.
pub(crate) fn log_remove_dimension_correction(dims: &[usize], j: usize, weights: &[f64]) -> f64 {
    if weights_are_uniform(weights) {
        return 0.0;
    }
    let d = dims.len();
    let e = elementary_symmetric(weights, d);
    let (w_in, w_out) = weight_split(dims, weights);
    mathsfn::ln(e[d] * w_in / (e[d - 1] * weights[j] * (w_out + weights[j])))
}

/// The Swap correction for outgoing `i`, incoming `j`:
/// `ln[ W_out / (W_out − s_j + s_i) ]` (the subset-prior ratio s_j/s_i cancels
/// against the selection probabilities). Exactly 0 under uniform weights.
pub(crate) fn log_swap_correction(dims: &[usize], i: usize, j: usize, weights: &[f64]) -> f64 {
    if weights_are_uniform(weights) {
        return 0.0;
    }
    let (_, w_out) = weight_split(dims, weights);
    mathsfn::ln(w_out / (w_out - weights[j] + weights[i]))
}
