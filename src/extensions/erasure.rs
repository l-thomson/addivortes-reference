//! Internal type-erasure plumbing shared by the cell-model, response and scale
//! points, **not an extension point**, and nothing here is public.
//!
//! The config is `Clone`, so a chosen component is held as a *factory* that
//! mints a fresh erased instance per fit; the sampler then stores `dyn` at
//! batch granularity (one virtual call per (tessellation, proposal) batch, hot
//! loops statically dispatched inside).
//!
//! Contributors implementing a trait never touch this file; it is the machinery
//! that lets them be swapped.

use std::any::Any;
use std::sync::Arc;

use crate::engine::error::{AddiVortesError, Result};
use crate::extensions::cell_model::{
    CellModel, CellStats, GaussianCellModel, WeightedGaussianModel,
};
use crate::extensions::response::ResponseModel;
use crate::extensions::scale::{GlobalSigma, ScaleCtx, ScaleModel};

/// This extension point's fit-time default cell kernel: the Gaussian conjugate model,
/// type-erased so the sampler resolves the default without naming a
/// concrete approach.
pub(crate) fn default_cell_kernel(sigma_mu_sq: f64) -> Box<dyn ErasedCellKernel> {
    Box::new(KernelOf(GaussianCellModel::new(sigma_mu_sq)))
}

/// The weight-aware sibling of [`default_cell_kernel`], assembled by the
/// weight-producing response families (robust-t): the same conjugate
/// Gaussian cells, accumulated under per-observation precisions.
pub(crate) fn weighted_cell_kernel(sigma_mu_sq: f64) -> Box<dyn ErasedCellKernel> {
    Box::new(KernelOf(WeightedGaussianModel::new(sigma_mu_sq)))
}

/// This extension point's fit-time default scale model: the global σ² Gibbs draw,
/// type-erased for the same reason.
pub(crate) fn default_scale_model(nu: f64, lambda: f64) -> Box<dyn ErasedScaleModel> {
    Box::new(GlobalSigma::new(nu, lambda))
}

// ---------------------------------------------------------------------------
// Config-level storage (the ErasedInclusionModel precedent): the config is
// Clone, so seam choices are held as factories that mint a fresh erased
// instance per fit. The `Clone` bound is the price of config-level selection;
// the Sampler-level `with_*` entries remain for non-Clone models.
// ---------------------------------------------------------------------------

/// Mints a fresh erased cell kernel per fit (config storage, the deep seam).
pub(crate) trait CellModelFactory: std::fmt::Debug + Send + Sync {
    fn kernel(&self) -> Box<dyn ErasedCellKernel>;
}

impl<M: CellModel + Clone + 'static> CellModelFactory for M {
    fn kernel(&self) -> Box<dyn ErasedCellKernel> {
        Box::new(KernelOf(self.clone()))
    }
}

/// Mints a fresh erased scale model per fit (config storage, the deep seam).
pub(crate) trait ScaleModelFactory: std::fmt::Debug + Send + Sync {
    fn scale(&self) -> Box<dyn ErasedScaleModel>;
    /// The model's own [`ScaleModel::heteroscedastic`] claim, readable from the
    /// config before any fit: the sampler needs it to choose the mean-cell
    /// statistic at assembly time.
    fn heteroscedastic(&self) -> bool;
}

impl<S: ScaleModel + Clone + 'static> ScaleModelFactory for S {
    fn scale(&self) -> Box<dyn ErasedScaleModel> {
        Box::new(self.clone())
    }
    fn heteroscedastic(&self) -> bool {
        ScaleModel::heteroscedastic(self)
    }
}

/// Mints a fresh erased kernel step per fit (config storage, the deep seam).
pub(crate) trait ResponseModelFactory: std::fmt::Debug + Send + Sync {
    fn step(&self) -> Box<dyn ErasedResponseModel>;
}

impl<K: ResponseModel + Clone + 'static> ResponseModelFactory for K {
    fn step(&self) -> Box<dyn ErasedResponseModel> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Crate-internal erasure (the CellAssigner/ErasedInclusionModel precedent):
// the sampler stores `dyn` at batch granularity: one virtual call per
// (tessellation, proposal) batch, hot loops statically dispatched inside.
// ---------------------------------------------------------------------------

/// Opaque accumulated statistics, produced and consumed by the same kernel.
pub(crate) type StatsBox = Box<dyn Any>;

/// The per-fit basis rows z(xᵢ), row-major `n × q`. Built once per fit
/// (the basis is a fixed set of columns, so it does not move with the
/// tessellation) and handed to every accumulation.
#[derive(Debug)]
pub(crate) struct BasisRows {
    pub(crate) values: Vec<f64>,
    pub(crate) q: usize,
}

impl BasisRows {
    /// Observation `i`'s basis row (length q).
    pub(crate) fn row(&self, i: usize) -> &[f64] {
        &self.values[i * self.q..(i + 1) * self.q]
    }
}

pub(crate) trait ErasedCellKernel: std::fmt::Debug + Send + Sync {
    /// Accumulate per-cell statistics in ascending observation order (pinned).
    /// `weights = None` means hard assignment (weight ≡ 1, no buffer).
    /// `basis = None` is the scalar payload; `Some` records each observation
    /// against its basis row.
    fn accumulate(
        &self,
        assignments: &[usize],
        values: &[f64],
        weights: Option<&[f64]>,
        n_cells: usize,
        basis: Option<&BasisRows>,
    ) -> StatsBox;
    /// The empty-cell guard: is every cell occupied?
    fn all_occupied(&self, stats: &StatsBox) -> bool;
    fn log_marginal(&self, stats: &StatsBox, sigma_sq: f64) -> Result<f64>;
    /// The per-cell payload, flattened row-major (q values per cell, ascending
    /// cell index): exactly the layout of `Tessellation::mus`.
    fn draw_cell_values(
        &self,
        stats: &StatsBox,
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<Vec<f64>>;
    /// Payload width q (1 for the scalar families).
    fn payload_width(&self) -> usize;
    /// Whether this payload needs basis rows.
    fn cell_basis(&self) -> bool;
}

/// Wraps a typed [`CellModel`] into the erased kernel.
#[derive(Debug)]
pub(crate) struct KernelOf<M: CellModel>(pub(crate) M);

impl<M: CellModel + 'static> KernelOf<M> {
    fn stats<'a>(&self, stats: &'a StatsBox) -> &'a Vec<M::Stats> {
        stats
            .downcast_ref::<Vec<M::Stats>>()
            .expect("stats are only ever built by the kernel that consumes them")
    }
}

impl<M: CellModel + 'static> ErasedCellKernel for KernelOf<M> {
    fn accumulate(
        &self,
        assignments: &[usize],
        values: &[f64],
        weights: Option<&[f64]>,
        n_cells: usize,
        basis: Option<&BasisRows>,
    ) -> StatsBox {
        let mut stats = vec![M::Stats::default(); n_cells];
        match (weights, basis) {
            (None, None) => {
                for (i, &cell) in assignments.iter().enumerate() {
                    stats[cell].record(values[i], 1.0);
                }
            }
            (Some(weights), None) => {
                for (i, &cell) in assignments.iter().enumerate() {
                    stats[cell].record(values[i], weights[i]);
                }
            }
            (None, Some(basis)) => {
                for (i, &cell) in assignments.iter().enumerate() {
                    stats[cell].record_basis(basis.row(i), values[i], 1.0);
                }
            }
            (Some(weights), Some(basis)) => {
                for (i, &cell) in assignments.iter().enumerate() {
                    stats[cell].record_basis(basis.row(i), values[i], weights[i]);
                }
            }
        }
        Box::new(stats)
    }

    fn all_occupied(&self, stats: &StatsBox) -> bool {
        self.stats(stats).iter().all(CellStats::occupied)
    }

    fn log_marginal(&self, stats: &StatsBox, sigma_sq: f64) -> Result<f64> {
        self.0
            .log_marginal_terms(self.stats(stats), sigma_sq)
            .map_err(|e| AddiVortesError::Extension {
                source: Arc::new(e),
            })
    }

    fn draw_cell_values(
        &self,
        stats: &StatsBox,
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<Vec<f64>> {
        self.0
            .draw_cell_payload(self.stats(stats), sigma_sq, rng)
            .map_err(|e| AddiVortesError::Extension {
                source: Arc::new(e),
            })
    }

    fn payload_width(&self) -> usize {
        self.0.payload_width()
    }

    fn cell_basis(&self) -> bool {
        self.0.cell_basis()
    }
}

pub(crate) trait ErasedScaleModel: std::fmt::Debug + Send + Sync {
    fn update(&mut self, ctx: &ScaleCtx<'_>, rng: &mut dyn rand_core::Rng) -> Result<()>;
    fn sigma_sq(&self) -> f64;
    fn precisions(&self) -> Option<&[f64]>;
}

impl<S: ScaleModel> ErasedScaleModel for S {
    fn update(&mut self, ctx: &ScaleCtx<'_>, rng: &mut dyn rand_core::Rng) -> Result<()> {
        ScaleModel::update(self, ctx, rng).map_err(|e| AddiVortesError::Extension {
            source: Arc::new(e),
        })
    }
    fn sigma_sq(&self) -> f64 {
        ScaleModel::sigma_sq(self)
    }
    fn precisions(&self) -> Option<&[f64]> {
        ScaleModel::precisions(self)
    }
}

pub(crate) trait ErasedResponseModel: std::fmt::Debug + Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn augment(
        &mut self,
        y: &[f64],
        fit: &[f64],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
        working: &mut [f64],
        weights: &mut [f64],
    ) -> Result<()>;
}

impl<K: ResponseModel> ErasedResponseModel for K {
    fn augment(
        &mut self,
        y: &[f64],
        fit: &[f64],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
        working: &mut [f64],
        weights: &mut [f64],
    ) -> Result<()> {
        ResponseModel::augment(self, y, fit, sigma_sq, rng, working, weights).map_err(|e| {
            AddiVortesError::Extension {
                source: Arc::new(e),
            }
        })
    }
}
