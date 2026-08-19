//! Crate-internal construction surface: the one place components meet the
//! engine. Model files and the spec layer wire here; the public config
//! stays plain data.

use std::sync::Arc;

use crate::engine::config::AddiVortesConfig;
use crate::engine::data::Data;
use crate::engine::error::Result;
use crate::engine::model::FittedAddiVortes;
use crate::engine::sampler::Sampler;
use crate::extensions::basis::CellBasis;
use crate::extensions::cell_model::CellModel;
use crate::extensions::coord::CoordinateDistribution;
use crate::extensions::count_priors::CountPriors;
use crate::extensions::distance::{CellAssigner, PairwiseDistance};
use crate::extensions::erasure::{CellModelFactory, ResponseModelFactory, ScaleModelFactory};
use crate::extensions::inclusion::{ErasedInclusionModel, InclusionModel};
use crate::extensions::membership::MembershipKernel;
use crate::extensions::moves::MoveSet;
use crate::extensions::response::ResponseModel;
use crate::extensions::scale::ScaleModel;

/// Component wiring for one fit; `None` on every axis is the paper model.
#[derive(Debug, Default)]
pub(crate) struct Components {
    /// Move set; `None` = the paper's six moves.
    pub(crate) move_set: Option<Arc<MoveSet>>,
    /// Per-raw-column coordinate laws; `None` = per-metric defaults.
    pub(crate) coords: Option<Vec<Arc<dyn CoordinateDistribution>>>,
    /// Cell assigner; `None` = compound `ColumnMetrics`.
    pub(crate) assigner: Option<Arc<dyn CellAssigner>>,
    /// Soft-membership kernel; `None` = hard membership.
    pub(crate) membership: Option<Arc<dyn MembershipKernel>>,
    /// Inclusion model; `None` = uniform.
    pub(crate) inclusion: Option<Arc<dyn ErasedInclusionModel>>,
    /// Cell model; `None` = conjugate Gaussian.
    pub(crate) cell_model: Option<Arc<dyn CellModelFactory>>,
    /// Cell basis; `None` = scalar payload.
    pub(crate) basis: Option<Arc<dyn CellBasis>>,
    /// Augmentation step; `None` = none.
    pub(crate) response_model: Option<Arc<dyn ResponseModelFactory>>,
    /// Scale model; `None` = calibrated global sigma-squared.
    pub(crate) scale_model: Option<Arc<dyn ScaleModelFactory>>,
    /// Count priors; `None` = shifted Poisson/Binomial.
    pub(crate) count_priors: Option<Arc<dyn CountPriors>>,
}

impl Components {
    /// True when any axis was set explicitly (such a fit refuses to
    /// serialise: trait objects have no portable form).
    pub(crate) fn any_custom(&self) -> bool {
        self.move_set.is_some()
            || self.coords.is_some()
            || self.assigner.is_some()
            || self.membership.is_some()
            || self.inclusion.is_some()
            || self.cell_model.is_some()
            || self.basis.is_some()
            || self.response_model.is_some()
            || self.scale_model.is_some()
            || self.count_priors.is_some()
    }
}

/// Plain config + component wiring -> sampler or fitted model. Consuming
/// setters, last wins; every setter mirrors a former config hook.
#[derive(Debug)]
pub(crate) struct SamplerBuilder {
    config: AddiVortesConfig,
    components: Components,
}

impl SamplerBuilder {
    /// Start from a plain-data config; all axes at their defaults.
    pub(crate) fn new(config: AddiVortesConfig) -> Self {
        Self {
            config,
            components: Components::default(),
        }
    }

    /// Pass-through for the plain-data family field (closure ergonomics in
    /// the batteries; the config stays the field's home).
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_response_family(
        mut self,
        family: crate::engine::model::ResponseFamily,
    ) -> Self {
        self.config = self.config.with_response_family(family);
        self
    }

    /// A custom move set.
    #[must_use]
    pub(crate) fn with_move_set(mut self, move_set: MoveSet) -> Self {
        // Single-threaded sharing; Send + Sync is never required of moves.
        #[allow(clippy::arc_with_non_send_sync)]
        let move_set = Arc::new(move_set);
        self.components.move_set = Some(move_set);
        self
    }

    /// Custom centre-coordinate laws, one per raw column.
    #[must_use]
    pub(crate) fn with_coords(mut self, coords: Vec<Arc<dyn CoordinateDistribution>>) -> Self {
        self.components.coords = Some(coords);
        self
    }

    /// A custom assignment geometry (sugar over `with_assigner`).
    #[must_use]
    pub(crate) fn with_distance<D: PairwiseDistance + 'static>(mut self, distance: D) -> Self {
        self.components.assigner = Some(Arc::new(distance));
        self
    }

    /// A custom cell assigner.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_assigner(mut self, assigner: Arc<dyn CellAssigner>) -> Self {
        self.components.assigner = Some(assigner);
        self
    }

    /// Soft membership.
    #[must_use]
    pub(crate) fn with_membership<K: MembershipKernel + 'static>(mut self, kernel: K) -> Self {
        self.components.membership = Some(Arc::new(kernel));
        self
    }

    /// A custom inclusion model.
    #[must_use]
    pub(crate) fn with_inclusion<M: InclusionModel + Clone + 'static>(mut self, model: M) -> Self {
        self.components.inclusion = Some(Arc::new(model));
        self
    }

    /// A custom cell model.
    #[must_use]
    pub(crate) fn with_cell_model<M: CellModel + Clone + 'static>(mut self, model: M) -> Self {
        self.components.cell_model = Some(Arc::new(model));
        self
    }

    /// The cell basis for a basis payload.
    #[must_use]
    pub(crate) fn with_cell_basis<B: CellBasis + 'static>(mut self, basis: B) -> Self {
        self.components.basis = Some(Arc::new(basis));
        self
    }

    /// A data-augmentation step.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_response_model<K: ResponseModel + Clone + 'static>(
        mut self,
        step: K,
    ) -> Self {
        self.components.response_model = Some(Arc::new(step));
        self
    }

    /// A custom scale model.
    #[must_use]
    pub(crate) fn with_scale_model<S: ScaleModel + Clone + 'static>(mut self, scale: S) -> Self {
        self.components.scale_model = Some(Arc::new(scale));
        self
    }

    /// Custom count priors.
    #[must_use]
    pub(crate) fn with_count_priors<C: CountPriors + 'static>(mut self, priors: C) -> Self {
        self.components.count_priors = Some(Arc::new(priors));
        self
    }

    /// The plain-config half (crate-internal accessor).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn config(&self) -> &AddiVortesConfig {
        &self.config
    }

    /// Construct the sampler (boundary checks run; the config is not
    /// validated here, matching `Sampler::new`).
    pub(crate) fn build(self, x: &Data, y: &[f64]) -> Result<Sampler> {
        Sampler::with_components(self.config, self.components, x, y)
    }

    /// Validate, construct, and run the standard burn-in/thinning loop.
    pub(crate) fn fit(self, x: &Data, y: &[f64]) -> Result<FittedAddiVortes> {
        self.config.validate()?;
        let sampler = self.build(x, y)?;
        crate::engine::model::fit_sampler(sampler, x, y)
    }

    /// The pinned-prior battery constructor (see `Sampler::pinned_prior`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn pinned_prior(
        self,
        x_enc: Data,
        metrics_enc: Vec<crate::engine::data::Metric>,
        y_scaled: Vec<f64>,
        lambda: f64,
        move_set: MoveSet,
    ) -> Result<Sampler> {
        Sampler::pinned_prior(
            self.config,
            self.components,
            x_enc,
            metrics_enc,
            y_scaled,
            lambda,
            move_set,
        )
    }
}
