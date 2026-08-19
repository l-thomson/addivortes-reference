//! **Scale / precision**, *"the noise level isn't one global constant."*
//!
//! Implement [`ScaleModel`]: a per-sweep `update` receiving a [`ScaleCtx`] (the
//! working response, the current fit, the scaled design, and the shared backfit
//! machinery), the scalar `sigma_sq()`, and optional per-observation precisions.
//! The conductor composes those with the response point's weights
//! (`wᵢ = wᵢ^resp · wᵢ^scale`).
//!
//! A variance *ensemble* is an ordinary supplier: [`HVariance`] runs a second
//! instance of the shared backfit block inside `update`, sharing moves,
//! coordinate laws, assigner and count priors by construction.
//!
//! Shelf: [`GlobalSigma`] (the paper's draw, and the fit-time default),
//! [`PinnedSigma`] (σ² held constant, probit pins 1), [`WeightedGlobalSigma`]
//! (the precision-weighted sibling), [`HVariance`] (the H product-of-
//! tessellations ensemble). Template: `examples/template_scale.rs`.
//!
//! Conformance checks: `conformance::check_scale_model` (σ² validity,
//! precision well-formedness, bit-exact update determinism — none of which
//! says whether σ² responds to the data at all), plus
//! `check_scale_model_learns_from_data`, which you **opt in to** if your
//! model is meant to learn σ². It is separate on purpose: [`PinnedSigma`]
//! ignores the data *correctly*, and a mandatory probe would fail correct
//! code.
//!
//! Note this is the *scale model*, not the data scaler: the response/covariate
//! scaling and prior calibration live in the engine (`crate::engine::scaler`).
//!
//! Sources: [`GlobalSigma`] and its (ν, q, λ) calibration are BART's σ²
//! treatment (Chipman, George & McCulloch 2010) as adopted by the paper
//! (Stone & Gosling 2025); [`PinnedSigma`] is the probit identifiability
//! constraint (Albert & Chib 1993); [`HVariance`] is H-AddiVortes (Stone &
//! Gosling 2025b §3.2–3.3), whose parent is heteroscedastic BART via
//! multiplicative trees (Pratola, Chipman, George & McCulloch 2020);
//! [`WeightedGlobalSigma`] is the σ² step of Geweke's (1993) t-model Gibbs.

use std::sync::Arc;

use crate::engine::data::Data;
use crate::extensions::coord::CoordinateDistribution;
use crate::extensions::distance::CellAssigner;
use crate::extensions::moves::MoveSet;

mod global_sigma;
mod h_variance;
mod pinned;
mod weighted_global_sigma;

pub use global_sigma::GlobalSigma;
pub use h_variance::{HVariance, h_variance_prior};
pub use pinned::PinnedSigma;
pub use weighted_global_sigma::WeightedGlobalSigma;

// ---------------------------------------------------------------------------
// ScaleModel: the pluggable per-sweep scale/precision supplier
// ---------------------------------------------------------------------------

/// The per-sweep update context handed to [`ScaleModel::update`]:
/// the working response, the current mean-ensemble fit,
/// the scaled/encoded design, and the shared backfit machinery (move set,
/// assigner, coordinate laws, inclusion weights, count-prior parameters).
/// A variance *ensemble* (H-AddiVortes) is then an ordinary supplier, it
/// runs its own tessellations over the same design with the same structural
/// machinery inside `update`: not a fork of the sampler.
///
/// Construction is crate-internal (the conductor assembles one per sweep);
/// extension authors receive it in `update` and exercise their model through
/// [`check_scale_model`](crate::conformance::check_scale_model).
#[derive(Debug)]
pub struct ScaleCtx<'a> {
    pub(crate) y: &'a [f64],
    pub(crate) fit: &'a [f64],
    pub(crate) x: &'a Data,
    pub(crate) move_set: &'a MoveSet,
    pub(crate) assigner: &'a dyn CellAssigner,
    pub(crate) omega: f64,
    pub(crate) lambda_c: f64,
    pub(crate) nu: f64,
    pub(crate) lambda: f64,
    pub(crate) p_enc: usize,
    pub(crate) coord_dists: &'a [Arc<dyn CoordinateDistribution>],
    pub(crate) weights_enc: &'a [f64],
    pub(crate) response_weights: Option<&'a [f64]>,
}

impl<'a> ScaleCtx<'a> {
    /// The working response this sweep (**scaled space**, ascending index).
    pub fn y(&self) -> &'a [f64] {
        self.y
    }

    /// The current mean-ensemble fit (**scaled space**, ascending index).
    pub fn fit(&self) -> &'a [f64] {
        self.fit
    }

    /// The scaled, encoded design matrix (**scaled space**).
    pub fn x(&self) -> &'a Data {
        self.x
    }

    /// The shared structural move set: a variance ensemble
    /// shares it with the mean ensemble by construction.
    pub fn move_set(&self) -> &'a MoveSet {
        self.move_set
    }

    /// The shared cell assigner.
    pub fn assigner(&self) -> &'a dyn CellAssigner {
        self.assigner
    }

    /// Dimension-count prior parameter ω (a count-prior input, read by the
    /// default count-prior hook).
    pub fn omega(&self) -> f64 {
        self.omega
    }

    /// Centre-count prior parameter λ_c (a count-prior input, read by the
    /// default count-prior hook).
    pub fn lambda_c(&self) -> f64 {
        self.lambda_c
    }

    /// σ² prior degrees of freedom ν (dimensionless prior parameter: a
    /// count of degrees of freedom; config input).
    pub fn nu(&self) -> f64 {
        self.nu
    }

    /// The calibrated λ of the σ² ~ χ⁻²(ν, λ) prior (**scaled space**;
    /// `scale`'s one-time data calibration, or the pinned value on the
    /// statistical-gate path). Exposed so a scale model, e.g. the H
    /// variance ensemble's §3.3 (ν′, λ′) matching, calibrates consistently
    /// with the engine's own prior.
    pub fn calibrated_lambda(&self) -> f64 {
        self.lambda
    }

    /// Number of **encoded** covariates.
    pub fn p_enc(&self) -> usize {
        self.p_enc
    }

    /// Per-covariate coordinate laws, indexed by **global** encoded column,
    /// shared with the mean ensemble.
    pub fn coord_dists(&self) -> &'a [Arc<dyn CoordinateDistribution>] {
        self.coord_dists
    }

    /// Current inclusion weights, indexed by **global** encoded
    /// column, as of the previous sweep's inclusion update (the pinned hook
    /// order runs the scale update *before* this sweep's inclusion update).
    pub fn inclusion_weights(&self) -> &'a [f64] {
        self.weights_enc
    }

    /// This sweep's response-side accumulation weights
    /// (dimensionless relative weights, ascending index), as written by the
    /// [`ResponseModel::augment`](crate::response::ResponseModel::augment) hook that ran just before this update,
    /// `None` when no kernel step is attached (the working likelihood is
    /// unit-weight). A scale model whose σ² draw must stay consistent with
    /// a weight-producing response family reads them here (the shelf
    /// [`WeightedGlobalSigma`] prices the precision-weighted RSS Σᵢ wᵢeᵢ²).
    pub fn response_weights(&self) -> Option<&'a [f64]> {
        self.response_weights
    }
}

/// The pluggable scale/precision supplier: generalises the single
/// global σ² Gibbs draw. Updated once per sweep at the pinned point (after
/// [`ResponseModel::augment`](crate::response::ResponseModel::augment), before the inclusion update and the j-loop),
/// receiving a [`ScaleCtx`]: the scaled design plus the shared backfit
/// machinery, so a multiplicatively-composed variance ensemble
/// (H-AddiVortes) runs its own backfitting inside `update` as an ordinary
/// supplier. Beyond the scalar σ² the model may supply **per-observation
/// precisions**; the conductor composes them with the response-side
/// augmentation weights (`wᵢ = wᵢ^resp · wᵢ^scale`).
pub trait ScaleModel: std::fmt::Debug + Send + Sync {
    /// The extension-error channel: surfaced by the sampler
    /// as [`AddiVortesError::Extension`](crate::AddiVortesError::Extension).
    type Error: std::error::Error + Send + Sync + 'static;

    /// One per-sweep update given the update context (**scaled space**
    /// throughout, ascending index).
    fn update(
        &mut self,
        ctx: &ScaleCtx<'_>,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error>;

    /// The current error variance σ² (**scaled space**), must be finite and
    /// positive after every `update` (release-asserted by the sampler).
    fn sigma_sq(&self) -> f64;

    /// Per-observation precisions w_i^scale multiplying the response-side
    /// weights in the conjugate accumulation (`wᵢ = wᵢ^resp · wᵢ^scale`;
    /// dimensionless relative weights, ascending index,
    /// length n). `None`: the default, means homoscedastic: no
    /// per-observation term, no buffer, and the conductor composes nothing.
    /// Every value must be finite and strictly positive (validated by the
    /// conductor each sweep; a violation surfaces as
    /// [`AddiVortesError::Extension`](crate::AddiVortesError::Extension)).
    fn precisions(&self) -> Option<&[f64]> {
        None
    }

    /// Whether this model will ever publish [`precisions`](Self::precisions):
    /// `false` (homoscedastic) by default, `true` for a heteroscedastic model.
    ///
    /// This is *declared* rather than inferred because the two happen at
    /// different times. The engine must pick the matching mean-cell statistic
    /// when it **assembles** the sampler — per-observation precisions make the
    /// accumulation weights fractional, which the hard-assignment
    /// `GaussianCellStats` refuses — but `precisions()` is still `None` at that
    /// point for any model whose state is built lazily at the first update
    /// (`HVariance` is exactly such a model). Answering from `precisions()`
    /// would therefore always say "homoscedastic", and the mispairing would
    /// surface as a panic several sweeps later.
    ///
    /// Same role as [`CellModel::cell_basis`](crate::cell_model::CellModel::cell_basis):
    /// a claim the engine reads up front, so it can assemble the pairing the
    /// caller should never have to know about.
    fn heteroscedastic(&self) -> bool {
        false
    }
}

/// Mid-chain precision corruption from a custom [`ScaleModel`],
/// surfaced as [`AddiVortesError::Extension`], mirroring the inclusion point's
/// weight validation.
#[derive(Debug, thiserror::Error)]
#[error("scale model returned invalid precisions: {detail}")]
pub(crate) struct InvalidScalePrecisions {
    pub(crate) detail: String,
}

/// (shape, scale) of the Gamma draw whose reciprocal is the σ² Gibbs draw:
/// σ² ~ IG((ν+n)/2, (νλ+RSS)/2) realised as 1/Gamma(shape, scale) with
/// shape = (ν+n)/2 and scale = 2/(νλ+RSS).
pub(crate) fn sigma_sq_gamma_params(nu: f64, lambda: f64, rss: f64, n: usize) -> (f64, f64) {
    let shape = 0.5 * (nu + n as f64);
    let scale = 2.0 / (nu * lambda + rss);
    (shape, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::AddiVortesError;
    use crate::test_support::{assert_abs_eq, assert_rel_eq};

    /// The global-σ² models check (ν, λ) in every build profile; λ = 0 (the
    /// zero-residual calibration) is accepted.
    #[test]
    fn global_sigma_constructors_reject_out_of_domain_arguments() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                GlobalSigma::new(bad, 0.02),
                Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "nu"
            ));
            assert!(matches!(
                WeightedGlobalSigma::new(bad, 0.02),
                Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "nu"
            ));
        }
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                GlobalSigma::new(6.0, bad),
                Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "lambda"
            ));
            assert!(matches!(
                WeightedGlobalSigma::new(6.0, bad),
                Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "lambda"
            ));
        }
        assert!(GlobalSigma::new(6.0, 0.0).is_ok());
        assert!(WeightedGlobalSigma::new(6.0, 0.0).is_ok());
    }

    #[test]
    fn sigma_sq_gamma_params_match_hand_values() {
        // ν = 6, λ = 0.02, RSS = 1.5, n = 10:
        // shape = (6+10)/2 = 8; scale = 2/(6·0.02 + 1.5) = 2/1.62.
        let (shape, scale) = sigma_sq_gamma_params(6.0, 0.02, 1.5, 10);
        assert_abs_eq(shape, 8.0, 0.0);
        assert_rel_eq(scale, 2.0 / 1.62, 1e-15);
    }
}
