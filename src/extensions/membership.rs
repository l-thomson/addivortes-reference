//! Membership: the
//! [`MembershipKernel`] trait (the researcher-facing surface of the dense
//! path) and its shelf (`membership/`): [`SoftmaxKernel`], the
//! softmax(−d²/τ) deterministic-weights entry with fixed τ.
//!
//! Soft membership is mostly a selection, not an implementation
//! (`AddiVortesConfig::with_membership` switches the mean ensemble onto the
//! dense path): the joint within-tessellation draw, the dense marginal, and
//! the predict-time weighting are engine code held to the cross-path oracle;
//! authoring a new blending recipe is this one small trait. No single
//! formulation is endorsed: the crate ships the
//! deterministic-weights family as an open research surface, each
//! candidate recipe is validated through the battery like any other
//! component. Latent-categorical soft assignment (per-sweep re-drawn hard
//! assignment) is deliberately out of scope: it makes assignment sampler
//! state rather than a function of X, which no seam expresses.
//!
//! The keys the kernel sees are the distance point's own comparison keys
//! (`CellAssigner::membership_keys`), so distance geometry composes with
//! softness by construction. Start from `examples/template_membership.rs`
//! (an inverse-quadratic blend); the conformance check is
//! `conformance::check_membership_kernel`.
//!
//! Sources: the closest literature is smooth/soft tree ensembles, SBART
//! (Linero & Yang 2018), where hard splits become logistic gates, with soft
//! decision trees (Irsoy, Yıldız & Alpaydın 2012) and fuzzy assignment
//! (Bezdek 1981) further back. [`SoftmaxKernel`]'s softmax(−d²/τ) over
//! Voronoi distance keys is this crate's transplant of that idea to
//! tessellations, which is why no formulation is endorsed and every recipe
//! earns its validity through the battery.
//!
//! The empty-cell guard under soft membership: a proposed structure is
//! rejected before the acceptance draw
//! iff any cell's total membership mass `Σᵢ φᵢₖ` is not strictly
//! positive: the direct generalisation of the hard guard ("some observation
//! supports every cell"), which it reduces to as τ → 0. Under
//! [`SoftmaxKernel`]'s shift-stable weights every observation's nearest cell
//! carries weight 1, so a cell fails the guard only when it wins no
//! meaningful mass from anywhere.
//!
//! τ handling: τ is a fixed hyperparameter of the kernel (scaled-space
//! squared-distance units), chosen by the researcher; CV/prior treatments
//! of τ are extension territory, deliberately not built in.

mod softmax;

pub use softmax::SoftmaxKernel;

/// The membership kernel: turn one observation's per-cell
/// comparison keys into unnormalised membership weights. The engine owns
/// everything around it: key computation through the distance assigner, row
/// normalisation to Σₖ φᵢₖ = 1, the guard, the dense conjugate algebra.
///
/// # Contract
///
/// - Pure and deterministic: same keys, same bits, no interior state,
///   no RNG (randomised membership is the out-of-scope latent-categorical
///   formulation).
/// - `keys` are the distance comparison keys (squared-distance scale, scaled
///   space) from one observation to every cell, ascending cell index.
/// - Every written weight must be finite and ≥ 0, with at least one > 0 per
///   row (be shift-stable: subtract the row's best key before
///   exponentiating, as [`SoftmaxKernel`] does; a kernel whose whole row
///   underflows to zero surfaces as [`AddiVortesError::Extension`]).
///
/// [`AddiVortesError::Extension`]: crate::AddiVortesError::Extension
pub trait MembershipKernel: std::fmt::Debug + Send + Sync {
    /// Write the unnormalised membership weight for every cell
    /// (dimensionless relative weights; `weights.len() == keys.len()`,
    /// ascending cell index).
    fn weights(&self, keys: &[f64], weights: &mut [f64]);
}
