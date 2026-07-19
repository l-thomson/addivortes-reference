//! **Response / likelihood**, *"my response isn't Gaussian."*
//!
//! Implement [`ResponseModel`]: once per sweep, first in the pinned hook order,
//! produce the Gaussian working response plus per-observation weights.
//!
//! **Scope rule:** a family fits iff augmentation restores the working form
//! `rᵢ | μ ~ N(μ, σ²/wᵢ)`, Albert–Chib probit, Pólya-Gamma logit,
//! Kozumi–Kobayashi quantile, robust-t scale mixtures. No augmentation means
//! out of scope, on purpose.
//!
//! Shelf: [`AlbertChibProbit`] (binary probit), [`RobustTStep`] (Student-t).
//! Template: `examples/template_response.rs`. A weight-producing step must
//! be paired with a weight-aware cell model and a weighted σ² draw: the
//! pairing rules are on the template, and `ResponseFamily` applies them for you.
//!
//! The conformance check is `conformance::check_response_model` (every
//! working entry written and finite, weights strictly positive, the
//! augmentation deterministic, and which pairing rule the weights put you
//! under). Statistical validity is the battery's job, not the check's: the
//! probit and robust-t families validate through the externalised battery
//! from outside the crate in `tests/calibration_acceptance.rs`.
//!
//! Sources: [`AlbertChibProbit`] is the Albert & Chib (1993) truncated-normal
//! augmentation, as used by Binary-AddiVortes (Stone, Ogundimu & Gosling
//! 2025) and probit BART before it (Chipman, George & McCulloch 2010 §4; the
//! σ_μ = 3/(k√m) latent widening is that section's eq. 24, which the Binary
//! paper reuses).
//! [`RobustTStep`] is the scale-mixture-of-normals representation of the t
//! (Andrews & Mallows 1974) with the Gibbs sampler of Geweke (1993). The
//! named-but-unbuilt siblings: Pólya–Gamma logit (Polson, Scott & Windle
//! 2013), quantile/ALD (Kozumi & Kobayashi 2011).

mod albert_chib;
mod robust_t;

pub use albert_chib::AlbertChibProbit;
pub use robust_t::RobustTStep;

// ---------------------------------------------------------------------------
// ResponseModel, latent-variable data augmentation
// ---------------------------------------------------------------------------

/// The data-augmentation hook (the deep seam): once per sweep, **first** in
/// the pinned hook order, produce the working response and per-observation
/// weights the kernel runs on. This is what keeps the response point cheap,
/// Albert–Chib probit fills `working` with truncated-normal latents
/// (weights ≡ 1, σ² pinned to 1); precision-weighted models fill `weights`
/// with 1/s²(xᵢ). When no step is configured the sampler runs directly on the
/// scaled response with **zero copies and zero RNG**: the default chain is
/// provably independent of this hook.
pub trait ResponseModel: std::fmt::Debug + Send + Sync {
    /// The extension-error channel: surfaced by the sampler
    /// as [`AddiVortesError::Extension`](crate::AddiVortesError::Extension).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fill `working` (the response the kernel sees this sweep) and `weights`
    /// (per-observation accumulation weights), given the **scaled** response
    /// `y`, the current ensemble fit and the previous sweep's σ² (all scaled
    /// space; on the very first sweep `sigma_sq` is the sampler's
    /// initialisation value, pin σ² inside the step if the model fixes it).
    #[allow(clippy::too_many_arguments)]
    fn augment(
        &mut self,
        y: &[f64],
        fit: &[f64],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
        working: &mut [f64],
        weights: &mut [f64],
    ) -> std::result::Result<(), Self::Error>;
}
