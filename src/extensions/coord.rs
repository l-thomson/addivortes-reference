//! Centre-coordinate distributions: the
//! [`CoordinateDistribution`] trait; each shelf approach is one file in
//! `coord/`: [`EuclideanNormal`] (the paper's law) and [`WrappedNormal`].
//!
//! A coordinate law serves as both the prior and the within-move proposal.
//! The two coincide and cancel in every built-in ratio, as in the paper, so a
//! custom law is valid by construction. All values are in scaled space. Custom
//! laws are checked by `conformance::check_coordinate_distribution`; a worked
//! starting point is `examples/template_coord.rs`.
//!
//! Sources: [`EuclideanNormal`] is the paper's centre law (Stone & Gosling
//! 2025); [`WrappedNormal`] is the wrapped normal of directional statistics
//! (Mardia & Jupp 2000; N. I. Fisher 1993).

mod normal;
mod wrapped_normal;

pub use normal::EuclideanNormal;
pub use wrapped_normal::WrappedNormal;

/// A centre-coordinate law, used as both the prior and the within-move
/// proposal for that coordinate (they coincide and cancel in every built-in
/// ratio, as in the paper).
pub trait CoordinateDistribution: std::fmt::Debug + Send + Sync {
    /// Draw one coordinate (scaled space).
    fn sample(&self, rng: &mut dyn rand_core::Rng) -> f64;
    /// Natural log of the density at `x` (scaled space). Must integrate to 1
    /// over the coordinate's domain; the conformance check verifies that `sample`
    /// and `log_density` agree.
    fn log_density(&self, x: f64) -> f64;
}

/// Wrap a real angle into [−π, π] (deterministic; `rem_euclid` is exact IEEE
/// remainder arithmetic, no transcendentals).
pub(crate) fn wrap_to_pi(x: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let wrapped = (x + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI;
    // Rounding at the seam can land just outside; clamp back to the closed domain.
    wrapped.clamp(-std::f64::consts::PI, std::f64::consts::PI)
}
