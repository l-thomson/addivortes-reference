//! The paper's count priors: shifted Poisson cell counts, shifted Binomial
//! dimension counts: the fit-time default.

use crate::engine::mathsfn;
use crate::extensions::count_priors::CountPriors;
use crate::extensions::moves::ModelCtx;

/// The paper's count priors, and the fit-time default: shifted Poisson cells
/// (b − 1 ~ Poisson(λ_c)) and shifted Binomial dimensions
/// (d − 1 ~ Binomial(p − 1, ω/p): **p − 1** trials; pricing p instead breaks
/// AD×RD detailed balance). Parameters are read from the context (`lambda_c`,
/// `omega`, `p`), so this hook carries no state of its own.
///
/// Selecting it explicitly reproduces the unset-default chain bit for bit
/// (a standing gate asserts this).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShiftedPoissonBinomial;

impl CountPriors for ShiftedPoissonBinomial {
    fn log_cell_count_ratio(&self, b: usize, ctx: &ModelCtx) -> f64 {
        // Telescoped Poisson ratio: P(b)/P(b−1) = λ_c/(b−1).
        mathsfn::ln(ctx.lambda_c) - mathsfn::ln((b - 1) as f64)
    }

    fn log_dim_count_ratio(&self, d: usize, ctx: &ModelCtx) -> f64 {
        // Binomial(p−1, θ) count ratio, θ = ω/p:
        // P(d)/P(d−1) = ((p − d + 1)/(d − 1)) · θ/(1 − θ).
        let p = ctx.p as f64;
        let d = d as f64;
        let theta = ctx.omega / p;
        mathsfn::ln(p - d + 1.0) - mathsfn::ln(d - 1.0) + mathsfn::ln(theta)
            - mathsfn::ln(1.0 - theta)
    }
}
