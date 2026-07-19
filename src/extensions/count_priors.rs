//! **Count priors**, *"I don't want Poisson cell counts / Binomial dimension counts."*
//!
//! Implement [`CountPriors`]: two ratio hooks on the model context
//! (`log_cell_count_ratio` = log P(b)/P(b−1), and `log_dim_count_ratio`), each
//! evaluated at the larger adjacent count. Select on the config with
//! `AddiVortesConfig::with_count_priors`, then `.fit()` as normal;
//! `ModelCtx::with_count_priors` overrides a single context directly.
//!
//! Every move prices structures through these hooks, so **a custom count prior
//! never touches a move**: and count-neutral moves have nothing to price
//! (`P(b)/P(b) = 1` under any prior). This is deliberately as far as the
//! "structure prior" is centralised: subset/selection corrections stay on the
//! inclusion point, and every remaining prior ≡ proposal cancellation is
//! interlocked per move. There is no `StructurePrior` mega-component, on
//! purpose: the per-move cancellations must stay per-move.
//!
//! Shelf: [`ShiftedPoissonBinomial`]: the paper's shifted Poisson(λ_c) cell
//! count and shifted Binomial(p−1, ω/p) dimension count, and the fit-time
//! default (bit-identical to the pre-hook pricing; an in-crate gate asserts
//! that selecting the default *explicitly* reproduces the unset chain bit
//! for bit).
//! Template: `examples/template_count_priors.rs`. Conformance checks:
//! `conformance::check_count_priors` (both hooks finite and pure over every
//! adjacent count a sampler can ask about, plus a portability digest) and
//! `check_count_priors_against_density`, where you supply the prior a second
//! time as an unnormalised log-pmf and each hook must be its adjacent
//! difference — the one check in the crate whose reference does not come
//! from the code under test. Neither can prove the ratios describe a
//! normalisable prior: only SBC sees that, so a custom count prior destined
//! for real inference needs its own battery configuration.
//!
//! `ModelCtx`'s parameters are readable out of crate (`lambda_c`, `omega`,
//! `p` and the rest are public fields; `#[non_exhaustive]` blocks
//! construction, not reads), so a prior may take them from the context or
//! carry its own.
//!
//! Sources: the shifted Poisson(λ_c) / shifted Binomial(p−1, ω/p) defaults
//! are the paper's tessellation-size priors (Stone & Gosling 2025); the hook
//! design is this crate's engineering.

use crate::extensions::moves::ModelCtx;

mod shifted_poisson_binomial;

pub use shifted_poisson_binomial::ShiftedPoissonBinomial;

/// The count-prior point: the two count-prior ratio hooks. The count
/// priors factor cleanly out of every move's acceptance ratio, so they are
/// supplied here once and every move prices structures through them: a
/// custom count prior never touches a move. This is deliberately as far as
/// the "structure prior" is centralised: subset/selection corrections stay in
/// the inclusion point, and every remaining prior ≡ proposal cancellation is
/// interlocked per move (there is no `StructurePrior` mega-component, on
/// purpose).
///
/// Contract: each hook returns the log prior ratio between two **adjacent**
/// counts, evaluated at the larger one; both hooks must be pure and
/// deterministic, and any transcendentals must route through
/// `addivortes::mathsfn` (std float results are platform-dependent and
/// break bit-exact reproducibility). Moves whose
/// proposal leaves a count unchanged need no hook call: `P(b)/P(b) = 1` for
/// *any* count prior, so the term vanishes generically.
///
/// `Send + Sync` because a selected prior is held on the (shared)
/// configuration and read from the sampler, exactly as on every other
/// extension point.
pub trait CountPriors: std::fmt::Debug + Send + Sync {
    /// `ln P(b) − ln P(b−1)` of the cell-count prior at the larger count `b`
    /// (`b ≥ 2`; log scale). AddCentre adds this at the proposed count;
    /// RemoveCentre subtracts it at the current count.
    fn log_cell_count_ratio(&self, b: usize, ctx: &ModelCtx) -> f64;
    /// `ln P(d) − ln P(d−1)` of the dimension-count prior at the larger count
    /// `d` (`2 ≤ d ≤ p`; log scale). AddDimension adds this at the proposed
    /// count; RemoveDimension subtracts it at the current count.
    fn log_dim_count_ratio(&self, d: usize, ctx: &ModelCtx) -> f64;
}
