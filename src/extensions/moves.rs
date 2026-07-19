//! Structural MCMC moves: the [`ProposalMove`] trait
//! and the weighted [`MoveSet`] machinery (state-dependent selection and the
//! generic selection-probability ratio: the paper's folded move table, with
//! boundary corrections computed generically).
//! Each shelf approach is one file in `moves/`: `stone_gosling` holds the six
//! Stone & Gosling (2025) moves as one approach (the fit-time default). Start a
//! custom move from `examples/template_moves.rs`; the conformance check is
//! `conformance::check_move_set`.
//!
//! The sibling extension points live in their own modules: coordinate laws in
//! `crate::extensions::coord`, distance/assignment in `crate::extensions::distance`,
//! variable inclusion in `crate::extensions::inclusion`, and the deep cell-model
//! seam in `crate::extensions::cell_model`.
//!
//! Everything operates in **scaled, encoded** space; covariate indices are
//! always **global** encoded-column indices.
//!
//! Four facts to know before writing a move:
//!
//! - Proposals carry a structural-delta claim. [`Proposal::new`] claims
//!   nothing: cached assignments are fully recomputed, which is always
//!   correct. [`Proposal::with_delta`] declares the exact structural change
//!   so the assigner can update incrementally; the claim must be exact, and
//!   an inaccurate delta silently corrupts the chain. When in doubt, use
//!   `Proposal::new`.
//! - Custom moves change the kernel. As soon as any custom move is present,
//!   move selection switches from the paper's folded table to weight
//!   renormalisation over the currently-valid moves. You never write
//!   boundary corrections: the `MoveSet` computes selection-probability
//!   ratios generically.
//! - Count priors are priced through [`ModelCtx`], never hand-coded. A move
//!   that changes the cell or dimension count prices it through
//!   `ctx.log_cell_count_ratio(b)` / `ctx.log_dim_count_ratio(d)` (evaluated
//!   at the larger adjacent count; added by the growing move, subtracted by
//!   the shrinking one), never by writing the Poisson/Binomial forms
//!   directly. That keeps every move composable with a custom `CountPriors`;
//!   count-neutral moves have nothing to price under any hook.
//! - `log_structure_ratio` is strictly log[prior ratio × within-move
//!   proposal ratio]: no σ² term, no selection probabilities, no likelihood.
//!
//! Sources: the six moves are Stone & Gosling (2025); their lineage is BART's
//! grow/prune/change set (Chipman, George & McCulloch 2010) inside the
//! reversible-jump framework (Green 1995). The AC/RC acceptance ratio rests
//! on two separate cancellations: the sampled coordinates' prior against the
//! proposal density (prior ≡ proposal, with μ integrated out — Chipman,
//! George & McCulloch), and the reverse move's uniform pick against the
//! exchangeable orderings of the enlarged centre set (the labelled ↔
//! unlabelled multiplicity, the same order-statistics factor Richardson &
//! Green 1997 carry in their mixture birth-death move, eq. 12). There it
//! survives, paired against a pick over empty components only; cells here are
//! always non-empty, so the pick runs over all b+1 centres and the factor
//! cancels outright — Richardson & Green's empty-component argument is not
//! what holds the ratio together. See `stone_gosling::AddCentre` for the full
//! derivation.

mod registry;
mod stone_gosling;
#[cfg(test)]
mod tests;

pub(crate) use registry::default_move_set;
pub use registry::{ModelCtx, MoveSet, MoveSetBuilder};
pub use stone_gosling::{AddCentre, AddDimension, Change, RemoveCentre, RemoveDimension, Swap};

use crate::engine::tessellation::Tessellation;
use crate::extensions::distance::AssignmentDelta;

// ---------------------------------------------------------------------------
// The moves point, proposal moves
// ---------------------------------------------------------------------------

/// How a move names its reverse for detailed balance (validated as **mutual**
/// by `MoveSetBuilder::build`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reverse {
    /// The move is its own reverse (Change, Swap).
    SelfInverse,
    /// The reverse is the move registered under this name.
    Named(&'static str),
}

/// A proposed tessellation (passive record; fields stay private).
/// Cell μ values in the proposal are placeholders: the sampler redraws every μ
/// conjugately after the structural accept/reject.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    /// The proposed tessellation structure.
    pub tessellation: Tessellation,
    // Private: the constructors below keep custom moves
    // correct-by-default (`new` claims nothing → full recompute).
    delta: AssignmentDelta,
}

impl Proposal {
    /// A proposal carrying no structural-delta claim: cached assignments are
    /// fully recomputed. Always correct: the default for custom moves.
    pub fn new(tessellation: Tessellation) -> Self {
        Self {
            tessellation,
            delta: AssignmentDelta::FullRecompute,
        }
    }

    /// A proposal declaring the exact structural change it makes, letting the
    /// assigner update cached assignments incrementally. The claim
    /// must be **exact**: an inaccurate delta silently corrupts the chain.
    pub fn with_delta(tessellation: Tessellation, delta: AssignmentDelta) -> Self {
        Self {
            tessellation,
            delta,
        }
    }

    /// The declared structural delta (see [`AssignmentDelta`]).
    pub fn delta(&self) -> AssignmentDelta {
        self.delta
    }
}

/// A structural MCMC move. Object-safe; RNG enters only as
/// `&mut dyn rand_core::Rng`.
///
/// `log_structure_ratio` is **strictly** log[prior ratio × within-move proposal
/// ratio], no σ² term, no move-selection probabilities (the `MoveSet` owns
/// those), no likelihood (the sampler owns that).
pub trait ProposalMove: std::fmt::Debug + Send + Sync {
    /// Unique name within a `MoveSet`.
    fn name(&self) -> &'static str;
    /// The move's reverse, for mutual-pairing validation and selection ratios.
    fn reverse(&self) -> Reverse;
    /// Whether the move can fire from this state (all validity lives here,
    /// `propose` is infallible by design).
    fn is_valid(&self, tessellation: &Tessellation, ctx: &ModelCtx) -> bool;
    /// Draw a proposed tessellation. Only called when `is_valid` is true.
    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal;
    /// log[prior × within-move proposal] ratio for `old → proposed`.
    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64;
}

// ---------------------------------------------------------------------------
// Shared draw helpers (part of the pinned RNG-stream surface)
// ---------------------------------------------------------------------------

/// One uniform f64 in [0, 1): the top 53 bits of a single `next_u64` draw.
/// Pinned, changing this changes every chain.
pub(crate) fn uniform_f64(rng: &mut dyn rand_core::Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// One uniform index in `0..n` from a single draw (n ≥ 1).
pub(crate) fn uniform_index(n: usize, rng: &mut dyn rand_core::Rng) -> usize {
    debug_assert!(n >= 1);
    let i = (uniform_f64(rng) * n as f64) as usize;
    i.min(n - 1)
}

/// The dimension present in `after` but not `before` (AD incoming, Swap
/// incoming), if any.
pub(crate) fn dim_added(before: &[usize], after: &[usize]) -> Option<usize> {
    after.iter().copied().find(|d| !before.contains(d))
}
