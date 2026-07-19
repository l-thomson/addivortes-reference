//! The move registry: the per-sweep model context handed to every move, and
//! the move set that selects among them (state-dependent selection plus the
//! generic selection-ratio correction).
//!
//! This is the moves point's machinery, not its trait, [`ProposalMove`] lives
//! in the parent module and is the only thing a contributor implements.

use std::sync::Arc;

use crate::engine::error::{AddiVortesError, Result};
use crate::engine::mathsfn;
use crate::engine::tessellation::Tessellation;
use crate::extensions::coord::CoordinateDistribution;
use crate::extensions::count_priors::{CountPriors, ShiftedPoissonBinomial};
use crate::extensions::moves::{ProposalMove, Reverse, uniform_f64};

// ---------------------------------------------------------------------------
// ModelCtx
// ---------------------------------------------------------------------------

/// Read-only model state handed to moves (public constructor so extension
/// moves are unit-testable). `#[non_exhaustive]`: fields may be added.
///
/// λ/σ̂ prior-calibration values are deliberately NOT here (`scale` owns
/// them); this is exactly the state a structural proposal may read.
#[non_exhaustive]
#[derive(Debug)]
pub struct ModelCtx<'a> {
    /// Current error variance σ² (scaled space).
    pub sigma_sq: f64,
    /// Dimension-count prior parameter ω (read by the default count-prior hook:
    /// d − 1 ~ Binomial(p − 1, ω/p)).
    pub omega: f64,
    /// Centre-count prior parameter λ_c (read by the default count-prior hook:
    /// b − 1 ~ Poisson(λ_c)).
    pub lambda_c: f64,
    /// Cell-output prior variance σ_μ² (scaled space).
    pub sigma_mu_sq: f64,
    /// Number of **encoded** covariates.
    pub p: usize,
    /// Per-covariate coordinate distributions, indexed by **global** encoded
    /// column index.
    pub coord_dists: &'a [Arc<dyn CoordinateDistribution>],
    /// Current inclusion weights, indexed by **global** encoded
    /// column index; strictly positive (validated at config). Relative values
    /// only, they need not sum to 1.
    pub weights: &'a [f64],
    /// The count-prior ratio hooks; defaults to the paper's
    /// [`ShiftedPoissonBinomial`]. Moves price count changes through
    /// [`log_cell_count_ratio`](ModelCtx::log_cell_count_ratio) and
    /// [`log_dim_count_ratio`](ModelCtx::log_dim_count_ratio), never through
    /// `lambda_c`/`omega` directly.
    pub count_priors: &'a dyn CountPriors,
}

impl<'a> ModelCtx<'a> {
    /// Assemble a context with the default count priors (the struct is
    /// `#[non_exhaustive]`, so this is the only way to build one outside the
    /// crate; override the count-prior point with
    /// [`with_count_priors`](ModelCtx::with_count_priors)).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sigma_sq: f64,
        omega: f64,
        lambda_c: f64,
        sigma_mu_sq: f64,
        p: usize,
        coord_dists: &'a [Arc<dyn CoordinateDistribution>],
        weights: &'a [f64],
    ) -> Self {
        Self {
            sigma_sq,
            omega,
            lambda_c,
            sigma_mu_sq,
            p,
            coord_dists,
            weights,
            count_priors: &ShiftedPoissonBinomial,
        }
    }

    /// Replace the count-prior hooks (last wins).
    #[must_use]
    pub fn with_count_priors(mut self, priors: &'a dyn CountPriors) -> Self {
        self.count_priors = priors;
        self
    }

    /// `ln P(b) − ln P(b−1)` of the cell-count prior, evaluated at the
    /// **larger** of the two adjacent cell counts `b` (log scale; delegates
    /// to the count-prior hook, see [`CountPriors`]).
    pub fn log_cell_count_ratio(&self, b: usize) -> f64 {
        self.count_priors.log_cell_count_ratio(b, self)
    }

    /// `ln P(d) − ln P(d−1)` of the dimension-count prior, evaluated at the
    /// **larger** of the two adjacent dimension counts `d` (log scale;
    /// delegates to the count-prior hook, see [`CountPriors`]).
    pub fn log_dim_count_ratio(&self, d: usize) -> f64 {
        self.count_priors.log_dim_count_ratio(d, self)
    }
}

/// How selection probabilities respond to invalid moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionRule {
    /// The paper's folded table (Appendix B): an invalid move's weight folds
    /// into its designated partner (RC→AC, RD→AD, AD→RD, Swap→Change), then
    /// the valid mass is normalised. The boundary log(0.5)/log(2) corrections
    /// **emerge** from this computation on *pre-move* state, never
    /// hand-coded, so an off-by-one-state error is structurally impossible.
    PaperFolded,
    /// Weight renormalisation over the valid moves only, used as soon as any
    /// custom move is present.
    WeightedValid,
}

/// The paper move set's fold partners by name (PaperFolded only).
fn fold_target_name(name: &str) -> Option<&'static str> {
    match name {
        "RemoveCentre" => Some("AddCentre"),
        "RemoveDimension" => Some("AddDimension"),
        "AddDimension" => Some("RemoveDimension"),
        "Swap" => Some("Change"),
        _ => None,
    }
}

/// A validated, weighted set of proposal moves. Owns state-dependent selection
/// and the generic selection-probability correction; users never write
/// boundary corrections.
#[derive(Debug)]
pub struct MoveSet {
    moves: Vec<Box<dyn ProposalMove>>,
    weights: Vec<f64>,
    /// Index of each move's reverse (self-index for self-inverse moves).
    reverse_of: Vec<usize>,
    /// Index each *invalid* move's weight folds into (PaperFolded only).
    fold_into: Vec<Option<usize>>,
    rule: SelectionRule,
}

impl MoveSet {
    /// Number of registered moves.
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Whether the set is empty (a built set never is).
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// The registered move names, in registration order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.moves.iter().map(|m| m.name())
    }

    /// The move at `index` (selection order = registration order).
    #[allow(dead_code)] // consumed by the sampler
    pub(crate) fn move_at(&self, index: usize) -> &dyn ProposalMove {
        self.moves[index].as_ref()
    }

    /// Index of the reverse of the move at `index`.
    #[allow(dead_code)] // consumed by the sampler
    pub(crate) fn reverse_index(&self, index: usize) -> usize {
        self.reverse_of[index]
    }

    /// Selection probabilities in `state` (normalised over the valid mass;
    /// all-zero only if no move is valid). Crate-visible for the oracle tests.
    pub(crate) fn selection_probs(&self, state: &Tessellation, ctx: &ModelCtx) -> Vec<f64> {
        let valid: Vec<bool> = self.moves.iter().map(|m| m.is_valid(state, ctx)).collect();
        let mut probs = vec![0.0_f64; self.moves.len()];
        for (i, &is_valid) in valid.iter().enumerate() {
            if is_valid {
                probs[i] += self.weights[i];
            } else if self.rule == SelectionRule::PaperFolded {
                if let Some(target) = self.fold_into[i] {
                    if valid[target] {
                        probs[target] += self.weights[i];
                    }
                }
            }
        }
        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            for probability in &mut probs {
                *probability /= total;
            }
        }
        probs
    }

    /// Draw a move index for `state` with a single uniform draw (ascending
    /// cumulative walk). `None` when no registered move is valid in `state`
    /// (impossible for the paper set, AddCentre and Change are always
    /// valid).
    pub fn select(
        &self,
        state: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Option<usize> {
        let probs = self.selection_probs(state, ctx);
        if probs.iter().sum::<f64>() <= 0.0 {
            return None;
        }
        let target = uniform_f64(rng);
        let mut cumulative = 0.0_f64;
        for (index, &probability) in probs.iter().enumerate() {
            cumulative += probability;
            if target < cumulative {
                return Some(index);
            }
        }
        // Top-edge rounding: the last move with positive probability wins.
        probs.iter().rposition(|&q| q > 0.0)
    }

    /// The generic selection-probability correction for the acceptance ratio:
    /// `ln q(reverse | proposed) − ln q(move | old)`. The boundary log(0.5)
    /// and log(2) values of the paper's Appendix B emerge from this, computed
    /// from **pre-move** state on both sides.
    pub fn log_selection_ratio(
        &self,
        move_index: usize,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        let forward = self.selection_probs(old, ctx)[move_index];
        let reverse = self.selection_probs(proposed, ctx)[self.reverse_of[move_index]];
        mathsfn::ln(reverse) - mathsfn::ln(forward)
    }
}

/// Builder for [`MoveSet`]: infallible appends, all validation in
/// [`build`](MoveSetBuilder::build).
#[derive(Debug, Default)]
pub struct MoveSetBuilder {
    pub(crate) moves: Vec<Box<dyn ProposalMove>>,
    pub(crate) weights: Vec<f64>,
    pub(crate) custom: bool,
}

impl MoveSetBuilder {
    /// An empty set to be filled with custom moves (selection rule:
    /// renormalisation over valid moves).
    #[allow(dead_code)] // public API surface (feature-gated export)
    pub fn empty() -> Self {
        Self {
            moves: Vec::new(),
            weights: Vec::new(),
            custom: true,
        }
    }

    /// Append a move (infallible, validation happens in `build`). Adding any
    /// move to the paper set switches selection to valid-move
    /// renormalisation.
    #[must_use]
    #[allow(dead_code)] // public API surface (feature-gated export)
    pub fn with_move(mut self, proposal_move: Box<dyn ProposalMove>, weight: f64) -> Self {
        self.push(proposal_move, weight);
        self.custom = true;
        self
    }

    pub(crate) fn push(&mut self, proposal_move: Box<dyn ProposalMove>, weight: f64) {
        self.moves.push(proposal_move);
        self.weights.push(weight);
    }

    /// Validate and build: unique names, strictly positive finite weights, and
    /// **mutual** reverse pairing (a move naming `R` as its reverse requires
    /// `R` to be registered and to name it back).
    pub fn build(self) -> Result<MoveSet> {
        if self.moves.is_empty() {
            return Err(AddiVortesError::InvalidMoveSet {
                move_name: "(none)".into(),
                reason: "is required: the move set is empty".into(),
            });
        }
        for (i, m) in self.moves.iter().enumerate() {
            if self.moves[..i].iter().any(|other| other.name() == m.name()) {
                return Err(AddiVortesError::InvalidMoveSet {
                    move_name: m.name().into(),
                    reason: "is registered more than once".into(),
                });
            }
            let w = self.weights[i];
            if !w.is_finite() || w <= 0.0 {
                return Err(AddiVortesError::InvalidMoveSet {
                    move_name: m.name().into(),
                    reason: format!("has invalid weight {w} (must be finite and positive)"),
                });
            }
        }

        let mut reverse_of = Vec::with_capacity(self.moves.len());
        for m in &self.moves {
            match m.reverse() {
                Reverse::SelfInverse => {
                    reverse_of.push(reverse_of.len());
                }
                Reverse::Named(reverse_name) => {
                    let Some(j) = self
                        .moves
                        .iter()
                        .position(|other| other.name() == reverse_name)
                    else {
                        return Err(AddiVortesError::InvalidMoveSet {
                            move_name: m.name().into(),
                            reason: format!(
                                "names `{reverse_name}` as its reverse, which is not registered"
                            ),
                        });
                    };
                    let mutual = match self.moves[j].reverse() {
                        Reverse::SelfInverse => reverse_name == m.name(),
                        Reverse::Named(back) => back == m.name(),
                    };
                    if !mutual {
                        return Err(AddiVortesError::InvalidMoveSet {
                            move_name: m.name().into(),
                            reason: format!(
                                "is not mutually paired: `{reverse_name}` does not name it back"
                            ),
                        });
                    }
                    reverse_of.push(j);
                }
            }
        }

        let rule = if self.custom {
            SelectionRule::WeightedValid
        } else {
            SelectionRule::PaperFolded
        };
        let fold_into: Vec<Option<usize>> = self
            .moves
            .iter()
            .map(|m| {
                fold_target_name(m.name())
                    .and_then(|t| self.moves.iter().position(|other| other.name() == t))
            })
            .collect();

        Ok(MoveSet {
            moves: self.moves,
            weights: self.weights,
            reverse_of,
            fold_into,
            rule,
        })
    }
}

/// This extension point's fit-time default move set. Owns the "the default is
/// Stone & Gosling" choice so the sampler resolves the default without naming a
/// concrete approach.
pub(crate) fn default_move_set() -> Result<MoveSet> {
    MoveSetBuilder::stone_gosling().build()
}
