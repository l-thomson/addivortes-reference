//! Stone & Gosling's (2025) move set, as one shelf entry (paper p.863): the
//! six structural moves (AddCentre/RemoveCentre, AddDimension/RemoveDimension,
//! Change and Swap) that
//! [`MoveSetBuilder::stone_gosling`](crate::extensions::moves::MoveSetBuilder::stone_gosling)
//! registers with the published weights under the folded selection table. This
//! is the fit-time default.
//!
//! A new research approach to this extension point is a sibling file (or an external
//! crate): implement [`ProposalMove`] and register it with
//! [`MoveSetBuilder`](crate::extensions::moves::MoveSetBuilder). Template:
//! `examples/template_moves.rs`.

use crate::engine::tessellation::Tessellation;
use crate::extensions::distance::AssignmentDelta;
use crate::extensions::inclusion::{
    log_add_dimension_correction, log_remove_dimension_correction, log_swap_correction,
    weighted_choice,
};
use crate::extensions::moves::{
    ModelCtx, MoveSetBuilder, Proposal, ProposalMove, Reverse, dim_added, uniform_index,
};

// ---------------------------------------------------------------------------
// AC: add a centre
// ---------------------------------------------------------------------------

/// Add a centre: sample one coordinate per active dimension from that
/// dimension's [`CoordinateDistribution`](crate::extensions::coord::CoordinateDistribution)
/// (pinned draw order: `dims` order) and append the new centre.
///
/// Structure ratio (b = current centre count): the cell-count prior
/// contributes P(b+1)/P(b), priced through the count-prior hook
/// ([`ModelCtx::log_cell_count_ratio`] at the enlarged count, default
/// shifted Poisson λ_c/b); the sampled coordinates' prior cancels against
/// the proposal density (prior ≡ proposal); and the reverse RemoveCentre's
/// uniform pick 1/(b+1) cancels against the b+1 exchangeable orderings of
/// the enlarged centre set (the labelled ↔ unlabelled multiplicity, the
/// same order-statistics factor as Richardson & Green 1997's mixture
/// birth-death move, eq. 12 — though their death picks among empty
/// components only, so their (k+1) survives where ours cancels; centres
/// here are exchangeable continuous marks and every cell is non-empty, so
/// the pick factor must not survive into the ratio). Net, at the default:
///
/// ```text
/// ln λ_c − ln b
/// ```
///
/// (Exactly the telescoped Poisson ratio. A spurious −ln(b+1) pick
/// term here would thin the sampled cell-count marginal to
/// P(b+1)/P(b) = λ_c/(b(b+1)) — and would pass every local
/// detailed-balance check, because the matching +ln b in the reverse move
/// cancels it; only a joint-distribution battery can see it.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AddCentre;

impl ProposalMove for AddCentre {
    fn name(&self) -> &'static str {
        "AddCentre"
    }

    fn reverse(&self) -> Reverse {
        Reverse::Named("RemoveCentre")
    }

    fn is_valid(&self, _tessellation: &Tessellation, _ctx: &ModelCtx) -> bool {
        true
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        let mut centres = tessellation.centres.clone();
        for &dim in &tessellation.dims {
            centres.push(ctx.coord_dists[dim].sample(rng));
        }
        // Placeholder payload for the new cell: the sampler redraws every cell
        // after accept/reject, so only the length matters. One value per cell
        // for the scalar families, q for a basis payload.
        let mut mus = tessellation.mus.clone();
        mus.extend(std::iter::repeat_n(0.0, tessellation.q()));
        Proposal::with_delta(
            Tessellation {
                centres,
                dims: tessellation.dims.clone(),
                mus,
            },
            AssignmentDelta::CentreAdded,
        )
    }

    fn log_structure_ratio(
        &self,
        _old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        ctx.log_cell_count_ratio(proposed.n_cells())
    }
}

// ---------------------------------------------------------------------------
// RC: remove a centre
// ---------------------------------------------------------------------------

/// Remove a uniformly-chosen centre. Valid only with at least two centres.
///
/// Structure ratio (b = current centre count): exact negative of
/// [`AddCentre`] from the smaller state: the count-prior hook subtracted at the
/// current count ([`ModelCtx::log_cell_count_ratio`]); at the default
/// shifted Poisson,
///
/// ```text
/// ln(b − 1) − ln λ_c
/// ```
///
/// (Poisson ratio P(b−2)/P(b−1) = (b−1)/λ_c; the removed centre's coordinate
/// prior cancels against the reverse AddCentre proposal density; and the
/// forward uniform pick 1/b cancels against the b exchangeable orderings of
/// the current centre set; see the AddCentre derivation note.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RemoveCentre;

impl ProposalMove for RemoveCentre {
    fn name(&self) -> &'static str {
        "RemoveCentre"
    }

    fn reverse(&self) -> Reverse {
        Reverse::Named("AddCentre")
    }

    fn is_valid(&self, tessellation: &Tessellation, _ctx: &ModelCtx) -> bool {
        tessellation.n_cells() >= 2
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        _ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        let d = tessellation.dims.len();
        let q = tessellation.q();
        let removed = uniform_index(tessellation.n_cells(), rng);
        let mut centres = tessellation.centres.clone();
        centres.drain(removed * d..(removed + 1) * d);
        // Drop the removed cell's whole payload block (q values; q = 1 scalar).
        let mut mus = tessellation.mus.clone();
        mus.drain(removed * q..(removed + 1) * q);
        Proposal::with_delta(
            Tessellation {
                centres,
                dims: tessellation.dims.clone(),
                mus,
            },
            AssignmentDelta::CentreRemoved { index: removed },
        )
    }

    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        _proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        -ctx.log_cell_count_ratio(old.n_cells())
    }
}

// ---------------------------------------------------------------------------
// AD: add a dimension (Binomial(p−1) count prior)
// ---------------------------------------------------------------------------

/// Add a dimension: choose the incoming covariate among the unused ones by the
/// single-draw weighted choice (the inclusion point: uniform under `UniformInclusion`,
/// exactly the paper), then sample one coordinate for every centre from the
/// incoming covariate's distribution (pinned draw order: choice first, then
/// coordinates in ascending centre index). The new dimension is appended to
/// `dims`.
///
/// Structure ratio (d = current dimension count): the dimension-count prior
/// contributes P(d+1)/P(d), priced through the count-prior hook
/// ([`ModelCtx::log_dim_count_ratio`] at the enlarged count); under uniform
/// weights the subset-prior ratio and both within-move pick probabilities
/// collapse, leaving (at the default shifted Binomial(p−1, θ), θ = ω/p) the
/// paper value
///
/// ```text
/// ln(p − d) − ln d + ln θ − ln(1 − θ)
/// ```
///
/// plus the weight-generic subset correction
/// `log_add_dimension_correction`, which is exactly 0 at uniform weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AddDimension;

impl ProposalMove for AddDimension {
    fn name(&self) -> &'static str {
        "AddDimension"
    }

    fn reverse(&self) -> Reverse {
        Reverse::Named("RemoveDimension")
    }

    fn is_valid(&self, tessellation: &Tessellation, ctx: &ModelCtx) -> bool {
        tessellation.dims.len() < ctx.p
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        let old_d = tessellation.dims.len();
        // Unused covariates in ascending global order (deterministic candidate list).
        let unused: Vec<usize> = (0..ctx.p)
            .filter(|c| !tessellation.dims.contains(c))
            .collect();
        let incoming = unused[weighted_choice(&unused, ctx.weights, rng)];

        let mut dims = tessellation.dims.clone();
        dims.push(incoming);
        let n_cells = tessellation.n_cells();
        let mut centres = Vec::with_capacity(n_cells * (old_d + 1));
        for cell in 0..n_cells {
            centres.extend_from_slice(&tessellation.centres[cell * old_d..(cell + 1) * old_d]);
            centres.push(ctx.coord_dists[incoming].sample(rng));
        }
        // `dims` changed: every cached pairwise distance is stale, so the
        // delta stays at the correct-by-default FullRecompute.
        Proposal::new(Tessellation {
            centres,
            dims,
            mus: tessellation.mus.clone(),
        })
    }

    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        let incoming = dim_added(&old.dims, &proposed.dims)
            .expect("AddDimension proposal must add a dimension");
        ctx.log_dim_count_ratio(proposed.dims.len())
            + log_add_dimension_correction(&old.dims, incoming, ctx.weights)
    }
}

// ---------------------------------------------------------------------------
// RD: remove a dimension (Binomial(p−1) count prior)
// ---------------------------------------------------------------------------

/// Remove a dimension: choose the outgoing covariate among the active ones by
/// the single-draw weighted choice (the inclusion point: uniform under
/// `UniformInclusion`, exactly the paper) and delete its coordinate from every
/// centre. Valid only with at least two active dimensions.
///
/// Structure ratio (d = current dimension count): the exact negative of
/// [`AddDimension`] from the smaller state: the count-prior hook subtracted at
/// the current count ([`ModelCtx::log_dim_count_ratio`]); at the default
/// shifted Binomial(p−1, θ), θ = ω/p,
///
/// ```text
/// ln(d − 1) − ln(p − d + 1) − ln θ + ln(1 − θ)
/// ```
///
/// plus the weight-generic subset correction
/// `log_remove_dimension_correction`, exactly 0 at uniform weights.
/// (The default Binomial count prior uses p − 1 trials: the paper's
/// 1/C(p, d) subset factor cancels the uniform pick exactly; pricing p
/// trials instead would break AD×RD detailed balance.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RemoveDimension;

impl ProposalMove for RemoveDimension {
    fn name(&self) -> &'static str {
        "RemoveDimension"
    }

    fn reverse(&self) -> Reverse {
        Reverse::Named("AddDimension")
    }

    fn is_valid(&self, tessellation: &Tessellation, _ctx: &ModelCtx) -> bool {
        tessellation.dims.len() >= 2
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        let old_d = tessellation.dims.len();
        let out_position = weighted_choice(&tessellation.dims, ctx.weights, rng);

        let mut dims = tessellation.dims.clone();
        dims.remove(out_position);
        let n_cells = tessellation.n_cells();
        let mut centres = Vec::with_capacity(n_cells * (old_d - 1));
        for cell in 0..n_cells {
            for di in 0..old_d {
                if di != out_position {
                    centres.push(tessellation.centres[cell * old_d + di]);
                }
            }
        }
        // `dims` changed: every cached pairwise distance is stale, so the
        // delta stays at the correct-by-default FullRecompute.
        Proposal::new(Tessellation {
            centres,
            dims,
            mus: tessellation.mus.clone(),
        })
    }

    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        let outgoing = dim_added(&proposed.dims, &old.dims)
            .expect("RemoveDimension proposal must remove a dimension");
        -ctx.log_dim_count_ratio(old.dims.len())
            + log_remove_dimension_correction(&old.dims, outgoing, ctx.weights)
    }
}

// ---------------------------------------------------------------------------
// Change: move a centre
// ---------------------------------------------------------------------------

/// Change a centre's position: pick a centre uniformly and resample all of its
/// coordinates from the per-dimension distributions (pinned draw order: pick
/// first, then coordinates in `dims` order).
///
/// Structure ratio: 0. The pick probability is 1/b both ways, the old/new
/// coordinate priors cancel against the reverse/forward proposal densities
/// (prior ≡ proposal), and both counts are unchanged, so the count priors
/// contribute P(b)/P(b) = P(d)/P(d) = 1 for any hook, nothing to price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Change;

impl ProposalMove for Change {
    fn name(&self) -> &'static str {
        "Change"
    }

    fn reverse(&self) -> Reverse {
        Reverse::SelfInverse
    }

    fn is_valid(&self, _tessellation: &Tessellation, _ctx: &ModelCtx) -> bool {
        true
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        let d = tessellation.dims.len();
        let cell = uniform_index(tessellation.n_cells(), rng);
        let mut centres = tessellation.centres.clone();
        for (di, &dim) in tessellation.dims.iter().enumerate() {
            centres[cell * d + di] = ctx.coord_dists[dim].sample(rng);
        }
        Proposal::with_delta(
            Tessellation {
                centres,
                dims: tessellation.dims.clone(),
                mus: tessellation.mus.clone(),
            },
            AssignmentDelta::CentreMoved { index: cell },
        )
    }

    fn log_structure_ratio(
        &self,
        _old: &Tessellation,
        _proposed: &Tessellation,
        _ctx: &ModelCtx,
    ) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Swap: exchange an active covariate for an unused one
// ---------------------------------------------------------------------------

/// Swap a covariate: pick the outgoing dimension uniformly (paper), the
/// incoming covariate among the unused ones by the single-draw weighted
/// choice, and resample the affected coordinate of every centre from the
/// incoming covariate's distribution (pinned draw order: outgoing, incoming,
/// then coordinates in ascending centre index). The incoming covariate takes
/// the outgoing one's slot in `dims`. Valid only when an unused covariate
/// exists (d < p).
///
/// Structure ratio: 0 under uniform weights: the dimension count is
/// unchanged (so the count prior contributes P(d)/P(d) = 1 for
/// any hook), the uniform-subset prior ratio is 1, and the pick
/// probabilities mirror each other. The weight-generic correction is
/// `log_swap_correction` (exactly 0 at uniform weights).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Swap;

impl ProposalMove for Swap {
    fn name(&self) -> &'static str {
        "Swap"
    }

    fn reverse(&self) -> Reverse {
        Reverse::SelfInverse
    }

    fn is_valid(&self, tessellation: &Tessellation, ctx: &ModelCtx) -> bool {
        tessellation.dims.len() < ctx.p
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        let d = tessellation.dims.len();
        let out_position = uniform_index(d, rng);
        let unused: Vec<usize> = (0..ctx.p)
            .filter(|c| !tessellation.dims.contains(c))
            .collect();
        let incoming = unused[weighted_choice(&unused, ctx.weights, rng)];

        let mut dims = tessellation.dims.clone();
        dims[out_position] = incoming;
        let mut centres = tessellation.centres.clone();
        for cell in 0..tessellation.n_cells() {
            centres[cell * d + out_position] = ctx.coord_dists[incoming].sample(rng);
        }
        // `dims` changed: every cached pairwise distance is stale, so the
        // delta stays at the correct-by-default FullRecompute.
        Proposal::new(Tessellation {
            centres,
            dims,
            mus: tessellation.mus.clone(),
        })
    }

    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        let incoming =
            dim_added(&old.dims, &proposed.dims).expect("Swap proposal must introduce a covariate");
        let outgoing =
            dim_added(&proposed.dims, &old.dims).expect("Swap proposal must retire a covariate");
        log_swap_correction(&old.dims, outgoing, incoming, ctx.weights)
    }
}

// ---------------------------------------------------------------------------
// The Stone & Gosling move set (the fit-time default)
// ---------------------------------------------------------------------------

impl MoveSetBuilder {
    /// Stone & Gosling's (2025) six moves with their published weights
    /// (AC/RC/AD/RD 0.2, Change/Swap 0.1) under the folded selection table:
    /// the complete `stone_gosling` shelf approach, defined here next to the
    /// moves it combines. This is the fit-time default move set (the machinery
    /// in `moves.rs` never names it; one file holds the whole approach,
    /// recipe and weights included).
    pub fn stone_gosling() -> Self {
        let mut builder = Self {
            moves: Vec::new(),
            weights: Vec::new(),
            custom: false,
        };
        builder.push(Box::new(AddCentre), 0.2);
        builder.push(Box::new(RemoveCentre), 0.2);
        builder.push(Box::new(AddDimension), 0.2);
        builder.push(Box::new(RemoveDimension), 0.2);
        builder.push(Box::new(Change), 0.1);
        builder.push(Box::new(Swap), 0.1);
        builder
    }
}
