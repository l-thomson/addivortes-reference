//! Template for a custom structural move: copy this file, rename
//! the moves, fill the marked blocks, and run:
//!
//! ```sh
//! cargo run --example template_moves
//! ```
//!
//! The file contains three worked patterns, and the third is the one to read
//! if your move is not a variation on a shipped one:
//! - `Relocate`: a self-inverse move (its own reverse), the simplest shape;
//! - `Grow`/`Shrink`: a mutually-paired couple (each names the other), the
//!   shape used for anything that changes a count;
//! - `NudgeCentre`: a move whose **proposal is not the prior**. Every move on
//!   the shipped shelf samples coordinates straight from the coordinate law, so
//!   the prior cancels against the proposal density and the coordinate term
//!   disappears. That cancellation is a property of *those proposals*, not a
//!   rule of the seam. A random walk does not get it, and its ratio carries a
//!   real coordinate term.
//!
//! Everything runs against the one-command check at the bottom: reverse-pair
//! ratio identities, selection symmetry, and the proposal contract.
//!
//! Read `src/conformance.rs`'s module note before you trust a green run. The
//! reverse-pair identity compares your move against **its own reverse**, so a
//! derivation that is wrong in a *mirrored* way cancels and passes anyway — as
//! the published add/remove-centre ratio did (see `Grow` below). Statistical
//! validity comes from the Geweke/SBC battery (`calibration::getting_it_right`);
//! `tests/calibration_acceptance.rs` is a copyable, out-of-crate worked example.

use std::sync::Arc;

use addivortes::{
    AssignmentDelta, CoordinateDistribution, EuclideanNormal, ModelCtx, MoveSetBuilder, Proposal,
    ProposalMove, Reverse, Tessellation,
};
use addivortes::{conformance, mathsfn};

// ---------------------------------------------------------------------------
// Pattern 1: a self-inverse move
// ---------------------------------------------------------------------------

/// Relocate one centre by resampling all of its coordinates from the
/// per-covariate distributions (prior ≡ proposal, so the densities cancel).
#[derive(Debug)]
struct Relocate;

impl ProposalMove for Relocate {
    fn name(&self) -> &'static str {
        "Relocate"
    }

    /// Relocate undoes itself: another Relocate can restore the old state.
    fn reverse(&self) -> Reverse {
        Reverse::SelfInverse
    }

    /// All validity logic lives here; `propose` is only called when this is
    /// true and must then be infallible.
    fn is_valid(&self, _tessellation: &Tessellation, _ctx: &ModelCtx) -> bool {
        true
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        // ----- your proposal logic here ------------------------------------
        // Pick a centre with one uniform draw, then resample its coordinates.
        let n_cells = tessellation.n_cells();
        let d = tessellation.dims().len();
        let cell =
            ((rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) * n_cells as f64) as usize;
        let cell = cell.min(n_cells - 1);
        let mut centres = tessellation.centres().to_vec();
        for (di, &dim) in tessellation.dims().iter().enumerate() {
            // Look distributions up by global covariate index, never by the
            // local slot position (a classic bug the test suite injects for).
            centres[cell * d + di] = ctx.coord_dists[dim].sample(rng);
        }
        // --------------------------------------------------------------------
        // Declaring the structural delta lets the assigner update cached
        // assignments incrementally instead of recomputing them. The claim
        // must be exact; when in doubt, `Proposal::new` (no claim, full
        // recompute) is always correct.
        Proposal::with_delta(
            Tessellation::new(
                centres,
                tessellation.dims().to_vec(),
                tessellation.mus().to_vec(),
            )
            .expect("structurally unchanged"),
            AssignmentDelta::CentreMoved { index: cell },
        )
    }

    fn log_structure_ratio(
        &self,
        _old: &Tessellation,
        _proposed: &Tessellation,
        _ctx: &ModelCtx,
    ) -> f64 {
        // ----- your structure-ratio here ------------------------------------
        // log[prior ratio × within-move proposal ratio], and nothing else:
        // no σ² term, no selection probabilities, no likelihood.
        // Here: the pick probability is 1/b both ways and the coordinate
        // prior cancels against the proposal density → exactly 0.
        0.0
        // --------------------------------------------------------------------
    }
}

// ---------------------------------------------------------------------------
// Pattern 2: a mutually-paired couple
// ---------------------------------------------------------------------------

/// Grow: append a centre sampled from the coordinate distributions.
/// (The same maths as the built-in AddCentre, kept explicit so the
/// derivation is visible as a worked example.)
///
/// Read the `log_structure_ratio` derivation below before writing your own: the
/// tempting `1/(b+1)` reverse-pick term is **wrong**, and it is wrong in a way
/// no local check can see.
#[derive(Debug)]
struct Grow;

/// Shrink: remove a uniformly-chosen centre. The reverse of Grow.
#[derive(Debug)]
struct Shrink;

impl ProposalMove for Grow {
    fn name(&self) -> &'static str {
        "Grow"
    }
    fn reverse(&self) -> Reverse {
        Reverse::Named("Shrink") // mutual: Shrink names Grow back
    }
    fn is_valid(&self, _t: &Tessellation, _ctx: &ModelCtx) -> bool {
        true
    }
    fn propose(&self, t: &Tessellation, ctx: &ModelCtx, rng: &mut dyn rand_core::Rng) -> Proposal {
        let mut centres = t.centres().to_vec();
        for &dim in t.dims() {
            centres.push(ctx.coord_dists[dim].sample(rng));
        }
        // Placeholder payload for the new cell: the sampler redraws every cell
        // after accept/reject, so only the *length* matters. Extend by `q()`,
        // not by 1: a scalar family has q = 1, a basis payload carries
        // q coefficients per cell.
        let mut mus = t.mus().to_vec();
        mus.extend(std::iter::repeat_n(0.0, t.q()));
        Proposal::with_delta(
            Tessellation::new(centres, t.dims().to_vec(), mus).expect("coherent"),
            AssignmentDelta::CentreAdded, // appended at the highest index
        )
    }
    fn log_structure_ratio(
        &self,
        _old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        // Worked derivation (b = old centre count). Price the count change
        // through the count-prior hook, never through `ctx.lambda_c` directly: that
        // way a custom count prior composes for free, and this stays identical
        // to the built-in AddCentre by construction.
        //
        //   prior:    P(b+1)/P(b), the hook, evaluated at the LARGER count.
        //             At the default (b−1 ~ Poisson(λ_c)) that is ln λ_c − ln b.
        //   proposal: the sampled coordinates cancel (prior ≡ proposal).
        //
        // AND NOTHING ELSE. In particular there is **no** `−ln(b+1)` term for
        // the reverse Shrink's uniform pick of which centre to drop. The pick
        // probability cancels against the (b+1) exchangeable orderings of the
        // enlarged centre set: centres are exchangeable continuous marks, so
        // the labelled↔unlabelled multiplicity exactly absorbs it (the
        // Richardson & Green 1997 birth–death cancellation).
        //
        // Keeping the pick factor thins the sampled cell-count marginal to
        // P(b+1)/P(b) = λ_c/(b(b+1)) instead of λ_c/b, so tessellations come
        // out far smaller than the prior you stated.
        //
        // NOTE FOR YOUR OWN MOVE: `check_move_set` CANNOT catch this. It checks
        // that Grow and Shrink cancel against each other, and a pick factor
        // dropped into both halves cancels perfectly. Only the Geweke/SBC
        // battery (`calibration`) sees this class of error.
        ctx.log_cell_count_ratio(proposed.n_cells())
    }
}

impl ProposalMove for Shrink {
    fn name(&self) -> &'static str {
        "Shrink"
    }
    fn reverse(&self) -> Reverse {
        Reverse::Named("Grow")
    }
    fn is_valid(&self, t: &Tessellation, _ctx: &ModelCtx) -> bool {
        t.n_cells() >= 2 // never empty the tessellation
    }
    fn propose(&self, t: &Tessellation, _ctx: &ModelCtx, rng: &mut dyn rand_core::Rng) -> Proposal {
        let n_cells = t.n_cells();
        let d = t.dims().len();
        let removed =
            ((rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) * n_cells as f64) as usize;
        let removed = removed.min(n_cells - 1);
        let q = t.q();
        let mut centres = t.centres().to_vec();
        centres.drain(removed * d..(removed + 1) * d);
        // Drop the removed cell's whole payload block (q values, not one).
        let mut mus = t.mus().to_vec();
        mus.drain(removed * q..(removed + 1) * q);
        Proposal::with_delta(
            Tessellation::new(centres, t.dims().to_vec(), mus).expect("coherent"),
            AssignmentDelta::CentreRemoved { index: removed },
        )
    }
    fn log_structure_ratio(&self, old: &Tessellation, _p: &Tessellation, ctx: &ModelCtx) -> f64 {
        // The exact negative of Grow evaluated from the smaller state: the same
        // hook, subtracted at the current (larger) count. At the default that is
        // ln(b−1) − ln λ_c. Again: no `+ln b` forward-pick term, for the same
        // exchangeability reason as Grow.
        //
        // Detailed balance holds *by construction* because this is literally
        // `-Grow`: derive one side, negate it, and the pair cannot drift apart.
        // (That is also why `check_move_set`'s cancellation test proves so
        // little — see Grow.)
        -ctx.log_cell_count_ratio(old.n_cells())
    }
}

// ---------------------------------------------------------------------------
// Pattern 3: a move whose proposal is NOT the prior
// ---------------------------------------------------------------------------

/// Nudge one centre coordinate by a Gaussian random walk of spread `step_sd`.
///
/// This is the pattern to copy if your move is genuinely new, and the one the
/// other two cannot teach. `Relocate`, `Grow` and every move on the shipped
/// shelf resample coordinates *from the coordinate law itself*, so the prior
/// cancels against the proposal density and the coordinate term vanishes. A
/// random walk gets no such gift: the proposal is symmetric (so *it* cancels),
/// but the prior does not, and `log_structure_ratio` carries a real,
/// non-vanishing coordinate term.
///
/// Copying `Relocate`'s `0.0` into this move would be wrong, would look
/// entirely reasonable, and **passes `check_move_set` green** (verified). The
/// reverse-pair identity asks whether `f(a→b) + f(b→a) = 0`, and `0.0` satisfies
/// that as perfectly as the correct ratio does — every antisymmetric expression
/// does, including the trivial one. So the check cannot distinguish "the prior
/// genuinely cancels" from "I forgot the prior".
///
/// That is the whole reason this pattern is in the template: the seam's own
/// check cannot teach it to you, so the template has to.
#[derive(Debug)]
struct NudgeCentre {
    step_sd: f64,
}

impl NudgeCentre {
    fn new(step_sd: f64) -> Self {
        assert!(
            step_sd.is_finite() && step_sd > 0.0,
            "step_sd must be finite and strictly positive"
        );
        Self { step_sd }
    }
}

/// One uniform on [0, 1).
fn unit_uniform(rng: &mut dyn rand_core::Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// One standard normal, Box–Muller through the pinned `mathsfn` transcendentals.
/// A `rand_distr` draw would route through std `f64` `ln`/`cos`, whose results
/// are platform-dependent, and void the same-seed promise across targets.
fn standard_normal(rng: &mut dyn rand_core::Rng) -> f64 {
    let u1 = unit_uniform(rng).max(f64::MIN_POSITIVE); // ln(0) guard
    let u2 = unit_uniform(rng);
    // sqrt is IEEE-754 correctly rounded, so it needs no pinning.
    (-2.0 * mathsfn::ln(u1)).sqrt() * mathsfn::cos(2.0 * std::f64::consts::PI * u2)
}

impl ProposalMove for NudgeCentre {
    fn name(&self) -> &'static str {
        "NudgeCentre"
    }

    /// A nudge of −δ undoes a nudge of +δ, and the walk is symmetric.
    fn reverse(&self) -> Reverse {
        Reverse::SelfInverse
    }

    fn is_valid(&self, _tessellation: &Tessellation, _ctx: &ModelCtx) -> bool {
        true // every tessellation has at least one coordinate to nudge
    }

    fn propose(
        &self,
        tessellation: &Tessellation,
        _ctx: &ModelCtx,
        rng: &mut dyn rand_core::Rng,
    ) -> Proposal {
        // ----- your proposal logic here ------------------------------------
        // One uniform to pick a flat coordinate slot, one normal for the step.
        let d = tessellation.dims().len();
        let n_coords = tessellation.centres().len();
        let pick = ((unit_uniform(rng) * n_coords as f64) as usize).min(n_coords - 1);

        let mut centres = tessellation.centres().to_vec();
        centres[pick] += self.step_sd * standard_normal(rng);
        // --------------------------------------------------------------------
        // One coordinate moved in place: `dims` and the centre count are
        // unchanged, so the assigner can reassign around the one moved centre.
        Proposal::with_delta(
            Tessellation::new(
                centres,
                tessellation.dims().to_vec(),
                tessellation.mus().to_vec(),
            )
            .expect("structurally unchanged"),
            AssignmentDelta::CentreMoved { index: pick / d },
        )
    }

    fn log_structure_ratio(
        &self,
        old: &Tessellation,
        proposed: &Tessellation,
        ctx: &ModelCtx,
    ) -> f64 {
        // ----- your structure-ratio here ------------------------------------
        // log[prior ratio × within-move proposal ratio]:
        //   proposal: the random walk is symmetric and the pick probability is
        //             1/(b·d) both ways  →  cancels exactly;
        //   prior:    one coordinate moved, so its covariate's law SURVIVES.
        //             This is the term Relocate does not have.
        //
        // Priced through the per-covariate law on the ctx, never a hard-coded
        // normal, so the move stays correct under a wrapped or spherical law
        // (the same discipline as pricing counts through the count-prior hook).
        let d = old.dims().len();
        let before = old.centres();
        let after = proposed.centres();
        for i in 0..before.len() {
            // Bitwise, not epsilon: exactly one coordinate differs (a zero-step
            // draw differs nowhere and correctly returns log 1 = 0).
            if before[i] != after[i] {
                // Look the law up by global covariate index, never by the local
                // slot position.
                let law = &ctx.coord_dists[old.dims()[i % d]];
                return law.log_density(after[i]) - law.log_density(before[i]);
            }
        }
        0.0
        // --------------------------------------------------------------------
    }
}

// ---------------------------------------------------------------------------
// The one-command check
// ---------------------------------------------------------------------------

fn main() {
    use rand_core::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);

    // A tiny fixture: p = 3 covariates, 2 cells on dims {0, 2}.
    let dists: Vec<Arc<dyn CoordinateDistribution>> = (0..3)
        .map(|_| Arc::new(EuclideanNormal::new(0.8)) as Arc<_>)
        .collect();
    let weights = vec![1.0; 3];
    let ctx = ModelCtx::new(1.0, 1.5, 10.0, 0.01, 3, &dists, &weights);
    let state = Tessellation::new(vec![0.1, 0.2, 0.3, 0.4], vec![0, 2], vec![0.0, 0.0]).unwrap();

    // Custom sets use weight renormalisation over valid moves automatically;
    // you never write boundary corrections (the MoveSet owns selection).
    let move_set = MoveSetBuilder::empty()
        .with_move(Box::new(Relocate), 0.4)
        .with_move(Box::new(Grow), 0.3)
        .with_move(Box::new(Shrink), 0.3)
        .with_move(Box::new(NudgeCentre::new(0.1)), 0.2)
        .build()
        .expect("names unique, pairing mutual, weights positive");

    let results = conformance::check_move_set(&move_set, &state, &ctx, &mut rng);
    let ok = conformance::report(&results);
    if !ok {
        std::process::exit(1);
    }
    println!("template_moves: all checks passed");
}
