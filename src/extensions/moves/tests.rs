//! The moves oracle suite: the selection/structure-ratio oracles, weighted
//! oracles with the uniform-degeneracy check, builder validation,
//! the global-index fixture, and the detailed-balance oracle.
//! All expected values are hand-derived closed forms; no external
//! implementation is used as an oracle.

use crate::extensions::count_priors::CountPriors;
use std::sync::Arc;

use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};

use super::*;
use crate::engine::error::AddiVortesError;
use crate::extensions::coord::{
    CoordinateDistribution, EuclideanNormal, WrappedNormal, wrap_to_pi,
};
use crate::extensions::inclusion::{
    InclusionModel, InclusionUsage, UniformInclusion, WeightedInclusion,
    log_add_dimension_correction, log_remove_dimension_correction, log_swap_correction,
    weighted_choice,
};
use crate::test_support::assert_abs_eq;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/// A coordinate distribution that always returns `value`, making proposal
/// mechanics deterministic and global-index lookups detectable.
#[derive(Debug)]
struct ConstDist {
    value: f64,
}

impl CoordinateDistribution for ConstDist {
    fn sample(&self, _rng: &mut dyn rand_core::Rng) -> f64 {
        self.value
    }
    fn log_density(&self, _x: f64) -> f64 {
        0.0
    }
}

/// p constant distributions where covariate g returns 100·(g+1); any
/// local-index regression shows up as the wrong constant.
fn tagged_dists(p: usize) -> Vec<Arc<dyn CoordinateDistribution>> {
    (0..p)
        .map(|g| {
            Arc::new(ConstDist {
                value: 100.0 * (g as f64 + 1.0),
            }) as Arc<_>
        })
        .collect()
}

fn uniform_weights(p: usize) -> Vec<f64> {
    vec![1.0; p]
}

fn ctx<'a>(
    omega: f64,
    lambda_c: f64,
    p: usize,
    dists: &'a [Arc<dyn CoordinateDistribution>],
    weights: &'a [f64],
) -> ModelCtx<'a> {
    ModelCtx::new(1.0, omega, lambda_c, 0.01, p, dists, weights)
}

fn rng(seed: u8) -> ChaCha8Rng {
    ChaCha8Rng::from_seed([seed; 32])
}

fn ln(x: f64) -> f64 {
    crate::engine::mathsfn::ln(x)
}

/// b cells on the given dims, arbitrary distinct coordinates, zero μs.
fn tess(dims: Vec<usize>, b: usize) -> Tessellation {
    let d = dims.len();
    let centres: Vec<f64> = (0..b * d).map(|i| 0.01 * (i as f64 + 1.0)).collect();
    Tessellation {
        centres,
        dims,
        mus: vec![0.0; b],
    }
}

fn standard_set() -> MoveSet {
    MoveSetBuilder::stone_gosling().build().unwrap()
}

// ---------------------------------------------------------------------------
// Assignment-delta wiring
// ---------------------------------------------------------------------------

/// Each built-in move declares the exact structural delta it performs; the
/// dims-changing moves claim nothing (FullRecompute). The declared index must
/// be the block the proposal actually touched.
#[test]
fn built_in_moves_declare_their_assignment_deltas() {
    use crate::extensions::distance::AssignmentDelta;

    let dists = tagged_dists(4);
    let w = uniform_weights(4);
    let c = ctx(2.0, 25.0, 4, &dists, &w);
    let t = tess(vec![0, 1], 3);
    let d = 2;

    let proposal = AddCentre.propose(&t, &c, &mut rng(9));
    assert_eq!(proposal.delta(), AssignmentDelta::CentreAdded);

    // Change: the declared index is the one resampled block (ConstDist values
    // 100·(g+1) are unmistakable against the 0.01·i fixture coordinates).
    let proposal = Change.propose(&t, &c, &mut rng(9));
    let AssignmentDelta::CentreMoved { index } = proposal.delta() else {
        panic!(
            "Change must declare CentreMoved, got {:?}",
            proposal.delta()
        );
    };
    for cell in 0..3 {
        let block = &proposal.tessellation.centres()[cell * d..(cell + 1) * d];
        if cell == index {
            assert_eq!(block, &[100.0, 200.0]);
        } else {
            assert_eq!(block, &t.centres()[cell * d..(cell + 1) * d]);
        }
    }

    // RemoveCentre: the declared index is the drained block.
    let proposal = RemoveCentre.propose(&t, &c, &mut rng(9));
    let AssignmentDelta::CentreRemoved { index } = proposal.delta() else {
        panic!(
            "RemoveCentre must declare CentreRemoved, got {:?}",
            proposal.delta()
        );
    };
    let mut expected = t.centres().to_vec();
    expected.drain(index * d..(index + 1) * d);
    assert_eq!(proposal.tessellation.centres(), &expected[..]);

    // dims-changing moves invalidate every cached distance: no claim.
    for (name, proposal) in [
        ("Swap", Swap.propose(&t, &c, &mut rng(9))),
        ("AddDimension", AddDimension.propose(&t, &c, &mut rng(9))),
        (
            "RemoveDimension",
            RemoveDimension.propose(&t, &c, &mut rng(9)),
        ),
    ] {
        assert_eq!(
            proposal.delta(),
            AssignmentDelta::FullRecompute,
            "{name} must declare FullRecompute"
        );
    }
}

// ---------------------------------------------------------------------------
// Structure-ratio oracles (uniform weights: the paper's constants)
// ---------------------------------------------------------------------------

#[test]
fn add_and_remove_centre_ratios_match_closed_forms() {
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w);
    let t3 = tess(vec![0, 1], 3);
    let t4 = tess(vec![0, 1], 4);

    // AC from b = 3: ln λ − ln 3 (pure Poisson count ratio; RC's uniform pick
    // cancels against the centre-set ordering multiplicity. Corrected 2026-07-02
    // after the Geweke gate caught the spurious −ln(b+1) pick term the original
    // oracle had blessed).
    let ac = AddCentre.log_structure_ratio(&t3, &t4, &c);
    assert_abs_eq(ac, ln(25.0) - ln(3.0), 1e-10);
    // RC from b = 4: ln 3 − ln λ, the exact reverse.
    let rc = RemoveCentre.log_structure_ratio(&t4, &t3, &c);
    assert_abs_eq(rc, ln(3.0) - ln(25.0), 1e-10);
    // Reversibility: the pair telescopes to zero.
    assert_abs_eq(ac + rc, 0.0, 1e-12);
}

#[test]
fn add_and_remove_dimension_ratios_match_closed_forms() {
    // p = 5, ω = 2 ⇒ θ = 0.4.
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w);
    let theta: f64 = 0.4;

    // AD from d = 2 (add covariate 4): ln(p−d) − ln d + ln θ − ln(1−θ).
    let from = tess(vec![0, 1], 2);
    let to = tess(vec![0, 1, 4], 2);
    let ad = AddDimension.log_structure_ratio(&from, &to, &c);
    assert_abs_eq(ad, ln(3.0) - ln(2.0) + ln(theta) - ln(1.0 - theta), 1e-10);

    // RD from d = 3: ln(d−1) − ln(p−d+1) − ln θ + ln(1−θ), exact reverse of AD.
    let rd = RemoveDimension.log_structure_ratio(&to, &from, &c);
    assert_abs_eq(rd, ln(2.0) - ln(3.0) - ln(theta) + ln(1.0 - theta), 1e-10);
    assert_abs_eq(ad + rd, 0.0, 1e-12);

    // Boundary d = 1 (RD folds away): AD ratio is ln(p−1) + ln θ − ln(1−θ).
    let from1 = tess(vec![2], 2);
    let to1 = tess(vec![2, 0], 2);
    let ad1 = AddDimension.log_structure_ratio(&from1, &to1, &c);
    assert_abs_eq(ad1, ln(4.0) + ln(theta) - ln(1.0 - theta), 1e-10);

    // Boundary d = p: RD from the full set: ln(p−1) − ln(1) − ln θ + ln(1−θ).
    let full = tess(vec![0, 1, 2, 3, 4], 2);
    let down = tess(vec![0, 1, 2, 3], 2);
    let rdp = RemoveDimension.log_structure_ratio(&full, &down, &c);
    assert_abs_eq(rdp, ln(4.0) - ln(1.0) - ln(theta) + ln(1.0 - theta), 1e-10);
}

#[test]
fn change_and_swap_ratios_are_zero_under_uniform_weights() {
    let dists = tagged_dists(4);
    let w = uniform_weights(4);
    let c = ctx(1.5, 10.0, 4, &dists, &w);
    let from = tess(vec![0, 2], 3);
    let to_change = tess(vec![0, 2], 3);
    assert_eq!(Change.log_structure_ratio(&from, &to_change, &c), 0.0);
    let to_swap = tess(vec![0, 3], 3); // 2 out, 3 in
    assert_eq!(Swap.log_structure_ratio(&from, &to_swap, &c), 0.0);
}

// ---------------------------------------------------------------------------
// The count-prior point: count-prior hook oracles
// ---------------------------------------------------------------------------

/// The default hooks reproduce the paper's closed forms bit for bit: the
/// same values the move oracles above pin through the moves (the golden
/// chain proves the same fact over full sweeps).
#[test]
fn shifted_poisson_binomial_hooks_match_closed_forms() {
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w);
    // Cells: P(4)/P(3) = λ_c/3.
    assert_abs_eq(c.log_cell_count_ratio(4), ln(25.0) - ln(3.0), 0.0);
    // Dimensions (p = 5, θ = 0.4): P(3)/P(2) = ((5−3+1)/2)·θ/(1−θ).
    let theta: f64 = 0.4;
    assert_abs_eq(
        c.log_dim_count_ratio(3),
        ln(3.0) - ln(2.0) + ln(theta) - ln(1.0 - theta),
        0.0,
    );
}

/// A custom hook reprices all four count-changing moves without touching a
/// move: the hook's value lands (with the correct sign) in each structure
/// ratio, and the count-neutral moves stay at exactly zero under any hook.
#[test]
fn custom_count_priors_reprice_the_count_moves() {
    #[derive(Debug)]
    struct ConstPriors;
    impl CountPriors for ConstPriors {
        fn log_cell_count_ratio(&self, _b: usize, _ctx: &ModelCtx) -> f64 {
            -7.0
        }
        fn log_dim_count_ratio(&self, _d: usize, _ctx: &ModelCtx) -> f64 {
            11.0
        }
    }
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w).with_count_priors(&ConstPriors);

    let t3 = tess(vec![0, 1], 3);
    let t4 = tess(vec![0, 1], 4);
    assert_eq!(AddCentre.log_structure_ratio(&t3, &t4, &c), -7.0);
    assert_eq!(RemoveCentre.log_structure_ratio(&t4, &t3, &c), 7.0);

    // Uniform weights ⇒ the subset corrections are exactly 0, isolating the hook.
    let from = tess(vec![0, 1], 2);
    let to = tess(vec![0, 1, 4], 2);
    assert_eq!(AddDimension.log_structure_ratio(&from, &to, &c), 11.0);
    assert_eq!(RemoveDimension.log_structure_ratio(&to, &from, &c), -11.0);

    assert_eq!(Change.log_structure_ratio(&from, &from, &c), 0.0);
    let to_swap = tess(vec![0, 3], 2);
    assert_eq!(Swap.log_structure_ratio(&from, &to_swap, &c), 0.0);
}

/// State-dependent custom hooks preserve the AC×RC and AD×RD round trips
/// exactly: forward and reverse price the same adjacent-count boundary (the
/// larger count), so the pair telescopes to zero by construction; detailed
/// balance holds for any count-prior prior without touching a move.
#[test]
fn custom_count_priors_keep_detailed_balance_round_trips() {
    #[derive(Debug)]
    struct StateDependent;
    impl CountPriors for StateDependent {
        fn log_cell_count_ratio(&self, b: usize, _ctx: &ModelCtx) -> f64 {
            // Geometric-flavoured: P(b)/P(b−1) = 1/(b+1).
            -ln((b + 1) as f64)
        }
        fn log_dim_count_ratio(&self, d: usize, ctx: &ModelCtx) -> f64 {
            ln(ctx.p as f64) - 2.0 * ln(d as f64)
        }
    }
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w).with_count_priors(&StateDependent);

    let t3 = tess(vec![0, 1], 3);
    let t4 = tess(vec![0, 1], 4);
    let forward = AddCentre.log_structure_ratio(&t3, &t4, &c);
    let reverse = RemoveCentre.log_structure_ratio(&t4, &t3, &c);
    assert_eq!(forward + reverse, 0.0);

    let from = tess(vec![0, 1], 2);
    let to = tess(vec![0, 1, 4], 2);
    let forward = AddDimension.log_structure_ratio(&from, &to, &c);
    let reverse = RemoveDimension.log_structure_ratio(&to, &from, &c);
    assert_eq!(forward + reverse, 0.0);
}

// ---------------------------------------------------------------------------
// Selection oracles (PaperFolded: the boundary values emerge)
// ---------------------------------------------------------------------------

#[test]
fn folded_selection_table_matches_hand_values() {
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w);
    let set = standard_set();
    // Registration order: AC, RC, AD, RD, Change, Swap.

    // Interior state (nC ≥ 2, 2 ≤ d ≤ p−1): the paper's base weights.
    let interior = tess(vec![0, 1], 3);
    assert_eq!(
        set.selection_probs(&interior, &c),
        vec![0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    );

    // nC = 1: RC invalid, its mass folds into AC.
    let single_cell = tess(vec![0, 1], 1);
    assert_eq!(
        set.selection_probs(&single_cell, &c),
        vec![0.4, 0.0, 0.2, 0.2, 0.1, 0.1]
    );

    // d = 1: RD invalid, folds into AD.
    let one_dim = tess(vec![3], 2);
    assert_eq!(
        set.selection_probs(&one_dim, &c),
        vec![0.2, 0.2, 0.4, 0.0, 0.1, 0.1]
    );

    // d = p: AD folds into RD, Swap folds into Change.
    let full = tess(vec![0, 1, 2, 3, 4], 2);
    assert_eq!(
        set.selection_probs(&full, &c),
        vec![0.2, 0.2, 0.0, 0.4, 0.2, 0.0]
    );
}

#[test]
fn compound_p_equals_two_states() {
    let dists = tagged_dists(2);
    let w = uniform_weights(2);
    let c = ctx(1.0, 25.0, 2, &dists, &w);
    let set = standard_set();

    // p = 2, d = 1, nC = 1: RC→AC and RD→AD fold simultaneously.
    let t = tess(vec![1], 1);
    assert_eq!(
        set.selection_probs(&t, &c),
        vec![0.4, 0.0, 0.4, 0.0, 0.1, 0.1]
    );

    // p = 2, d = 2 = p, nC = 1: RC→AC, AD→RD, Swap→Change.
    let t = tess(vec![0, 1], 1);
    assert_eq!(
        set.selection_probs(&t, &c),
        vec![0.4, 0.0, 0.0, 0.4, 0.2, 0.0]
    );
}

#[test]
fn boundary_selection_ratios_emerge_generically() {
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w);
    let set = standard_set();
    let names: Vec<_> = set.names().collect();
    let ac = names.iter().position(|n| *n == "AddCentre").unwrap();
    let rc = names.iter().position(|n| *n == "RemoveCentre").unwrap();

    // AC from nC = 1 (forward q = 0.4) to nC = 2 (reverse RC q = 0.2): ln(1/2).
    let from = tess(vec![0, 1], 1);
    let to = tess(vec![0, 1], 2);
    assert_abs_eq(set.log_selection_ratio(ac, &from, &to, &c), ln(0.5), 1e-12);
    // …and RC back down: ln 2, the mirrored correction.
    assert_abs_eq(set.log_selection_ratio(rc, &to, &from, &c), ln(2.0), 1e-12);
}

// ---------------------------------------------------------------------------
// Weighted-inclusion oracles
// ---------------------------------------------------------------------------

/// Weights {1,2,3,4,5}: e₁ = 15, e₂ = 85 (hand-summed pair products).
const W5: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

#[test]
fn weighted_add_dimension_correction_matches_hand_value() {
    // dims = [4] (W_in = 5, W_out = 10), add j = 1 (s = 2):
    // corr = s·e₁·W_out / (e₂·(W_in+s)) = 2·15·10 / (85·7) = 300/595.
    let corr = log_add_dimension_correction(&[4], 1, &W5);
    assert_abs_eq(corr, ln(300.0 / 595.0), 1e-10);

    // Full AD ratio (p = 5, d = 1, ω = 2 ⇒ θ = 0.4), via the move itself,
    // with the global-index layout (global covariate 4 at local slot 0)
    // so the weights lookup is part of the fixture.
    let dists = tagged_dists(5);
    let c = ctx(2.0, 25.0, 5, &dists, &W5);
    let from = tess(vec![4], 2);
    let to = tess(vec![4, 1], 2);
    let expected = ln(4.0) - ln(1.0) + ln(0.4) - ln(0.6) + ln(300.0 / 595.0);
    assert_abs_eq(
        AddDimension.log_structure_ratio(&from, &to, &c),
        expected,
        1e-10,
    );
}

#[test]
fn weighted_remove_dimension_correction_is_the_reciprocal() {
    // dims = [4, 1] (W_in = 7, W_out = 8), remove j = 1 (s = 2):
    // corr = e₂·W_in / (e₁·s·(W_out+s)) = 85·7 / (15·2·10) = 595/300.
    let corr = log_remove_dimension_correction(&[4, 1], 1, &W5);
    assert_abs_eq(corr, ln(595.0 / 300.0), 1e-10);
    // Detailed balance of the corrections themselves.
    assert_abs_eq(
        corr + log_add_dimension_correction(&[4], 1, &W5),
        0.0,
        1e-12,
    );
}

#[test]
fn weighted_swap_correction_matches_hand_value() {
    // dims = [4] (W_out = 10), out i = 4 (s = 5), in j = 1 (s = 2):
    // corr = W_out / (W_out − s_j + s_i) = 10/13.
    let corr = log_swap_correction(&[4], 4, 1, &W5);
    assert_abs_eq(corr, ln(10.0 / 13.0), 1e-10);
    // Reverse swap from [1]: out 1, in 4: W_out(D′) = 13 → ln(13/10), which telescopes.
    let reverse = log_swap_correction(&[1], 1, 4, &W5);
    assert_abs_eq(corr + reverse, 0.0, 1e-12);
}

#[test]
fn uniform_degeneracy_is_exact_not_approximate() {
    // Under UniformInclusion (weights all exactly 1.0) the weight-generic
    // ratios must be bit-identical to the paper constants.
    let n = 7;
    let model = UniformInclusion::new(n);
    let dists = tagged_dists(n);
    let weights = InclusionModel::weights(&model);
    let c = ctx(2.0, 25.0, n, &dists, weights);

    assert_eq!(
        log_add_dimension_correction(&[4], 1, c.weights).to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        log_remove_dimension_correction(&[4, 1], 1, c.weights).to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        log_swap_correction(&[4], 4, 1, c.weights).to_bits(),
        0.0_f64.to_bits()
    );

    // Through the move: identical bits to the hand-assembled constant.
    let theta = c.omega / c.p as f64;
    let from = tess(vec![4], 2);
    let to = tess(vec![4, 1], 2);
    let generic = AddDimension.log_structure_ratio(&from, &to, &c);
    let constant = ln((n - 1) as f64) - ln(1.0) + ln(theta) - ln(1.0 - theta);
    assert_eq!(generic.to_bits(), constant.to_bits());
}

// ---------------------------------------------------------------------------
// Deviation-2 global-index fixture (proposal lookups by global covariate index)
// ---------------------------------------------------------------------------

#[test]
fn proposals_look_up_distributions_by_global_index() {
    // Tessellation lives on global covariate 4 at local slot 0. ConstDist tags
    // covariate g with value 100(g+1): a local-index regression would sample
    // 100 instead of 500.
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w);
    let t = tess(vec![4], 2);
    let mut r = rng(1);

    // Change resamples the chosen centre's coordinate from covariate 4's law.
    let changed = Change.propose(&t, &c, &mut r).tessellation;
    let moved: Vec<f64> = changed
        .centres
        .iter()
        .copied()
        .filter(|v| *v != 0.01 && *v != 0.02)
        .collect();
    assert_eq!(moved, vec![500.0]);

    // AC samples the new centre from covariate 4's law.
    let grown = AddCentre.propose(&t, &c, &mut r).tessellation;
    assert_eq!(grown.centres.last().copied(), Some(500.0));

    // AD's new column samples from the incoming covariate's law.
    let wider = AddDimension.propose(&t, &c, &mut r).tessellation;
    let incoming = *wider.dims.last().unwrap();
    assert_ne!(incoming, 4);
    for cell in 0..wider.n_cells() {
        assert_eq!(wider.centres[cell * 2 + 1], 100.0 * (incoming as f64 + 1.0));
    }
}

// ---------------------------------------------------------------------------
// Proposal mechanics + draw-stream discipline
// ---------------------------------------------------------------------------

#[test]
fn remove_centre_and_dimension_edit_the_right_slices() {
    let dists = tagged_dists(3);
    let w = uniform_weights(3);
    let c = ctx(1.0, 10.0, 3, &dists, &w);

    let t = tess(vec![0, 2], 3); // centres 0.01..0.06 row-major
    let mut r = rng(2);
    let smaller = RemoveCentre.propose(&t, &c, &mut r).tessellation;
    assert_eq!(smaller.n_cells(), 2);
    assert_eq!(smaller.dims(), t.dims());
    // The two surviving centres are original rows.
    let rows: Vec<&[f64]> = (0..3).map(|i| &t.centres[i * 2..(i + 1) * 2]).collect();
    for i in 0..2 {
        assert!(rows.contains(&&smaller.centres[i * 2..(i + 1) * 2]));
    }

    let narrower = RemoveDimension.propose(&t, &c, &mut r).tessellation;
    assert_eq!(narrower.dims().len(), 1);
    assert!(t.dims().contains(&narrower.dims()[0]));
    assert_eq!(narrower.centres().len(), 3);
}

#[test]
fn swap_replaces_slot_and_resamples_only_that_column() {
    let dists = tagged_dists(3);
    let w = uniform_weights(3);
    let c = ctx(1.0, 10.0, 3, &dists, &w);
    let t = tess(vec![0, 2], 2);
    let mut r = rng(3);
    let swapped = Swap.propose(&t, &c, &mut r).tessellation;
    assert_eq!(swapped.dims().len(), 2);
    // Covariate 1 must have entered (it is the only unused one).
    assert!(swapped.dims().contains(&1));
    let slot = swapped.dims().iter().position(|&d| d == 1).unwrap();
    for cell in 0..2 {
        assert_eq!(swapped.centres[cell * 2 + slot], 200.0); // ConstDist for cov 1
        assert_eq!(
            swapped.centres[cell * 2 + (1 - slot)],
            t.centres[cell * 2 + (1 - slot)]
        );
    }
}

#[test]
fn uniform_inclusion_update_consumes_no_rng() {
    let mut model = UniformInclusion::new(4);
    let usage = InclusionUsage::new(4);
    let mut r = rng(4);
    let mirror = r.clone();
    InclusionModel::update(&mut model, &usage, &mut r).unwrap();
    // The stream is untouched: the next draw agrees with an untouched clone.
    let mut mirror = mirror;
    assert_eq!(r.next_u64(), mirror.next_u64());
}

#[test]
fn weighted_choice_consumes_exactly_one_draw_and_respects_weights() {
    let weights = [1.0, 999.0, 3.0, 2.0]; // 999 unused: candidate 1 excluded
    let candidates = [0usize, 2, 3];
    let mut r = rng(5);
    let mut mirror = r.clone();
    let _ = weighted_choice(&candidates, &weights, &mut r);
    mirror.next_u64();
    assert_eq!(r.next_u64(), mirror.next_u64());

    // Frequencies over a fixed seeded stream ≈ 1:3:2 over candidates {0,2,3}.
    let mut counts = [0usize; 4];
    let mut r = rng(6);
    let draws = 60_000;
    for _ in 0..draws {
        counts[candidates[weighted_choice(&candidates, &weights, &mut r)]] += 1;
    }
    for (candidate, expected_weight) in [(0usize, 1.0), (2, 3.0), (3, 2.0)] {
        let observed = counts[candidate] as f64 / draws as f64;
        let expected = expected_weight / 6.0;
        assert!(
            (observed - expected).abs() < 0.01,
            "candidate {candidate}: observed {observed}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// Builder validation
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct NamedMove {
    name: &'static str,
    reverse: Reverse,
}

impl ProposalMove for NamedMove {
    fn name(&self) -> &'static str {
        self.name
    }
    fn reverse(&self) -> Reverse {
        self.reverse
    }
    fn is_valid(&self, _t: &Tessellation, _c: &ModelCtx) -> bool {
        true
    }
    fn propose(&self, t: &Tessellation, _c: &ModelCtx, _r: &mut dyn rand_core::Rng) -> Proposal {
        Proposal::new(t.clone())
    }
    fn log_structure_ratio(&self, _o: &Tessellation, _p: &Tessellation, _c: &ModelCtx) -> f64 {
        0.0
    }
}

#[test]
fn builder_rejects_bad_sets() {
    // Duplicate name.
    let err = MoveSetBuilder::stone_gosling()
        .with_move(Box::new(AddCentre), 0.1)
        .build()
        .unwrap_err();
    assert_eq!(
        err,
        AddiVortesError::InvalidMoveSet {
            move_name: "AddCentre".into(),
            reason: "is registered more than once".into(),
        }
    );

    // Missing reverse.
    let err = MoveSetBuilder::empty()
        .with_move(
            Box::new(NamedMove {
                name: "A",
                reverse: Reverse::Named("Ghost"),
            }),
            1.0,
        )
        .build()
        .unwrap_err();
    assert_eq!(
        err,
        AddiVortesError::InvalidMoveSet {
            move_name: "A".into(),
            reason: "names `Ghost` as its reverse, which is not registered".into(),
        }
    );

    // Non-mutual pairing: A → B but B → itself.
    let err = MoveSetBuilder::empty()
        .with_move(
            Box::new(NamedMove {
                name: "A",
                reverse: Reverse::Named("B"),
            }),
            1.0,
        )
        .with_move(
            Box::new(NamedMove {
                name: "B",
                reverse: Reverse::SelfInverse,
            }),
            1.0,
        )
        .build()
        .unwrap_err();
    assert_eq!(
        err,
        AddiVortesError::InvalidMoveSet {
            move_name: "A".into(),
            reason: "is not mutually paired: `B` does not name it back".into(),
        }
    );

    // Empty set and invalid weight.
    assert!(MoveSetBuilder::empty().build().is_err());
    assert!(
        MoveSetBuilder::empty()
            .with_move(
                Box::new(NamedMove {
                    name: "A",
                    reverse: Reverse::SelfInverse
                }),
                0.0
            )
            .build()
            .is_err()
    );
}

#[test]
fn mutual_custom_pair_registers_in_either_order() {
    let pair = || {
        (
            Box::new(NamedMove {
                name: "Fwd",
                reverse: Reverse::Named("Bwd"),
            }),
            Box::new(NamedMove {
                name: "Bwd",
                reverse: Reverse::Named("Fwd"),
            }),
        )
    };
    let (a, b) = pair();
    assert!(
        MoveSetBuilder::empty()
            .with_move(a, 1.0)
            .with_move(b, 1.0)
            .build()
            .is_ok()
    );
    let (a, b) = pair();
    assert!(
        MoveSetBuilder::empty()
            .with_move(b, 1.0)
            .with_move(a, 1.0)
            .build()
            .is_ok()
    );
}

#[test]
fn custom_move_switches_to_weighted_valid_renormalisation() {
    let dists = tagged_dists(5);
    let w = uniform_weights(5);
    let c = ctx(2.0, 25.0, 5, &dists, &w);
    let set = MoveSetBuilder::stone_gosling()
        .with_move(
            Box::new(NamedMove {
                name: "Jitter",
                reverse: Reverse::SelfInverse,
            }),
            0.1,
        )
        .build()
        .unwrap();

    // nC = 1: RC is invalid. Under WeightedValid its mass is not folded into
    // AC: the remaining valid weights renormalise, total = 1.1 − 0.2 = 0.9.
    let from = tess(vec![0, 1], 1);
    let to = tess(vec![0, 1], 2);
    let names: Vec<_> = set.names().collect();
    let ac = names.iter().position(|n| *n == "AddCentre").unwrap();
    // Forward q(AC | nC=1) = 0.2/0.9; reverse q(RC | nC=2) = 0.2/1.1.
    let expected = ln(0.2 / 1.1) - ln(0.2 / 0.9);
    assert_abs_eq(set.log_selection_ratio(ac, &from, &to, &c), expected, 1e-12);

    // Detailed-balance spot check: the reverse traversal mirrors it exactly.
    let rc = names.iter().position(|n| *n == "RemoveCentre").unwrap();
    let back = set.log_selection_ratio(rc, &to, &from, &c);
    assert_abs_eq(
        set.log_selection_ratio(ac, &from, &to, &c) + back,
        0.0,
        1e-12,
    );
}

// ---------------------------------------------------------------------------
// Detailed-balance oracle
// ---------------------------------------------------------------------------

/// Tiny fixed tessellation, hand-computed closed form for the full structural
/// correction `log_structure_ratio + log_selection_ratio`, forward and
/// reverse. (The likelihood term joins in the sampler tests, where the
/// 2-state toy's analytic full acceptance probability completes this oracle;
/// the structure and selection parts here are the per-commit guard for the
/// selection-ratio / reverse-pairing bug class.)
#[test]
fn l15_detailed_balance_uniform() {
    // p = 3, ω = 1 (θ = 1/3), b = 2 cells, dims = [0] → AD adds covariate 2.
    let dists = tagged_dists(3);
    let w = uniform_weights(3);
    let c = ctx(1.0, 2.0, 3, &dists, &w);
    let set = standard_set();
    let names: Vec<_> = set.names().collect();
    let ad = names.iter().position(|n| *n == "AddDimension").unwrap();
    let rd = names.iter().position(|n| *n == "RemoveDimension").unwrap();

    let from = tess(vec![0], 2);
    let to = tess(vec![0, 2], 2);

    // Hand derivation:
    //   structure(AD, d=1): ln(p−d) − ln d + ln θ − ln(1−θ)
    //                     = ln 2 − 0 + ln(1/3) − ln(2/3) = ln 2 − ln 2 = 0.
    //   selection: q(AD | d=1) = 0.4 (RD folds in), q(RD | d=2) = 0.2 → ln(1/2).
    // Total forward correction: ln(1/2).
    let forward = AddDimension.log_structure_ratio(&from, &to, &c)
        + set.log_selection_ratio(ad, &from, &to, &c);
    assert_abs_eq(forward, ln(0.5), 1e-10);

    // Reverse traversal: structure(RD, d=2) = 0; selection ln(0.4/0.2) = ln 2.
    let reverse = RemoveDimension.log_structure_ratio(&to, &from, &c)
        + set.log_selection_ratio(rd, &to, &from, &c);
    assert_abs_eq(reverse, ln(2.0), 1e-10);
    assert_abs_eq(forward + reverse, 0.0, 1e-12);
}

#[test]
fn l15_detailed_balance_weighted() {
    // Same tiny fixture with WeightedInclusion weights {1,2,3}.
    // Hand derivation for AD adding covariate 2 from dims = [0]:
    //   uniform part: 0 (as in the uniform test);
    //   correction: s₂·e₁·W_out / (e₂·(W_in+s₂)) with e₁ = 6, e₂ = 11,
    //               W_in = 1, W_out = 5, s₂ = 3 → 3·6·5 / (11·4) = 90/44;
    //   selection: ln(1/2) (the validity pattern is unchanged).
    // Total forward: ln(90/44) + ln(1/2) = ln(45/44).
    let dists = tagged_dists(3);
    let model = WeightedInclusion::new(vec![1.0, 2.0, 3.0]);
    let weights = InclusionModel::weights(&model);
    let c = ctx(1.0, 2.0, 3, &dists, weights);
    let set = standard_set();
    let names: Vec<_> = set.names().collect();
    let ad = names.iter().position(|n| *n == "AddDimension").unwrap();
    let rd = names.iter().position(|n| *n == "RemoveDimension").unwrap();

    let from = tess(vec![0], 2);
    let to = tess(vec![0, 2], 2);

    let forward = AddDimension.log_structure_ratio(&from, &to, &c)
        + set.log_selection_ratio(ad, &from, &to, &c);
    assert_abs_eq(forward, ln(45.0 / 44.0), 1e-10);

    let reverse = RemoveDimension.log_structure_ratio(&to, &from, &c)
        + set.log_selection_ratio(rd, &to, &from, &c);
    assert_abs_eq(forward + reverse, 0.0, 1e-12);
    assert_abs_eq(reverse, -ln(45.0 / 44.0), 1e-10);
}

// ---------------------------------------------------------------------------
// Coordinate distributions
// ---------------------------------------------------------------------------

#[test]
fn wrap_to_pi_wraps_correctly() {
    let pi = std::f64::consts::PI;
    assert_abs_eq(wrap_to_pi(pi + 0.1), -pi + 0.1, 1e-12);
    assert_abs_eq(wrap_to_pi(-pi - 0.1), pi - 0.1, 1e-12);
    assert_abs_eq(wrap_to_pi(0.3), 0.3, 1e-15); // one ±π round trip of rounding
    assert!(wrap_to_pi(1e6) >= -pi && wrap_to_pi(1e6) <= pi);
}

#[test]
fn coordinate_densities_integrate_to_one() {
    // Trapezoid integration of exp(log_density): N(0, 0.8²) over ±10σ and the
    // wrapped normal over its full domain [−π, π].
    let numeric_integral = |f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, n: usize| -> f64 {
        let h = (hi - lo) / n as f64;
        let mut sum = 0.5 * (f(lo) + f(hi));
        for i in 1..n {
            sum += f(lo + h * i as f64);
        }
        sum * h
    };
    let normal = EuclideanNormal::new(0.8).unwrap();
    let mass = numeric_integral(
        &|x| crate::engine::mathsfn::exp(normal.log_density(x)),
        -8.0,
        8.0,
        4000,
    );
    assert_abs_eq(mass, 1.0, 1e-8);

    let wrapped = WrappedNormal::new(0.8).unwrap();
    let pi = std::f64::consts::PI;
    let mass = numeric_integral(
        &|x| crate::engine::mathsfn::exp(wrapped.log_density(x)),
        -pi,
        pi,
        4000,
    );
    assert_abs_eq(mass, 1.0, 1e-8);

    // Samples stay inside the wrapped domain.
    let mut r = rng(7);
    for _ in 0..500 {
        let s = wrapped.sample(&mut r);
        assert!((-pi..=pi).contains(&s));
    }
}
