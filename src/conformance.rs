//! One-command conformance checks for extensions: cheap, local,
//! plain-language checks to run before the expensive global gates.
//!
//! # Read this before you trust a green run
//!
//! **These checks prove self-consistency, not correctness.** Passing means
//! "not broken in the obvious ways", never "valid".
//!
//! The reason is structural, and it applies to nearly every check here: the
//! reference each one compares against is *drawn from the component under
//! test*. The move check asks whether a move cancels against **its own
//! reverse**. The cell-model check estimates the Bayes factor from **the
//! model's own prior draws**. The coordinate check asks whether `sample` and
//! `log_density` agree **with each other**. The assigner check compares
//! `assign_cells` against a recompute that **calls `assign_cells`**.
//!
//! So a component that is wrong *consistently* passes. And that is precisely
//! the failure mode of a competent researcher: derive one thing incorrectly,
//! then implement it faithfully everywhere. The canonical instance: an
//! add/remove-centre acceptance ratio carrying a spurious centre-pick factor
//! in **both** halves cancels perfectly in every local detailed-balance
//! check; only the joint-distribution battery can see it.
//!
//! What these checks *are* good for: catching mechanical faults (a wrong index,
//! an asymmetric slip, a non-finite return, a stale cache, an inverted sign)
//! in seconds, before you spend minutes on the battery.
//!
//! **For statistical validity there is exactly one route: the
//! joint-distribution battery** ([`crate::calibration`]), which generates the
//! prior *independently* of your component and can therefore see what these
//! cannot. It is fast — bugs the checks here miss go red in about half a
//! second — and `tests/calibration_acceptance.rs` is a copyable,
//! public-API-only worked example you can lift wholesale into your own crate.
//! An adaptive `InclusionModel::update` in particular *cannot* be validated
//! here: AddiVortes dimension sets are distinct subsets, so DART-style
//! conjugate Dirichlet updates are not the true conditional (the e_d(s)
//! normalisers do not cancel), and only SBC/Geweke can tell you whether your
//! correction is right.
//!
//! Everything here is deterministic given the RNG you pass in.

// Test-build infrastructure: each shelf entry exercises its own check
// suite, so any single build uses a subset.
#![allow(dead_code)]

use crate::engine::data::Data;
use crate::engine::error::Result;
use crate::engine::mathsfn;
use crate::engine::tessellation::Tessellation;
use crate::extensions::basis::CellBasis;
use crate::extensions::cell_model::{CellModel, CellStats};
use crate::extensions::coord::CoordinateDistribution;
use crate::extensions::count_priors::CountPriors;
use crate::extensions::distance::{
    AssignmentCache, AssignmentDelta, CellAssigner, PairwiseDistance,
};
use crate::extensions::inclusion::{InclusionModel, InclusionUsage};
use crate::extensions::membership::MembershipKernel;
use crate::extensions::moves::{ModelCtx, MoveSet};
use crate::extensions::response::ResponseModel;
use crate::extensions::scale::{ScaleCtx, ScaleModel};

/// Outcome of one conformance check: a name, a verdict, and a plain-language
/// explanation of what was checked (and, on failure, what went wrong).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckResult {
    /// Short name of the check (stable, greppable).
    pub name: String,
    /// Did the check pass?
    pub passed: bool,
    /// Plain-language detail: what was verified, or what failed and where.
    pub detail: String,
}

impl CheckResult {
    fn pass(name: &str, detail: String) -> Self {
        Self {
            name: name.into(),
            passed: true,
            detail,
        }
    }
    fn fail(name: &str, detail: String) -> Self {
        Self {
            name: name.into(),
            passed: false,
            detail,
        }
    }
}

/// True when every check passed (convenience for examples/CI).
pub fn all_passed(results: &[CheckResult]) -> bool {
    results.iter().all(|r| r.passed)
}

/// Print the results like a small test run and return `all_passed`.
pub fn report(results: &[CheckResult]) -> bool {
    for r in results {
        println!(
            "{} {}: {}",
            if r.passed { "PASS" } else { "FAIL" },
            r.name,
            r.detail
        );
    }
    all_passed(results)
}

/// Check every move of a `MoveSet` from the given state: reverse-pair
/// structure-ratio identity, selection-ratio symmetry, `is_valid` purity, and
/// the proposal contract (structurally coherent tessellations).
///
/// The detailed-balance identity checked here is the cheap local gate: for a
/// move M with reverse R, `log_structure_ratio(M, s→s′) +
/// log_structure_ratio(R, s′→s) = 0` and the same for the selection ratios.
/// A move can pass this and still be wrong globally; run the statistical
/// gates before trusting new kernels on real inference.
pub fn check_move_set(
    move_set: &MoveSet,
    state: &Tessellation,
    ctx: &ModelCtx,
    rng: &mut dyn rand_core::Rng,
) -> Vec<CheckResult> {
    let mut results = Vec::new();
    let names: Vec<&'static str> = move_set.names().collect();

    for (index, &name) in names.iter().enumerate() {
        let mv = move_set.move_at(index);

        // is_valid purity: same state, same answer, twice.
        let v1 = mv.is_valid(state, ctx);
        let v2 = mv.is_valid(state, ctx);
        if v1 != v2 {
            results.push(CheckResult::fail(
                "is_valid_pure",
                format!("move `{name}`: is_valid returned {v1} then {v2} for the same state"),
            ));
            continue;
        }
        if !v1 {
            results.push(CheckResult::pass(
                "move_skipped",
                format!("move `{name}` is not valid from this state (fine): pick a state where it is to exercise it"),
            ));
            continue;
        }

        let proposal = mv.propose(state, ctx, rng);
        let declared = proposal.delta();
        let proposed = proposal.tessellation;

        // The delta claim, checked against the change the move actually made.
        //
        // A move *declares* what it changed so the assigner can update cached
        // assignments incrementally instead of rescanning. The engine takes that
        // claim on trust — nothing downstream re-derives it — so a wrong claim
        // silently corrupts the chain with no error anywhere, exactly like a
        // stale reassign fast path.
        //
        // This is a rare thing among these checks: a genuine oracle. The check does not
        // ask the move what changed, it diffs the two tessellations itself.
        results.push(match validate_delta(state, &proposed, declared) {
            Ok(detail) => {
                CheckResult::pass("delta_claim_exact", format!("move `{name}`: {detail}"))
            }
            Err(detail) => CheckResult::fail(
                "delta_claim_exact",
                format!(
                    "move `{name}` declared {declared:?}, but {detail}. The assigner trusts this \
                     claim and updates only what it names, so an inexact one silently corrupts \
                     the chain. When in doubt, `Proposal::new` (claim nothing, full recompute) is \
                     always correct."
                ),
            ),
        });

        // Proposal contract: the proposed tessellation is structurally coherent.
        if let Err(e) = Tessellation::new(
            proposed.centres().to_vec(),
            proposed.dims().to_vec(),
            proposed.mus().to_vec(),
        ) {
            results.push(CheckResult::fail(
                "proposal_contract",
                format!("move `{name}` proposed an incoherent tessellation: {e}"),
            ));
            continue;
        }
        if let Some(&bad) = proposed.dims().iter().find(|d| **d >= ctx.p) {
            results.push(CheckResult::fail(
                "proposal_contract",
                format!("move `{name}` proposed covariate {bad} but p = {}", ctx.p),
            ));
            continue;
        }

        // Reverse-pair structure-ratio identity.
        let reverse_index = move_set.reverse_index(index);
        let reverse = move_set.move_at(reverse_index);
        let forward = mv.log_structure_ratio(state, &proposed, ctx);
        let backward = reverse.log_structure_ratio(&proposed, state, ctx);
        let residual = (forward + backward).abs();
        // The finiteness test is load-bearing: `NaN > 1e-8` is *false*, so a bare
        // `residual > 1e-8` would send a NaN structure ratio down the pass branch.
        if !residual.is_finite() || residual > 1e-8 {
            results.push(CheckResult::fail(
                "detailed_balance_structure",
                format!(
                    "move `{name}` + reverse `{}`: forward {forward:+.6e} and backward \
                     {backward:+.6e} do not cancel (residual {residual:.3e}): one of the \
                     two structure ratios is wrong, the pairing is not truly mutual, or a \
                     ratio is non-finite",
                    names[reverse_index]
                ),
            ));
        } else {
            results.push(CheckResult::pass(
                "detailed_balance_structure",
                format!(
                    "move `{name}` ↔ `{}` structure ratios cancel",
                    names[reverse_index]
                ),
            ));
        }

        // Selection-ratio symmetry.
        let sel_forward = move_set.log_selection_ratio(index, state, &proposed, ctx);
        let sel_backward = move_set.log_selection_ratio(reverse_index, &proposed, state, ctx);
        let sel_residual = (sel_forward + sel_backward).abs();
        // Finiteness first, for the same reason: a NaN must not pass.
        if !sel_residual.is_finite() || sel_residual > 1e-8 {
            results.push(CheckResult::fail(
                "selection_ratio_symmetry",
                format!(
                    "move `{name}`: selection corrections do not mirror (residual \
                     {sel_residual:.3e}): validity flags are likely inconsistent between \
                     the two states, or a correction is non-finite"
                ),
            ));
        } else {
            results.push(CheckResult::pass(
                "selection_ratio_symmetry",
                format!("move `{name}` selection corrections mirror"),
            ));
        }
    }
    results
}

/// Verify a move's declared [`AssignmentDelta`] against the change it actually
/// made, by diffing the two tessellations directly.
///
/// Bitwise comparison throughout, not epsilon: the claim is about *structure*
/// ("these centres are untouched"), and the assigner reuses their cached keys
/// verbatim. A coordinate that changed in the last ulp is still a coordinate
/// that changed, and the cached key for it is stale.
fn validate_delta(
    old: &Tessellation,
    new: &Tessellation,
    delta: AssignmentDelta,
) -> std::result::Result<String, String> {
    // Every claim except FullRecompute asserts `dims` is unchanged.
    if delta != AssignmentDelta::FullRecompute && old.dims() != new.dims() {
        return Err(format!(
            "the active dimensions changed ({:?} -> {:?}), which only `FullRecompute` permits",
            old.dims(),
            new.dims()
        ));
    }
    let d = old.dims().len();
    let (b_old, b_new) = (old.n_cells(), new.n_cells());
    // Centre `k`'s coordinate block, bitwise.
    let block = |t: &Tessellation, k: usize| -> Vec<u64> {
        t.centres()[k * d..(k + 1) * d]
            .iter()
            .map(|c| c.to_bits())
            .collect()
    };

    match delta {
        // Claims nothing; the assigner rescans. Always honest.
        AssignmentDelta::FullRecompute => Ok("claims no structural delta (full recompute)".into()),

        AssignmentDelta::CentreAdded => {
            if b_new != b_old + 1 {
                return Err(format!(
                    "the centre count went {b_old} -> {b_new}, not {b_old} -> {}",
                    b_old + 1
                ));
            }
            // The claim is "appended at the highest index", so every pre-existing
            // centre must survive untouched at its own index.
            for k in 0..b_old {
                if block(old, k) != block(new, k) {
                    return Err(format!(
                        "centre {k} also changed. `CentreAdded` promises the new centre is \
                         appended at the highest index and every existing centre is untouched"
                    ));
                }
            }
            Ok(format!(
                "appended centre {b_old}, leaving {b_old} existing centres untouched"
            ))
        }

        AssignmentDelta::CentreRemoved { index } => {
            if index >= b_old {
                return Err(format!(
                    "centre {index} does not exist in the old tessellation (b = {b_old})"
                ));
            }
            if b_new != b_old - 1 {
                return Err(format!(
                    "the centre count went {b_old} -> {b_new}, not {b_old} -> {}",
                    b_old - 1
                ));
            }
            // Everything below `index` keeps its slot; everything above shifts down one.
            for k in 0..b_new {
                let source = if k < index { k } else { k + 1 };
                if block(old, source) != block(new, k) {
                    return Err(format!(
                        "old centre {source} should have landed at new index {k} but did not. \
                         `CentreRemoved {{ index: {index} }}` promises the remaining centres keep \
                         their order, with those above the removed one shifting down by one"
                    ));
                }
            }
            Ok(format!(
                "removed centre {index}; the remaining {b_new} shifted down in order"
            ))
        }

        AssignmentDelta::CentreMoved { index } => {
            if b_new != b_old {
                return Err(format!(
                    "the centre count changed ({b_old} -> {b_new}); `CentreMoved` promises it does not"
                ));
            }
            if index >= b_old {
                return Err(format!("centre {index} does not exist (b = {b_old})"));
            }
            // Only the named centre may differ. (It need not: a zero-magnitude
            // step legitimately lands back where it started.)
            for k in 0..b_old {
                if k != index && block(old, k) != block(new, k) {
                    return Err(format!(
                        "centre {k} changed too. `CentreMoved {{ index: {index} }}` promises \
                         centre {index} is the *only* one whose coordinates differ, and the \
                         assigner reuses every other centre's cached key untouched"
                    ));
                }
            }
            Ok(format!("only centre {index} moved, as declared"))
        }
    }
}

/// Check an assigner on the given inputs: repeat-call determinism (the
/// `PairwiseDistance` contract: pure, deterministic, no interior state) and
/// incremental-reassign consistency. For every declarable
/// [`AssignmentDelta`], `reassign` must return exactly what a fresh recompute
/// of the new tessellation would, since a divergence silently corrupts the
/// chain with no error anywhere.
pub fn check_assigner(
    assigner: &dyn CellAssigner,
    x: &Data,
    tessellation: &Tessellation,
) -> Vec<CheckResult> {
    let run = |label: &str| -> std::result::Result<Vec<usize>, CheckResult> {
        assigner
            .assign_cells(x, tessellation)
            .map_err(|e| CheckResult::fail("assigner_runs", format!("{label} call failed: {e}")))
    };
    let first = match run("first") {
        Ok(v) => v,
        Err(r) => return vec![r],
    };
    let second = match run("second") {
        Ok(v) => v,
        Err(r) => return vec![r],
    };
    if first != second {
        return vec![CheckResult::fail(
            "assigner_deterministic",
            "two identical calls returned different assignments: the metric has hidden \
             state or nondeterminism, which breaks the reproducibility contract"
                .into(),
        )];
    }
    let mut results = vec![CheckResult::pass(
        "assigner_deterministic",
        format!(
            "two identical calls agreed on all {} observations",
            first.len()
        ),
    )];
    results.push(check_reassign_consistency(assigner, x, tessellation));
    results
}

/// The incremental-reassign invariant, exercised over deterministic structural edits of the
/// given tessellation: every delta's incremental result must equal a fresh
/// full recompute (assignments exactly; winning keys bit for bit when both
/// sides report them).
fn check_reassign_consistency(
    assigner: &dyn CellAssigner,
    x: &Data,
    tessellation: &Tessellation,
) -> CheckResult {
    const NAME: &str = "reassign_consistent";
    let cold = || AssignmentCache::new(vec![0; x.n_rows()], Vec::new());
    let fresh = |t: &Tessellation| assigner.reassign(x, t, AssignmentDelta::FullRecompute, &cold());
    // The reference for *assignments* is `assign_cells`, deliberately not
    // `reassign(FullRecompute)`. The trait's default `reassign` delegates to
    // `assign_cells`, so taking the reference from `reassign` would compare an
    // override against itself and pass a memoising cache that never invalidates.
    // `fresh` is still called, and cross-checked below, because it is the only
    // path that reports winning keys.
    let recompute = |t: &Tessellation| assigner.assign_cells(x, t);
    let prev = match fresh(tessellation) {
        Ok(cache) => cache,
        Err(e) => return CheckResult::fail(NAME, format!("full recompute failed: {e}")),
    };

    let d = tessellation.dims().len();
    let b = tessellation.n_cells();
    let mut variants: Vec<(String, Tessellation, AssignmentDelta)> = Vec::new();
    // CentreAdded: centre 0's coordinates, deterministically shifted.
    {
        let mut centres = tessellation.centres().to_vec();
        for di in 0..d {
            centres.push(centres[di] + 0.31);
        }
        let mut mus = tessellation.mus().to_vec();
        mus.push(0.0);
        let new = Tessellation::new(centres, tessellation.dims().to_vec(), mus)
            .expect("appended centre keeps the tessellation coherent");
        variants.push(("CentreAdded".into(), new, AssignmentDelta::CentreAdded));
    }
    // CentreMoved: every index, coordinates deterministically shifted.
    for moved in 0..b {
        let mut centres = tessellation.centres().to_vec();
        for di in 0..d {
            centres[moved * d + di] += 0.17;
        }
        let new = Tessellation::new(
            centres,
            tessellation.dims().to_vec(),
            tessellation.mus().to_vec(),
        )
        .expect("moving a centre keeps the tessellation coherent");
        variants.push((
            format!("CentreMoved {moved}"),
            new,
            AssignmentDelta::CentreMoved { index: moved },
        ));
    }
    // CentreRemoved: every index (needs ≥ 2 cells to stay valid).
    if b >= 2 {
        for removed in 0..b {
            let mut centres = tessellation.centres().to_vec();
            centres.drain(removed * d..(removed + 1) * d);
            let mut mus = tessellation.mus().to_vec();
            mus.remove(removed);
            let new = Tessellation::new(centres, tessellation.dims().to_vec(), mus)
                .expect("removing a centre keeps the tessellation coherent");
            variants.push((
                format!("CentreRemoved {removed}"),
                new,
                AssignmentDelta::CentreRemoved { index: removed },
            ));
        }
    }

    let count = variants.len();
    for (label, new, delta) in variants {
        let (incremental, recomputed, reference) = match (
            assigner.reassign(x, &new, delta, &prev),
            fresh(&new),
            recompute(&new),
        ) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                return CheckResult::fail(NAME, format!("{label}: reassign failed: {e}"));
            }
        };
        if incremental.assignment() != reference {
            return CheckResult::fail(
                NAME,
                format!(
                    "{label}: the incremental assignment differs from `assign_cells` on the \
                     same tessellation: the reassign fast path is wrong and would silently \
                     corrupt the chain"
                ),
            );
        }
        // `reassign(FullRecompute)` must also agree with `assign_cells`. An
        // override that caches across *both* paths keeps them consistent with
        // each other while both drift from the geometry; only comparing against
        // `assign_cells` sees that.
        if recomputed.assignment() != reference {
            return CheckResult::fail(
                NAME,
                format!(
                    "{label}: `reassign(FullRecompute)` differs from `assign_cells` on the \
                     same tessellation: the two paths disagree, so one of them is stale"
                ),
            );
        }
        let keys_agree = incremental.best_keys().is_empty()
            || recomputed.best_keys().is_empty()
            || incremental
                .best_keys()
                .iter()
                .zip(recomputed.best_keys())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !keys_agree {
            return CheckResult::fail(
                NAME,
                format!(
                    "{label}: incremental winning keys are not bit-identical to a fresh \
                     recompute: stale keys poison every later incremental step"
                ),
            );
        }
    }
    CheckResult::pass(
        NAME,
        format!("{count} delta variants match a full recompute bit for bit"),
    )
}

/// Check a [`PairwiseDistance`] metric on the given fixture: correctness
/// properties, not just repeatability.
///
/// - `keys_finite` / `distance_deterministic`: every (observation, centre)
///   key is finite and bit-identical across two passes.
/// - `self_key_minimal`: an observation's key against itself must not be
///   beaten by any centre. Every strictly increasing transform of a true
///   distance satisfies this; sign errors and inverted keys fail it.
/// - `all_euclidean_claim`: a metric claiming the fast path must return
///   exactly the sum of squared active differences, bit for bit. The blanket
///   kernel bypasses `distance()` entirely on that path, so a wrong claim
///   silently replaces the geometry with Euclidean and no other check sees it.
/// - `key_digest`: an order-pinned FNV-1a digest of every key's bits,
///   reported in the detail. Run the checks on each platform you target and
///   compare digests. A difference means the metric's arithmetic is not
///   bit-portable (usually a std `f64` transcendental where
///   `addivortes::mathsfn` was required), and the same-seed reproducibility
///   promise is void for chains using it. No single-machine check can catch
///   this breach.
pub fn check_distance(
    metric: &dyn PairwiseDistance,
    x: &Data,
    tessellation: &Tessellation,
) -> Vec<CheckResult> {
    let mut results = Vec::new();
    let dims = tessellation.dims();
    let d = dims.len();
    let centres = tessellation.centres();
    let n_cells = tessellation.n_cells();

    // One synthesised-row pass over every (observation, centre) pair, in
    // pinned row-major order, the same synthesis the general kernel performs.
    let keys_pass = || -> Vec<f64> {
        let mut keys = Vec::with_capacity(x.n_rows() * n_cells);
        let mut synthesised = vec![0.0_f64; x.n_cols()];
        for row_index in 0..x.n_rows() {
            let row = x.row(row_index);
            synthesised.copy_from_slice(row);
            for cell in 0..n_cells {
                for (di, &dim) in dims.iter().enumerate() {
                    synthesised[dim] = centres[cell * d + di];
                }
                keys.push(metric.distance(row, &synthesised, dims));
            }
        }
        keys
    };
    let keys = keys_pass();

    if let Some(position) = keys.iter().position(|k| !k.is_finite()) {
        return vec![CheckResult::fail(
            "keys_finite",
            format!(
                "the key for observation {row}, centre {cell} is not finite",
                row = position / n_cells,
                cell = position % n_cells
            ),
        )];
    }
    let repeat = keys_pass();
    let identical = keys
        .iter()
        .zip(&repeat)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    if identical {
        results.push(CheckResult::pass(
            "distance_deterministic",
            format!("{} keys bit-identical across two passes", keys.len()),
        ));
    } else {
        return vec![CheckResult::fail(
            "distance_deterministic",
            "two identical passes returned different key bits: the metric has hidden \
             state or nondeterminism, which breaks the reproducibility contract"
                .into(),
        )];
    }

    // Self-minimality: no centre may score strictly below the observation's
    // key against itself.
    let mut minimal = true;
    'rows: for row_index in 0..x.n_rows() {
        let row = x.row(row_index);
        let self_key = metric.distance(row, row, dims);
        for cell in 0..n_cells {
            let key = keys[row_index * n_cells + cell];
            if key.total_cmp(&self_key) == std::cmp::Ordering::Less {
                results.push(CheckResult::fail(
                    "self_key_minimal",
                    format!(
                        "centre {cell} scores {key:.6e} against observation {row_index}, \
                         below the observation's self-key {self_key:.6e}: the key is not a \
                         monotone transform of a distance (inverted sign or wrong direction?)"
                    ),
                ));
                minimal = false;
                break 'rows;
            }
        }
    }
    if minimal {
        results.push(CheckResult::pass(
            "self_key_minimal",
            "no centre beats an observation's key against itself".into(),
        ));
    }

    // The fast-path claim: `all_euclidean() == true` promises the key is the
    // sum of squared active differences (the kernel then never calls
    // `distance()` at all).
    if metric.all_euclidean() {
        let mut consistent = true;
        'pairs: for row_index in 0..x.n_rows() {
            let row = x.row(row_index);
            for cell in 0..n_cells {
                let mut expected = 0.0_f64;
                for (di, &dim) in dims.iter().enumerate() {
                    let diff = row[dim] - centres[cell * d + di];
                    expected += diff * diff;
                }
                let key = keys[row_index * n_cells + cell];
                if key.to_bits() != expected.to_bits() {
                    results.push(CheckResult::fail(
                        "all_euclidean_claim",
                        format!(
                            "the metric claims the all-Euclidean fast path but its key for \
                             observation {row_index}, centre {cell} is {key:.6e} where the \
                             squared-difference sum is {expected:.6e}: on the fast path the \
                             kernel never calls distance(), so this geometry would be \
                             SILENTLY replaced by Euclidean"
                        ),
                    ));
                    consistent = false;
                    break 'pairs;
                }
            }
        }
        if consistent {
            results.push(CheckResult::pass(
                "all_euclidean_claim",
                "claimed fast-path keys equal the squared-difference sum bit for bit".into(),
            ));
        }
    } else {
        results.push(CheckResult::pass(
            "all_euclidean_claim",
            "the metric does not claim the fast path (nothing to verify)".into(),
        ));
    }

    let digest = bit_digest(keys.iter().copied());
    results.push(CheckResult::pass(
        "key_digest",
        format!(
            "digest of {} keys on this fixture: {digest:016x}. Run the checks on every \
             platform you target and compare; a differing digest means un-pinned maths \
             (std f64 transcendentals instead of addivortes::mathsfn) and voids the \
             same-seed reproducibility promise for chains using this metric",
            keys.len()
        ),
    ));
    results
}

/// Check a coordinate distribution: `sample` draws must match
/// `exp(log_density)` (a Kolmogorov–Smirnov comparison against the numerically
/// integrated density over the sampled range).
///
/// Read the module note first: both sides of this comparison come from the
/// *same object*, so it proves `sample` and `log_density` agree with each
/// other, not that either is the law you meant. A distribution that is the
/// wrong family *consistently* — a Laplace where you intended a Normal,
/// implemented faithfully in both methods — passes. What it catches is the two
/// halves **disagreeing**: a density that was edited without its sampler (or
/// the reverse), a mis-set scale on one side only, a mis-normalisation large
/// enough to bend the CDF.
pub fn check_coordinate_distribution(
    dist: &dyn CoordinateDistribution,
    n_samples: usize,
    rng: &mut dyn rand_core::Rng,
) -> Vec<CheckResult> {
    let mut samples: Vec<f64> = (0..n_samples).map(|_| dist.sample(rng)).collect();
    samples.sort_by(f64::total_cmp);
    let lo = samples[0];
    let hi = samples[n_samples - 1];
    // The window is the observed sample range, with no padding, and that matters.
    //
    // The comparison renormalises the density over the window, so the window must
    // contain no region the sampler cannot reach — otherwise the renormaliser
    // counts mass no sample can ever supply, and the whole theoretical CDF shifts.
    // A 5% pad did exactly that to any law with bounded support: `WrappedNormal`
    // is periodic, its density is perfectly happy outside the circle, and at
    // σ_c ≥ 3 (where it wraps towards uniform) the pad carried ~9% phantom mass
    // and failed the crate's own shipped, correct law on every seed.
    //
    // Unpadded, the window is exactly where the samples are, and the theoretical
    // CDF is the true one conditioned on it — which is the right comparison for a
    // density with mass outside the sample range too (a Normal's tails). Measured
    // against the padded version, this costs nothing: it still reds a 20% scale
    // error and a 0.10 location shift on the same seeds.
    //
    // The epsilon is only for the degenerate case of a point mass, where every
    // sample is identical and the window would otherwise have zero width.
    let pad = if hi - lo < 1e-9 { 1e-6 } else { 0.0 };
    let (lo, hi) = (lo - pad, hi + pad);

    // Numeric CDF of exp(log_density) over [lo, hi] (trapezoid, 4000 panels),
    // normalised over the window.
    let panels = 4000usize;
    let h = (hi - lo) / panels as f64;
    let mut cumulative = vec![0.0_f64; panels + 1];
    let mut previous = mathsfn::exp(dist.log_density(lo));
    for i in 1..=panels {
        let x = lo + h * i as f64;
        let density = mathsfn::exp(dist.log_density(x));
        cumulative[i] = cumulative[i - 1] + 0.5 * (previous + density) * h;
        previous = density;
    }
    let total = cumulative[panels];
    if !(total.is_finite() && total > 0.0) {
        return vec![CheckResult::fail(
            "density_integrates",
            format!("exp(log_density) integrated to {total} over the sampled range"),
        )];
    }

    // KS distance between the empirical CDF and the numeric CDF.
    let cdf = |x: f64| -> f64 {
        let position = ((x - lo) / h).clamp(0.0, panels as f64);
        let index = (position as usize).min(panels - 1);
        let fraction = position - index as f64;
        (cumulative[index] + fraction * (cumulative[index + 1] - cumulative[index])) / total
    };
    let mut d_stat = 0.0_f64;
    for (i, &x) in samples.iter().enumerate() {
        let empirical_hi = (i + 1) as f64 / n_samples as f64;
        let empirical_lo = i as f64 / n_samples as f64;
        let theoretical = cdf(x);
        d_stat = d_stat.max((empirical_hi - theoretical).abs());
        d_stat = d_stat.max((theoretical - empirical_lo).abs());
    }
    // 1.63/√n is the α ≈ 0.01 KS critical value; ×1.5 slack for the numeric
    // CDF approximation. Deterministic given the seeded RNG.
    let threshold = 1.5 * 1.63 / (n_samples as f64).sqrt();
    if d_stat > threshold {
        vec![CheckResult::fail(
            "sample_matches_density",
            format!(
                "KS distance {d_stat:.4} exceeds {threshold:.4}: sample() and \
                 log_density() disagree (different family, location, or scale?)"
            ),
        )]
    } else {
        vec![CheckResult::pass(
            "sample_matches_density",
            format!("KS distance {d_stat:.4} within {threshold:.4} over {n_samples} draws"),
        )]
    }
}

/// Check an inclusion model: weight validity (finite, strictly
/// positive, expected length) **before and after** an update, and update
/// determinism (two identically-seeded runs must agree).
///
/// The post-update check is the one with teeth. Validity before an update only
/// says the constructor works; a model that corrupts its weights on every sweep
/// is a real and common failure (a normaliser dividing by a zero usage total, a
/// concentration that drifts non-positive) that a construct-only check misses.
///
/// These local checks cannot prove an adaptive update statistically *valid*;
/// only the calibration battery can (see the module docs).
pub fn check_inclusion_model<M, F>(
    make_model: F,
    n_covariates: usize,
    usage: &InclusionUsage,
    seed_a: &mut dyn rand_core::Rng,
    seed_b: &mut dyn rand_core::Rng,
) -> Vec<CheckResult>
where
    M: InclusionModel,
    F: Fn() -> M,
{
    let mut results = Vec::new();
    let model = make_model();
    let weights = model.weights();
    if weights.len() != n_covariates {
        results.push(CheckResult::fail(
            "weights_length",
            format!("{} weights for {n_covariates} covariates", weights.len()),
        ));
    } else {
        results.push(CheckResult::pass(
            "weights_length",
            format!("one weight per covariate ({n_covariates})"),
        ));
    }
    match weights.iter().find(|w| !w.is_finite() || **w <= 0.0) {
        Some(w) => results.push(CheckResult::fail(
            "weights_valid",
            format!("weight {w} is not finite and strictly positive"),
        )),
        None => results.push(CheckResult::pass(
            "weights_valid",
            "all weights finite and strictly positive".into(),
        )),
    }

    // Update determinism: same start, same usage, identically-seeded RNGs.
    let run = |rng: &mut dyn rand_core::Rng| -> Result<Vec<f64>> {
        let mut model = make_model();
        InclusionModel::update(&mut model, usage, rng).map_err(|e| {
            crate::engine::error::AddiVortesError::Extension {
                source: std::sync::Arc::new(CheckWrappedError(format!("{e}"))),
            }
        })?;
        Ok(model.weights().to_vec())
    };
    match (run(seed_a), run(seed_b)) {
        (Ok(a), Ok(b)) => {
            // Re-validate the weights the update *produced*. The checks above ran
            // against the freshly-built model, so they only ever saw the starting
            // weights — a model that corrupts its own weights on every sweep (a
            // normaliser that divides by a zero usage total, an α that drifts
            // negative) was green here, and the engine met the damage a long way
            // downstream. The invariant belongs on the post-state too.
            match a
                .iter()
                .position(|w| !w.is_finite() || *w <= 0.0)
                .or((a.len() != n_covariates).then_some(usize::MAX))
            {
                None => results.push(CheckResult::pass(
                    "weights_valid_after_update",
                    format!("all {n_covariates} weights still finite and strictly positive"),
                )),
                Some(usize::MAX) => results.push(CheckResult::fail(
                    "weights_valid_after_update",
                    format!(
                        "update returned {} weights for {n_covariates} covariates: the update \
                         must preserve one weight per covariate",
                        a.len()
                    ),
                )),
                Some(i) => results.push(CheckResult::fail(
                    "weights_valid_after_update",
                    format!(
                        "after update, weight {i} is {}: an inclusion weight must stay finite \
                         and strictly positive. The pre-update weights were valid, so the update \
                         itself is destroying them.",
                        a[i]
                    ),
                )),
            }

            let identical =
                a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
            if identical {
                results.push(CheckResult::pass(
                    "update_deterministic",
                    "two identically-seeded updates produced bit-identical weights".into(),
                ));
            } else {
                results.push(CheckResult::fail(
                    "update_deterministic",
                    "two identically-seeded updates produced different weights: the update \
                     uses hidden state or entropy outside the RNG it is given, which breaks \
                     the reproducibility contract"
                        .into(),
                ));
            }
        }
        (Err(e), _) | (_, Err(e)) => {
            results.push(CheckResult::fail(
                "update_runs",
                format!("update failed: {e}"),
            ));
        }
    }
    results
}

/// The minimal fixture context every scale check shares: one evenly-spaced
/// scaled column and the default structural machinery, enough for a variance
/// ensemble to run its own backfitting inside `update`.
fn with_scale_ctx<R>(y: &[f64], fit: &[f64], body: impl FnOnce(&ScaleCtx) -> R) -> R {
    let n = y.len();
    let column: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
    let x = Data::new(column, n, 1).expect("fixture shape is coherent");
    let move_set = crate::extensions::moves::default_move_set().expect("the paper move set builds");
    let assigner =
        crate::extensions::distance::default_assigner(vec![crate::engine::data::Metric::Euclidean]);
    let coord_dists: Vec<std::sync::Arc<dyn CoordinateDistribution>> = vec![std::sync::Arc::new(
        crate::extensions::coord::EuclideanNormal::new(0.8).unwrap(),
    )];
    let weights_enc = [1.0_f64];
    let ctx = ScaleCtx {
        y,
        fit,
        x: &x,
        move_set: &move_set,
        assigner: assigner.as_ref(),
        omega: 0.5,
        lambda_c: 25.0,
        nu: 6.0,
        lambda: 0.02,
        p_enc: 1,
        coord_dists: &coord_dists,
        weights_enc: &weights_enc,
        response_weights: None,
    };
    body(&ctx)
}

/// Check that a [`ScaleModel`] actually **learns σ² from the data**.
///
/// **Opt in only if your model is supposed to.** This check is deliberately not
/// part of [`check_scale_model`], because a pinned scale legitimately ignores
/// the data: the crate's own [`PinnedSigma`](crate::extensions::scale::PinnedSigma)
/// holds σ² = 1 for the probit augmentation and is *correct* to do so. Making
/// the probe mandatory would fail correct code, which is worse than missing a
/// bug — it teaches you to distrust every other verdict here.
///
/// So this is where you declare intent: "my model learns σ² from the residuals."
/// Both properties below then follow from that intent alone. They hold for any
/// recipe — conjugate, weighted, or a whole variance ensemble — because neither
/// assumes a functional form:
///
/// - `sigma_sq_responds_to_residuals`: inflate every residual fourfold and σ²
///   must come out **larger**. A model that quietly ignores the data and returns
///   prior draws is finite, positive, deterministic and perfectly reproducible,
///   so [`check_scale_model`] passes it happily; only this sees it. So does an
///   update with an inverted sense, where σ² *shrinks* as the data gets noisier.
///
/// - `sigma_sq_uses_residuals`: shift `y` and `fit` by the same constant. The
///   residuals `y − fit` are unchanged, so σ² must be **bit-identical**. This is
///   what catches the classic slip of scoring the raw response instead of the
///   residual — a bug that responds to the data perfectly well, and so passes the
///   first probe. It is why `fit` must be **non-zero** in your fixture: at
///   `fit = 0` the residual *is* the response and the bug is invisible.
///
/// `y` and `fit` are a small scaled-space fixture (equal lengths, `fit` not
/// identically zero). The two runs are identically seeded, so any difference is
/// the data talking, not the RNG.
///
/// As ever: local necessary conditions, not proof. A model that passes both is
/// still only validated for real inference by the Geweke/SBC battery
/// (`tests/calibration_acceptance.rs` is the copyable worked example).
pub fn check_scale_model_learns_from_data<S, F>(
    make_model: F,
    y: &[f64],
    fit: &[f64],
    seed: u64,
) -> Vec<CheckResult>
where
    S: ScaleModel,
    F: Fn() -> S,
{
    debug_assert!(y.len() >= 2 && y.len() == fit.len());
    let mut results = Vec::new();

    if fit.iter().all(|f| *f == 0.0) {
        return vec![CheckResult::fail(
            "sigma_sq_uses_residuals",
            "the fixture's `fit` is identically zero, so the residual y − fit *is* y and a model \
             scoring the raw response cannot be told apart from one scoring residuals. Supply a \
             non-zero fit."
                .into(),
        )];
    }

    // Same seed every time: any difference in σ² is the data, not the RNG.
    let sigma_sq_for = |y: &[f64], fit: &[f64]| -> Result<f64> {
        let mut model = make_model();
        let mut rng = <rand_chacha::ChaCha8Rng as rand_core::SeedableRng>::seed_from_u64(seed);
        with_scale_ctx(y, fit, |ctx| {
            ScaleModel::update(&mut model, ctx, &mut rng).map_err(|e| {
                crate::engine::error::AddiVortesError::Extension {
                    source: std::sync::Arc::new(CheckWrappedError(format!("{e}"))),
                }
            })
        })?;
        Ok(model.sigma_sq())
    };

    // Probe 1: fourfold residuals must raise σ².
    let loud_y: Vec<f64> = y.iter().zip(fit).map(|(y, f)| f + 4.0 * (y - f)).collect();
    match (sigma_sq_for(y, fit), sigma_sq_for(&loud_y, fit)) {
        (Ok(base), Ok(loud)) => results.push(if loud > base {
            CheckResult::pass(
                "sigma_sq_responds_to_residuals",
                format!("fourfold residuals raise σ² from {base:.6e} to {loud:.6e}"),
            )
        } else {
            CheckResult::fail(
                "sigma_sq_responds_to_residuals",
                format!(
                    "fourfold residuals moved σ² from {base:.6e} to {loud:.6e}, which is not an \
                     increase. A model that ignores the data and returns prior draws lands \
                     exactly here, and it is finite, positive and perfectly deterministic, so \
                     every other scale check passes it. So does an update whose sense is \
                     inverted."
                ),
            )
        }),
        (Err(e), _) | (_, Err(e)) => {
            return vec![CheckResult::fail(
                "update_runs",
                format!("update failed: {e}"),
            )];
        }
    }

    // Probe 2: shifting y and fit together leaves the residuals alone.
    const SHIFT: f64 = 0.75;
    let shifted_y: Vec<f64> = y.iter().map(|v| v + SHIFT).collect();
    let shifted_fit: Vec<f64> = fit.iter().map(|v| v + SHIFT).collect();
    match (sigma_sq_for(y, fit), sigma_sq_for(&shifted_y, &shifted_fit)) {
        // A tolerance, not a bit compare, and the template proved why: in floating
        // point `(y + c) − (fit + c)` is not bit-identical to `y − fit`, so a
        // *correct* model shifts σ² in the last few ulps and a bitwise test reds
        // it. The bug being hunted moves σ² by orders of magnitude; it does not
        // hide down there.
        (Ok(base), Ok(shifted)) if (base - shifted).abs() <= 1e-9 * base.abs().max(1.0) => {
            results.push(CheckResult::pass(
                "sigma_sq_uses_residuals",
                format!(
                    "shifting y and fit together by {SHIFT} leaves σ² unchanged \
                     ({base:.6e}): the update scores residuals, not the raw response"
                ),
            ));
        }
        (Ok(base), Ok(shifted)) => results.push({
            CheckResult::fail(
                "sigma_sq_uses_residuals",
                format!(
                    "shifting y and fit together by {SHIFT} moved σ² from {base:.6e} to \
                     {shifted:.6e}. The residuals y − fit are unchanged, so σ² must not move: \
                     the update is scoring the raw response somewhere it should be scoring the \
                     residual. That bug responds to the data perfectly well, so the \
                     residual-magnitude probe does not see it."
                ),
            )
        }),
        (Err(e), _) | (_, Err(e)) => results.push(CheckResult::fail(
            "update_runs",
            format!("update failed: {e}"),
        )),
    }

    results
}

/// Check a [`ScaleModel`]: the update runs on a minimal
/// single-column fixture context, σ² is finite and strictly positive after
/// it, any per-observation precisions are well-formed (length n, finite,
/// strictly positive), and two identically-seeded updates agree bit for bit.
///
/// This check says nothing about whether σ² **responds to the data** — a model
/// that ignores it entirely and returns prior draws passes every line of it. If
/// your model is meant to learn σ², opt in to
/// [`check_scale_model_learns_from_data`] as well. It is deliberately separate:
/// [`PinnedSigma`](crate::extensions::scale::PinnedSigma) legitimately ignores
/// the data, and a mandatory probe would fail correct code.
///
/// `y` and `fit` are a small scaled-space fixture (equal lengths). The
/// context handed to `update` carries a deterministic one-column design over
/// `[−0.5, 0.5]`, the paper move set, the default assigner and coordinate
/// law, and unit inclusion weights, enough for a variance ensemble to run
/// its own backfitting.
///
/// Local necessary conditions, not proof: a scale model destined for real
/// inference is validated by the golden chain (defaults) and the
/// Geweke/SBC calibration battery, exactly like a cell model.
pub fn check_scale_model<S, F>(
    make_model: F,
    y: &[f64],
    fit: &[f64],
    seed_a: &mut dyn rand_core::Rng,
    seed_b: &mut dyn rand_core::Rng,
) -> Vec<CheckResult>
where
    S: ScaleModel,
    F: Fn() -> S,
{
    debug_assert!(y.len() >= 2 && y.len() == fit.len());
    let mut results = Vec::new();
    let n = y.len();

    // One update from each identically-seeded RNG; capture (σ², precisions).
    let run = |rng: &mut dyn rand_core::Rng| -> Result<(f64, Option<Vec<f64>>)> {
        let mut model = make_model();
        with_scale_ctx(y, fit, |ctx| {
            ScaleModel::update(&mut model, ctx, rng).map_err(|e| {
                crate::engine::error::AddiVortesError::Extension {
                    source: std::sync::Arc::new(CheckWrappedError(format!("{e}"))),
                }
            })
        })?;
        Ok((
            model.sigma_sq(),
            ScaleModel::precisions(&model).map(<[f64]>::to_vec),
        ))
    };
    let (a, b) = (run(seed_a), run(seed_b));
    match (&a, &b) {
        (Ok((sigma_sq, precisions)), Ok(_)) => {
            if sigma_sq.is_finite() && *sigma_sq > 0.0 {
                results.push(CheckResult::pass(
                    "sigma_sq_valid",
                    format!("sigma_sq {sigma_sq} is finite and strictly positive after update"),
                ));
            } else {
                results.push(CheckResult::fail(
                    "sigma_sq_valid",
                    format!(
                        "sigma_sq {sigma_sq} after update: must be finite and strictly \
                         positive (the sampler release-asserts this every sweep)"
                    ),
                ));
            }
            match precisions {
                None => results.push(CheckResult::pass(
                    "precisions_valid",
                    "no per-observation precisions (homoscedastic model)".into(),
                )),
                Some(p) if p.len() != n => results.push(CheckResult::fail(
                    "precisions_valid",
                    format!("{} precisions for {n} observations", p.len()),
                )),
                Some(p) => match p.iter().find(|w| !w.is_finite() || **w <= 0.0) {
                    Some(w) => results.push(CheckResult::fail(
                        "precisions_valid",
                        format!("precision {w} is not finite and strictly positive"),
                    )),
                    None => results.push(CheckResult::pass(
                        "precisions_valid",
                        format!("{n} precisions, all finite and strictly positive"),
                    )),
                },
            }
        }
        (Err(e), _) | (_, Err(e)) => {
            results.push(CheckResult::fail(
                "update_runs",
                format!("update failed: {e}"),
            ));
            return results;
        }
    }
    let (Ok((sigma_a, prec_a)), Ok((sigma_b, prec_b))) = (a, b) else {
        unreachable!("errors returned above");
    };
    let identical = sigma_a.to_bits() == sigma_b.to_bits()
        && match (&prec_a, &prec_b) {
            (None, None) => true,
            (Some(pa), Some(pb)) => {
                pa.len() == pb.len() && pa.iter().zip(pb).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        };
    if identical {
        results.push(CheckResult::pass(
            "update_deterministic",
            "two identically-seeded updates produced bit-identical σ² and precisions".into(),
        ));
    } else {
        results.push(CheckResult::fail(
            "update_deterministic",
            "two identically-seeded updates disagree: the update uses hidden state or \
             entropy outside the RNG it is given, which breaks the reproducibility contract"
                .into(),
        ));
    }
    results
}

/// Check a [`CellBasis`]: the per-observation basis row z(x).
///
/// This is the half of the basis point a researcher actually writes — the payload model
/// is usually the shelf's — and it had no check at all.
///
/// - `basis_width`: `q >= 1`, and `row` must fill every one of the `q` entries
///   with a finite value. The engine hands `row` an already-sized buffer and
///   trusts it to write all of it; a partially-filled row silently carries
///   whatever was in the buffer before.
/// - `row_deterministic`: the same input row must give bit-identical output.
///   The engine evaluates z(x) once per fit and again per row at predict time,
///   so a basis with interior state makes prediction disagree with training.
/// - `max_column_claim_exact`: **the oracle.** The engine bounds-checks
///   `max_column()` against the design exactly once, then trusts it. A basis
///   that reads column 7 while declaring `Some(2)` sails through that check on a
///   3-column design and then indexes out of range in the hot loop. So the check
///   *discovers* which columns the basis reads — by perturbing each one and
///   watching whether the output row changes — and requires every column it
///   actually reads to be within what it declared.
///
///   Perturbation gives a **lower bound** on what is read: a column multiplied
///   by zero, or entering through a step function, can be read without any
///   perturbation showing it. So this check has no false positives (anything it
///   reports is genuinely read and genuinely out of the declared range) but it
///   can miss. Over-declaring `max_column` is safe and is never reported.
/// - `row_digest`: the cross-platform portability probe.
///
/// `x_rows` are scaled-space design rows, all of the same width; use rows with
/// distinct, non-zero entries, since an all-zero column cannot reveal a read.
pub fn check_cell_basis(basis: &dyn CellBasis, x_rows: &[Vec<f64>]) -> Vec<CheckResult> {
    debug_assert!(!x_rows.is_empty());
    let mut results = Vec::new();
    let q = basis.q();
    let p = x_rows[0].len();
    debug_assert!(x_rows.iter().all(|row| row.len() == p));

    if q == 0 {
        return vec![CheckResult::fail(
            "basis_width",
            "q() is 0: a basis writes at least one value per row (an intercept is q = 1)".into(),
        )];
    }

    // `row` must fill the whole buffer. Seeding it with NaN is what makes an
    // unwritten entry visible: the engine would otherwise carry stale values.
    let evaluate = |x_row: &[f64]| -> Vec<f64> {
        let mut out = vec![f64::NAN; q];
        basis.row(x_row, &mut out);
        out
    };

    let rows: Vec<Vec<f64>> = x_rows.iter().map(|x| evaluate(x)).collect();
    match rows
        .iter()
        .enumerate()
        .find_map(|(i, z)| z.iter().position(|v| !v.is_finite()).map(|j| (i, j)))
    {
        Some((i, j)) => results.push(CheckResult::fail(
            "basis_width",
            format!(
                "z(x) entry {j} of row {i} is {}: `row` must write all {q} entries with finite \
                 values. The engine hands you an already-sized buffer and trusts you to fill it; \
                 an entry you never write keeps whatever was there before.",
                rows[i][j]
            ),
        )),
        None => results.push(CheckResult::pass(
            "basis_width",
            format!("every row writes all q = {q} entries, all finite"),
        )),
    }

    let repeat: Vec<Vec<f64>> = x_rows.iter().map(|x| evaluate(x)).collect();
    let pure = rows
        .iter()
        .flatten()
        .zip(repeat.iter().flatten())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    results.push(if pure {
        CheckResult::pass(
            "row_deterministic",
            "the same design row gives a bit-identical basis row when asked twice".into(),
        )
    } else {
        CheckResult::fail(
            "row_deterministic",
            "the same design row gave different basis rows on two calls: a CellBasis must be \
             pure. The engine evaluates z(x) once per fit and again per row at predict time, so \
             interior state makes prediction disagree with training."
                .into(),
        )
    });

    // The oracle: find out what this basis really reads.
    let declared = basis.max_column();
    let mut read_beyond: Vec<usize> = Vec::new();
    for column in 0..p {
        let within = declared.is_some_and(|max| column <= max);
        if within {
            continue; // reading it is exactly what was declared
        }
        // Two probes, because one perturbation can land on a stationary point.
        let sensitive = [0.37_f64, -0.61].iter().any(|delta| {
            x_rows.iter().enumerate().any(|(i, x)| {
                let mut probe = x.clone();
                probe[column] += delta;
                evaluate(&probe)
                    .iter()
                    .zip(&rows[i])
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            })
        });
        if sensitive {
            read_beyond.push(column);
        }
    }
    results.push(if read_beyond.is_empty() {
        CheckResult::pass(
            "max_column_claim_exact",
            match declared {
                Some(max) => format!(
                    "declares max_column() = {max}, and no column beyond it changes z(x)"
                ),
                None => "declares max_column() = None, and no column changes z(x) (a pure \
                         intercept)"
                    .into(),
            },
        )
    } else {
        CheckResult::fail(
            "max_column_claim_exact",
            format!(
                "z(x) changes with column(s) {read_beyond:?}, but max_column() declares {declared:?}. \
                 The engine bounds-checks that claim against the design exactly once and then \
                 trusts it, so an under-declared basis passes `fit` and then indexes out of range \
                 in the hot loop. Declare the largest column you read (over-declaring is safe)."
            ),
        )
    });

    let digest = bit_digest(rows.iter().flatten().copied());
    results.push(CheckResult::pass(
        "row_digest",
        format!(
            "digest of {} basis values on this fixture: {digest:016x}. Run the check on every \
             platform you target and compare; a differing digest means un-pinned maths (std f64 \
             transcendentals instead of addivortes::mathsfn) and voids the same-seed \
             reproducibility promise for chains using this basis",
            rows.len() * q
        ),
    ));

    results
}

/// Check a basis-payload [`CellModel`]: the same three verifications as
/// [`check_cell_model`], extended to vector payloads. Sufficiency ×2 over
/// [`CellStats::record_basis`] accumulation, the marginal-vs-Monte-Carlo Bayes
/// factor under the linear working likelihood `rᵢ ~ N(zᵢ·β, σ²/wᵢ)`, and
/// cell-local SBC of the coefficient draw along the linear predictor at the first
/// fixture row.
///
/// Generic over any `CellModel` whose `cell_basis()` is `true`, not just the
/// shelf [`LinearGaussianModel`](crate::extensions::basis::LinearGaussianModel):
/// everything it needs ([`CellStats::record_basis`],
/// [`CellModel::payload_width`], [`CellModel::draw_cell_payload`]) is on the
/// traits, so a caller-written basis payload is checked exactly like the
/// shipped one.
///
/// `basis_rows` are the per-observation basis rows (length `payload_width()`
/// each), `observations`/`weights` the working fixture, all scaled space.
///
/// Its Monte-Carlo Bayes factor and SBC machinery are the same shapes the
/// sibling checks exercise with negative controls
/// (`dropped_normalising_term_fails_bayes_factor`,
/// `dropped_variance_prior_term_fails_bayes_factor`), and the same caveat
/// applies: the Bayes-factor reference is built from the **model's own** prior
/// draws, so it tests the marginal *given* that prior, not the prior itself.
pub fn check_basis_cell_model<M: CellModel>(
    model: &M,
    basis_rows: &[Vec<f64>],
    observations: &[f64],
    weights: &[f64],
    sigma_sq: f64,
    rng: &mut dyn rand_core::Rng,
) -> Vec<CheckResult> {
    debug_assert!(observations.len() >= 4);
    debug_assert!(basis_rows.len() == observations.len() && weights.len() == observations.len());
    let mut results = Vec::new();
    let n = observations.len();
    let q = model.payload_width();

    // The model must actually be a basis payload, and its width must match the
    // rows it is being fed. The engine validates this at `fit`; saying it here
    // turns a late, confusing error into an immediate one.
    if !model.cell_basis() {
        return vec![CheckResult::fail(
            "declares_cell_basis",
            "the model's `cell_basis()` is false, so the engine will treat it as a scalar \
             family and never call `record_basis` or `draw_cell_payload`. A basis payload must \
             return true, or it silently ignores the basis entirely."
                .into(),
        )];
    }
    if let Some(bad) = basis_rows.iter().position(|z| z.len() != q) {
        return vec![CheckResult::fail(
            "declares_cell_basis",
            format!(
                "the model declares payload_width() = {q} but basis row {bad} has length {}: \
                 the basis and the payload must agree on q, and `fit` rejects a mismatch",
                basis_rows[bad].len()
            ),
        )];
    }
    results.push(CheckResult::pass(
        "declares_cell_basis",
        format!("declares a basis payload of width q = {q}, matching the fixture rows"),
    ));

    // A payload draw for one cell: `draw_cell_payload` is flattened row-major,
    // q values per cell, so cell 0's coefficient vector is the first q of them.
    let draw_first_payload = |stats: &[M::Stats],
                              rng: &mut dyn rand_core::Rng|
     -> std::result::Result<Vec<f64>, String> {
        let payload = model
            .draw_cell_payload(stats, sigma_sq, rng)
            .map_err(|e| format!("draw_cell_payload failed: {e}"))?;
        if payload.len() < q {
            return Err(format!(
                "draw_cell_payload returned {} values for a 1-cell statistic, expected q = {q}",
                payload.len()
            ));
        }
        Ok(payload[..q].to_vec())
    };

    let accumulate = |order: &[usize], assignment: &dyn Fn(usize) -> usize, n_cells: usize| {
        let mut stats = vec![M::Stats::default(); n_cells];
        for &i in order {
            stats[assignment(i)].record_basis(&basis_rows[i], observations[i], weights[i]);
        }
        stats
    };
    let forward: Vec<usize> = (0..n).collect();
    let reversed: Vec<usize> = (0..n).rev().collect();

    // --- order invariance ---------------------------------------------------
    let one_cell = accumulate(&forward, &|_| 0, 1);
    let one_cell_reversed = accumulate(&reversed, &|_| 0, 1);
    let (a, b) = (
        model.log_marginal_terms(&one_cell, sigma_sq).unwrap(),
        model
            .log_marginal_terms(&one_cell_reversed, sigma_sq)
            .unwrap(),
    );
    if (a - b).abs() <= 1e-9 * a.abs().max(1.0) {
        results.push(CheckResult::pass(
            "stats_order_invariance",
            format!("marginal terms agree under reversed accumulation ({a:.6e})"),
        ));
    } else {
        results.push(CheckResult::fail(
            "stats_order_invariance",
            format!(
                "marginal terms differ under reversed accumulation ({a:.6e} vs {b:.6e}): \
                 the block statistic is not sufficient"
            ),
        ));
    }

    // --- merge / remove consistency ------------------------------------------
    let half = n / 2;
    let mut merged = accumulate(&forward[..half], &|_| 0, 1).remove(0);
    let second = accumulate(&forward[half..], &|_| 0, 1).remove(0);
    merged.merge(&second);
    let merged_terms = model
        .log_marginal_terms(std::slice::from_ref(&merged), sigma_sq)
        .unwrap();
    if (merged_terms - a).abs() <= 1e-9 * a.abs().max(1.0) {
        let mut undone = merged.clone();
        undone.remove(&second);
        let first = accumulate(&forward[..half], &|_| 0, 1).remove(0);
        let undone_terms = model
            .log_marginal_terms(std::slice::from_ref(&undone), sigma_sq)
            .unwrap();
        let first_terms = model
            .log_marginal_terms(std::slice::from_ref(&first), sigma_sq)
            .unwrap();
        if (undone_terms - first_terms).abs() <= 1e-6 * first_terms.abs().max(1.0) {
            results.push(CheckResult::pass(
                "stats_merge_consistency",
                "merge equals sequential accumulation and remove undoes merge".into(),
            ));
        } else {
            results.push(CheckResult::fail(
                "stats_merge_consistency",
                "remove does not undo merge: the subtract path of the block statistic is wrong"
                    .into(),
            ));
        }
    } else {
        results.push(CheckResult::fail(
            "stats_merge_consistency",
            format!(
                "merging two partial accumulations gives {merged_terms:.6e} but sequential \
                 accumulation gives {a:.6e}: the add path of the block statistic is wrong"
            ),
        ));
    }

    // --- marginal ↔ Bayes-factor agreement -----------------------------------
    let two_cell = accumulate(&forward, &|i| usize::from(i >= half), 2);
    let claimed = model.log_marginal_terms(&two_cell, sigma_sq).unwrap()
        - model.log_marginal_terms(&one_cell, sigma_sq).unwrap();
    let n_mc = 40_000;
    let prior_stats = vec![M::Stats::default()];
    let log_bayes = |indices: &[usize], rng: &mut dyn rand_core::Rng| -> f64 {
        let mut max = f64::NEG_INFINITY;
        let mut terms = Vec::with_capacity(n_mc);
        for _ in 0..n_mc {
            // The zero-data posterior is the prior, for any conjugate model.
            let beta = draw_first_payload(&prior_stats, rng).expect("prior payload draw");
            let mut log_lik = 0.0;
            for &i in indices {
                let predictor: f64 = basis_rows[i].iter().zip(&beta).map(|(z, b)| z * b).sum();
                let residual = observations[i] - predictor;
                log_lik += 0.5 * mathsfn::ln(weights[i] / (2.0 * std::f64::consts::PI * sigma_sq))
                    - weights[i] * residual * residual / (2.0 * sigma_sq);
            }
            max = max.max(log_lik);
            terms.push(log_lik);
        }
        let sum: f64 = terms.iter().map(|t| mathsfn::exp(t - max)).sum();
        max + mathsfn::ln(sum / n_mc as f64)
    };
    let estimate = (log_bayes(&forward[..half], rng) + log_bayes(&forward[half..], rng))
        - log_bayes(&forward, rng);
    if (estimate - claimed).abs() <= 0.15 {
        results.push(CheckResult::pass(
            "marginal_bayes_factor",
            format!(
                "claimed 2-vs-1-cell log Bayes factor {claimed:.4} matches the Monte-Carlo \
                 estimate {estimate:.4}"
            ),
        ));
    } else {
        results.push(CheckResult::fail(
            "marginal_bayes_factor",
            format!(
                "claimed 2-vs-1-cell log Bayes factor {claimed:.4} disagrees with the \
                 Monte-Carlo estimate {estimate:.4}: log_marginal_terms is not the \
                 integrated linear-working-likelihood marginal for this prior"
            ),
        ));
    }

    // --- cell-local SBC of the coefficient draw ------------------------------
    // Rank the prior-drawn linear predictor at the first fixture row among
    // posterior draws of the same functional (a joint direction of β).
    let replications = 400;
    let posterior_draws = 19;
    let cell_size = 4.min(n);
    let probe = &basis_rows[0];
    let mut bins = vec![0usize; posterior_draws + 1];
    for _ in 0..replications {
        let prior_stats = vec![M::Stats::default()];
        let beta_star = draw_first_payload(&prior_stats, rng).expect("prior payload draw");
        let predictor_star: f64 = probe.iter().zip(&beta_star).map(|(z, b)| z * b).sum();
        let mut stats = M::Stats::default();
        for i in 0..cell_size {
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut *rng);
            let mean: f64 = basis_rows[i]
                .iter()
                .zip(&beta_star)
                .map(|(z, b)| z * b)
                .sum();
            let value = mean + (sigma_sq / weights[i]).sqrt() * z;
            stats.record_basis(&basis_rows[i], value, weights[i]);
        }
        let stats = vec![stats];
        let mut rank = 0usize;
        for _ in 0..posterior_draws {
            let beta = draw_first_payload(&stats, rng).expect("posterior payload draw");
            let predictor: f64 = probe.iter().zip(&beta).map(|(z, b)| z * b).sum();
            if predictor < predictor_star {
                rank += 1;
            }
        }
        bins[rank] += 1;
    }
    let expected = replications as f64 / bins.len() as f64;
    let chi_sq: f64 = bins
        .iter()
        .map(|&count| {
            let diff = count as f64 - expected;
            diff * diff / expected
        })
        .sum();
    if chi_sq <= 43.8 {
        results.push(CheckResult::pass(
            "cell_value_sbc",
            format!(
                "coefficient-draw SBC ranks are uniform along the probe direction \
                 (χ² = {chi_sq:.1} over {} bins)",
                bins.len()
            ),
        ));
    } else {
        results.push(CheckResult::fail(
            "cell_value_sbc",
            format!(
                "coefficient-draw SBC ranks are non-uniform (χ² = {chi_sq:.1} over {} bins): \
                 the coefficient draw is not the conjugate posterior",
                bins.len()
            ),
        ));
    }
    results
}

/// Check a variance-family [`CellModel`] (the second conjugate
/// family of the universal triple, where cells hold a variance factor s², e.g.
/// [`InvChiSqCellModel`](crate::extensions::cell_model::InvChiSqCellModel)): the same
/// three verifications as [`check_cell_model`] (sufficiency ×2,
/// marginal-vs-Monte-Carlo Bayes factor, cell-local SBC) under the variance
/// working likelihood `ẽᵢ ~ N(0, s²)`, where the recorded value is the
/// squared residual `ẽ²ᵢ` (H paper Eq. 8). `check_cell_model` itself
/// hardcodes the mean-family Gaussian working likelihood, so variance cells
/// get this sibling instead of a false red there.
///
/// `sq_observations` is a small fixture of squared scale-free residuals
/// (non-negative, hard assignment; weights are 1 by contract).
pub fn check_variance_cell_model<M: CellModel>(
    model: &M,
    sq_observations: &[f64],
    rng: &mut dyn rand_core::Rng,
) -> Vec<CheckResult> {
    debug_assert!(sq_observations.len() >= 4);
    debug_assert!(sq_observations.iter().all(|v| *v >= 0.0));
    let mut results = Vec::new();
    let n = sq_observations.len();

    let accumulate = |order: &[usize], assignment: &dyn Fn(usize) -> usize, n_cells: usize| {
        let mut stats = vec![M::Stats::default(); n_cells];
        for &i in order {
            stats[assignment(i)].record(sq_observations[i], 1.0);
        }
        stats
    };
    let forward: Vec<usize> = (0..n).collect();
    let reversed: Vec<usize> = (0..n).rev().collect();
    // The sigma_sq argument is inert for variance families (the noise level
    // is the cell value); 1.0 keeps the call sites uniform.
    let sigma_sq = 1.0;

    // --- order invariance ---------------------------------------------------
    let one_cell = accumulate(&forward, &|_| 0, 1);
    let one_cell_reversed = accumulate(&reversed, &|_| 0, 1);
    match (
        model.log_marginal_terms(&one_cell, sigma_sq),
        model.log_marginal_terms(&one_cell_reversed, sigma_sq),
    ) {
        (Ok(a), Ok(b)) if (a - b).abs() <= 1e-9 * a.abs().max(1.0) => {
            results.push(CheckResult::pass(
                "stats_order_invariance",
                format!("marginal terms agree under reversed accumulation ({a:.6e})"),
            ));
        }
        (Ok(a), Ok(b)) => {
            results.push(CheckResult::fail(
                "stats_order_invariance",
                format!(
                    "marginal terms differ under reversed accumulation ({a:.6e} vs {b:.6e}): \
                     the statistic is not sufficient (it depends on observation order)"
                ),
            ));
        }
        (Err(e), _) | (_, Err(e)) => {
            results.push(CheckResult::fail(
                "stats_order_invariance",
                format!("log_marginal_terms failed: {e}"),
            ));
        }
    }

    // --- merge / remove consistency ------------------------------------------
    let half = n / 2;
    let mut merged = accumulate(&forward[..half], &|_| 0, 1).remove(0);
    let second = accumulate(&forward[half..], &|_| 0, 1).remove(0);
    merged.merge(&second);
    match (
        model.log_marginal_terms(std::slice::from_ref(&merged), sigma_sq),
        model.log_marginal_terms(&one_cell, sigma_sq),
    ) {
        (Ok(a), Ok(b)) if (a - b).abs() <= 1e-9 * b.abs().max(1.0) => {
            let mut undone = merged.clone();
            undone.remove(&second);
            let first = accumulate(&forward[..half], &|_| 0, 1).remove(0);
            let undone_terms = model
                .log_marginal_terms(std::slice::from_ref(&undone), sigma_sq)
                .ok();
            let first_terms = model
                .log_marginal_terms(std::slice::from_ref(&first), sigma_sq)
                .ok();
            match (undone_terms, first_terms) {
                (Some(u), Some(f)) if (u - f).abs() <= 1e-6 * f.abs().max(1.0) => {
                    results.push(CheckResult::pass(
                        "stats_merge_consistency",
                        "merge equals sequential accumulation and remove undoes merge".into(),
                    ));
                }
                _ => {
                    results.push(CheckResult::fail(
                        "stats_merge_consistency",
                        "remove does not undo merge: the subtract path of the sufficient \
                         statistic is wrong"
                            .into(),
                    ));
                }
            }
        }
        (Ok(a), Ok(b)) => {
            results.push(CheckResult::fail(
                "stats_merge_consistency",
                format!(
                    "merging two partial accumulations gives {a:.6e} but sequential \
                     accumulation gives {b:.6e}: the add path of the sufficient statistic \
                     is wrong (double-counting or dropped terms?)"
                ),
            ));
        }
        (Err(e), _) | (_, Err(e)) => {
            results.push(CheckResult::fail(
                "stats_merge_consistency",
                format!("log_marginal_terms failed: {e}"),
            ));
        }
    }

    // --- marginal ↔ Bayes-factor agreement (conjugacy of the marginal) ------
    let two_cell = accumulate(&forward, &|i| usize::from(i >= half), 2);
    let claimed = match (
        model.log_marginal_terms(&two_cell, sigma_sq),
        model.log_marginal_terms(&one_cell, sigma_sq),
    ) {
        (Ok(two), Ok(one)) => Some(two - one),
        _ => None,
    };
    if let Some(claimed) = claimed {
        let n_mc = 40_000;
        let prior_stats = vec![M::Stats::default()];
        // log E_prior[∏_{i∈cell} N(ẽᵢ; 0, s²)] per cell, streaming
        // log-sum-exp over prior draws of s² (zero-data posterior = prior).
        let log_bayes = |indices: &[usize], rng: &mut dyn rand_core::Rng| -> Option<f64> {
            let mut max = f64::NEG_INFINITY;
            let mut terms = Vec::with_capacity(n_mc);
            for _ in 0..n_mc {
                let s_sq = model.draw_cell_values(&prior_stats, sigma_sq, rng).ok()?[0];
                let mut log_lik = 0.0;
                for &i in indices {
                    log_lik += -0.5 * mathsfn::ln(2.0 * std::f64::consts::PI * s_sq)
                        - sq_observations[i] / (2.0 * s_sq);
                }
                max = max.max(log_lik);
                terms.push(log_lik);
            }
            let sum: f64 = terms.iter().map(|t| mathsfn::exp(t - max)).sum();
            Some(max + mathsfn::ln(sum / n_mc as f64))
        };
        let estimate = (|| {
            let cell_a = log_bayes(&forward[..half], rng)?;
            let cell_b = log_bayes(&forward[half..], rng)?;
            let joint = log_bayes(&forward, rng)?;
            Some((cell_a + cell_b) - joint)
        })();
        match estimate {
            Some(mc) if (mc - claimed).abs() <= 0.15 => {
                results.push(CheckResult::pass(
                    "marginal_bayes_factor",
                    format!(
                        "claimed 2-vs-1-cell log Bayes factor {claimed:.4} matches the \
                         Monte-Carlo estimate {mc:.4}"
                    ),
                ));
            }
            Some(mc) => {
                results.push(CheckResult::fail(
                    "marginal_bayes_factor",
                    format!(
                        "claimed 2-vs-1-cell log Bayes factor {claimed:.4} disagrees with the \
                         Monte-Carlo estimate {mc:.4}: log_marginal_terms is not the \
                         integrated variance-working-likelihood marginal for this prior \
                         (wrong normalising term, wrong prior, or a dropped \
                         cell-count-dependent factor)"
                    ),
                ));
            }
            None => {
                results.push(CheckResult::fail(
                    "marginal_bayes_factor",
                    "draw_cell_values failed while sampling the prior".into(),
                ));
            }
        }
    } else {
        results.push(CheckResult::fail(
            "marginal_bayes_factor",
            "log_marginal_terms failed on the fixture".into(),
        ));
    }

    // --- cell-local SBC of the cell-value draw (conjugacy of the draw) ------
    let replications = 400;
    let posterior_draws = 19;
    let cell_size = 4.min(n);
    let mut bins = vec![0usize; posterior_draws + 1];
    let mut sbc_failed = None;
    'sbc: for _ in 0..replications {
        let prior_stats = vec![M::Stats::default()];
        let s_sq_star = match model.draw_cell_values(&prior_stats, sigma_sq, rng) {
            Ok(v) => v[0],
            Err(e) => {
                sbc_failed = Some(format!("prior draw failed: {e}"));
                break 'sbc;
            }
        };
        let mut stats = M::Stats::default();
        for _ in 0..cell_size {
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut *rng);
            let e_tilde = s_sq_star.sqrt() * z;
            stats.record(e_tilde * e_tilde, 1.0);
        }
        let stats = vec![stats];
        let mut rank = 0usize;
        for _ in 0..posterior_draws {
            match model.draw_cell_values(&stats, sigma_sq, rng) {
                Ok(v) if v[0] < s_sq_star => rank += 1,
                Ok(_) => {}
                Err(e) => {
                    sbc_failed = Some(format!("posterior draw failed: {e}"));
                    break 'sbc;
                }
            }
        }
        bins[rank] += 1;
    }
    if let Some(reason) = sbc_failed {
        results.push(CheckResult::fail("cell_value_sbc", reason));
    } else {
        let expected = replications as f64 / bins.len() as f64;
        let chi_sq: f64 = bins
            .iter()
            .map(|&count| {
                let diff = count as f64 - expected;
                diff * diff / expected
            })
            .sum();
        if chi_sq <= 43.8 {
            results.push(CheckResult::pass(
                "cell_value_sbc",
                format!(
                    "cell-local SBC ranks are uniform (χ² = {chi_sq:.1} over {} bins)",
                    bins.len()
                ),
            ));
        } else {
            results.push(CheckResult::fail(
                "cell_value_sbc",
                format!(
                    "cell-local SBC ranks are non-uniform (χ² = {chi_sq:.1} over {} bins): \
                     the cell-value draw is not the conjugate posterior of the variance \
                     working likelihood",
                    bins.len()
                ),
            ));
        }
    }
    results
}

/// Check a [`CellModel`] (the deep seam): sufficient-statistic
/// consistency and conditional conjugacy, against the seam's scope rule that
/// the kernel always sees a Gaussian working likelihood
/// `rᵢ | μ ~ N(μ, σ²/wᵢ)` (that is what "conditionally
/// conjugate" means here; augmentation restores exactly this form).
///
/// `observations` and `weights` are a small scaled-space fixture (weights all
/// 1.0 for a hard-assignment model). Four checks:
///
/// - `stats_order_invariance`: a sufficient statistic cannot depend on
///   observation order (accumulation is re-run on a reversed fixture).
/// - `stats_merge_consistency`: `merge`ing two partial accumulations must
///   equal one sequential accumulation, and `remove` must undo `merge`. This
///   is what the sampler's incremental bookkeeping relies on, and the check
///   that catches a wrong add/subtract implementation.
/// - `marginal_bayes_factor`: the claimed marginal-likelihood difference
///   between the 2-cell and 1-cell partitions of the fixture is compared
///   against a Monte-Carlo estimate of the true Bayes factor computed from
///   the model's own prior draws (`draw_cell_values` on an empty statistic;
///   for any conjugate model the zero-data posterior is the prior). Catches a
///   `log_marginal_terms` that disagrees with the model's *own* payload draw.
/// - `cell_value_sbc`: cell-local simulation-based calibration of the
///   cell-value draw: μ* from the prior, data from the working likelihood,
///   rank of μ* among posterior draws must be uniform (wide-band χ²).
///
/// **What this cannot catch, and it is the important case.** Both statistical
/// checks take their reference *from the model under test*: the Bayes factor is
/// estimated from the model's own prior draws, and SBC ranks the model's own
/// draws against its own prior. So they prove the marginal and the draw are
/// **mutually consistent**, not that either is the law you intended. A model
/// whose prior is the wrong family, or the right family at a wildly wrong
/// scale, is self-consistent and passes everything here. That is exactly the
/// error a competent researcher makes: derive one thing wrong, then implement
/// it faithfully on both sides.
///
/// Only the joint-distribution battery (`crate::calibration`) compares the
/// sampler against an *independently generated* prior, so only it can see this
/// class. Local necessary conditions, not proof: run the battery before
/// trusting real inference.
pub fn check_cell_model<M: CellModel>(
    model: &M,
    sigma_sq: f64,
    observations: &[f64],
    weights: &[f64],
    rng: &mut dyn rand_core::Rng,
) -> Vec<CheckResult> {
    debug_assert!(observations.len() >= 4 && observations.len() == weights.len());
    let mut results = Vec::new();
    let n = observations.len();

    let accumulate = |order: &[usize], assignment: &dyn Fn(usize) -> usize, n_cells: usize| {
        let mut stats = vec![M::Stats::default(); n_cells];
        for &i in order {
            stats[assignment(i)].record(observations[i], weights[i]);
        }
        stats
    };
    let forward: Vec<usize> = (0..n).collect();
    let reversed: Vec<usize> = (0..n).rev().collect();

    // --- order invariance -------------------------------------------------
    let one_cell = accumulate(&forward, &|_| 0, 1);
    let one_cell_reversed = accumulate(&reversed, &|_| 0, 1);
    match (
        model.log_marginal_terms(&one_cell, sigma_sq),
        model.log_marginal_terms(&one_cell_reversed, sigma_sq),
    ) {
        (Ok(a), Ok(b)) if (a - b).abs() <= 1e-9 * a.abs().max(1.0) => {
            results.push(CheckResult::pass(
                "stats_order_invariance",
                format!("marginal terms agree under reversed accumulation ({a:.6e})"),
            ));
        }
        (Ok(a), Ok(b)) => {
            results.push(CheckResult::fail(
                "stats_order_invariance",
                format!(
                    "marginal terms differ under reversed accumulation ({a:.6e} vs {b:.6e}): \
                     the statistic is not sufficient (it depends on observation order)"
                ),
            ));
        }
        (Err(e), _) | (_, Err(e)) => {
            results.push(CheckResult::fail(
                "stats_order_invariance",
                format!("log_marginal_terms failed: {e}"),
            ));
        }
    }

    // --- merge / remove consistency ----------------------------------------
    let half = n / 2;
    let mut merged = accumulate(&forward[..half], &|_| 0, 1).remove(0);
    let second = accumulate(&forward[half..], &|_| 0, 1).remove(0);
    merged.merge(&second);
    let merged_terms = model.log_marginal_terms(std::slice::from_ref(&merged), sigma_sq);
    let sequential_terms = model.log_marginal_terms(&one_cell, sigma_sq);
    match (merged_terms, sequential_terms) {
        (Ok(a), Ok(b)) if (a - b).abs() <= 1e-9 * b.abs().max(1.0) => {
            let mut undone = merged.clone();
            undone.remove(&second);
            let first = accumulate(&forward[..half], &|_| 0, 1).remove(0);
            let undone_terms = model
                .log_marginal_terms(std::slice::from_ref(&undone), sigma_sq)
                .ok();
            let first_terms = model
                .log_marginal_terms(std::slice::from_ref(&first), sigma_sq)
                .ok();
            match (undone_terms, first_terms) {
                (Some(u), Some(f)) if (u - f).abs() <= 1e-6 * f.abs().max(1.0) => {
                    results.push(CheckResult::pass(
                        "stats_merge_consistency",
                        "merge equals sequential accumulation and remove undoes merge".into(),
                    ));
                }
                _ => {
                    results.push(CheckResult::fail(
                        "stats_merge_consistency",
                        "remove does not undo merge: the subtract path of the sufficient \
                         statistic is wrong"
                            .into(),
                    ));
                }
            }
        }
        (Ok(a), Ok(b)) => {
            results.push(CheckResult::fail(
                "stats_merge_consistency",
                format!(
                    "merging two partial accumulations gives {a:.6e} but sequential \
                     accumulation gives {b:.6e}: the add path of the sufficient statistic \
                     is wrong (double-counting or dropped terms?)"
                ),
            ));
        }
        (Err(e), _) | (_, Err(e)) => {
            results.push(CheckResult::fail(
                "stats_merge_consistency",
                format!("log_marginal_terms failed: {e}"),
            ));
        }
    }

    // --- marginal ↔ Bayes-factor agreement (conjugacy of the marginal) ----
    // Split the fixture into two cells (first half / second half) and compare
    // exp(terms(2-cell) − terms(1-cell)) with a Monte-Carlo Bayes factor from
    // the model's own prior draws. The structure-independent factors a model
    // may omit cancel in this ratio by construction.
    let two_cell = accumulate(&forward, &|i| usize::from(i >= half), 2);
    let claimed = match (
        model.log_marginal_terms(&two_cell, sigma_sq),
        model.log_marginal_terms(&one_cell, sigma_sq),
    ) {
        (Ok(two), Ok(one)) => Some(two - one),
        _ => None,
    };
    if let Some(claimed) = claimed {
        let n_mc = 40_000;
        let prior_stats = vec![M::Stats::default()];
        // log E_prior[∏_{i∈cell} N(r_i; μ, σ²/w_i)] per cell, by streaming
        // log-sum-exp over prior draws (drawn one at a time through the
        // model's own zero-data posterior).
        let log_bayes = |indices: &[usize], rng: &mut dyn rand_core::Rng| -> Option<f64> {
            let mut max = f64::NEG_INFINITY;
            let mut terms = Vec::with_capacity(n_mc);
            for _ in 0..n_mc {
                let mu = model.draw_cell_values(&prior_stats, sigma_sq, rng).ok()?[0];
                let mut log_lik = 0.0;
                for &i in indices {
                    let residual = observations[i] - mu;
                    log_lik += 0.5
                        * mathsfn::ln(weights[i] / (2.0 * std::f64::consts::PI * sigma_sq))
                        - weights[i] * residual * residual / (2.0 * sigma_sq);
                }
                max = max.max(log_lik);
                terms.push(log_lik);
            }
            let sum: f64 = terms.iter().map(|t| mathsfn::exp(t - max)).sum();
            Some(max + mathsfn::ln(sum / n_mc as f64))
        };
        let estimate = (|| {
            let cell_a = log_bayes(&forward[..half], rng)?;
            let cell_b = log_bayes(&forward[half..], rng)?;
            let joint = log_bayes(&forward, rng)?;
            Some((cell_a + cell_b) - joint)
        })();
        match estimate {
            Some(mc) if (mc - claimed).abs() <= 0.15 => {
                results.push(CheckResult::pass(
                    "marginal_bayes_factor",
                    format!(
                        "claimed 2-vs-1-cell log Bayes factor {claimed:.4} matches the \
                         Monte-Carlo estimate {mc:.4}"
                    ),
                ));
            }
            Some(mc) => {
                results.push(CheckResult::fail(
                    "marginal_bayes_factor",
                    format!(
                        "claimed 2-vs-1-cell log Bayes factor {claimed:.4} disagrees with the \
                         Monte-Carlo estimate {mc:.4}: log_marginal_terms is not the \
                         integrated Gaussian-working-likelihood marginal for this prior \
                         (wrong normalising term, wrong prior variance, or a dropped \
                         cell-count-dependent factor)"
                    ),
                ));
            }
            None => {
                results.push(CheckResult::fail(
                    "marginal_bayes_factor",
                    "draw_cell_values failed while sampling the prior".into(),
                ));
            }
        }
    } else {
        results.push(CheckResult::fail(
            "marginal_bayes_factor",
            "log_marginal_terms failed on the fixture".into(),
        ));
    }

    // --- cell-local SBC of the cell-value draw (conjugacy of the draw) -----
    let replications = 400;
    let posterior_draws = 19; // ranks in 0..=19 → 20 bins
    let cell_size = 4.min(n);
    let mut bins = vec![0usize; posterior_draws + 1];
    let mut sbc_failed = None;
    'sbc: for _ in 0..replications {
        let prior_stats = vec![M::Stats::default()];
        let mu_star = match model.draw_cell_values(&prior_stats, sigma_sq, rng) {
            Ok(v) => v[0],
            Err(e) => {
                sbc_failed = Some(format!("prior draw failed: {e}"));
                break 'sbc;
            }
        };
        let mut stats = M::Stats::default();
        for &weight in &weights[..cell_size] {
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut *rng);
            let value = mu_star + (sigma_sq / weight).sqrt() * z;
            stats.record(value, weight);
        }
        let stats = vec![stats];
        let mut rank = 0usize;
        for _ in 0..posterior_draws {
            match model.draw_cell_values(&stats, sigma_sq, rng) {
                Ok(v) if v[0] < mu_star => rank += 1,
                Ok(_) => {}
                Err(e) => {
                    sbc_failed = Some(format!("posterior draw failed: {e}"));
                    break 'sbc;
                }
            }
        }
        bins[rank] += 1;
    }
    if let Some(reason) = sbc_failed {
        results.push(CheckResult::fail("cell_value_sbc", reason));
    } else {
        let expected = replications as f64 / bins.len() as f64;
        let chi_sq: f64 = bins
            .iter()
            .map(|&count| {
                let diff = count as f64 - expected;
                diff * diff / expected
            })
            .sum();
        // χ²(19 df) 99.9% ≈ 43.8; generous local band (deterministic seed).
        if chi_sq <= 43.8 {
            results.push(CheckResult::pass(
                "cell_value_sbc",
                format!(
                    "cell-local SBC ranks are uniform (χ² = {chi_sq:.1} over {} bins)",
                    bins.len()
                ),
            ));
        } else {
            results.push(CheckResult::fail(
                "cell_value_sbc",
                format!(
                    "cell-local SBC ranks are NOT uniform (χ² = {chi_sq:.1} over {} bins): \
                     draw_cell_values does not sample the conjugate posterior implied by the \
                     Gaussian working likelihood (wrong posterior mean/variance formula?)",
                    bins.len()
                ),
            ));
        }
    }

    results
}

/// Check a [`ResponseModel`] (the response point): the augmentation runs, writes
/// a finite working response and strictly positive weights for every
/// observation, and two identically-seeded runs agree bit for bit.
///
/// `y` and `fit` are a small scaled-space fixture (equal lengths); `sigma_sq`
/// is the previous sweep's variance, as the sampler passes it.
///
/// The weights are reported, not just validated, because they carry the
/// pairing rule: a step that returns anything other than unit weights must be
/// paired with a weight-aware cell model and a precision-weighted σ² draw, or
/// the ensemble accumulates with weights the variance draw never sees. No
/// local check can catch that mispairing; only the calibration battery can, and
/// `ResponseFamily` exists to apply the rule for you.
pub fn check_response_model<K, F>(
    make_step: F,
    y: &[f64],
    fit: &[f64],
    sigma_sq: f64,
    seed_a: &mut dyn rand_core::Rng,
    seed_b: &mut dyn rand_core::Rng,
) -> Vec<CheckResult>
where
    K: ResponseModel,
    F: Fn() -> K,
{
    debug_assert!(!y.is_empty() && y.len() == fit.len());
    let mut results = Vec::new();
    let n = y.len();

    // The sampler hands the step buffers it has already sized; a step that
    // leaves an entry untouched is a bug the NaN fill below exposes.
    let run = |rng: &mut dyn rand_core::Rng| -> Result<(Vec<f64>, Vec<f64>)> {
        let mut step = make_step();
        let mut working = vec![f64::NAN; n];
        let mut weights = vec![f64::NAN; n];
        step.augment(y, fit, sigma_sq, rng, &mut working, &mut weights)
            .map_err(|e| crate::engine::error::AddiVortesError::Extension {
                source: std::sync::Arc::new(CheckWrappedError(format!("{e}"))),
            })?;
        Ok((working, weights))
    };

    let (a, b) = (run(seed_a), run(seed_b));
    let ((working, weights), (working_b, weights_b)) = match (a, b) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            results.push(CheckResult::fail(
                "augment_runs",
                format!("augment failed: {e}"),
            ));
            return results;
        }
    };

    match working.iter().position(|v| !v.is_finite()) {
        Some(i) => results.push(CheckResult::fail(
            "working_finite",
            format!(
                "working[{i}] is {} after augment: every entry must be written and finite \
                 (the kernel regresses on this response directly)",
                working[i]
            ),
        )),
        None => results.push(CheckResult::pass(
            "working_finite",
            format!("all {n} working-response entries are written and finite"),
        )),
    }

    match weights.iter().position(|w| !w.is_finite() || *w <= 0.0) {
        Some(i) => results.push(CheckResult::fail(
            "weights_valid",
            format!(
                "weights[{i}] is {}: every weight must be finite and strictly positive \
                 (a zero weight silently drops an observation; a negative one corrupts \
                 the cell sufficient statistics)",
                weights[i]
            ),
        )),
        None => {
            let unit = weights.iter().all(|w| *w == 1.0);
            results.push(CheckResult::pass(
                "weights_valid",
                if unit {
                    format!(
                        "all {n} weights are exactly 1: an unweighted step, which pairs with \
                         any cell model and the unweighted sigma^2 draw"
                    )
                } else {
                    format!(
                        "all {n} weights are finite and strictly positive, and are NOT all 1: \
                         this step must be paired with a weight-aware cell model \
                         (WeightedGaussianModel) and a precision-weighted sigma^2 draw \
                         (WeightedGlobalSigma), or the ensemble accumulates with weights the \
                         variance draw never sees"
                    )
                },
            ));
        }
    }

    let identical = working
        .iter()
        .zip(&working_b)
        .all(|(x, y)| x.to_bits() == y.to_bits())
        && weights
            .iter()
            .zip(&weights_b)
            .all(|(x, y)| x.to_bits() == y.to_bits());
    results.push(if identical {
        CheckResult::pass(
            "augment_deterministic",
            "two identically-seeded augmentations agree bit for bit".into(),
        )
    } else {
        CheckResult::fail(
            "augment_deterministic",
            "two identically-seeded augmentations differ: the step draws from a source other \
             than the handed RNG (interior state, or a thread-local/system RNG), which voids \
             the same-seed reproducibility promise"
                .into(),
        )
    });

    results
}

/// Check a [`MembershipKernel`] (the membership point): the weights are finite
/// and non-negative, every row carries strictly positive mass, the kernel is
/// pure, a distant row keeps its mass, and the weights are **weakly
/// monotone** in the key (a nearer cell never receives less weight than a
/// further one).
///
/// It does **not** check shift-invariance, and must not: weights may
/// legitimately shrink with absolute distance (an inverse-quadratic blend
/// does), only an exponential family is genuinely shift-invariant, and the
/// trait endorses no single formulation.
///
/// `keys` are one observation's comparison keys to every cell (the
/// squared-distance scale the assigner produces, ascending cell index).
///
/// `distant_row_keeps_mass` is the check worth having. A kernel that
/// exponentiates the raw key, instead of subtracting the row's best one first,
/// passes every other check here and then underflows an entire row to zero the
/// moment an observation sits far from every centre, which surfaces a long way
/// away as an `AddiVortesError::Extension` from the guard. Note this is a
/// mass check, not an invariance one: weights may legitimately shrink with
/// absolute distance (an inverse-quadratic blend does), only an exponential
/// family is genuinely shift-invariant, and the trait endorses no single
/// formulation.
pub fn check_membership_kernel(kernel: &dyn MembershipKernel, keys: &[f64]) -> Vec<CheckResult> {
    debug_assert!(!keys.is_empty());
    let mut results = Vec::new();
    let k = keys.len();

    let weigh = |keys: &[f64]| -> Vec<f64> {
        let mut weights = vec![f64::NAN; keys.len()];
        kernel.weights(keys, &mut weights);
        weights
    };
    let weights = weigh(keys);

    match weights.iter().position(|w| !w.is_finite() || *w < 0.0) {
        Some(i) => results.push(CheckResult::fail(
            "weights_finite_nonneg",
            format!(
                "weights[{i}] is {}: every membership weight must be written, finite and >= 0",
                weights[i]
            ),
        )),
        None => results.push(CheckResult::pass(
            "weights_finite_nonneg",
            format!("all {k} membership weights are written, finite and >= 0"),
        )),
    }

    let mass: f64 = weights.iter().sum();
    results.push(if mass > 0.0 {
        CheckResult::pass(
            "row_has_mass",
            format!("the row carries strictly positive total mass ({mass:e})"),
        )
    } else {
        CheckResult::fail(
            "row_has_mass",
            "the row's total membership mass is not strictly positive: the engine normalises \
             by it, so this row has no representable membership at all"
                .into(),
        )
    });

    let again = weigh(keys);
    results.push(
        if weights
            .iter()
            .zip(&again)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        {
            CheckResult::pass(
                "weights_deterministic",
                "two calls on the same keys agree bit for bit".into(),
            )
        } else {
            CheckResult::fail(
                "weights_deterministic",
                "two calls on the same keys differ: the kernel must be pure, with no interior \
                 state and no RNG (randomised membership is the out-of-scope \
                 latent-categorical formulation)"
                    .into(),
            )
        },
    );

    // A row from an observation far from every centre. The weights may legitimately
    // shrink (an inverse-quadratic blend depends on absolute distance, and only an
    // exponential family is genuinely shift-invariant), but the row must still carry
    // representable mass: the engine divides by it.
    let shift = 1.0e3;
    let distant: Vec<f64> = keys.iter().map(|key| key + shift).collect();
    let distant_weights = weigh(&distant);
    let distant_mass: f64 = distant_weights.iter().sum();
    if distant_mass.is_finite() && distant_mass > 0.0 {
        results.push(CheckResult::pass(
            "distant_row_keeps_mass",
            format!(
                "a row {shift} further from every centre still carries positive mass \
                 ({distant_mass:e})"
            ),
        ));
    } else {
        results.push(CheckResult::fail(
            "distant_row_keeps_mass",
            format!(
                "a row {shift} further from every centre collapsed to {distant_mass} total mass: \
                 an exponentiating kernel must subtract the row's best key before exponentiating \
                 (as SoftmaxKernel does), or a distant observation underflows the whole row and \
                 the guard rejects structures it should accept"
            ),
        ));
    }

    // Weak monotonicity: a nearer cell (smaller key) never receives strictly
    // less weight than a further one. This is the one real correctness property
    // the point has that does not depend on the recipe: every membership
    // formulation is a decreasing function of the key, whatever its shape. It
    // is what catches an inverted sign (`exp(+d²/τ)`), which every other check
    // here passes happily — the weights are still finite, positive, deterministic
    // and massive, they are simply backwards.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    let violation = order.windows(2).find(|pair| {
        let (near, far) = (pair[0], pair[1]);
        // Strict violation only: equal keys may carry equal weight, and ties are
        // resolved by the engine, not the kernel.
        keys[near] < keys[far] && weights[near] < weights[far]
    });
    results.push(match violation {
        None => CheckResult::pass(
            "weights_weakly_monotone",
            "a nearer cell never receives less weight than a further one".into(),
        ),
        Some(pair) => {
            let (near, far) = (pair[0], pair[1]);
            CheckResult::fail(
                "weights_weakly_monotone",
                format!(
                    "cell {near} is nearer than cell {far} (key {:.6e} < {:.6e}) but receives \
                     less weight ({:.6e} < {:.6e}): the kernel must be decreasing in the key. \
                     A flipped sign (exp(+d²/τ) for exp(−d²/τ)) passes every other check here.",
                    keys[near], keys[far], weights[near], weights[far]
                ),
            )
        }
    });

    // Permutation equivariance: cells carry no identity. Relabel them and every
    // weight must follow its own key. A kernel is a function of the *keys*, so a
    // weight that reads a cell's position in the row — a per-index lookup table,
    // an accumulator that never resets — is wrong, and is otherwise finite,
    // positive, deterministic, monotone and completely invisible.
    //
    // Compared **after row normalisation**, and that is load-bearing, not a
    // detail. The engine normalises every row, so two kernels differing by a
    // per-row positive constant are the same kernel. A recipe that subtracts
    // `keys[0]` rather than the row's minimum before exponentiating shifts by
    // such a constant, and a raw bitwise comparison would red it — a legitimate
    // recipe failed for a difference the engine divides away. (The same trap as
    // demanding shift-invariance; see the note above.) Only what survives
    // normalisation is a property of the kernel.
    if k >= 2 {
        let normalise = |w: &[f64]| -> Option<Vec<f64>> {
            let total: f64 = w.iter().sum();
            (total.is_finite() && total > 0.0)
                .then(|| w.iter().map(|weight| weight / total).collect())
        };
        let reversed: Vec<f64> = keys.iter().rev().copied().collect();
        match (normalise(&weights), normalise(&weigh(&reversed))) {
            (Some(base), Some(reversed_weights)) => {
                // Reassociating the sum reorders the rounding, so this is a
                // tolerance, not a bit compare. The bug it hunts is structural
                // and enormous; it does not hide in the last ulp.
                const TOL: f64 = 1e-9;
                let mismatch =
                    (0..k).find(|&i| (reversed_weights[i] - base[k - 1 - i]).abs() > TOL);
                results.push(match mismatch {
                    None => CheckResult::pass(
                        "weights_permutation_equivariant",
                        format!(
                            "relabelling the {k} cells permutes the normalised weights identically"
                        ),
                    ),
                    Some(i) => CheckResult::fail(
                        "weights_permutation_equivariant",
                        format!(
                            "reversing the key order moved normalised weight {i} from {:.6e} to \
                             {:.6e}: the kernel is reading a cell's *position* in the row, not \
                             its key. Cells carry no identity, and the engine may present them \
                             in any order.",
                            base[k - 1 - i],
                            reversed_weights[i]
                        ),
                    ),
                });
            }
            _ => results.push(CheckResult::fail(
                "weights_permutation_equivariant",
                "a row lost all its mass, so equivariance could not be assessed".into(),
            )),
        }
    }

    let digest = bit_digest(weights.iter().copied());
    results.push(CheckResult::pass(
        "weight_digest",
        format!(
            "digest of {k} weights on this fixture: {digest:016x}. Run the check on every \
             platform you target and compare; a differing digest means un-pinned maths \
             (std f64 transcendentals instead of addivortes::mathsfn) and voids the \
             same-seed reproducibility promise for chains using this kernel"
        ),
    ));

    results
}

/// Check a [`CountPriors`] (the count-priors point): both ratio hooks are
/// finite and pure over the adjacent counts a sampler will actually ask about,
/// and their bits are reported for cross-platform comparison.
///
/// The hooks are evaluated at every cell count `2..=max_cells` and every
/// dimension count `2..=ctx.p`, which is the whole domain: moves only ever ask
/// about adjacent counts, and a count-neutral move asks nothing at all.
///
/// What this cannot prove: that the ratios correspond to a normalisable prior.
/// A pair of hooks can be finite, pure and portable and still describe no
/// distribution. Only the calibration battery sees that, which is why a custom count
/// prior needs its own gate config.
pub fn check_count_priors(
    priors: &dyn CountPriors,
    ctx: &ModelCtx,
    max_cells: usize,
) -> Vec<CheckResult> {
    debug_assert!(max_cells >= 2);
    let mut results = Vec::new();

    let cells: Vec<f64> = (2..=max_cells)
        .map(|b| priors.log_cell_count_ratio(b, ctx))
        .collect();
    match cells.iter().position(|r| !r.is_finite()) {
        Some(i) => results.push(CheckResult::fail(
            "cell_ratios_finite",
            format!(
                "log_cell_count_ratio({}) is {}: every adjacent-count ratio must be finite \
                 (a non-finite one poisons the acceptance ratio of every add/remove move)",
                i + 2,
                cells[i]
            ),
        )),
        None => results.push(CheckResult::pass(
            "cell_ratios_finite",
            format!("log_cell_count_ratio is finite for every cell count 2..={max_cells}"),
        )),
    }

    // The dimension hook's domain is 2..=p; a single-covariate context has no
    // adjacent dimension counts to price.
    let dims: Vec<f64> = (2..=ctx.p)
        .map(|d| priors.log_dim_count_ratio(d, ctx))
        .collect();
    if dims.is_empty() {
        results.push(CheckResult::pass(
            "dim_ratios_finite",
            format!(
                "p = {} leaves no adjacent dimension counts to price (nothing to verify)",
                ctx.p
            ),
        ));
    } else {
        match dims.iter().position(|r| !r.is_finite()) {
            Some(i) => results.push(CheckResult::fail(
                "dim_ratios_finite",
                format!(
                    "log_dim_count_ratio({}) is {}: every adjacent-count ratio must be finite",
                    i + 2,
                    dims[i]
                ),
            )),
            None => results.push(CheckResult::pass(
                "dim_ratios_finite",
                format!(
                    "log_dim_count_ratio is finite for every dimension count 2..={}",
                    ctx.p
                ),
            )),
        }
    }

    let repeat_cells: Vec<f64> = (2..=max_cells)
        .map(|b| priors.log_cell_count_ratio(b, ctx))
        .collect();
    let repeat_dims: Vec<f64> = (2..=ctx.p)
        .map(|d| priors.log_dim_count_ratio(d, ctx))
        .collect();
    let pure = cells
        .iter()
        .zip(&repeat_cells)
        .chain(dims.iter().zip(&repeat_dims))
        .all(|(x, y)| x.to_bits() == y.to_bits());
    results.push(if pure {
        CheckResult::pass(
            "ratios_deterministic",
            "both hooks return bit-identical ratios when asked twice".into(),
        )
    } else {
        CheckResult::fail(
            "ratios_deterministic",
            "the hooks returned different bits when asked twice: a count prior must be pure \
             and deterministic (no interior state, no RNG)"
                .into(),
        )
    });

    let digest = bit_digest(cells.iter().chain(&dims).copied());
    results.push(CheckResult::pass(
        "ratio_digest",
        format!(
            "digest of {} cell and {} dimension ratios on this context: {digest:016x}. Run the \
             check on every platform you target and compare; a differing digest means un-pinned \
             maths (std f64 transcendentals instead of addivortes::mathsfn) and voids the \
             same-seed reproducibility promise for chains using this prior",
            cells.len(),
            dims.len()
        ),
    ));

    results
}

/// Cross-examine a [`CountPriors`] against the prior you *meant*: you supply the
/// unnormalised log-pmfs, and each ratio hook must be their adjacent difference.
///
/// **This is the only check in the module whose reference does not come from the
/// component under test**, and it is worth understanding why it can exist here
/// and nowhere else.
///
/// No *free* oracle for a count prior is possible, even in principle: any finite,
/// pure hook is the exact adjacent log-ratio of *some* positive sequence, so
/// there is nothing to contradict. The escape is that you already wrote this
/// prior down as a density — in your paper, in the line above the derivation —
/// and deriving the ratio from it is the step where the mistake happens. Writing
/// the prior twice, once as `log P(b)` and once as `log P(b) − log P(b−1)`, and
/// making the two agree, is a real test of the derivation. The cost is that if
/// the *density* is what you got wrong, both sides are wrong together and this
/// passes — it checks your algebra, not your model.
///
/// The pmfs may be unnormalised (any constant cancels in the difference), and
/// only ratios at adjacent counts are ever exercised, because that is all a move
/// asks for.
///
/// ```no_run
/// # use addivortes::{conformance, mathsfn, ModelCtx, ShiftedPoissonBinomial};
/// # fn demo(ctx: &ModelCtx) {
/// // The paper's cell prior: b − 1 ~ Poisson(λ_c), so log P(b) ∝ (b−1)·ln λ_c − ln (b−1)!
/// let log_cell_pmf = |b: usize| {
///     let ln_factorial: f64 = (1..b).map(|k| mathsfn::ln(k as f64)).sum();
///     (b - 1) as f64 * mathsfn::ln(ctx.lambda_c) - ln_factorial
/// };
/// # let log_dim_pmf = |_d: usize| 0.0;
/// let results = conformance::check_count_priors_against_density(
///     &ShiftedPoissonBinomial, ctx, &log_cell_pmf, &log_dim_pmf, 40,
/// );
/// # let _ = results;
/// # }
/// ```
///
/// `log_cell_pmf` is evaluated at `1..=max_cells` and `log_dim_pmf` at
/// `1..=ctx.p`.
pub fn check_count_priors_against_density(
    priors: &dyn CountPriors,
    ctx: &ModelCtx,
    log_cell_pmf: &dyn Fn(usize) -> f64,
    log_dim_pmf: &dyn Fn(usize) -> f64,
    max_cells: usize,
) -> Vec<CheckResult> {
    debug_assert!(max_cells >= 2);
    // The hooks and the differences are computed by different arithmetic, so a
    // few ulps of disagreement are expected and meaningless. The bugs this hunts
    // — an inverted ratio, a stray factor of b, a dropped term — are enormous.
    const TOL: f64 = 1e-8;

    let mut results = Vec::new();

    let mut compare = |name: &'static str,
                       hook_name: &str,
                       counts: std::ops::RangeInclusive<usize>,
                       hook: &dyn Fn(usize) -> f64,
                       log_pmf: &dyn Fn(usize) -> f64| {
        let mut worst: Option<(usize, f64, f64)> = None;
        let mut checked = 0usize;
        for count in counts {
            let (upper, lower) = (log_pmf(count), log_pmf(count - 1));
            if !upper.is_finite() || !lower.is_finite() {
                results.push(CheckResult::fail(
                    name,
                    format!(
                        "the supplied log-pmf is {} at count {count}: it must be finite over the \
                         whole domain the sampler can reach",
                        if upper.is_finite() { lower } else { upper }
                    ),
                ));
                return;
            }
            let expected = upper - lower;
            let actual = hook(count);
            let error = (actual - expected).abs();
            checked += 1;
            if worst.is_none_or(|(_, _, w)| error > w) {
                worst = Some((count, actual - expected, error));
            }
        }
        match worst {
            Some((count, signed, error)) if error > TOL => {
                let expected = log_pmf(count) - log_pmf(count - 1);
                results.push(CheckResult::fail(
                    name,
                    format!(
                        "{hook_name}({count}) returns {:.9} but the density you supplied implies \
                         {expected:.9} (out by {signed:+.3e}). The hook must be \
                         `log P(n) − log P(n−1)` at the *larger* count. An inverted subtraction, \
                         or a ratio taken at the wrong count, lands here.",
                        hook(count)
                    ),
                ));
            }
            Some((_, _, error)) => results.push(CheckResult::pass(
                name,
                format!(
                    "{hook_name} matches the supplied density at all {checked} adjacent counts \
                     (worst disagreement {error:.2e})"
                ),
            )),
            None => results.push(CheckResult::pass(
                name,
                format!("{hook_name} has no adjacent counts to price on this context"),
            )),
        }
    };

    compare(
        "cell_ratio_matches_density",
        "log_cell_count_ratio",
        2..=max_cells,
        &|b| priors.log_cell_count_ratio(b, ctx),
        log_cell_pmf,
    );
    compare(
        "dim_ratio_matches_density",
        "log_dim_count_ratio",
        2..=ctx.p,
        &|d| priors.log_dim_count_ratio(d, ctx),
        log_dim_pmf,
    );

    results
}

/// Order-pinned FNV-1a digest over the bits of a value sequence: the
/// cross-platform portability probe every check that reports one shares. A
/// digest that differs across targets means the component's arithmetic is not
/// bit-portable, which no single-machine check can catch.
fn bit_digest(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x100_0000_01b3);
        }
    }
    digest
}

/// Internal wrapper so check failures surface uniformly.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct CheckWrappedError(String);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::Metric;
    use crate::extensions::coord::{EuclideanNormal, WrappedNormal};
    use crate::extensions::distance::{ColumnMetrics, PairwiseDistance};
    use crate::extensions::inclusion::UniformInclusion;
    use crate::extensions::moves::{MoveSetBuilder, Proposal, ProposalMove, Reverse};

    fn ctx_fixture<'a>(
        dists: &'a [std::sync::Arc<dyn CoordinateDistribution>],
        weights: &'a [f64],
    ) -> ModelCtx<'a> {
        ModelCtx::new(1.0, 1.5, 10.0, 0.01, weights.len(), dists, weights)
    }

    fn dists(p: usize) -> Vec<std::sync::Arc<dyn CoordinateDistribution>> {
        (0..p)
            .map(|_| std::sync::Arc::new(EuclideanNormal::new(0.8).unwrap()) as std::sync::Arc<_>)
            .collect()
    }

    fn rng(seed: u8) -> ChaCha8Rng {
        ChaCha8Rng::from_seed([seed; 32])
    }

    // ---- positive controls: the built-ins pass every check ----

    #[test]
    fn standard_move_set_passes() {
        let dists = dists(3);
        let weights = vec![1.0; 3];
        let ctx = ctx_fixture(&dists, &weights);
        let state =
            Tessellation::new(vec![0.1, 0.2, 0.3, 0.4], vec![0, 2], vec![0.0, 0.0]).unwrap();
        let set = MoveSetBuilder::stone_gosling().build().unwrap();
        let results = check_move_set(&set, &state, &ctx, &mut rng(1));
        assert!(all_passed(&results), "{results:#?}");
    }

    #[test]
    fn built_in_assigner_and_distributions_pass() {
        let x = Data::from_rows(&[[0.1, 0.5], [0.9, 0.2], [0.4, 0.8]]).unwrap();
        let t = Tessellation::new(vec![0.2, 0.8], vec![0], vec![0.0, 0.0]).unwrap();
        let cm = ColumnMetrics::new(vec![Metric::Euclidean, Metric::Euclidean]);
        assert!(all_passed(&check_assigner(&cm, &x, &t)), "{:#?}", {
            check_assigner(&cm, &x, &t)
        });
        // The metric-level correctness checks pass for the built-ins: the
        // fast-path claimant (all-Euclidean) and the general path (spherical).
        assert!(all_passed(&check_distance(&cm, &x, &t)), "{:#?}", {
            check_distance(&cm, &x, &t)
        });
        let mixed = ColumnMetrics::new(vec![Metric::Euclidean, Metric::Spherical]);
        assert!(all_passed(&check_distance(&mixed, &x, &t)), "{:#?}", {
            check_distance(&mixed, &x, &t)
        });

        assert!(all_passed(&check_coordinate_distribution(
            &EuclideanNormal::new(0.8).unwrap(),
            2000,
            &mut rng(2)
        )));
        assert!(all_passed(&check_coordinate_distribution(
            &WrappedNormal::new(0.8).unwrap(),
            2000,
            &mut rng(3)
        )));
    }

    #[test]
    fn uniform_inclusion_passes() {
        let usage = InclusionUsage::new(4);
        let results = check_inclusion_model(
            || UniformInclusion::new(4),
            4,
            &usage,
            &mut rng(4),
            &mut rng(4),
        );
        assert!(all_passed(&results), "{results:#?}");
    }

    // ---- negative controls: each broken extension fails its check ----

    /// A "Relocate"-style move whose structure ratio is simply wrong.
    #[derive(Debug)]
    struct WrongRatio;
    impl ProposalMove for WrongRatio {
        fn name(&self) -> &'static str {
            "WrongRatio"
        }
        fn reverse(&self) -> Reverse {
            Reverse::SelfInverse
        }
        fn is_valid(&self, _t: &Tessellation, _c: &ModelCtx) -> bool {
            true
        }
        fn propose(
            &self,
            t: &Tessellation,
            ctx: &ModelCtx,
            rng: &mut dyn rand_core::Rng,
        ) -> Proposal {
            // Relocate the first centre along the first dim (fine)…
            let mut centres = t.centres().to_vec();
            centres[0] = ctx.coord_dists[t.dims()[0]].sample(rng);
            Proposal::new(Tessellation::new(centres, t.dims().to_vec(), t.mus().to_vec()).unwrap())
        }
        fn log_structure_ratio(&self, _o: &Tessellation, _p: &Tessellation, _c: &ModelCtx) -> f64 {
            0.7 // …but claim a non-zero, non-cancelling ratio: wrong
        }
    }

    /// A move whose structure ratio is NaN. `NaN > 1e-8` is *false*, so the
    /// pre-fix comparison sent this straight down the pass branch: a move that
    /// returns garbage was certified green.
    #[derive(Debug)]
    struct NanRatio;
    impl ProposalMove for NanRatio {
        fn name(&self) -> &'static str {
            "NanRatio"
        }
        fn reverse(&self) -> Reverse {
            Reverse::SelfInverse
        }
        fn is_valid(&self, _t: &Tessellation, _c: &ModelCtx) -> bool {
            true
        }
        fn propose(
            &self,
            t: &Tessellation,
            ctx: &ModelCtx,
            rng: &mut dyn rand_core::Rng,
        ) -> Proposal {
            let mut centres = t.centres().to_vec();
            centres[0] = ctx.coord_dists[t.dims()[0]].sample(rng);
            Proposal::new(Tessellation::new(centres, t.dims().to_vec(), t.mus().to_vec()).unwrap())
        }
        fn log_structure_ratio(&self, _o: &Tessellation, _p: &Tessellation, _c: &ModelCtx) -> f64 {
            f64::NAN
        }
    }

    /// Moves centre 1 but declares `CentreMoved { index: 0 }`. The assigner then
    /// rescans centre 0 (which did not move) and reuses centre 1's cached key
    /// (which is now stale), so observations keep an assignment that no longer
    /// reflects the geometry — silently, forever, with no error anywhere.
    ///
    /// It is perfectly self-consistent: its structure ratio is a correct 0.0, it
    /// cancels against its own reverse, and it proposes a coherent tessellation.
    /// Every other move check passes it.
    #[derive(Debug)]
    struct LyingDelta;
    impl ProposalMove for LyingDelta {
        fn name(&self) -> &'static str {
            "LyingDelta"
        }
        fn reverse(&self) -> Reverse {
            Reverse::SelfInverse
        }
        fn is_valid(&self, t: &Tessellation, _ctx: &ModelCtx) -> bool {
            t.n_cells() >= 2
        }
        fn propose(
            &self,
            t: &Tessellation,
            _ctx: &ModelCtx,
            _rng: &mut dyn rand_core::Rng,
        ) -> Proposal {
            let d = t.dims().len();
            let mut centres = t.centres().to_vec();
            centres[d] += 0.25; // centre 1's first coordinate
            Proposal::with_delta(
                Tessellation::new(centres, t.dims().to_vec(), t.mus().to_vec()).unwrap(),
                AssignmentDelta::CentreMoved { index: 0 }, // ...but claims centre 0
            )
        }
        fn log_structure_ratio(&self, _o: &Tessellation, _p: &Tessellation, _c: &ModelCtx) -> f64 {
            0.0
        }
    }

    #[test]
    fn a_lying_delta_claim_fails_and_nothing_else_sees_it() {
        let dists = dists(2);
        let weights = vec![1.0; 2];
        let ctx = ctx_fixture(&dists, &weights);
        let state = Tessellation::new(vec![0.1, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let set = MoveSetBuilder::empty()
            .with_move(Box::new(LyingDelta), 1.0)
            .build()
            .unwrap();
        let results = check_move_set(&set, &state, &ctx, &mut rng(5));

        let claim = results
            .iter()
            .find(|r| r.name == "delta_claim_exact")
            .expect("the delta-claim check runs");
        assert!(
            !claim.passed,
            "a move that moves centre 1 while declaring centre 0 must fail: {}",
            claim.detail
        );
        // The point of the control: the move is entirely self-consistent, so
        // every other check on the point passes it.
        assert!(
            results
                .iter()
                .filter(|r| r.name != "delta_claim_exact")
                .all(|r| r.passed),
            "every other move check should pass this move: {results:#?}"
        );
    }

    #[test]
    fn a_nan_structure_ratio_fails_detailed_balance() {
        let dists = dists(2);
        let weights = vec![1.0; 2];
        let ctx = ctx_fixture(&dists, &weights);
        let state = Tessellation::new(vec![0.1, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let set = MoveSetBuilder::empty()
            .with_move(Box::new(NanRatio), 1.0)
            .build()
            .unwrap();
        let results = check_move_set(&set, &state, &ctx, &mut rng(5));
        let db = results
            .iter()
            .find(|r| r.name == "detailed_balance_structure")
            .expect("the detailed-balance check runs");
        assert!(
            !db.passed,
            "a NaN structure ratio must fail, not pass through the comparison"
        );
    }

    #[test]
    fn wrong_structure_ratio_fails_detailed_balance() {
        let dists = dists(2);
        let weights = vec![1.0; 2];
        let ctx = ctx_fixture(&dists, &weights);
        let state = Tessellation::new(vec![0.1, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let set = MoveSetBuilder::empty()
            .with_move(Box::new(WrongRatio), 1.0)
            .build()
            .unwrap();
        let results = check_move_set(&set, &state, &ctx, &mut rng(5));
        let db = results
            .iter()
            .find(|r| r.name == "detailed_balance_structure")
            .unwrap();
        assert!(!db.passed);
        assert!(db.detail.contains("do not cancel"));
    }

    /// An assigner whose metric changes on every call (hidden state).
    #[derive(Debug)]
    struct FlickeringMetric {
        calls: std::sync::atomic::AtomicU64,
    }
    impl PairwiseDistance for FlickeringMetric {
        fn distance(&self, x_row: &[f64], centre_row: &[f64], dims: &[usize]) -> f64 {
            let call = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let d: f64 = dims.iter().map(|&k| (x_row[k] - centre_row[k]).abs()).sum();
            if call % 3 == 0 { -d } else { d } // sign flips: argmin flips
        }
    }

    /// Manhattan wrongly claiming the all-Euclidean fast path: assignments
    /// would silently become Euclidean; the claim check must go red.
    #[derive(Debug)]
    struct FalseClaimManhattan;
    impl PairwiseDistance for FalseClaimManhattan {
        fn distance(&self, x_row: &[f64], centre_row: &[f64], dims: &[usize]) -> f64 {
            dims.iter().map(|&k| (x_row[k] - centre_row[k]).abs()).sum()
        }
        fn all_euclidean(&self) -> bool {
            true // wrong: the key is not the squared-difference sum
        }
    }

    #[test]
    fn false_all_euclidean_claim_fails() {
        let x = Data::from_rows(&[[0.1], [0.9], [0.4]]).unwrap();
        let t = Tessellation::new(vec![0.2, 0.8], vec![0], vec![0.0, 0.0]).unwrap();
        let results = check_distance(&FalseClaimManhattan, &x, &t);
        let claim = results
            .iter()
            .find(|r| r.name == "all_euclidean_claim")
            .unwrap();
        assert!(!claim.passed);
        assert!(claim.detail.contains("SILENTLY"), "{}", claim.detail);
    }

    /// An inverted key (negated distance): nearest becomes farthest, which
    /// repeat-call determinism alone would wave through.
    #[derive(Debug)]
    struct InvertedKey;
    impl PairwiseDistance for InvertedKey {
        fn distance(&self, x_row: &[f64], centre_row: &[f64], dims: &[usize]) -> f64 {
            -dims
                .iter()
                .map(|&k| (x_row[k] - centre_row[k]).abs())
                .sum::<f64>()
        }
    }

    #[test]
    fn inverted_key_fails_self_minimality() {
        let x = Data::from_rows(&[[0.1], [0.9]]).unwrap();
        let t = Tessellation::new(vec![0.2, 0.8], vec![0], vec![0.0, 0.0]).unwrap();
        // The old determinism-only check would pass this broken metric…
        assert!(all_passed(&check_assigner(&InvertedKey, &x, &t)));
        // …the correctness check does not.
        let results = check_distance(&InvertedKey, &x, &t);
        let minimal = results
            .iter()
            .find(|r| r.name == "self_key_minimal")
            .unwrap();
        assert!(!minimal.passed);
        assert!(minimal.detail.contains("self-key"), "{}", minimal.detail);
    }

    /// A custom batch assigner whose incremental `reassign` ignores the move
    /// (returns the previous cache): the silent-corruption class.
    #[derive(Debug)]
    struct StaleReassign;
    impl CellAssigner for StaleReassign {
        fn assign_cells(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<usize>> {
            ColumnMetrics::new(vec![Metric::Euclidean]).assign_cells(x, tessellation)
        }
        fn reassign(
            &self,
            x: &Data,
            new: &Tessellation,
            delta: crate::extensions::distance::AssignmentDelta,
            prev: &crate::extensions::distance::AssignmentCache,
        ) -> Result<crate::extensions::distance::AssignmentCache> {
            match delta {
                // wrong: pretends a moved centre changes nothing.
                crate::extensions::distance::AssignmentDelta::CentreMoved { .. } => {
                    Ok(prev.clone())
                }
                _ => Ok(crate::extensions::distance::AssignmentCache::new(
                    self.assign_cells(x, new)?,
                    Vec::new(),
                )),
            }
        }
    }

    #[test]
    fn stale_reassign_fails_consistency() {
        // The check shifts each centre by +0.17: centre 0 (0.0 → 0.17)
        // captures the observation at 0.15 from centre 1 (0.2) on a fresh
        // recompute; the stale cache keeps the old winner.
        let x = Data::from_rows(&[[0.15]]).unwrap();
        let t = Tessellation::new(vec![0.0, 0.2], vec![0], vec![0.0, 0.0]).unwrap();
        let results = check_assigner(&StaleReassign, &x, &t);
        let consistency = results
            .iter()
            .find(|r| r.name == "reassign_consistent")
            .unwrap();
        assert!(!consistency.passed);
        assert!(
            consistency.detail.contains("CentreMoved"),
            "{}",
            consistency.detail
        );
    }

    /// A cache that never invalidates: `reassign` echoes `prev` for *every*
    /// delta, `FullRecompute` included. `assign_cells` itself is honest.
    ///
    /// This is the case the check could not see while it took its reference
    /// from `reassign(FullRecompute)`: both sides of that comparison came from
    /// this same stale path, agreed with each other perfectly, and passed. Only
    /// a reference computed by `assign_cells` exposes it.
    #[derive(Debug)]
    struct NeverInvalidates;
    impl CellAssigner for NeverInvalidates {
        fn assign_cells(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<usize>> {
            ColumnMetrics::new(vec![Metric::Euclidean]).assign_cells(x, tessellation)
        }
        fn reassign(
            &self,
            _x: &Data,
            _new: &Tessellation,
            _delta: crate::extensions::distance::AssignmentDelta,
            prev: &crate::extensions::distance::AssignmentCache,
        ) -> Result<crate::extensions::distance::AssignmentCache> {
            Ok(prev.clone())
        }
    }

    #[test]
    fn a_cache_that_never_invalidates_fails_against_assign_cells() {
        // The honest assignment of x = 0.15 against centres {0.0, 0.2} is cell
        // 1; the never-invalidating cache echoes the cold cache's cell 0.
        let x = Data::from_rows(&[[0.15]]).unwrap();
        let t = Tessellation::new(vec![0.0, 0.2], vec![0], vec![0.0, 0.0]).unwrap();
        let results = check_assigner(&NeverInvalidates, &x, &t);
        let consistency = results
            .iter()
            .find(|r| r.name == "reassign_consistent")
            .unwrap();
        assert!(
            !consistency.passed,
            "a never-invalidating cache must fail: {}",
            consistency.detail
        );
        assert!(
            consistency.detail.contains("assign_cells"),
            "the failure must name the independent reference: {}",
            consistency.detail
        );
    }

    #[test]
    fn key_digest_is_stable_and_reported() {
        let x = Data::from_rows(&[[0.1, 0.5], [0.9, 0.2]]).unwrap();
        let t = Tessellation::new(vec![0.2, 0.8], vec![0], vec![0.0, 0.0]).unwrap();
        let cm = ColumnMetrics::new(vec![Metric::Euclidean, Metric::Euclidean]);
        let digest = |results: &[CheckResult]| {
            results
                .iter()
                .find(|r| r.name == "key_digest")
                .unwrap()
                .detail
                .clone()
        };
        let a = digest(&check_distance(&cm, &x, &t));
        let b = digest(&check_distance(&cm, &x, &t));
        assert_eq!(a, b); // same fixture, same digest; comparable across runs
        assert!(a.contains("digest of 4 keys"), "{a}");
    }

    #[test]
    fn nondeterministic_assigner_fails() {
        let x = Data::from_rows(&[[0.1], [0.9], [0.4], [0.6]]).unwrap();
        let t = Tessellation::new(vec![0.2, 0.8], vec![0], vec![0.0, 0.0]).unwrap();
        let flickering = FlickeringMetric {
            calls: std::sync::atomic::AtomicU64::new(0),
        };
        let results = check_assigner(&flickering, &x, &t);
        assert!(!all_passed(&results));
        assert!(results[0].detail.contains("different assignments"));
    }

    /// Samples N(0, 2²) but reports the density of N(0, 0.5²).
    #[derive(Debug)]
    struct LyingDistribution;
    impl CoordinateDistribution for LyingDistribution {
        fn sample(&self, rng: &mut dyn rand_core::Rng) -> f64 {
            EuclideanNormal::new(2.0).unwrap().sample(rng)
        }
        fn log_density(&self, x: f64) -> f64 {
            EuclideanNormal::new(0.5).unwrap().log_density(x)
        }
    }

    #[test]
    fn mismatched_density_fails_ks() {
        let results = check_coordinate_distribution(&LyingDistribution, 2000, &mut rng(6));
        assert!(!all_passed(&results));
        assert!(results[0].detail.contains("disagree"));
    }

    /// The crate's own shipped laws must not fail their own check. `WrappedNormal`
    /// did: it is periodic, so its density is non-zero outside the circle, and the
    /// 5% window pad renormalised over mass no sample could ever occupy. At
    /// σ_c ≥ 3 (wrapping towards uniform) that phantom mass reached ~9% and the
    /// check went red on *every* seed.
    ///
    /// A false red is worse than a false green: it tells a researcher the crate's
    /// own law is broken, and teaches them to distrust every other verdict.
    #[test]
    fn the_shipped_coordinate_laws_pass_their_own_check_at_every_scale() {
        for sigma in [0.5, 1.0, 2.0, 3.0, 5.0] {
            for seed in 1..=4u8 {
                let wrapped = crate::extensions::coord::WrappedNormal::new(sigma).unwrap();
                let results = check_coordinate_distribution(&wrapped, 4000, &mut rng(seed));
                assert!(
                    all_passed(&results),
                    "WrappedNormal(sigma_c = {sigma}) is shipped and correct; it must not fail \
                     its own check (seed {seed}): {results:#?}"
                );

                let euclidean = EuclideanNormal::new(sigma).unwrap();
                let results = check_coordinate_distribution(&euclidean, 4000, &mut rng(seed));
                assert!(
                    all_passed(&results),
                    "EuclideanNormal(sigma_c = {sigma}) must pass its own check (seed {seed}): \
                     {results:#?}"
                );
            }
        }
    }

    /// Samples N(0, 1) but reports the density of N(0, 1.2²): a 20% scale error,
    /// the sensitivity floor the unpadded window must keep. This is the guard
    /// against "fixing" the false red by simply making the check blind.
    #[derive(Debug)]
    struct MildlyWrongScale;
    impl CoordinateDistribution for MildlyWrongScale {
        fn sample(&self, rng: &mut dyn rand_core::Rng) -> f64 {
            EuclideanNormal::new(1.0).unwrap().sample(rng)
        }
        fn log_density(&self, x: f64) -> f64 {
            EuclideanNormal::new(1.2).unwrap().log_density(x)
        }
    }

    #[test]
    fn the_unpadded_window_still_catches_a_scale_error() {
        let results = check_coordinate_distribution(&MildlyWrongScale, 4000, &mut rng(3));
        assert!(
            !all_passed(&results),
            "a 20% scale mismatch must still be caught: {results:#?}"
        );
    }

    /// An update that pulls entropy from shared state outside the RNG.
    #[derive(Debug, Clone)]
    struct LeakyInclusion {
        weights: Vec<f64>,
        counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
    }
    #[derive(Debug, thiserror::Error)]
    #[error("unreachable")]
    struct Never;
    impl InclusionModel for LeakyInclusion {
        type Error = Never;
        fn weights(&self) -> &[f64] {
            &self.weights
        }
        fn update(
            &mut self,
            _usage: &InclusionUsage,
            _rng: &mut dyn rand_core::Rng,
        ) -> std::result::Result<(), Self::Error> {
            let next = self
                .counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            self.weights[0] = 1.0 + next as f64;
            Ok(())
        }
    }

    #[test]
    fn nondeterministic_inclusion_update_fails() {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let make = {
            let counter = std::sync::Arc::clone(&counter);
            move || LeakyInclusion {
                weights: vec![1.0, 1.0],
                counter: std::sync::Arc::clone(&counter),
            }
        };
        let usage = InclusionUsage::new(2);
        let results = check_inclusion_model(make, 2, &usage, &mut rng(7), &mut rng(7));
        let update = results
            .iter()
            .find(|r| r.name == "update_deterministic")
            .unwrap();
        assert!(!update.passed);
        assert!(update.detail.contains("outside the RNG"));
    }

    /// Valid weights at construction, destroyed by its own update: the
    /// normalise-by-usage-total shape, which divides by zero the moment no
    /// covariate has been used yet (exactly the state of the first sweep).
    /// Deterministic, correctly-lengthed, and it passed every inclusion check
    /// while validity was only ever asserted against the freshly-built model.
    #[derive(Debug, Clone)]
    struct SelfDestructingInclusion {
        weights: Vec<f64>,
    }
    impl InclusionModel for SelfDestructingInclusion {
        type Error = Never;
        fn weights(&self) -> &[f64] {
            &self.weights
        }
        fn update(
            &mut self,
            usage: &InclusionUsage,
            _rng: &mut dyn rand_core::Rng,
        ) -> std::result::Result<(), Self::Error> {
            let total: f64 = usage.counts().iter().map(|c| *c as f64).sum();
            for weight in &mut self.weights {
                *weight /= total; // total == 0 on the first sweep -> NaN
            }
            Ok(())
        }
    }

    #[test]
    fn an_update_that_destroys_its_weights_fails_after_update() {
        let make = || SelfDestructingInclusion {
            weights: vec![1.0, 1.0],
        };
        let usage = InclusionUsage::new(2); // all-zero usage: the first sweep
        let results = check_inclusion_model(make, 2, &usage, &mut rng(7), &mut rng(7));

        let after = results
            .iter()
            .find(|r| r.name == "weights_valid_after_update")
            .unwrap();
        assert!(
            !after.passed,
            "an update producing NaN weights must fail: {}",
            after.detail
        );
        // The point: the pre-update validity check saw nothing wrong, and the
        // update is perfectly deterministic. Without the post-update oracle this
        // model is green.
        assert!(
            results
                .iter()
                .filter(|r| r.name != "weights_valid_after_update")
                .all(|r| r.passed),
            "every other inclusion check should pass this model: {results:#?}"
        );
    }

    // ---- scale checks ----

    #[test]
    fn global_sigma_and_pinned_sigma_pass_scale_checks() {
        let y = [0.31, -0.12, 0.07, 0.22, -0.44, 0.18];
        let fit = [0.1, -0.1, 0.0, 0.2, -0.3, 0.1];
        let results = check_scale_model(
            || crate::extensions::scale::GlobalSigma::new(6.0, 0.02).unwrap(),
            &y,
            &fit,
            &mut rng(31),
            &mut rng(31),
        );
        assert!(all_passed(&results), "{results:#?}");
        let results = check_scale_model(
            crate::extensions::scale::PinnedSigma::unit,
            &y,
            &fit,
            &mut rng(32),
            &mut rng(32),
        );
        assert!(all_passed(&results), "{results:#?}");
    }

    // ---- the scale point: the opt-in "does it learn from the data?" oracle ----

    /// The fixture. `fit` is deliberately **non-zero**: at `fit = 0` the residual
    /// *is* the response, and a model scoring the raw response cannot be told
    /// apart from one scoring residuals.
    fn scale_fixture() -> ([f64; 6], [f64; 6]) {
        (
            [0.31, -0.12, 0.07, 0.22, -0.44, 0.18],
            [0.1, -0.1, 0.0, 0.2, -0.3, 0.1],
        )
    }

    #[test]
    fn the_shelf_scale_model_learns_from_the_data() {
        let (y, fit) = scale_fixture();
        let results = check_scale_model_learns_from_data(
            || crate::extensions::scale::GlobalSigma::new(6.0, 0.02).unwrap(),
            &y,
            &fit,
            31,
        );
        assert!(all_passed(&results), "{results:#?}");
    }

    /// `PinnedSigma` is *correct* to ignore the data — it holds σ² = 1 for the
    /// probit augmentation — and it would fail this probe. That is precisely why
    /// the probe is opt-in and not part of `check_scale_model`.
    ///
    /// This test pins the reasoning: a pinned scale passes the mandatory checks
    /// and fails the opt-in one. Fold the probe into `check_scale_model` and you
    /// break correct code.
    #[test]
    fn a_pinned_scale_passes_the_mandatory_checks_and_would_fail_the_opt_in_probe() {
        let (y, fit) = scale_fixture();
        assert!(
            all_passed(&check_scale_model(
                crate::extensions::scale::PinnedSigma::unit,
                &y,
                &fit,
                &mut rng(32),
                &mut rng(32),
            )),
            "a pinned scale is correct and must pass the mandatory checks"
        );
        let opt_in = check_scale_model_learns_from_data(
            crate::extensions::scale::PinnedSigma::unit,
            &y,
            &fit,
            32,
        );
        assert!(
            !all_passed(&opt_in),
            "a pinned scale does not learn from the data, so the opt-in probe must reject it: \
             {opt_in:#?}"
        );
    }

    /// A correct precision-weighted model, the shape of `template_scale`'s own
    /// recipe. It exists to pin the **tolerance** in the shift probe.
    ///
    /// In floating point `(y + c) − (fit + c)` is not bit-identical to
    /// `y − fit`, so a correct model's σ² moves in the last few ulps under the
    /// shift. A bitwise shift probe therefore reds correct code — the template
    /// caught exactly that before this shipped. The bug the probe hunts moves σ²
    /// by orders of magnitude, so a relative tolerance loses nothing:
    /// `scoring_the_raw_response_fails_only_the_shift_probe` proves it still
    /// bites.
    #[derive(Debug, Clone)]
    struct WeightedProfileSigma {
        precisions: Vec<f64>,
        sigma_sq: f64,
    }
    impl ScaleModel for WeightedProfileSigma {
        type Error = Never;
        fn update(
            &mut self,
            ctx: &ScaleCtx,
            rng: &mut dyn rand_core::Rng,
        ) -> std::result::Result<(), Self::Error> {
            let rss: f64 = ctx
                .y
                .iter()
                .zip(ctx.fit)
                .zip(&self.precisions)
                .map(|((y, f), w)| w * (y - f) * (y - f))
                .sum();
            let n = ctx.y.len() as f64;
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
            self.sigma_sq = (6.0 * 0.02 + rss) / (6.0 + n + z.abs());
            Ok(())
        }
        fn sigma_sq(&self) -> f64 {
            self.sigma_sq
        }
        fn precisions(&self) -> Option<&[f64]> {
            Some(&self.precisions)
        }
    }

    #[test]
    fn a_correct_model_is_not_red_flagged_by_floating_point_shift_noise() {
        let (y, fit) = scale_fixture();
        let results = check_scale_model_learns_from_data(
            || WeightedProfileSigma {
                precisions: vec![1.0, 1.0, 0.5, 0.5, 2.0, 2.0],
                sigma_sq: 0.02,
            },
            &y,
            &fit,
            35,
        );
        assert!(
            all_passed(&results),
            "a correct residual-scoring model must pass the shift probe: (y+c) − (fit+c) differs \
             from y − fit in the last ulp, so this must be a tolerance, not a bit compare. \
             {results:#?}"
        );
    }

    /// Ignores the data entirely and returns a draw from the prior. Finite,
    /// strictly positive, deterministic given its RNG — so every mandatory
    /// scale check passes it. Only the opt-in probe sees it.
    #[derive(Debug, Clone)]
    struct PriorOnlySigma {
        sigma_sq: f64,
    }
    impl ScaleModel for PriorOnlySigma {
        type Error = Never;
        fn update(
            &mut self,
            _ctx: &ScaleCtx,
            rng: &mut dyn rand_core::Rng,
        ) -> std::result::Result<(), Self::Error> {
            // A scaled-inverse-χ² prior draw, never looking at ctx.y or ctx.fit.
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
            self.sigma_sq = 0.02 * 6.0 / (z * z + 1.0);
            Ok(())
        }
        fn sigma_sq(&self) -> f64 {
            self.sigma_sq
        }
    }

    #[test]
    fn a_scale_model_that_ignores_the_data_fails_only_the_opt_in_probe() {
        let (y, fit) = scale_fixture();
        let make = || PriorOnlySigma { sigma_sq: 0.02 };

        // Every mandatory check passes it: this is the blindness being closed.
        assert!(
            all_passed(&check_scale_model(
                make,
                &y,
                &fit,
                &mut rng(33),
                &mut rng(33)
            )),
            "a prior-only scale is finite, positive and deterministic: the mandatory checks \
             cannot see it"
        );

        let results = check_scale_model_learns_from_data(make, &y, &fit, 33);
        let responds = results
            .iter()
            .find(|r| r.name == "sigma_sq_responds_to_residuals")
            .unwrap();
        assert!(
            !responds.passed,
            "a model ignoring the data must fail the response probe: {}",
            responds.detail
        );
    }

    /// Scores the raw response instead of the residual: `Σ yᵢ²` where it should
    /// be `Σ (yᵢ − fitᵢ)²`. It responds to the data perfectly well, so the
    /// residual-magnitude probe passes it. Only the shift probe sees it — which
    /// is the whole reason there are two.
    #[derive(Debug, Clone)]
    struct ForgotToSubtractTheFit {
        sigma_sq: f64,
    }
    impl ScaleModel for ForgotToSubtractTheFit {
        type Error = Never;
        fn update(
            &mut self,
            ctx: &ScaleCtx,
            rng: &mut dyn rand_core::Rng,
        ) -> std::result::Result<(), Self::Error> {
            let n = ctx.y.len() as f64;
            let rss: f64 = ctx.y.iter().map(|y| y * y).sum(); // <-- the bug
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
            self.sigma_sq = (6.0 * 0.02 + rss) / (6.0 + n + z.abs());
            Ok(())
        }
        fn sigma_sq(&self) -> f64 {
            self.sigma_sq
        }
    }

    #[test]
    fn scoring_the_raw_response_fails_only_the_shift_probe() {
        let (y, fit) = scale_fixture();
        let make = || ForgotToSubtractTheFit { sigma_sq: 0.02 };
        let results = check_scale_model_learns_from_data(make, &y, &fit, 34);

        let shift = results
            .iter()
            .find(|r| r.name == "sigma_sq_uses_residuals")
            .unwrap();
        assert!(
            !shift.passed,
            "a model scoring the raw response must fail the shift probe: {}",
            shift.detail
        );
        // And it sails through the other one, which is why both exist.
        assert!(
            results
                .iter()
                .find(|r| r.name == "sigma_sq_responds_to_residuals")
                .unwrap()
                .passed,
            "the residual-magnitude probe cannot see this bug: {results:#?}"
        );
    }

    /// Negative control: an update drawing entropy outside the RNG it is
    /// handed fails `update_deterministic` with a pointed message.
    #[test]
    fn nondeterministic_scale_update_fails() {
        #[derive(Debug)]
        struct LeakyScale {
            counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
            sigma_sq: f64,
        }
        impl crate::extensions::scale::ScaleModel for LeakyScale {
            type Error = std::convert::Infallible;
            fn update(
                &mut self,
                _ctx: &crate::extensions::scale::ScaleCtx<'_>,
                _rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<(), Self::Error> {
                let next = self
                    .counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                self.sigma_sq = 1.0 + next as f64;
                Ok(())
            }
            fn sigma_sq(&self) -> f64 {
                self.sigma_sq
            }
        }

        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let make = {
            let counter = std::sync::Arc::clone(&counter);
            move || LeakyScale {
                counter: std::sync::Arc::clone(&counter),
                sigma_sq: 1.0,
            }
        };
        let y = [0.2, -0.1, 0.3, 0.0];
        let fit = [0.1, 0.0, 0.1, 0.0];
        let results = check_scale_model(make, &y, &fit, &mut rng(33), &mut rng(33));
        let update = results
            .iter()
            .find(|r| r.name == "update_deterministic")
            .unwrap();
        assert!(!update.passed);
        assert!(update.detail.contains("outside the RNG"));
    }

    /// Negative control: malformed per-observation precisions (wrong length,
    /// non-positive values) fail `precisions_valid`.
    #[test]
    fn malformed_scale_precisions_fail() {
        #[derive(Debug)]
        struct BadPrecisions(Vec<f64>);
        impl crate::extensions::scale::ScaleModel for BadPrecisions {
            type Error = std::convert::Infallible;
            fn update(
                &mut self,
                _ctx: &crate::extensions::scale::ScaleCtx<'_>,
                _rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<(), Self::Error> {
                Ok(())
            }
            fn sigma_sq(&self) -> f64 {
                1.0
            }
            fn precisions(&self) -> Option<&[f64]> {
                Some(&self.0)
            }
        }

        let y = [0.2, -0.1, 0.3, 0.0];
        let fit = [0.1, 0.0, 0.1, 0.0];
        for bad in [vec![1.0; 3], vec![1.0, -2.0, 1.0, 1.0]] {
            let results = check_scale_model(
                || BadPrecisions(bad.clone()),
                &y,
                &fit,
                &mut rng(34),
                &mut rng(34),
            );
            let precisions = results
                .iter()
                .find(|r| r.name == "precisions_valid")
                .unwrap();
            assert!(!precisions.passed, "{results:#?}");
        }
    }

    // ---- basis-payload checks ----

    #[test]
    fn linear_gaussian_model_passes_basis_conjugacy_checks() {
        let model = crate::extensions::basis::LinearGaussianModel::new(0.05, 2).unwrap();
        let basis_rows: Vec<Vec<f64>> = vec![
            vec![1.0, 0.2],
            vec![1.0, -0.4],
            vec![1.0, 0.1],
            vec![1.0, 0.5],
            vec![1.0, -0.3],
            vec![1.0, 0.35],
        ];
        let observations = [0.12, -0.05, 0.31, 0.07, -0.22, 0.18];
        let weights = [1.0, 0.5, 2.0, 1.5, 1.0, 0.8];
        let results = check_basis_cell_model(
            &model,
            &basis_rows,
            &observations,
            &weights,
            0.3,
            &mut rng(51),
        );
        assert!(all_passed(&results), "{results:#?}");
    }

    // ---- variance-family checks (second family) ----

    fn variance_fixture() -> [f64; 6] {
        // Squared scale-free residuals on a plausible scale for λ′ ≈ 0.5.
        [0.31, 0.12, 0.07, 0.52, 0.44, 0.18]
    }

    #[test]
    fn inv_chi_sq_cell_model_passes_variance_conjugacy_checks() {
        let model = crate::extensions::cell_model::InvChiSqCellModel::new(8.0, 0.5).unwrap();
        let results = check_variance_cell_model(&model, &variance_fixture(), &mut rng(41));
        assert!(all_passed(&results), "{results:#?}");
    }

    /// Negative control: dropping the per-cell prior normalising term
    /// ((ν′/2)·ln(ν′λ′/2) − lnΓ(ν′/2)), invisible to structure-preserving
    /// oracles, fails the Bayes-factor check.
    #[test]
    fn dropped_variance_prior_term_fails_bayes_factor() {
        #[derive(Debug)]
        struct DroppedPriorTerm(crate::extensions::cell_model::InvChiSqCellModel);
        impl CellModel for DroppedPriorTerm {
            type Stats = crate::extensions::cell_model::InvChiSqStats;
            type Error = std::convert::Infallible;
            fn log_marginal_terms(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
            ) -> std::result::Result<f64, Self::Error> {
                // wrong: re-adds the per-cell prior constant, cancelling it.
                let nu = 8.0_f64;
                let lambda = 0.5_f64;
                let prior_term =
                    0.5 * nu * mathsfn::ln(0.5 * nu * lambda) - mathsfn::lgamma(0.5 * nu);
                Ok(self.0.log_marginal_terms(stats, sigma_sq)? - prior_term * stats.len() as f64)
            }
            fn draw_cell_values(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
                rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<Vec<f64>, Self::Error> {
                self.0.draw_cell_values(stats, sigma_sq, rng)
            }
        }

        let model = DroppedPriorTerm(
            crate::extensions::cell_model::InvChiSqCellModel::new(8.0, 0.5).unwrap(),
        );
        let results = check_variance_cell_model(&model, &variance_fixture(), &mut rng(42));
        let bayes = results
            .iter()
            .find(|r| r.name == "marginal_bayes_factor")
            .unwrap();
        assert!(!bayes.passed, "{results:#?}");
    }

    // ---- deep-seam checks ----

    fn cell_fixture() -> ([f64; 6], [f64; 6]) {
        (
            [0.31, -0.12, 0.07, 0.22, -0.44, 0.18],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    }

    #[test]
    fn gaussian_cell_model_passes_conjugacy_checks() {
        let (observations, weights) = cell_fixture();
        let model = crate::extensions::cell_model::GaussianCellModel::new(0.02).unwrap();
        let results = check_cell_model(&model, 0.3, &observations, &weights, &mut rng(21));
        assert!(all_passed(&results), "{results:#?}");
    }

    #[test]
    fn weighted_gaussian_cell_model_passes_conjugacy_checks() {
        let observations = [0.31, -0.12, 0.07, 0.22, -0.44, 0.18];
        let weights = [0.5, 2.0, 1.0, 1.5, 0.8, 1.2];
        let model = crate::extensions::cell_model::WeightedGaussianModel::new(0.02).unwrap();
        let results = check_cell_model(&model, 0.3, &observations, &weights, &mut rng(22));
        assert!(all_passed(&results), "{results:#?}");
    }

    /// The spec's named negative control: a deliberately wrong
    /// sufficient-statistic accumulation (merge double-counts) fails with a
    /// pointed message.
    #[test]
    fn double_counting_merge_fails_sufficiency() {
        #[derive(Debug, Default, Clone)]
        struct DoubleCountingStats {
            inner: crate::extensions::cell_model::GaussianCellStats,
        }
        impl crate::extensions::cell_model::CellStats for DoubleCountingStats {
            fn record(&mut self, value: f64, weight: f64) {
                self.inner.record(value, weight);
            }
            fn merge(&mut self, other: &Self) {
                self.inner.merge(&other.inner);
                self.inner.merge(&other.inner); // wrong: double-counts
            }
            fn remove(&mut self, other: &Self) {
                self.inner.remove(&other.inner);
            }
            fn reset(&mut self) {
                self.inner.reset();
            }
            fn occupied(&self) -> bool {
                self.inner.occupied()
            }
        }
        #[derive(Debug)]
        struct Model;
        impl crate::extensions::cell_model::CellModel for Model {
            type Stats = DoubleCountingStats;
            type Error = std::convert::Infallible;
            fn log_marginal_terms(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
            ) -> std::result::Result<f64, Self::Error> {
                crate::extensions::cell_model::GaussianCellModel::new(0.02)
                    .unwrap()
                    .log_marginal_terms(
                        &stats.iter().map(|s| s.inner.clone()).collect::<Vec<_>>(),
                        sigma_sq,
                    )
            }
            fn draw_cell_values(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
                rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<Vec<f64>, Self::Error> {
                crate::extensions::cell_model::GaussianCellModel::new(0.02)
                    .unwrap()
                    .draw_cell_values(
                        &stats.iter().map(|s| s.inner.clone()).collect::<Vec<_>>(),
                        sigma_sq,
                        rng,
                    )
            }
        }
        let (observations, weights) = cell_fixture();
        let results = check_cell_model(&Model, 0.3, &observations, &weights, &mut rng(23));
        let merge = results
            .iter()
            .find(|r| r.name == "stats_merge_consistency")
            .unwrap();
        assert!(!merge.passed);
        assert!(merge.detail.contains("add path"), "{}", merge.detail);
    }

    /// A mis-derived marginal (the complete per-cell 0.5·ln term dropped)
    /// fails the Bayes-factor check: the wrong-but-consistent class the
    /// checks can catch locally.
    #[test]
    fn dropped_normalising_term_fails_bayes_factor() {
        #[derive(Debug)]
        struct DroppedTermModel;
        // σ_μ² comparable to σ² so the dropped normalising term actually
        // moves the 2-vs-1-cell Bayes factor (at σ_μ² ≪ σ² it is nearly
        // structure-flat and no local check could see it).
        const DROPPED_SIGMA_MU_SQ: f64 = 0.5;
        impl crate::extensions::cell_model::CellModel for DroppedTermModel {
            type Stats = crate::extensions::cell_model::GaussianCellStats;
            type Error = std::convert::Infallible;
            fn log_marginal_terms(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
            ) -> std::result::Result<f64, Self::Error> {
                // wrong: only the S² exponent term, no 0.5·ln(σ²/denom).
                let sigma_mu_sq = DROPPED_SIGMA_MU_SQ;
                let mut total = 0.0;
                for cell in stats {
                    let denominator = cell.count() * sigma_mu_sq + sigma_sq;
                    total += sigma_mu_sq * cell.sum() * cell.sum() / (2.0 * sigma_sq * denominator);
                }
                Ok(total)
            }
            fn draw_cell_values(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
                rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<Vec<f64>, Self::Error> {
                crate::extensions::cell_model::GaussianCellModel::new(DROPPED_SIGMA_MU_SQ)
                    .unwrap()
                    .draw_cell_values(stats, sigma_sq, rng)
            }
        }
        let (observations, weights) = cell_fixture();
        let results = check_cell_model(
            &DroppedTermModel,
            0.3,
            &observations,
            &weights,
            &mut rng(24),
        );
        let bayes = results
            .iter()
            .find(|r| r.name == "marginal_bayes_factor")
            .unwrap();
        assert!(!bayes.passed);
        assert!(bayes.detail.contains("disagrees"), "{}", bayes.detail);
    }

    /// A cell-value draw with the wrong posterior variance fails the
    /// cell-local SBC.
    #[test]
    fn wrong_posterior_variance_fails_cell_sbc() {
        #[derive(Debug)]
        struct OverconfidentModel;
        impl crate::extensions::cell_model::CellModel for OverconfidentModel {
            type Stats = crate::extensions::cell_model::GaussianCellStats;
            type Error = std::convert::Infallible;
            fn log_marginal_terms(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
            ) -> std::result::Result<f64, Self::Error> {
                crate::extensions::cell_model::GaussianCellModel::new(0.02)
                    .unwrap()
                    .log_marginal_terms(stats, sigma_sq)
            }
            fn draw_cell_values(
                &self,
                stats: &[Self::Stats],
                sigma_sq: f64,
                rng: &mut dyn rand_core::Rng,
            ) -> std::result::Result<Vec<f64>, Self::Error> {
                // wrong: posterior SD scaled down 4× (overconfident draws),
                // but only once data is in the cell, so the prior draw the
                // check relies on stays correct.
                let reference =
                    crate::extensions::cell_model::GaussianCellModel::new(0.02).unwrap();
                let sigma_mu_sq = 0.02_f64;
                let mut values = reference.draw_cell_values(stats, sigma_sq, rng)?;
                for (value, cell) in values.iter_mut().zip(stats) {
                    if cell.occupied() {
                        let denominator = cell.count() * sigma_mu_sq + sigma_sq;
                        let mean = sigma_mu_sq * cell.sum() / denominator;
                        *value = mean + (*value - mean) * 0.25;
                    }
                }
                Ok(values)
            }
        }
        let (observations, weights) = cell_fixture();
        let results = check_cell_model(
            &OverconfidentModel,
            0.3,
            &observations,
            &weights,
            &mut rng(25),
        );
        let sbc = results.iter().find(|r| r.name == "cell_value_sbc").unwrap();
        assert!(!sbc.passed);
        assert!(sbc.detail.contains("NOT uniform"), "{}", sbc.detail);
    }

    /// Wrong-length / invalid weights fail their checks.
    #[test]
    fn invalid_weights_fail() {
        let usage = InclusionUsage::new(3);
        let results = check_inclusion_model(
            || crate::extensions::inclusion::WeightedInclusion::new(vec![1.0, -2.0]),
            3,
            &usage,
            &mut rng(8),
            &mut rng(8),
        );
        assert!(
            results
                .iter()
                .any(|r| r.name == "weights_length" && !r.passed)
        );
        assert!(
            results
                .iter()
                .any(|r| r.name == "weights_valid" && !r.passed)
        );
    }

    // ---- response: the kernel step ----

    #[test]
    fn albert_chib_probit_passes_response_model_checks() {
        let y = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let fit = [0.4, -0.3, 0.9, 0.1, -0.8, 0.2];
        let results = check_response_model(
            || crate::extensions::response::AlbertChibProbit,
            &y,
            &fit,
            1.0,
            &mut rng(11),
            &mut rng(11),
        );
        assert!(all_passed(&results), "{results:#?}");
    }

    #[test]
    fn robust_t_step_reports_its_non_unit_weights() {
        let y = [0.4, -0.2, 0.5, 0.1, -0.3, 0.25];
        let fit = [0.1, -0.1, 0.2, 0.05, -0.2, 0.1];
        let results = check_response_model(
            || crate::extensions::response::RobustTStep::new(5.0).unwrap(),
            &y,
            &fit,
            0.04,
            &mut rng(12),
            &mut rng(12),
        );
        assert!(all_passed(&results), "{results:#?}");
        // The scale-mixture step produces precision weights, so the pairing
        // rule has to be surfaced rather than passed over in silence.
        let weights = results.iter().find(|r| r.name == "weights_valid").unwrap();
        assert!(
            weights.detail.contains("NOT all 1"),
            "the pairing rule must be reported: {weights:#?}"
        );
    }

    /// A step that leaves an observation untouched: the buffers the sampler
    /// hands over are reused across sweeps, so a partial write silently
    /// regresses on the previous sweep's latent.
    #[derive(Debug)]
    struct PartialAugment;
    impl ResponseModel for PartialAugment {
        type Error = std::convert::Infallible;
        fn augment(
            &mut self,
            _y: &[f64],
            fit: &[f64],
            _sigma_sq: f64,
            _rng: &mut dyn rand_core::Rng,
            working: &mut [f64],
            weights: &mut [f64],
        ) -> std::result::Result<(), Self::Error> {
            for i in 1..working.len() {
                working[i] = fit[i];
                weights[i] = 1.0;
            }
            Ok(())
        }
    }

    #[test]
    fn partially_written_working_response_fails() {
        let y = [1.0, 0.0, 1.0, 0.0];
        let fit = [0.2, -0.2, 0.3, -0.1];
        let results =
            check_response_model(|| PartialAugment, &y, &fit, 1.0, &mut rng(13), &mut rng(13));
        let working = results.iter().find(|r| r.name == "working_finite").unwrap();
        assert!(!working.passed, "{results:#?}");
    }

    /// An augmentation that pulls entropy from shared state outside the RNG
    /// (a process-global counter behaves exactly as a thread-local RNG would).
    #[derive(Debug)]
    struct LeakyStep {
        counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
    }
    impl ResponseModel for LeakyStep {
        type Error = std::convert::Infallible;
        fn augment(
            &mut self,
            _y: &[f64],
            fit: &[f64],
            _sigma_sq: f64,
            _rng: &mut dyn rand_core::Rng,
            working: &mut [f64],
            weights: &mut [f64],
        ) -> std::result::Result<(), Self::Error> {
            let next = self
                .counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            for i in 0..working.len() {
                working[i] = fit[i] + next as f64;
                weights[i] = 1.0;
            }
            Ok(())
        }
    }

    #[test]
    fn response_model_drawing_off_the_handed_rng_fails() {
        let y = [1.0, 0.0, 1.0, 0.0];
        let fit = [0.2, -0.2, 0.3, -0.1];
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let results = check_response_model(
            || LeakyStep {
                counter: std::sync::Arc::clone(&counter),
            },
            &y,
            &fit,
            1.0,
            &mut rng(14),
            &mut rng(14),
        );
        let det = results
            .iter()
            .find(|r| r.name == "augment_deterministic")
            .unwrap();
        assert!(!det.passed, "{results:#?}");
    }

    // ---- membership: the membership kernel ----

    fn membership_keys() -> Vec<f64> {
        // Squared-distance keys from one observation to four cells.
        vec![0.02, 0.35, 1.10, 0.60]
    }

    #[test]
    fn softmax_kernel_passes_membership_checks() {
        let kernel = crate::extensions::membership::SoftmaxKernel::new(0.25).unwrap();
        let results = check_membership_kernel(&kernel, &membership_keys());
        assert!(all_passed(&results), "{results:#?}");
    }

    /// A non-exponential blend whose weights genuinely depend on absolute
    /// distance (the template's own recipe). It is not shift-invariant, and it
    /// must still pass: the trait endorses no single formulation, so the check
    /// must test for surviving mass, not for softmax's invariance.
    #[derive(Debug)]
    struct InverseQuadratic {
        tau: f64,
    }
    impl MembershipKernel for InverseQuadratic {
        fn weights(&self, keys: &[f64], weights: &mut [f64]) {
            for (weight, key) in weights.iter_mut().zip(keys) {
                *weight = 1.0 / (1.0 + key / self.tau);
            }
        }
    }

    #[test]
    fn non_exponential_kernel_passes_membership_checks() {
        let results = check_membership_kernel(&InverseQuadratic { tau: 0.1 }, &membership_keys());
        assert!(all_passed(&results), "{results:#?}");
    }

    /// The negative control for `weights_permutation_equivariant`: a correct
    /// softmax multiplied by a per-cell bias read from the cell's *position* in
    /// the row (the shape of a per-index lookup table). It is finite, positive,
    /// deterministic, keeps its mass, and stays monotone in the key — every
    /// other membership check passes it. It is simply reading an identity the
    /// cells do not have.
    #[derive(Debug)]
    struct PositionBiasedSoftmax {
        tau: f64,
    }
    impl MembershipKernel for PositionBiasedSoftmax {
        fn weights(&self, keys: &[f64], weights: &mut [f64]) {
            let best = keys.iter().copied().fold(f64::INFINITY, f64::min);
            for (i, (weight, key)) in weights.iter_mut().zip(keys).enumerate() {
                *weight = mathsfn::exp(-(key - best) / self.tau) * (1.0 + 0.1 * i as f64);
            }
        }
    }

    #[test]
    fn a_position_biased_kernel_fails_only_the_equivariance_check() {
        let results =
            check_membership_kernel(&PositionBiasedSoftmax { tau: 0.25 }, &membership_keys());
        let equivariance = results
            .iter()
            .find(|r| r.name == "weights_permutation_equivariant")
            .unwrap();
        assert!(
            !equivariance.passed,
            "a position-keyed kernel must fail equivariance: {}",
            equivariance.detail
        );
        // The point of the control: nothing else on this extension point sees it.
        assert!(
            results
                .iter()
                .filter(|r| r.name != "weights_permutation_equivariant")
                .all(|r| r.passed),
            "every other membership check should pass this kernel: {results:#?}"
        );
    }

    /// A legitimate recipe that subtracts `keys[0]` instead of the row's
    /// minimum before exponentiating. It differs from `SoftmaxKernel` by a
    /// per-row positive constant, which the engine's row normalisation divides
    /// straight back out — so the two are the *same kernel*, and the
    /// equivariance check must not red it.
    ///
    /// This is the false-positive guard. A bitwise (un-normalised) equivariance
    /// test fails this kernel, and failing a correct recipe is worse than
    /// missing a wrong one.
    #[derive(Debug)]
    struct FirstKeyOffsetSoftmax {
        tau: f64,
    }
    impl MembershipKernel for FirstKeyOffsetSoftmax {
        fn weights(&self, keys: &[f64], weights: &mut [f64]) {
            let reference = keys[0];
            for (weight, key) in weights.iter_mut().zip(keys) {
                *weight = mathsfn::exp(-(key - reference) / self.tau);
            }
        }
    }

    #[test]
    fn a_constant_row_offset_is_not_an_equivariance_violation() {
        let results =
            check_membership_kernel(&FirstKeyOffsetSoftmax { tau: 0.25 }, &membership_keys());
        assert!(
            all_passed(&results),
            "a per-row constant offset is divided out by normalisation and must pass: {results:#?}"
        );
    }

    /// The negative control for `weights_weakly_monotone`: a kernel with the
    /// sign flipped (`exp(+d²/τ)`) gives the *furthest* cell the most weight.
    /// It is finite, positive, deterministic and keeps its mass, so every other
    /// membership check passes it — it is simply backwards. The monotonicity
    /// oracle is the only thing standing between that bug and a green run.
    #[derive(Debug)]
    struct SignFlippedSoftmax {
        tau: f64,
    }
    impl MembershipKernel for SignFlippedSoftmax {
        fn weights(&self, keys: &[f64], weights: &mut [f64]) {
            let best = keys.iter().copied().fold(f64::INFINITY, f64::min);
            for (weight, key) in weights.iter_mut().zip(keys) {
                *weight = mathsfn::exp((key - best) / self.tau);
            }
        }
    }

    #[test]
    fn a_sign_flipped_kernel_fails_only_the_monotonicity_check() {
        let results =
            check_membership_kernel(&SignFlippedSoftmax { tau: 0.25 }, &membership_keys());
        let monotone = results
            .iter()
            .find(|r| r.name == "weights_weakly_monotone")
            .expect("the monotonicity check runs");
        assert!(
            !monotone.passed,
            "an inverted membership kernel must fail the monotonicity oracle"
        );
        // The point of the control: every *other* check waves it through, which
        // is why the oracle had to be added.
        for result in results
            .iter()
            .filter(|r| r.name != "weights_weakly_monotone")
        {
            assert!(
                result.passed,
                "the sign-flipped kernel unexpectedly failed `{}`, which would make the \
                 monotonicity oracle look redundant: {result:#?}",
                result.name
            );
        }
    }

    /// The bug the distant-row check exists to catch: exponentiate the raw key
    /// instead of subtracting the row's best one first. Every other check
    /// passes; a distant observation underflows the whole row to zero.
    #[derive(Debug)]
    struct UnshiftedSoftmax {
        tau: f64,
    }
    impl MembershipKernel for UnshiftedSoftmax {
        fn weights(&self, keys: &[f64], weights: &mut [f64]) {
            for (weight, key) in weights.iter_mut().zip(keys) {
                *weight = mathsfn::exp(-key / self.tau);
            }
        }
    }

    #[test]
    fn unshifted_softmax_underflows_a_distant_row() {
        let kernel = UnshiftedSoftmax { tau: 0.25 };
        let results = check_membership_kernel(&kernel, &membership_keys());
        let distant = results
            .iter()
            .find(|r| r.name == "distant_row_keeps_mass")
            .unwrap();
        assert!(
            !distant.passed,
            "the unshifted kernel must underflow a distant row: {results:#?}"
        );
        // It is precisely the checks a naive kernel sails through that make
        // the one it fails worth having.
        assert!(
            results
                .iter()
                .find(|r| r.name == "row_has_mass")
                .unwrap()
                .passed
        );
        assert!(
            results
                .iter()
                .find(|r| r.name == "weights_finite_nonneg")
                .unwrap()
                .passed
        );
    }

    /// A kernel whose row carries no mass at all.
    #[derive(Debug)]
    struct DeadKernel;
    impl MembershipKernel for DeadKernel {
        fn weights(&self, _keys: &[f64], weights: &mut [f64]) {
            weights.fill(0.0);
        }
    }

    #[test]
    fn zero_mass_membership_row_fails() {
        let results = check_membership_kernel(&DeadKernel, &membership_keys());
        let mass = results.iter().find(|r| r.name == "row_has_mass").unwrap();
        assert!(!mass.passed, "{results:#?}");
    }

    // ---- count priors ----

    #[test]
    fn shifted_poisson_binomial_passes_count_prior_checks() {
        let dists = dists(4);
        let weights = vec![1.0; 4];
        let ctx = ctx_fixture(&dists, &weights);
        let results = check_count_priors(
            &crate::extensions::count_priors::ShiftedPoissonBinomial,
            &ctx,
            12,
        );
        assert!(all_passed(&results), "{results:#?}");
    }

    /// A prior returning a non-finite ratio at one count: it poisons the
    /// acceptance ratio of every add/remove move, and nothing downstream says so.
    #[derive(Debug)]
    struct NonFiniteAtFour;
    impl CountPriors for NonFiniteAtFour {
        fn log_cell_count_ratio(&self, b: usize, _ctx: &ModelCtx) -> f64 {
            if b == 4 { f64::NEG_INFINITY } else { -0.5 }
        }
        fn log_dim_count_ratio(&self, _d: usize, _ctx: &ModelCtx) -> f64 {
            -0.5
        }
    }

    #[test]
    fn non_finite_count_ratio_fails() {
        let dists = dists(4);
        let weights = vec![1.0; 4];
        let ctx = ctx_fixture(&dists, &weights);
        let results = check_count_priors(&NonFiniteAtFour, &ctx, 12);
        let finite = results
            .iter()
            .find(|r| r.name == "cell_ratios_finite")
            .unwrap();
        assert!(!finite.passed, "{results:#?}");
        assert!(
            finite.detail.contains("log_cell_count_ratio(4)"),
            "the failing count must be named: {finite:#?}"
        );
    }

    /// A prior with interior state: the same count prices differently on the
    /// second call, so the chain is not reproducible from its seed.
    #[derive(Debug)]
    struct DriftingPrior {
        calls: std::sync::atomic::AtomicU32,
    }
    impl CountPriors for DriftingPrior {
        fn log_cell_count_ratio(&self, _b: usize, _ctx: &ModelCtx) -> f64 {
            let next = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            -0.5 + f64::from(next) * 1.0e-3
        }
        fn log_dim_count_ratio(&self, _d: usize, _ctx: &ModelCtx) -> f64 {
            -0.5
        }
    }

    #[test]
    fn stateful_count_prior_fails_purity() {
        let dists = dists(4);
        let weights = vec![1.0; 4];
        let ctx = ctx_fixture(&dists, &weights);
        let results = check_count_priors(
            &DriftingPrior {
                calls: std::sync::atomic::AtomicU32::new(0),
            },
            &ctx,
            12,
        );
        let pure = results
            .iter()
            .find(|r| r.name == "ratios_deterministic")
            .unwrap();
        assert!(!pure.passed, "{results:#?}");
    }

    // ---- the basis point: the CellBasis check ----

    fn design_rows() -> Vec<Vec<f64>> {
        vec![
            vec![0.10, -0.25, 0.40, 0.05],
            vec![-0.30, 0.15, -0.20, 0.45],
            vec![0.35, 0.05, 0.10, -0.15],
        ]
    }

    #[test]
    fn the_shelf_basis_passes_its_own_check() {
        let basis = crate::extensions::basis::LinearBasis::new(vec![0, 2]);
        let results = check_cell_basis(&basis, &design_rows());
        assert!(all_passed(&results), "{results:#?}");
    }

    #[test]
    fn a_pure_intercept_basis_passes_with_no_declared_column() {
        let basis = crate::extensions::basis::LinearBasis::new(vec![]);
        let results = check_cell_basis(&basis, &design_rows());
        assert!(all_passed(&results), "{results:#?}");
    }

    /// Reads column 3 but declares `Some(1)`. The engine bounds-checks the claim
    /// against the design exactly once and then trusts it, so on a 2-column
    /// design this passes `fit` and then indexes out of range in the hot loop.
    ///
    /// Everything else about it is impeccable: it writes every entry, all
    /// values are finite, and it is perfectly deterministic.
    #[derive(Debug)]
    struct UnderDeclaredBasis;
    impl CellBasis for UnderDeclaredBasis {
        fn q(&self) -> usize {
            2
        }
        fn row(&self, x_row: &[f64], out: &mut [f64]) {
            out[0] = 1.0;
            out[1] = x_row[3]; // reads column 3...
        }
        fn max_column(&self) -> Option<usize> {
            Some(1) // ...but claims it never looks past column 1
        }
    }

    #[test]
    fn an_under_declared_max_column_fails_and_nothing_else_sees_it() {
        let results = check_cell_basis(&UnderDeclaredBasis, &design_rows());
        let claim = results
            .iter()
            .find(|r| r.name == "max_column_claim_exact")
            .expect("the max_column check runs");
        assert!(
            !claim.passed,
            "a basis reading column 3 while declaring Some(1) must fail: {}",
            claim.detail
        );
        assert!(claim.detail.contains('3'), "{}", claim.detail);
        // The point of the control: it is otherwise a perfectly well-behaved basis.
        assert!(
            results
                .iter()
                .filter(|r| r.name != "max_column_claim_exact")
                .all(|r| r.passed),
            "every other basis check should pass it: {results:#?}"
        );
    }

    /// Leaves an entry of the output buffer unwritten. The engine hands `row` an
    /// already-sized buffer, so the cell silently carries whatever was there.
    #[derive(Debug)]
    struct PartiallyWritingBasis;
    impl CellBasis for PartiallyWritingBasis {
        fn q(&self) -> usize {
            3
        }
        fn row(&self, x_row: &[f64], out: &mut [f64]) {
            out[0] = 1.0;
            out[1] = x_row[0];
            // out[2] never written
        }
        fn max_column(&self) -> Option<usize> {
            Some(0)
        }
    }

    #[test]
    fn a_partially_written_row_fails() {
        let results = check_cell_basis(&PartiallyWritingBasis, &design_rows());
        let width = results.iter().find(|r| r.name == "basis_width").unwrap();
        assert!(!width.passed, "{width:#?}");
    }

    /// A caller-written basis payload — the case that could not be checked at
    /// all while `check_basis_cell_model` took the concrete shelf type. This is
    /// the shelf's own algebra, re-expressed through the public traits, so it
    /// must pass every basis-payload check.
    #[test]
    fn a_caller_written_basis_payload_can_be_checked() {
        let model = crate::extensions::basis::LinearGaussianModel::new(0.5, 2).unwrap();
        let basis_rows = vec![
            vec![1.0, -0.4],
            vec![1.0, 0.1],
            vec![1.0, 0.3],
            vec![1.0, -0.2],
            vec![1.0, 0.45],
            vec![1.0, -0.35],
        ];
        let observations = [0.21, -0.05, 0.14, -0.11, 0.30, -0.22];
        let weights = [1.0; 6];
        let results = check_basis_cell_model(
            &model,
            &basis_rows,
            &observations,
            &weights,
            0.09,
            &mut rng(11),
        );
        assert!(all_passed(&results), "{results:#?}");
    }

    // ---- the count-prior density oracle ----

    /// The paper's cell prior as a *density*: b − 1 ~ Poisson(λ_c), so
    /// log P(b) ∝ (b−1)·ln λ_c − ln (b−1)!  (the normaliser cancels in the
    /// difference, so it is omitted).
    fn log_cell_pmf(b: usize, ctx: &ModelCtx) -> f64 {
        let ln_factorial: f64 = (1..b).map(|k| mathsfn::ln(k as f64)).sum();
        (b - 1) as f64 * mathsfn::ln(ctx.lambda_c) - ln_factorial
    }

    /// The paper's dimension prior as a density: d − 1 ~ Binomial(p−1, ω/p).
    fn log_dim_pmf(d: usize, ctx: &ModelCtx) -> f64 {
        let p = ctx.p;
        let theta = ctx.omega / p as f64;
        let ln_choose: f64 = (1..d).map(|k| mathsfn::ln((p - k) as f64)).sum::<f64>()
            - (1..d).map(|k| mathsfn::ln(k as f64)).sum::<f64>();
        ln_choose + (d - 1) as f64 * mathsfn::ln(theta) + (p - d) as f64 * mathsfn::ln(1.0 - theta)
    }

    #[test]
    fn the_shelf_prior_agrees_with_its_own_density() {
        let dists = dists(4);
        let weights = vec![1.0; 4];
        let ctx = ctx_fixture(&dists, &weights);
        let results = check_count_priors_against_density(
            &crate::extensions::count_priors::ShiftedPoissonBinomial,
            &ctx,
            &|b| log_cell_pmf(b, &ctx),
            &|d| log_dim_pmf(d, &ctx),
            12,
        );
        assert!(
            all_passed(&results),
            "the shelf prior's hooks must be the adjacent differences of the paper's \
             densities: {results:#?}"
        );
    }

    /// The ratio taken the wrong way round: `log P(b−1) − log P(b)`. Finite,
    /// pure, portable, and it passes every check in `check_count_priors` — the
    /// sampler then prefers exactly the tessellation sizes the prior disfavours.
    #[derive(Debug)]
    struct InvertedCellRatio;
    impl CountPriors for InvertedCellRatio {
        fn log_cell_count_ratio(&self, b: usize, ctx: &ModelCtx) -> f64 {
            -(mathsfn::ln(ctx.lambda_c) - mathsfn::ln((b - 1) as f64))
        }
        fn log_dim_count_ratio(&self, d: usize, ctx: &ModelCtx) -> f64 {
            crate::extensions::count_priors::ShiftedPoissonBinomial.log_dim_count_ratio(d, ctx)
        }
    }

    #[test]
    fn an_inverted_ratio_is_invisible_without_the_density_and_caught_with_it() {
        let dists = dists(4);
        let weights = vec![1.0; 4];
        let ctx = ctx_fixture(&dists, &weights);

        // Without the density there is no oracle at all: nothing to contradict.
        assert!(
            all_passed(&check_count_priors(&InvertedCellRatio, &ctx, 12)),
            "an inverted ratio is finite, pure and portable: check_count_priors cannot see it"
        );

        let results = check_count_priors_against_density(
            &InvertedCellRatio,
            &ctx,
            &|b| log_cell_pmf(b, &ctx),
            &|d| log_dim_pmf(d, &ctx),
            12,
        );
        let cell = results
            .iter()
            .find(|r| r.name == "cell_ratio_matches_density")
            .unwrap();
        assert!(!cell.passed, "{cell:#?}");
        assert!(
            results
                .iter()
                .find(|r| r.name == "dim_ratio_matches_density")
                .unwrap()
                .passed,
            "only the cell hook is wrong; the dimension hook must stay green"
        );
    }

    /// A spurious centre-pick factor that thins the cell-count prior to
    /// λ_c/(b(b+1)): the classic self-consistent wrong ratio. The density
    /// oracle is the cheapest thing that finds it — `check_move_set`
    /// cannot, because the factor cancels against its own reverse.
    #[derive(Debug)]
    struct ThinnedCellPrior;
    impl CountPriors for ThinnedCellPrior {
        fn log_cell_count_ratio(&self, b: usize, ctx: &ModelCtx) -> f64 {
            mathsfn::ln(ctx.lambda_c) - mathsfn::ln((b - 1) as f64) - mathsfn::ln(b as f64)
        }
        fn log_dim_count_ratio(&self, d: usize, ctx: &ModelCtx) -> f64 {
            crate::extensions::count_priors::ShiftedPoissonBinomial.log_dim_count_ratio(d, ctx)
        }
    }

    #[test]
    fn the_published_thinning_bug_fails_against_the_density() {
        let dists = dists(4);
        let weights = vec![1.0; 4];
        let ctx = ctx_fixture(&dists, &weights);
        let results = check_count_priors_against_density(
            &ThinnedCellPrior,
            &ctx,
            &|b| log_cell_pmf(b, &ctx),
            &|d| log_dim_pmf(d, &ctx),
            12,
        );
        let cell = results
            .iter()
            .find(|r| r.name == "cell_ratio_matches_density")
            .unwrap();
        assert!(
            !cell.passed,
            "the thinned prior must fail against the density it claims to be: {cell:#?}"
        );
    }

    // Unused-arc helper to silence the Arc import in no-feature builds.
    #[allow(dead_code)]
    fn _keep(_: Arc<()>) {}
}
