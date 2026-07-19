//! Distance and cell assignment: the `PairwiseDistance`
//! extension trait and the blanket `CellAssigner` batch kernel. Each shelf
//! approach is one file in `distance/`: [`Euclidean`] (the paper's metric),
//! [`Spherical`] (great-circle; the paper's model is Euclidean and gives
//! no spherical equation), [`ColumnMetrics`]
//! (`per_column`, the compound per-column mix and fit-time default),
//! [`Gower`] (mixed numeric + categorical), [`Manhattan`] (L1),
//! [`Minkowski`] (L_p), [`Cosine`] (directional), and [`Mahalanobis`]
//! (precision-weighted quadratic form).
//!
//! Everything here operates in scaled, encoded space. This point changes
//! arg-min geometry only: a `PairwiseDistance` is a pure, deterministic,
//! strictly-monotone comparison key deciding which centre is nearest, nothing
//! else (fractional membership is the membership point). A metric need not
//! own every column: `ColumnMetrics::with_group` hands a subset to your
//! geometry and keeps the built-in treatment for the rest; return a
//! squared-distance-scale key so the group contributions stay commensurate.
//! The engine provides the batch assignment kernel, lowest-index
//! tie-breaking, NaN guarding, and incremental reassignment caching.
//!
//! Bring-your-own data types go through `Metric::Prepared`: you prepare the
//! column before `Data` (mapped onto a range commensurate with the other
//! scaled columns, [−0.5, 0.5]), mark it `Prepared` in `with_metrics`, and
//! supply the geometry here (plus a coordinate law via `with_coords` if the
//! default normal does not match your domain, e.g. anything that wraps).
//!
//! Start from `examples/template_distance.rs`; the conformance checks are
//! `conformance::check_distance` and `conformance::check_assigner`, whose
//! `key_digest` probe is the cross-platform portability check: run it on each
//! platform you target and compare digests (a difference means un-pinned
//! maths, and the same-seed promise is void for chains using that geometry).
//!
//! Sources: Euclidean is the paper's metric (Stone & Gosling 2025); Spherical
//! is standard hyperspherical geometry (the paper's model is Euclidean),
//! derivation on the module;
//! `Gower` is Gower (1971), with its reduction to the scaled/encoded design
//! derived in the module docs; `Mahalanobis` is Mahalanobis (1936);
//! `Manhattan`/`Minkowski` are the classical L_p family, and `Cosine` the
//! standard information-retrieval similarity (e.g. Salton & McGill 1983); all
//! four transplanted onto the engine's monotone-key contract (powers instead
//! of roots, squared-form keys) as documented on each module.

mod cosine;
mod euclidean;
mod gower;
mod mahalanobis;
mod manhattan;
mod minkowski;
mod per_column;
mod spherical;

pub use cosine::Cosine;
pub use euclidean::Euclidean;
pub use gower::{Gower, GowerKind};
pub use mahalanobis::Mahalanobis;
pub use manhattan::Manhattan;
pub use minkowski::Minkowski;
pub use per_column::ColumnMetrics;
pub use spherical::Spherical;

use std::sync::Arc;

use crate::engine::data::{Data, Metric};
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::tessellation::{Centres, Tessellation};

/// This extension point's fit-time default assigner: the compound per-column
/// `ColumnMetrics` over the encoded metrics, boxed so the sampler resolves the
/// default without naming a concrete approach.
pub(crate) fn default_assigner(metrics: Vec<Metric>) -> Arc<dyn CellAssigner> {
    Arc::new(ColumnMetrics::new(metrics))
}

/// A pairwise comparison key between an observation and a centre: what
/// researchers implement to swap the assignment geometry.
///
/// # Contract
///
/// - **Pure and deterministic**: same inputs, same bits, no interior state, no
///   RNG, no caching with observable effects.
/// - **Strictly monotone key**: the return value is used only to compare
///   candidate centres for one observation (strict `<`); its magnitude is not
///   used for anything but the assigner's own nearest-centre bookkeeping (the
///   [`AssignmentCache`] winning keys), so any strictly increasing
///   transform of a true distance is valid (e.g. squared Euclidean).
/// - Both rows are full p-length rows in scaled space. The caller
///   synthesises the centre row equal to the observation everywhere except at
///   `active_dims` (required so a joint group metric, e.g. the spherical
///   great-circle group, can read the full coordinate context).
/// - A non-finite return value is caught by the assigner (always-on check) and
///   surfaced as [`AddiVortesError::NonFiniteDistance`]; it never propagates
///   into the chain.
pub trait PairwiseDistance: std::fmt::Debug + Send + Sync {
    /// The comparison key between `x_row` and `centre_row` given the
    /// tessellation's active dimensions (0-based global column indices).
    /// Rows are in scaled space (the sampler's coordinate system); the
    /// key itself is dimensionless; only its ordering matters.
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64;

    /// Opt-in fast path: when `true`, [`CellAssigner`]'s
    /// blanket impl assumes every key is a sum of non-negative, per-dimension
    /// squared differences (no group/joint metric contributes a
    /// cross-dimension term) and skips the synthesised centre-row buffer.
    /// Default `false`; every implementor keeps the general path unless it
    /// opts in.
    fn all_euclidean(&self) -> bool {
        false
    }
}

/// The structural change a proposal makes relative to the tessellation it was
/// proposed from, declared by the move so
/// [`CellAssigner::reassign`] can update a cached assignment instead of
/// recomputing it from scratch.
///
/// Carries no coordinates and no RNG: the tessellations hold the
/// coordinates. `FullRecompute` is always correct; a move that cannot describe
/// its change exactly must use it (and does, by `Proposal::new`'s default).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentDelta {
    /// One centre appended at the highest index; `dims` unchanged.
    CentreAdded,
    /// One centre removed; higher indices shift down by one; `dims` unchanged.
    CentreRemoved {
        /// 0-based index of the removed centre in the OLD tessellation.
        index: usize,
    },
    /// One centre's coordinates changed in place; `dims` unchanged.
    CentreMoved {
        /// 0-based index of the moved centre (the same in old and new).
        index: usize,
    },
    /// No structural guarantee (e.g. `dims` changed): recompute everything.
    FullRecompute,
}

/// A cached batch assignment: per observation, its nearest-centre
/// index and the winning comparison key, the same monotone key
/// [`PairwiseDistance::distance`] returns, in scaled space.
///
/// `best_keys` may be empty: a "cold" cache carrying no key information
/// (the sampler seeds one at chain init). [`CellAssigner::reassign`]
/// implementations must treat a cold cache as unusable and recompute in full.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentCache {
    assignment: Vec<usize>,
    best_keys: Vec<f64>,
}

impl AssignmentCache {
    /// Assemble from parts. `best_keys` must be empty (cold) or hold exactly
    /// one key per assignment entry.
    pub fn new(assignment: Vec<usize>, best_keys: Vec<f64>) -> Self {
        debug_assert!(best_keys.is_empty() || best_keys.len() == assignment.len());
        Self {
            assignment,
            best_keys,
        }
    }

    /// Per-observation nearest-centre indices.
    pub fn assignment(&self) -> &[usize] {
        &self.assignment
    }

    /// Per-observation winning comparison keys (scaled space, magnitude only
    /// meaningful to the assigner that produced them); empty for a cold cache.
    #[allow(dead_code)] // public API for custom CellAssigners (feature-gated export)
    pub fn best_keys(&self) -> &[f64] {
        &self.best_keys
    }
}

/// Batch cell assignment: map every observation to the index of its nearest
/// centre. Implemented once for every [`PairwiseDistance`] by the blanket impl
/// below (the n×nC loop is monomorphised per metric; the sampler makes one dyn
/// call per tessellation).
pub trait CellAssigner: std::fmt::Debug + Send + Sync {
    /// For each row of `x` (scaled, encoded space), the 0-based index of its
    /// nearest centre in `tessellation`: strict `<` comparison with
    /// lowest-index tie-breaking, total-order semantics, and an always-on
    /// finiteness check on every pair.
    fn assign_cells(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<usize>>;

    /// Update a cached assignment across a structural move: `new` is
    /// the proposed tessellation, `delta` the structural change the move
    /// declared, and `prev` the cache for the tessellation the move was
    /// proposed from. Must return exactly what a fresh recompute against
    /// `new` would: same assignment, bit for bit, whatever the delta.
    ///
    /// The default is a full recompute via [`assign_cells`] with a cold key
    /// cache: correct for every implementor with no edit. The blanket
    /// [`PairwiseDistance`] impl overrides it with incremental fast paths.
    ///
    /// [`assign_cells`]: CellAssigner::assign_cells
    fn reassign(
        &self,
        x: &Data,
        new: &Tessellation,
        delta: AssignmentDelta,
        prev: &AssignmentCache,
    ) -> Result<AssignmentCache> {
        let _ = (delta, prev);
        Ok(AssignmentCache::new(self.assign_cells(x, new)?, Vec::new()))
    }

    /// The dense-path entry: the comparison key from every
    /// row of `x` to every cell of `tessellation`, row-major n×b, in
    /// scaled space: the same monotone keys `assign_cells` minimises
    /// (the membership kernel consumes them, so the τ → 0 limit is the hard
    /// assignment by construction). Finiteness is checked per pair, like the
    /// hard path.
    ///
    /// Hard-membership assigners need not provide it: the default errors
    /// with [`AddiVortesError::MembershipUnsupported`]; selecting soft
    /// membership requires an assigner that opts in (every
    /// [`PairwiseDistance`], via the blanket impl, does).
    fn membership_keys(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<f64>> {
        let _ = (x, tessellation);
        Err(AddiVortesError::MembershipUnsupported {
            assigner: format!("{self:?}"),
        })
    }
}

impl<T: PairwiseDistance> CellAssigner for T {
    fn assign_cells(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<usize>> {
        Ok(assign_cells_cached(self, x, tessellation)?.assignment)
    }

    fn reassign(
        &self,
        x: &Data,
        new: &Tessellation,
        delta: AssignmentDelta,
        prev: &AssignmentCache,
    ) -> Result<AssignmentCache> {
        // A cold cache (no keys, or wrong length) supports no incremental path.
        let n = x.n_rows();
        if prev.assignment.len() != n || prev.best_keys.len() != n {
            return assign_cells_cached(self, x, new);
        }
        match delta {
            AssignmentDelta::FullRecompute => assign_cells_cached(self, x, new),
            AssignmentDelta::CentreAdded => reassign_added(self, x, new, prev),
            AssignmentDelta::CentreMoved { index } => reassign_moved(self, x, new, prev, index),
            AssignmentDelta::CentreRemoved { index } => reassign_removed(self, x, new, prev, index),
        }
    }

    fn membership_keys(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<f64>> {
        // The same per-pair key routines as the hard scans (`euclidean_key` /
        // `general_key`), so the membership kernel sees exactly the keys the
        // hard path minimises.
        let dims = tessellation.dims();
        let centres = tessellation.centres_view();
        let n_cells = tessellation.n_cells();
        let n = x.n_rows();
        let mut keys = Vec::with_capacity(n * n_cells);
        if self.all_euclidean() {
            let mut active = vec![0.0_f64; dims.len()];
            for observation in 0..n {
                gather_active(&mut active, x.row(observation), dims);
                for cell in 0..n_cells {
                    let key = euclidean_key(&active, centres.row(cell));
                    if !key.is_finite() {
                        return Err(AddiVortesError::NonFiniteDistance {
                            observation,
                            centre: cell,
                        });
                    }
                    keys.push(key);
                }
            }
        } else {
            let mut synthesised = vec![0.0_f64; x.n_cols()];
            for observation in 0..n {
                let row = x.row(observation);
                synthesised.copy_from_slice(row);
                for cell in 0..n_cells {
                    let key = general_key(self, row, &mut synthesised, dims, centres.row(cell));
                    if !key.is_finite() {
                        return Err(AddiVortesError::NonFiniteDistance {
                            observation,
                            centre: cell,
                        });
                    }
                    keys.push(key);
                }
            }
        }
        Ok(keys)
    }
}

// ---------------------------------------------------------------------------
// The blanket kernel (full + incremental paths)
//
// Every path routes each (observation, centre) pair through one key routine
// per branch (`euclidean_key` on the fast path, `general_key` on the general
// path), so cached keys are bit-identical however they were produced, and the
// incremental results match a full recompute bit for bit (the
// property-test invariant below). Incremental correctness rests on one fact: with
// `dims` unchanged, an untouched centre's key against any observation is
// unchanged, so only pairs involving the touched centre need new work.
// ---------------------------------------------------------------------------

/// Gather the observation's active coordinates (fast Euclidean path).
#[inline]
fn gather_active(active: &mut [f64], row: &[f64], dims: &[usize]) {
    for (slot, &dim) in active.iter_mut().zip(dims) {
        *slot = row[dim];
    }
}

/// THE per-pair key on the fast Euclidean path: Σ (active − centre)² in
/// ascending dim order. Sums of squares are never `-0.0`, so IEEE `<`/`==` on
/// finite keys agree with `total_cmp` here.
#[inline]
fn euclidean_key(active: &[f64], centre: &[f64]) -> f64 {
    let mut key = 0.0_f64;
    for (a, c) in active.iter().zip(centre) {
        let diff = a - c;
        key += diff * diff;
    }
    key
}

/// THE per-pair key on the general path: write the centre's coordinates into
/// the synthesised row (which must already equal the observation at every
/// inactive position) and ask the metric.
#[inline]
fn general_key<T: PairwiseDistance>(
    metric: &T,
    row: &[f64],
    synthesised: &mut [f64],
    dims: &[usize],
    centre: &[f64],
) -> f64 {
    for (di, &dim) in dims.iter().enumerate() {
        synthesised[dim] = centre[di];
    }
    metric.distance(row, synthesised, dims)
}

/// One observation's full scan on the fast Euclidean path: strict `<` with
/// lowest-index tie-breaking and the always-on finiteness check.
#[inline]
fn scan_euclidean(
    active: &[f64],
    centres: &Centres<'_>,
    n_cells: usize,
    observation: usize,
) -> Result<(usize, f64)> {
    let mut best = f64::INFINITY;
    let mut best_cell = 0usize;
    for cell in 0..n_cells {
        let key = euclidean_key(active, centres.row(cell));
        if !key.is_finite() {
            return Err(AddiVortesError::NonFiniteDistance {
                observation,
                centre: cell,
            });
        }
        if key < best {
            best = key;
            best_cell = cell;
        }
    }
    Ok((best_cell, best))
}

/// One observation's full scan on the general path: strict less under total
/// order (the lowest-index centre wins ties) and the always-on finiteness
/// check.
#[inline]
fn scan_general<T: PairwiseDistance>(
    metric: &T,
    row: &[f64],
    synthesised: &mut [f64],
    dims: &[usize],
    centres: &Centres<'_>,
    n_cells: usize,
    observation: usize,
) -> Result<(usize, f64)> {
    let mut best = f64::INFINITY;
    let mut best_cell = 0usize;
    for cell in 0..n_cells {
        let key = general_key(metric, row, synthesised, dims, centres.row(cell));
        if !key.is_finite() {
            return Err(AddiVortesError::NonFiniteDistance {
                observation,
                centre: cell,
            });
        }
        if key.total_cmp(&best) == std::cmp::Ordering::Less {
            best = key;
            best_cell = cell;
        }
    }
    Ok((best_cell, best))
}

/// The full n×nC recompute, recording the winning keys: the single kernel
/// behind `assign_cells` and every `reassign` full-recompute fallback.
fn assign_cells_cached<T: PairwiseDistance>(
    metric: &T,
    x: &Data,
    tessellation: &Tessellation,
) -> Result<AssignmentCache> {
    let dims = tessellation.dims();
    let centres = tessellation.centres_view();
    let n_cells = tessellation.n_cells();
    debug_assert!(
        dims.iter().all(|&d| d < x.n_cols()),
        "tessellation dims must index encoded columns"
    );

    let n = x.n_rows();
    let mut assignment = Vec::with_capacity(n);
    let mut best_keys = Vec::with_capacity(n);
    if metric.all_euclidean() {
        // Fast path: no synthesised row (every key is a
        // per-dimension sum, so no metric needs off-dimension context),
        // gathering the active coordinates once per observation instead of
        // rewriting a full-width buffer per cell.
        let mut active = vec![0.0_f64; dims.len()];
        for observation in 0..n {
            gather_active(&mut active, x.row(observation), dims);
            let (cell, key) = scan_euclidean(&active, &centres, n_cells, observation)?;
            assignment.push(cell);
            best_keys.push(key);
        }
    } else {
        // The synthesised centre row: equal to the observation everywhere
        // except at the active dims. Only the active positions change between
        // cells, so one buffer per observation is rewritten per cell.
        let mut synthesised = vec![0.0_f64; x.n_cols()];
        for observation in 0..n {
            let row = x.row(observation);
            synthesised.copy_from_slice(row);
            let (cell, key) = scan_general(
                metric,
                row,
                &mut synthesised,
                dims,
                &centres,
                n_cells,
                observation,
            )?;
            assignment.push(cell);
            best_keys.push(key);
        }
    }
    Ok(AssignmentCache {
        assignment,
        best_keys,
    })
}

/// `CentreAdded`: the appended (highest-index) centre wins observation `i` iff
/// its key beats the cached winner strictly; on an exact tie the incumbent
/// (lower index) keeps, exactly as the full scan's first-minimum rule decides.
fn reassign_added<T: PairwiseDistance>(
    metric: &T,
    x: &Data,
    new: &Tessellation,
    prev: &AssignmentCache,
) -> Result<AssignmentCache> {
    let dims = new.dims();
    let centres = new.centres_view();
    let added = new.n_cells() - 1;
    let mut assignment = prev.assignment.clone();
    let mut best_keys = prev.best_keys.clone();
    if metric.all_euclidean() {
        let mut active = vec![0.0_f64; dims.len()];
        let centre = centres.row(added);
        for observation in 0..x.n_rows() {
            gather_active(&mut active, x.row(observation), dims);
            let key = euclidean_key(&active, centre);
            if !key.is_finite() {
                return Err(AddiVortesError::NonFiniteDistance {
                    observation,
                    centre: added,
                });
            }
            if key < best_keys[observation] {
                best_keys[observation] = key;
                assignment[observation] = added;
            }
        }
    } else {
        let mut synthesised = vec![0.0_f64; x.n_cols()];
        for observation in 0..x.n_rows() {
            let row = x.row(observation);
            synthesised.copy_from_slice(row);
            let key = general_key(metric, row, &mut synthesised, dims, centres.row(added));
            if !key.is_finite() {
                return Err(AddiVortesError::NonFiniteDistance {
                    observation,
                    centre: added,
                });
            }
            if key.total_cmp(&best_keys[observation]) == std::cmp::Ordering::Less {
                best_keys[observation] = key;
                assignment[observation] = added;
            }
        }
    }
    Ok(AssignmentCache {
        assignment,
        best_keys,
    })
}

/// `CentreMoved { index }`: observations assigned to the moved centre rescan
/// in full (their cached key is stale); every other observation compares the
/// moved centre's new key against its cached winner: strictly less wins, and
/// an exact tie wins only from a lower index (the full scan's first-minimum
/// rule, decidable from the cache alone).
fn reassign_moved<T: PairwiseDistance>(
    metric: &T,
    x: &Data,
    new: &Tessellation,
    prev: &AssignmentCache,
    moved: usize,
) -> Result<AssignmentCache> {
    let dims = new.dims();
    let centres = new.centres_view();
    let n_cells = new.n_cells();
    let mut assignment = prev.assignment.clone();
    let mut best_keys = prev.best_keys.clone();
    if metric.all_euclidean() {
        let mut active = vec![0.0_f64; dims.len()];
        for observation in 0..x.n_rows() {
            gather_active(&mut active, x.row(observation), dims);
            let incumbent = prev.assignment[observation];
            if incumbent == moved {
                let (cell, key) = scan_euclidean(&active, &centres, n_cells, observation)?;
                assignment[observation] = cell;
                best_keys[observation] = key;
            } else {
                let key = euclidean_key(&active, centres.row(moved));
                if !key.is_finite() {
                    return Err(AddiVortesError::NonFiniteDistance {
                        observation,
                        centre: moved,
                    });
                }
                // IEEE `==` is exact equality here (finite, never -0.0 keys).
                if key < best_keys[observation]
                    || (key == best_keys[observation] && moved < incumbent)
                {
                    best_keys[observation] = key;
                    assignment[observation] = moved;
                }
            }
        }
    } else {
        let mut synthesised = vec![0.0_f64; x.n_cols()];
        for observation in 0..x.n_rows() {
            let row = x.row(observation);
            synthesised.copy_from_slice(row);
            let incumbent = prev.assignment[observation];
            if incumbent == moved {
                let (cell, key) = scan_general(
                    metric,
                    row,
                    &mut synthesised,
                    dims,
                    &centres,
                    n_cells,
                    observation,
                )?;
                assignment[observation] = cell;
                best_keys[observation] = key;
            } else {
                let key = general_key(metric, row, &mut synthesised, dims, centres.row(moved));
                if !key.is_finite() {
                    return Err(AddiVortesError::NonFiniteDistance {
                        observation,
                        centre: moved,
                    });
                }
                match key.total_cmp(&best_keys[observation]) {
                    std::cmp::Ordering::Less => {
                        best_keys[observation] = key;
                        assignment[observation] = moved;
                    }
                    std::cmp::Ordering::Equal if moved < incumbent => {
                        best_keys[observation] = key;
                        assignment[observation] = moved;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(AssignmentCache {
        assignment,
        best_keys,
    })
}

/// `CentreRemoved { index }`: observations assigned to the removed centre
/// rescan over the remaining centres; every other observation keeps its winner
/// and key, with indices above the removed one shifted down. Removal
/// introduces no new (observation, centre) pair, so the keep path cannot
/// surface a new non-finite key.
fn reassign_removed<T: PairwiseDistance>(
    metric: &T,
    x: &Data,
    new: &Tessellation,
    prev: &AssignmentCache,
    removed: usize,
) -> Result<AssignmentCache> {
    let dims = new.dims();
    let centres = new.centres_view();
    let n_cells = new.n_cells();
    let n = x.n_rows();
    let mut assignment = Vec::with_capacity(n);
    let mut best_keys = Vec::with_capacity(n);
    if metric.all_euclidean() {
        let mut active = vec![0.0_f64; dims.len()];
        for observation in 0..n {
            let incumbent = prev.assignment[observation];
            if incumbent == removed {
                gather_active(&mut active, x.row(observation), dims);
                let (cell, key) = scan_euclidean(&active, &centres, n_cells, observation)?;
                assignment.push(cell);
                best_keys.push(key);
            } else {
                assignment.push(if incumbent > removed {
                    incumbent - 1
                } else {
                    incumbent
                });
                best_keys.push(prev.best_keys[observation]);
            }
        }
    } else {
        let mut synthesised = vec![0.0_f64; x.n_cols()];
        for observation in 0..n {
            let incumbent = prev.assignment[observation];
            if incumbent == removed {
                let row = x.row(observation);
                synthesised.copy_from_slice(row);
                let (cell, key) = scan_general(
                    metric,
                    row,
                    &mut synthesised,
                    dims,
                    &centres,
                    n_cells,
                    observation,
                )?;
                assignment.push(cell);
                best_keys.push(key);
            } else {
                assignment.push(if incumbent > removed {
                    incumbent - 1
                } else {
                    incumbent
                });
                best_keys.push(prev.best_keys[observation]);
            }
        }
    }
    Ok(AssignmentCache {
        assignment,
        best_keys,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::data::Metric;

    // ---- assigner semantics ----

    fn euclidean_metrics(p: usize) -> ColumnMetrics {
        ColumnMetrics::new(vec![Metric::Euclidean; p])
    }

    fn one_dim_fixture() -> (Data, Tessellation) {
        let x = Data::from_rows(&[[-0.4], [0.4], [0.05]]).unwrap();
        let t = Tessellation::new(vec![-0.5, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        (x, t)
    }

    #[test]
    fn assigns_nearest_centre() {
        let (x, t) = one_dim_fixture();
        let cm = euclidean_metrics(1);
        assert_eq!(cm.assign_cells(&x, &t).unwrap(), vec![0, 1, 1]);
    }

    #[test]
    fn ties_break_to_lowest_centre_index() {
        // Two identical centres: strict `<` keeps the first.
        let x = Data::from_rows(&[[0.3]]).unwrap();
        let t = Tessellation::new(vec![0.1, 0.1], vec![0], vec![0.0, 0.0]).unwrap();
        let cm = euclidean_metrics(1);
        assert_eq!(cm.assign_cells(&x, &t).unwrap(), vec![0]);
        // Exactly equidistant distinct centres: lower index wins.
        let t = Tessellation::new(vec![0.2, 0.4], vec![0], vec![0.0, 0.0]).unwrap();
        assert_eq!(cm.assign_cells(&x, &t).unwrap(), vec![0]);
        // Repeat-call determinism.
        assert_eq!(
            cm.assign_cells(&x, &t).unwrap(),
            cm.assign_cells(&x, &t).unwrap()
        );
    }

    #[test]
    fn non_finite_distance_is_reported_with_indices() {
        // A NaN centre coordinate poisons the pair (0, 1).
        let x = Data::from_rows(&[[0.0]]).unwrap();
        let t = Tessellation::new(vec![0.1, f64::NAN], vec![0], vec![0.0, 0.0]).unwrap();
        let cm = euclidean_metrics(1);
        let err = cm.assign_cells(&x, &t).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::NonFiniteDistance {
                observation: 0,
                centre: 1
            }
        );
    }

    #[test]
    fn synthesised_centre_row_exposes_full_context() {
        // A metric that reads an INACTIVE column must see the observation's own
        // value there (the synthesis contract).
        #[derive(Debug)]
        struct ReadsInactive;
        impl PairwiseDistance for ReadsInactive {
            fn distance(&self, x_row: &[f64], centre_row: &[f64], _active: &[usize]) -> f64 {
                assert_eq!(
                    x_row[1], centre_row[1],
                    "inactive column must be synthesised equal"
                );
                (x_row[0] - centre_row[0]).abs()
            }
        }
        let x = Data::from_rows(&[[0.0, 7.0]]).unwrap();
        let t = Tessellation::new(vec![0.5], vec![0], vec![0.0]).unwrap();
        assert_eq!(ReadsInactive.assign_cells(&x, &t).unwrap(), vec![0]);
    }

    // ---- the ~10-line extension path ----

    // A user-authored L1 metric (the shelf now ships `Manhattan`; this stays
    // as the acceptance proof that the extension path needs no crate edits).
    #[derive(Debug)]
    struct CustomL1;

    impl PairwiseDistance for CustomL1 {
        fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
            active_dims
                .iter()
                .map(|&d| (x_row[d] - centre_row[d]).abs())
                .sum()
        }
    }

    #[test]
    fn custom_metric_extension_compiles_and_assigns() {
        let (x, t) = one_dim_fixture();
        assert_eq!(CustomL1.assign_cells(&x, &t).unwrap(), vec![0, 1, 1]);
        // …and works through dynamic dispatch, as the sampler will use it.
        let dyn_assigner: &dyn CellAssigner = &CustomL1;
        assert_eq!(dyn_assigner.assign_cells(&x, &t).unwrap(), vec![0, 1, 1]);
    }

    // ---- incremental reassign (delta-driven) ----

    use rand_chacha::ChaCha8Rng;
    use rand_core::{Rng, SeedableRng};

    fn uniform(rng: &mut ChaCha8Rng) -> f64 {
        (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Bit-level cache equality: same assignment AND bit-identical keys.
    fn assert_cache_bits_eq(incremental: &AssignmentCache, full: &AssignmentCache) {
        assert_eq!(incremental.assignment, full.assignment);
        let a: Vec<u64> = incremental.best_keys.iter().map(|k| k.to_bits()).collect();
        let b: Vec<u64> = full.best_keys.iter().map(|k| k.to_bits()).collect();
        assert_eq!(a, b);
    }

    /// For one (metric, data, tessellation) triple: every delta variant's
    /// incremental result is bit-for-bit the full recompute of `new`: the
    /// primary incremental-reassign invariant.
    fn check_all_deltas<T: PairwiseDistance>(
        metric: &T,
        x: &Data,
        old: &Tessellation,
        rng: &mut ChaCha8Rng,
    ) {
        let prev = assign_cells_cached(metric, x, old).unwrap();
        let d = old.dims().len();
        let b = old.n_cells();

        // CentreAdded: one random centre appended at the highest index.
        let mut centres = old.centres().to_vec();
        for _ in 0..d {
            centres.push(2.0 * uniform(rng) - 1.0);
        }
        let mut mus = old.mus().to_vec();
        mus.push(0.0);
        let new = Tessellation::new(centres, old.dims().to_vec(), mus).unwrap();
        let incremental = metric
            .reassign(x, &new, AssignmentDelta::CentreAdded, &prev)
            .unwrap();
        assert_cache_bits_eq(&incremental, &assign_cells_cached(metric, x, &new).unwrap());

        // CentreMoved: every index, coordinates resampled in place.
        for moved in 0..b {
            let mut centres = old.centres().to_vec();
            for di in 0..d {
                centres[moved * d + di] = 2.0 * uniform(rng) - 1.0;
            }
            let new = Tessellation::new(centres, old.dims().to_vec(), old.mus().to_vec()).unwrap();
            let incremental = metric
                .reassign(
                    x,
                    &new,
                    AssignmentDelta::CentreMoved { index: moved },
                    &prev,
                )
                .unwrap();
            assert_cache_bits_eq(&incremental, &assign_cells_cached(metric, x, &new).unwrap());
        }

        // CentreRemoved: every index (needs ≥ 2 cells to stay valid).
        if b >= 2 {
            for removed in 0..b {
                let mut centres = old.centres().to_vec();
                centres.drain(removed * d..(removed + 1) * d);
                let mut mus = old.mus().to_vec();
                mus.remove(removed);
                let new = Tessellation::new(centres, old.dims().to_vec(), mus).unwrap();
                let incremental = metric
                    .reassign(
                        x,
                        &new,
                        AssignmentDelta::CentreRemoved { index: removed },
                        &prev,
                    )
                    .unwrap();
                assert_cache_bits_eq(&incremental, &assign_cells_cached(metric, x, &new).unwrap());
            }
        }

        // FullRecompute: no claim, arbitrary new structure (fresh centres).
        let centres: Vec<f64> = (0..b * d).map(|_| 2.0 * uniform(rng) - 1.0).collect();
        let new = Tessellation::new(centres, old.dims().to_vec(), old.mus().to_vec()).unwrap();
        let incremental = metric
            .reassign(x, &new, AssignmentDelta::FullRecompute, &prev)
            .unwrap();
        assert_cache_bits_eq(&incremental, &assign_cells_cached(metric, x, &new).unwrap());
    }

    /// The bit-identity property test, over randomised data and
    /// tessellations, on both blanket paths (fast Euclidean + general) and a
    /// custom general metric. Duplicated observations force exact ties.
    #[test]
    fn reassign_matches_full_recompute_bit_for_bit() {
        let p = 3;
        for seed in 0..5u8 {
            let mut rng = ChaCha8Rng::from_seed([seed; 32]);
            // 12 random rows + 4 duplicates of the first rows (tie pressure).
            let mut rows: Vec<[f64; 3]> = (0..12)
                .map(|_| std::array::from_fn(|_| 2.0 * uniform(&mut rng) - 1.0))
                .collect();
            for i in 0..4 {
                rows.push(rows[i]);
            }
            let x = Data::from_rows(&rows).unwrap();

            for dims in [vec![0], vec![1, 2], vec![2, 0]] {
                for b in [1usize, 2, 5] {
                    let d = dims.len();
                    let centres: Vec<f64> =
                        (0..b * d).map(|_| 2.0 * uniform(&mut rng) - 1.0).collect();
                    let old = Tessellation::new(centres, dims.clone(), vec![0.0; b]).unwrap();

                    // Fast Euclidean path.
                    check_all_deltas(&euclidean_metrics(p), &x, &old, &mut rng);
                    // General path, built-in metric with a spherical group.
                    let mixed = ColumnMetrics::new(vec![
                        Metric::Euclidean,
                        Metric::Spherical,
                        Metric::Spherical,
                    ]);
                    check_all_deltas(&mixed, &x, &old, &mut rng);
                    // General path, custom extension metric.
                    check_all_deltas(&CustomL1, &x, &old, &mut rng);
                }
            }
        }
    }

    #[test]
    fn reassign_added_tie_keeps_incumbent() {
        // 0.25 is exactly equidistant from 0.0 and 0.5 (both diffs exactly
        // ±0.25): the appended centre ties and must NOT displace the incumbent.
        let x = Data::from_rows(&[[0.25]]).unwrap();
        let cm = euclidean_metrics(1);
        let old = Tessellation::new(vec![0.0], vec![0], vec![0.0]).unwrap();
        let prev = assign_cells_cached(&cm, &x, &old).unwrap();
        let new = Tessellation::new(vec![0.0, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let incremental = cm
            .reassign(&x, &new, AssignmentDelta::CentreAdded, &prev)
            .unwrap();
        assert_eq!(incremental.assignment(), &[0]);
        assert_cache_bits_eq(&incremental, &assign_cells_cached(&cm, &x, &new).unwrap());
    }

    #[test]
    fn reassign_moved_tie_wins_only_from_a_lower_index() {
        let cm = euclidean_metrics(1);
        let x = Data::from_rows(&[[0.25]]).unwrap();

        // A LOWER-index centre moves into an exact tie with the incumbent: the
        // full scan's first-minimum rule hands it the win.
        let old = Tessellation::new(vec![0.9, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let prev = assign_cells_cached(&cm, &x, &old).unwrap();
        assert_eq!(prev.assignment(), &[1]);
        let new = Tessellation::new(vec![0.0, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let incremental = cm
            .reassign(&x, &new, AssignmentDelta::CentreMoved { index: 0 }, &prev)
            .unwrap();
        assert_eq!(incremental.assignment(), &[0]);
        assert_cache_bits_eq(&incremental, &assign_cells_cached(&cm, &x, &new).unwrap());

        // A HIGHER-index centre moves into an exact tie: the incumbent keeps.
        let old = Tessellation::new(vec![0.0, 0.9], vec![0], vec![0.0, 0.0]).unwrap();
        let prev = assign_cells_cached(&cm, &x, &old).unwrap();
        assert_eq!(prev.assignment(), &[0]);
        let new = Tessellation::new(vec![0.0, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let incremental = cm
            .reassign(&x, &new, AssignmentDelta::CentreMoved { index: 1 }, &prev)
            .unwrap();
        assert_eq!(incremental.assignment(), &[0]);
        assert_cache_bits_eq(&incremental, &assign_cells_cached(&cm, &x, &new).unwrap());
    }

    #[test]
    fn reassign_general_path_ties_break_to_lowest_index() {
        // Same tie geometry through the general (total_cmp) path.
        let x = Data::from_rows(&[[0.25]]).unwrap();
        let old = Tessellation::new(vec![0.0], vec![0], vec![0.0]).unwrap();
        let prev = assign_cells_cached(&CustomL1, &x, &old).unwrap();
        let new = Tessellation::new(vec![0.0, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let incremental = CustomL1
            .reassign(&x, &new, AssignmentDelta::CentreAdded, &prev)
            .unwrap();
        assert_eq!(incremental.assignment(), &[0]);
        assert_cache_bits_eq(
            &incremental,
            &assign_cells_cached(&CustomL1, &x, &new).unwrap(),
        );

        let old = Tessellation::new(vec![0.9, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let prev = assign_cells_cached(&CustomL1, &x, &old).unwrap();
        let new = Tessellation::new(vec![0.0, 0.5], vec![0], vec![0.0, 0.0]).unwrap();
        let incremental = CustomL1
            .reassign(&x, &new, AssignmentDelta::CentreMoved { index: 0 }, &prev)
            .unwrap();
        assert_eq!(incremental.assignment(), &[0]);
        assert_cache_bits_eq(
            &incremental,
            &assign_cells_cached(&CustomL1, &x, &new).unwrap(),
        );
    }

    #[test]
    fn reassign_cold_cache_falls_back_to_full_recompute() {
        // A cold cache (no keys, the sampler's chain-init state) supports no
        // incremental path: reassign must full-recompute, whatever the delta.
        let (x, t) = one_dim_fixture();
        let cm = euclidean_metrics(1);
        let cold = AssignmentCache::new(vec![0; x.n_rows()], Vec::new());
        let out = cm
            .reassign(&x, &t, AssignmentDelta::CentreAdded, &cold)
            .unwrap();
        assert_cache_bits_eq(&out, &assign_cells_cached(&cm, &x, &t).unwrap());
    }

    #[test]
    fn reassign_incremental_nonfinite_reports_indices() {
        // A NaN coordinate on the APPENDED centre must surface through the
        // incremental path with the same indices a full scan would report.
        let x = Data::from_rows(&[[0.0]]).unwrap();
        let cm = euclidean_metrics(1);
        let old = Tessellation::new(vec![0.1], vec![0], vec![0.0]).unwrap();
        let prev = assign_cells_cached(&cm, &x, &old).unwrap();
        let new = Tessellation::new(vec![0.1, f64::NAN], vec![0], vec![0.0, 0.0]).unwrap();
        let err = cm
            .reassign(&x, &new, AssignmentDelta::CentreAdded, &prev)
            .unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::NonFiniteDistance {
                observation: 0,
                centre: 1
            }
        );
    }

    #[test]
    fn default_reassign_is_a_full_recompute_with_cold_keys() {
        // A CellAssigner that does NOT override reassign (and is not a
        // PairwiseDistance): the provided default reproduces assign_cells and
        // returns a cold cache, correct with zero edits.
        #[derive(Debug)]
        struct BruteForce;
        impl CellAssigner for BruteForce {
            fn assign_cells(&self, x: &Data, tessellation: &Tessellation) -> Result<Vec<usize>> {
                CustomL1.assign_cells(x, tessellation)
            }
        }
        let (x, t) = one_dim_fixture();
        let prev = AssignmentCache::new(vec![0; x.n_rows()], Vec::new());
        let out = BruteForce
            .reassign(&x, &t, AssignmentDelta::CentreMoved { index: 0 }, &prev)
            .unwrap();
        assert_eq!(out.assignment(), CustomL1.assign_cells(&x, &t).unwrap());
        assert!(out.best_keys().is_empty());
    }
}
