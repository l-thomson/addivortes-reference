//! The reusable per-tessellation backfit block: the unit
//! every AddiVortes ensemble shares:
//!
//! ```text
//! propose (move set) → reassign (assigner, incremental where declared)
//! → empty-cell guard → integrated-likelihood ratio (cell model)
//! → structure + count-prior + selection ratios → MH accept/reject
//! → conjugate payload redraw → composition update
//! ```
//!
//! Extracted once from `Sampler::backfit_one` and parameterised by the cell
//! kernel and the [`Composition`]; a model is a set of ensemble instances
//! of this block (the paper/Binary models are one instance; H-AddiVortes is
//! two, sharing moves, coordinate laws, assigner and count priors by
//! construction). The extraction is a pure refactor: the additive/diagonal
//! default samples the pre-extraction chain bit for bit (the golden chain is
//! the proof).

use crate::engine::data::Data;
use crate::engine::error::Result;
use crate::engine::mathsfn;
use crate::engine::tessellation::Tessellation;
use crate::extensions::distance::{AssignmentCache, CellAssigner};
use crate::extensions::erasure::{BasisRows, ErasedCellKernel, StatsBox};
use crate::extensions::moves::{ModelCtx, MoveSet, uniform_f64};

/// How an ensemble's per-tessellation contributions compose into the running
/// fit: additive on ℝ for mean ensembles; multiplicative
/// on ℝ₊ for variance ensembles (H-AddiVortes §3.2), whose partial "residual"
/// is a ratio, not a subtraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Composition {
    /// `F_i = Σ_j g_j(i)`: partial residual `R = y − F + g_j`, fit update
    /// `F ← F − g_old + g_new`.
    Additive,
    /// `S_i = ∏_l h_l(i)` on ℝ₊ (cell values strictly positive): the value
    /// the cell model sees is `ẽ²_i = y_i · h_l(i) / S_i = y_i / S₋ₗ(i)`
    /// (H paper Eq. 8, with `y` the squared mean-residuals), and the fit
    /// update is `S ← (S / h_old) · h_new`.
    Multiplicative,
}

/// Observation `i`'s contribution from the cell it lands in: `μ_k` for the
/// scalar families (q = 1, the golden path: a plain index, no arithmetic), and
/// the basis inner product `z(xᵢ) · β_k` for a basis payload.
pub(crate) fn cell_contribution(
    payload: &[f64],
    q: usize,
    cell: usize,
    basis: Option<&BasisRows>,
    i: usize,
) -> f64 {
    match basis {
        None => payload[cell],
        Some(basis) => {
            debug_assert_eq!(basis.q, q);
            payload[cell * q..(cell + 1) * q]
                .iter()
                .zip(basis.row(i))
                .map(|(beta, z)| beta * z)
                .sum()
        }
    }
}

impl Composition {
    /// The partial "residual" the cell model sees for tessellation j: the
    /// working response with this tessellation's contribution held out
    /// (ascending index). For the multiplicative composition `rest` is also
    /// filled with the held-out product `S₋ₗ(i) = S_i / h_l(i)`, the factor
    /// `update_fit` re-multiplies (kept separate so a zero working value
    /// cannot corrupt the fit through 0/0); additive leaves it empty.
    #[allow(clippy::too_many_arguments)] // the composition's full working state
    fn partial_residuals(
        self,
        y: &[f64],
        fit: &[f64],
        mus: &[f64],
        q: usize,
        basis: Option<&BasisRows>,
        assignment: &[usize],
        residuals: &mut [f64],
        rest: &mut Vec<f64>,
    ) {
        match self {
            Composition::Additive => {
                for i in 0..y.len() {
                    residuals[i] =
                        y[i] - fit[i] + cell_contribution(mus, q, assignment[i], basis, i);
                }
            }
            Composition::Multiplicative => {
                rest.resize(y.len(), 0.0);
                for i in 0..y.len() {
                    let held_out = fit[i] / mus[assignment[i]];
                    debug_assert!(held_out > 0.0 && held_out.is_finite());
                    rest[i] = held_out;
                    residuals[i] = y[i] / held_out;
                }
            }
        }
    }

    /// Fold tessellation j's redrawn contribution back into the running fit
    /// (ascending index). Additive: the residual vector already holds
    /// `y − F + g_old`, so `F_new = y − residuals + g_new`. Multiplicative:
    /// the held-out product was captured in `rest`, so `S_new = rest · h_new`.
    #[allow(clippy::too_many_arguments)] // the composition's full working state
    fn update_fit(
        self,
        y: &[f64],
        residuals: &[f64],
        rest: &[f64],
        mus: &[f64],
        q: usize,
        basis: Option<&BasisRows>,
        assignment: &[usize],
        fit: &mut [f64],
    ) {
        match self {
            Composition::Additive => {
                for i in 0..y.len() {
                    fit[i] =
                        y[i] - residuals[i] + cell_contribution(mus, q, assignment[i], basis, i);
                }
            }
            Composition::Multiplicative => {
                // The variance ensemble is scalar by construction (q = 1): a
                // basis payload is mean-side machinery.
                debug_assert!(basis.is_none() && q == 1);
                for i in 0..y.len() {
                    fit[i] = rest[i] * mus[assignment[i]];
                }
            }
        }
    }
}

/// The linear-algebra path of the mean-side conjugate calculation:
/// the golden-pinned per-cell diagonal path (hard
/// membership, scalar cells) or the dense path (soft membership: joint
/// within-tessellation draws over the full b×b system). Selected once at
/// assembly; the diagonal hot path is untouched by the dense machinery (the
/// dispatch is one branch per tessellation-update, at the same batch
/// granularity as the existing dyn calls).
#[derive(Debug)]
pub(crate) enum Path {
    /// Per-cell scalars through the [`ErasedCellKernel`]; the default.
    Diagonal,
    /// Soft membership: dense per-tessellation statistics and
    /// the joint payload draw, with membership weights from the membership
    /// kernel over the distance keys.
    Dense(DenseState),
}

/// The dense path's engine state: the membership kernel and, per tessellation,
/// the cached (row-normalised) membership matrix Φ (n×b row-major, scaled
/// space). Memberships are fully recomputed for every proposal: soft
/// membership breaks the incremental `AssignmentDelta` contract (a moved
/// centre changes every row's weights), so the dense path bypasses that
/// cache entirely; the reassign-consistency test pins the cached matrix
/// against a fresh recompute bit for bit.
#[derive(Debug)]
pub(crate) struct DenseState {
    pub(crate) kernel: std::sync::Arc<dyn crate::extensions::membership::MembershipKernel>,
    pub(crate) memberships: Vec<Vec<f64>>,
}

/// One ensemble instance of the backfit block: the tessellations, their
/// cached assignments, the cell kernel that prices and redraws their payloads,
/// and the composition that folds them into the running fit. The conductor
/// (`Sampler::step`) owns everything else (RNG, working response, weights,
/// move set, assigner) and passes it in per call (dataflow rule: calls go
/// down, data goes up).
#[derive(Debug)]
pub(crate) struct EnsembleUnit {
    /// The m tessellations (scaled space).
    pub(crate) tessellations: Vec<Tessellation>,
    /// Cached cell assignments (+ winning keys) per tessellation
    /// (structure- and X-dependent only; meaningful on the diagonal path;
    /// the dense path's membership matrices live in [`Path::Dense`]).
    pub(crate) assignments: Vec<AssignmentCache>,
    /// The cell kernel (deep seam; default: Gaussian conjugate).
    pub(crate) kernel: Box<dyn ErasedCellKernel>,
    composition: Composition,
    pub(crate) path: Path,
    /// The per-fit basis rows z(xᵢ), or `None` for a scalar payload.
    /// Built once at assembly: the basis is a fixed set of columns, so it does
    /// not move with the tessellation and never needs recomputing.
    pub(crate) basis: Option<BasisRows>,
}

impl EnsembleUnit {
    /// Assemble an ensemble instance from already-initialised state (the
    /// diagonal path, the golden-pinned default).
    pub(crate) fn new(
        tessellations: Vec<Tessellation>,
        assignments: Vec<AssignmentCache>,
        kernel: Box<dyn ErasedCellKernel>,
        composition: Composition,
    ) -> Self {
        Self {
            tessellations,
            assignments,
            kernel,
            composition,
            path: Path::Diagonal,
            basis: None,
        }
    }

    /// Attach the basis rows (builder-style; the payload model's
    /// `payload_width` is validated against `basis.q` by the caller).
    pub(crate) fn with_basis(mut self, basis: BasisRows) -> Self {
        self.basis = Some(basis);
        self
    }

    /// Assemble a dense-path (soft-membership) instance. `memberships` are
    /// the initial per-tessellation matrices (row-normalised n×b).
    pub(crate) fn new_dense(
        tessellations: Vec<Tessellation>,
        assignments: Vec<AssignmentCache>,
        kernel: Box<dyn ErasedCellKernel>,
        softness: std::sync::Arc<dyn crate::extensions::membership::MembershipKernel>,
        memberships: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            tessellations,
            assignments,
            kernel,
            composition: Composition::Additive,
            path: Path::Dense(DenseState {
                kernel: softness,
                memberships,
            }),
            // Soft membership and a basis payload do not compose: the dense
            // path owns its own joint draw (rejected at config validation).
            basis: None,
        }
    }

    /// The j-th backfitting step: structural MH move then conjugate payload
    /// redraw, updating the running `fit` in place. RNG-consumption order is
    /// pinned (see the RNG-order doc in `src/engine/sampler.rs`): move-selection
    /// uniform, the
    /// move's own proposal draws, then (only if the empty-cell guard passes)
    /// the acceptance uniform, and finally the cell-value draws. Dispatches
    /// once per call between the golden-pinned diagonal path and the dense
    /// (soft-membership) path.
    #[allow(clippy::too_many_arguments)] // the conductor's per-sweep state, passed not stored
    pub(crate) fn backfit_one(
        &mut self,
        j: usize,
        x: &Data,
        y: &[f64],
        fit: &mut [f64],
        obs_weights: Option<&[f64]>,
        ctx: &ModelCtx,
        move_set: &MoveSet,
        assigner: &dyn CellAssigner,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<()> {
        if matches!(self.path, Path::Dense(_)) {
            return self.backfit_one_dense(j, x, y, fit, obs_weights, ctx, move_set, assigner, rng);
        }
        self.backfit_one_diagonal(j, x, y, fit, obs_weights, ctx, move_set, assigner, rng)
    }

    /// The diagonal-path body: byte-for-byte the pre-dense `backfit_one`
    /// (the golden chain is the proof).
    #[allow(clippy::too_many_arguments)] // the conductor's per-sweep state, passed not stored
    fn backfit_one_diagonal(
        &mut self,
        j: usize,
        x: &Data,
        y: &[f64],
        fit: &mut [f64],
        obs_weights: Option<&[f64]>,
        ctx: &ModelCtx,
        move_set: &MoveSet,
        assigner: &dyn CellAssigner,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<()> {
        let n = x.n_rows();
        let sigma_sq = ctx.sigma_sq;

        // Partial residuals for this tessellation (composition-specific);
        // `rest` holds the multiplicative composition's held-out product
        // (empty on the additive path, no allocation).
        let q = self.kernel.payload_width();
        let basis = self.basis.as_ref();
        let mut residuals = vec![0.0_f64; n];
        let mut rest = Vec::new();
        self.composition.partial_residuals(
            y,
            fit,
            &self.tessellations[j].mus,
            q,
            basis,
            self.assignments[j].assignment(),
            &mut residuals,
            &mut rest,
        );

        // The move block already accumulates per-cell stats over the final
        // assignment (`proposed_stats` on accept, `current_stats` on a guarded
        // reject); carry that StatsBox out so the cell-value redraw below
        // reuses it instead of a third `accumulate` pass. `None` on the two
        // branches that never computed it (no move selected, or empty-cell
        // guard failed).
        let mut redraw_stats: Option<StatsBox> = None;

        // Structural move (selection draw; None = no valid move, skip structure).
        if let Some(move_index) = move_set.select(&self.tessellations[j], ctx, rng) {
            let chosen = move_set.move_at(move_index);
            let proposal = chosen.propose(&self.tessellations[j], ctx, rng);
            let delta = proposal.delta();
            let proposed = proposal.tessellation;
            // Incremental where the move's declared delta allows it, full
            // recompute otherwise; bit-identical either way.
            let proposed_cache = assigner.reassign(x, &proposed, delta, &self.assignments[j])?;

            // Empty-cell guard (sampler-side rejection: every cell must
            // keep observation support): rejected before the acceptance uniform, so no
            // acceptance draw is consumed. Cell statistics route through the
            // deep seam; the built-in Gaussian kernel is bit-identical to the
            // pre-seam code.
            let proposed_stats = self.kernel.accumulate(
                proposed_cache.assignment(),
                &residuals,
                obs_weights,
                proposed.n_cells(),
                basis,
            );
            if self.kernel.all_occupied(&proposed_stats) {
                let current_stats = self.kernel.accumulate(
                    self.assignments[j].assignment(),
                    &residuals,
                    obs_weights,
                    self.tessellations[j].n_cells(),
                    basis,
                );
                let log_lik_ratio = self.kernel.log_marginal(&proposed_stats, sigma_sq)?
                    - self.kernel.log_marginal(&current_stats, sigma_sq)?;
                let log_alpha = log_lik_ratio
                    + chosen.log_structure_ratio(&self.tessellations[j], &proposed, ctx)
                    + move_set.log_selection_ratio(
                        move_index,
                        &self.tessellations[j],
                        &proposed,
                        ctx,
                    );
                // Release-mode invariant: NaN log_α is impossible by
                // construction; a reject-on-NaN branch would be dead code.
                assert!(!log_alpha.is_nan(), "log_alpha must never be NaN");

                let u = uniform_f64(rng);
                if mathsfn::ln(u) < log_alpha {
                    // Accepted: final assignment == proposed, so its stats are
                    // exactly `proposed_stats`.
                    self.tessellations[j] = proposed;
                    self.assignments[j] = proposed_cache;
                    redraw_stats = Some(proposed_stats);
                } else {
                    // Rejected: final assignment stays current, whose stats are
                    // exactly `current_stats`.
                    redraw_stats = Some(current_stats);
                }
            }
        }

        // Conjugate cell-value redraw for the (possibly new) structure,
        // ascending cell index (through the seam); then update the running
        // fit. Reuse the stats accumulated in the move block when available
        // (identical assignment + residuals + ascending order ⇒
        // bit-identical), else accumulate fresh.
        let assignment = self.assignments[j].assignment();
        let stats = match redraw_stats {
            Some(stats) => stats,
            None => self.kernel.accumulate(
                assignment,
                &residuals,
                obs_weights,
                self.tessellations[j].n_cells(),
                basis,
            ),
        };
        let new_mus = self.kernel.draw_cell_values(&stats, sigma_sq, rng)?;
        // Release-mode invariant.
        for mu in &new_mus {
            assert!(mu.is_finite(), "sampled mu must be finite");
        }
        let tessellation = &mut self.tessellations[j];
        debug_assert_eq!(new_mus.len(), tessellation.n_cells() * q);
        tessellation.mus = new_mus;

        self.composition.update_fit(
            y,
            &residuals,
            &rest,
            &tessellation.mus,
            q,
            basis,
            assignment,
            fit,
        );
        // In debug builds only: the fit vector stays finite (O(n) sweep).
        debug_assert!(fit.iter().all(|f| f.is_finite()));
        Ok(())
    }

    /// The dense-path body: the same
    /// propose → guard → score → MH → redraw skeleton with the per-cell
    /// pieces replaced by the joint within-tessellation calculation:
    /// membership matrices from the membership kernel over the distance keys,
    /// the dense marginal `0.5(bᵀA⁻¹b − ln det(σ_μ²A))` with
    /// `A = ΦᵀWΦ/σ² + I/σ_μ²`, and the joint draw μ ~ N(A⁻¹b, A⁻¹). RNG
    /// order per tessellation: selection uniform, proposal draws, acceptance
    /// uniform (guard permitting), then b standard normals (ascending cell
    /// index) for the joint redraw.
    #[allow(clippy::too_many_arguments)] // the conductor's per-sweep state, passed not stored
    fn backfit_one_dense(
        &mut self,
        j: usize,
        x: &Data,
        y: &[f64],
        fit: &mut [f64],
        obs_weights: Option<&[f64]>,
        ctx: &ModelCtx,
        move_set: &MoveSet,
        assigner: &dyn CellAssigner,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<()> {
        let n = x.n_rows();
        let sigma_sq = ctx.sigma_sq;
        let sigma_mu_sq = ctx.sigma_mu_sq;
        let Path::Dense(dense) = &mut self.path else {
            unreachable!("dispatched on Path::Dense");
        };

        // Partial residuals: R_i = y_i − F_i + φ_i·μ (additive composition;
        // the dense path is mean-side machinery).
        let b_current = self.tessellations[j].n_cells();
        let mut residuals = vec![0.0_f64; n];
        for i in 0..n {
            let phi = &dense.memberships[j][i * b_current..(i + 1) * b_current];
            let g: f64 = phi
                .iter()
                .zip(&self.tessellations[j].mus)
                .map(|(p, mu)| p * mu)
                .sum();
            residuals[i] = y[i] - fit[i] + g;
        }

        // Structural move (same selection machinery as the diagonal path).
        if let Some(move_index) = move_set.select(&self.tessellations[j], ctx, rng) {
            let chosen = move_set.move_at(move_index);
            let proposal = chosen.propose(&self.tessellations[j], ctx, rng);
            // Soft membership bypasses the incremental AssignmentDelta cache
            // (a moved centre changes every row's weights): memberships are
            // fully recomputed for every proposal.
            let proposed = proposal.tessellation;
            let proposed_memberships =
                compute_memberships(assigner, dense.kernel.as_ref(), x, &proposed)?;

            // The soft empty-cell guard (documented in `crate::extensions::membership`):
            // every cell must carry strictly positive total membership mass:
            // the τ → 0 limit of the hard guard. Rejected before the
            // acceptance uniform, like the hard path.
            if occupied_under_membership(&proposed_memberships, proposed.n_cells()) {
                let (l_prop, _, quad_prop) = dense_posterior(
                    &proposed_memberships,
                    proposed.n_cells(),
                    &residuals,
                    obs_weights,
                    sigma_sq,
                    sigma_mu_sq,
                );
                let (l_cur, _, quad_cur) = dense_posterior(
                    &dense.memberships[j],
                    b_current,
                    &residuals,
                    obs_weights,
                    sigma_sq,
                    sigma_mu_sq,
                );
                let log_lik_ratio =
                    dense_log_marginal(&l_prop, proposed.n_cells(), quad_prop, sigma_mu_sq)
                        - dense_log_marginal(&l_cur, b_current, quad_cur, sigma_mu_sq);
                let log_alpha = log_lik_ratio
                    + chosen.log_structure_ratio(&self.tessellations[j], &proposed, ctx)
                    + move_set.log_selection_ratio(
                        move_index,
                        &self.tessellations[j],
                        &proposed,
                        ctx,
                    );
                assert!(!log_alpha.is_nan(), "log_alpha must never be NaN");

                let u = uniform_f64(rng);
                if mathsfn::ln(u) < log_alpha {
                    self.tessellations[j] = proposed;
                    dense.memberships[j] = proposed_memberships;
                }
            }
        }

        // Joint payload redraw for the (possibly new) structure:
        // μ ~ N(A⁻¹b, A⁻¹) via the pinned Cholesky: b standard normals in
        // ascending cell index, then the Lᵀ half-solve.
        let b_final = self.tessellations[j].n_cells();
        let (l, mean, _) = dense_posterior(
            &dense.memberships[j],
            b_final,
            &residuals,
            obs_weights,
            sigma_sq,
            sigma_mu_sq,
        );
        let mut z: Vec<f64> = (0..b_final)
            .map(|_| rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng))
            .collect();
        mathsfn::cholesky_solve_transposed(&l, b_final, &mut z);
        let new_mus: Vec<f64> = mean.iter().zip(&z).map(|(m, v)| m + v).collect();
        for mu in &new_mus {
            assert!(mu.is_finite(), "sampled mu must be finite");
        }
        self.tessellations[j].mus = new_mus;

        // Fit update: F_new = y − R + Φ_new·μ_new (dot products:
        // under soft membership, fit updates become dot products).
        let membership = &dense.memberships[j];
        let mus = &self.tessellations[j].mus;
        for i in 0..n {
            let phi = &membership[i * b_final..(i + 1) * b_final];
            let g: f64 = phi.iter().zip(mus).map(|(p, mu)| p * mu).sum();
            fit[i] = y[i] - residuals[i] + g;
        }
        debug_assert!(fit.iter().all(|f| f.is_finite()));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dense-path algebra (engine code, held to the cross-path oracle)
// ---------------------------------------------------------------------------

/// Compute a row-normalised membership matrix Φ (n×b row-major) for one
/// tessellation: distance comparison keys → membership-kernel weights →
/// validation (finite, ≥ 0, row sum > 0; a violation surfaces as the
/// Extension channel) → row normalisation to Σₖ φᵢₖ = 1.
pub(crate) fn compute_memberships(
    assigner: &dyn CellAssigner,
    kernel: &dyn crate::extensions::membership::MembershipKernel,
    x: &Data,
    tessellation: &Tessellation,
) -> Result<Vec<f64>> {
    let keys = assigner.membership_keys(x, tessellation)?;
    let n = x.n_rows();
    let b = tessellation.n_cells();
    debug_assert_eq!(keys.len(), n * b);
    let mut memberships = vec![0.0_f64; n * b];
    for i in 0..n {
        let row_keys = &keys[i * b..(i + 1) * b];
        let row = &mut memberships[i * b..(i + 1) * b];
        kernel.weights(row_keys, row);
        let mut total = 0.0_f64;
        for weight in row.iter() {
            if !(weight.is_finite() && *weight >= 0.0) {
                return Err(crate::engine::error::AddiVortesError::Extension {
                    source: std::sync::Arc::new(InvalidMembershipWeights {
                        detail: format!(
                            "weight {weight} for observation {i} is not finite and non-negative"
                        ),
                    }),
                });
            }
            total += weight;
        }
        if !(total.is_finite() && total > 0.0) {
            return Err(crate::engine::error::AddiVortesError::Extension {
                source: std::sync::Arc::new(InvalidMembershipWeights {
                    detail: format!("observation {i}'s membership row sums to {total}"),
                }),
            });
        }
        for weight in row.iter_mut() {
            *weight /= total;
        }
    }
    Ok(memberships)
}

/// Mid-chain membership corruption from a custom membership kernel:
/// the Extension channel, mirroring the inclusion and scale points.
#[derive(Debug, thiserror::Error)]
#[error("membership kernel returned invalid membership weights: {detail}")]
pub(crate) struct InvalidMembershipWeights {
    pub(crate) detail: String,
}

/// The soft empty-cell guard: every cell carries strictly positive total
/// membership mass (see `crate::extensions::membership` for the semantics decision).
pub(crate) fn occupied_under_membership(memberships: &[f64], b: usize) -> bool {
    let n = memberships.len() / b;
    (0..b).all(|cell| (0..n).any(|i| memberships[i * b + cell] > 0.0))
}

/// The dense conjugate pieces for one tessellation (mean-side family
/// of the universal conjugate triple): given the design Z (n×cols row-major: the membership
/// matrix, or any general design for the oracle's block leg), returns the
/// lower Cholesky factor of `A = ZᵀWZ/σ² + I/σ_μ²`, the posterior mean
/// `A⁻¹b` with `b = ZᵀWr/σ²`, and the quadratic `bᵀA⁻¹b` (all scaled space).
pub(crate) fn dense_posterior(
    z: &[f64],
    cols: usize,
    residuals: &[f64],
    obs_weights: Option<&[f64]>,
    sigma_sq: f64,
    sigma_mu_sq: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = residuals.len();
    debug_assert_eq!(z.len(), n * cols);
    let mut a = vec![0.0_f64; cols * cols];
    let mut b = vec![0.0_f64; cols];
    for i in 0..n {
        let weight = obs_weights.map_or(1.0, |w| w[i]);
        let row = &z[i * cols..(i + 1) * cols];
        for r in 0..cols {
            b[r] += weight * row[r] * residuals[i];
            for c in r..cols {
                a[r * cols + c] += weight * row[r] * row[c];
            }
        }
    }
    // Symmetrise the accumulated upper triangle first, then scale: scaling
    // row by row would re-divide mirrored entries (caught by the block leg
    // of the cross-path oracle; invisible on one-hot inputs, whose
    // off-diagonals are zero).
    for r in 0..cols {
        for c in 0..r {
            a[r * cols + c] = a[c * cols + r];
        }
    }
    for r in 0..cols {
        for c in 0..cols {
            a[r * cols + c] /= sigma_sq;
        }
        a[r * cols + r] += 1.0 / sigma_mu_sq;
        b[r] /= sigma_sq;
    }
    let spd = mathsfn::cholesky(&mut a, cols);
    assert!(spd, "A = ZᵀWZ/σ² + I/σ_μ² is SPD by construction");
    let mut mean = b.clone();
    mathsfn::cholesky_solve(&a, cols, &mut mean);
    let quadratic: f64 = b.iter().zip(&mean).map(|(bi, ui)| bi * ui).sum();
    (a, mean, quadratic)
}

/// The dense integrated marginal's structure-varying part (log scale):
/// `0.5·(bᵀA⁻¹b − ln det(σ_μ² A))` from [`dense_posterior`]'s Cholesky
/// factor and quadratic: algebraically the diagonal path's per-cell terms
/// when Z is one-hot, and the per-cell block terms when Z is block-diagonal
/// (the cross-path oracle pins both to 1e-12).
pub(crate) fn dense_log_marginal(l: &[f64], cols: usize, quadratic: f64, sigma_mu_sq: f64) -> f64 {
    let mut log_det = cols as f64 * mathsfn::ln(sigma_mu_sq);
    for r in 0..cols {
        log_det += 2.0 * mathsfn::ln(l[r * cols + r]);
    }
    0.5 * (quadratic - log_det)
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::extensions::basis::LinearGaussianModel;
    use crate::extensions::cell_model::{CellModel, CellStats};
    use crate::test_support::assert_rel_eq;

    // ---- the cross-path oracle ---------
    //
    // Runs in the fast per-commit suite: dense ≡ diagonal to 1e-12 on one-hot
    // inputs, and block ≡ dense to 1e-12 on hard-membership basis inputs.

    fn oracle_fixture() -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        // 8 observations over 3 cells with non-trivial weights.
        let assignment = vec![0usize, 1, 2, 0, 1, 2, 0, 1];
        let residuals = vec![0.12, -0.05, 0.31, 0.07, -0.22, 0.18, 0.02, -0.11];
        let weights = vec![1.0, 0.5, 2.0, 1.5, 1.0, 0.8, 1.2, 0.6];
        (assignment, residuals, weights)
    }

    /// One-hot membership matrix from a hard assignment.
    fn one_hot(assignment: &[usize], b: usize) -> Vec<f64> {
        let mut z = vec![0.0_f64; assignment.len() * b];
        for (i, &cell) in assignment.iter().enumerate() {
            z[i * b + cell] = 1.0;
        }
        z
    }

    /// Dense ≡ diagonal on one-hot inputs, ≤ 1e-12: the dense marginal over
    /// the one-hot membership matrix equals the diagonal path's per-cell
    /// Gaussian terms, and the joint draw equals the per-cell scalar draws
    /// under the same RNG (A is diagonal, so both consume b normals in
    /// ascending cell order).
    #[test]
    fn cross_path_oracle_dense_equals_diagonal_on_one_hot() {
        let (assignment, residuals, weights) = oracle_fixture();
        let (sigma_sq, sigma_mu_sq) = (0.3, 0.02);
        let b = 3;
        let z = one_hot(&assignment, b);

        // Marginal.
        let (l, _, quadratic) =
            dense_posterior(&z, b, &residuals, Some(&weights), sigma_sq, sigma_mu_sq);
        let dense = dense_log_marginal(&l, b, quadratic, sigma_mu_sq);
        let mut pairs = vec![(0.0_f64, 0.0_f64); b];
        for i in 0..assignment.len() {
            pairs[assignment[i]].0 += weights[i];
            pairs[assignment[i]].1 += weights[i] * residuals[i];
        }
        let diagonal = crate::extensions::cell_model::gaussian_marginal_terms(
            pairs.iter().copied(),
            sigma_sq,
            sigma_mu_sq,
        );
        assert_rel_eq(dense, diagonal, 1e-12);

        // Joint draw vs per-cell draws under the same RNG.
        let (l, mean, _) =
            dense_posterior(&z, b, &residuals, Some(&weights), sigma_sq, sigma_mu_sq);
        let mut rng_a = ChaCha8Rng::from_seed([61; 32]);
        let mut z_draw: Vec<f64> = (0..b)
            .map(|_| rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng_a))
            .collect();
        mathsfn::cholesky_solve_transposed(&l, b, &mut z_draw);
        let dense_draw: Vec<f64> = mean.iter().zip(&z_draw).map(|(m, v)| m + v).collect();

        let model = crate::extensions::cell_model::WeightedGaussianModel::new(sigma_mu_sq).unwrap();
        let mut stats = vec![crate::extensions::cell_model::WeightedGaussianStats::default(); b];
        for i in 0..assignment.len() {
            stats[assignment[i]].record(residuals[i], weights[i]);
        }
        let mut rng_b = ChaCha8Rng::from_seed([61; 32]);
        let diagonal_draw = model
            .draw_cell_values(&stats, sigma_sq, &mut rng_b)
            .unwrap();
        for (a, d) in dense_draw.iter().zip(&diagonal_draw) {
            assert_rel_eq(*a, *d, 1e-12);
        }
    }

    /// Block ≡ dense on hard-membership basis inputs, ≤ 1e-12: the linear
    /// family's per-cell q×q block marginal equals the dense marginal over
    /// the block-embedded (b·q)-column design.
    #[test]
    fn cross_path_oracle_block_equals_dense_on_hard_basis() {
        let (assignment, residuals, weights) = oracle_fixture();
        let (sigma_sq, sigma_beta_sq) = (0.3, 0.05);
        let (b, q) = (3usize, 2usize);
        let basis: Vec<[f64; 2]> = (0..assignment.len())
            .map(|i| [1.0, -0.4 + 0.13 * i as f64])
            .collect();

        // Block side: the linear family per cell.
        let model = LinearGaussianModel::new(sigma_beta_sq, q).unwrap();
        let mut stats = vec![crate::extensions::basis::LinearCellStats::default(); b];
        for i in 0..assignment.len() {
            stats[assignment[i]].record_row(&basis[i], residuals[i], weights[i]);
        }
        let block = model.log_marginal_terms(&stats, sigma_sq).unwrap();

        // Dense side: Z (n×bq) with each row's basis embedded in its cell's
        // column block; the same engine helper the soft path runs.
        let cols = b * q;
        let mut z = vec![0.0_f64; assignment.len() * cols];
        for (i, &cell) in assignment.iter().enumerate() {
            for c in 0..q {
                z[i * cols + cell * q + c] = basis[i][c];
            }
        }
        let (l, _, quadratic) = dense_posterior(
            &z,
            cols,
            &residuals,
            Some(&weights),
            sigma_sq,
            sigma_beta_sq,
        );
        let dense = dense_log_marginal(&l, cols, quadratic, sigma_beta_sq);
        assert_rel_eq(dense, block, 1e-12);
    }

    // ---- membership machinery ---------------------------------------------

    #[test]
    fn memberships_are_row_normalised_and_guarded() {
        let x = Data::from_rows(&[[0.0], [0.45], [1.0]]).unwrap();
        // Two centres near 0 and 1 (scaled space).
        let tessellation = Tessellation {
            centres: vec![0.0, 1.0],
            dims: vec![0],
            mus: vec![0.0, 0.0],
        };
        let assigner = crate::extensions::distance::default_assigner(vec![
            crate::engine::data::Metric::Euclidean,
        ]);
        let kernel = crate::extensions::membership::SoftmaxKernel::new(0.5).unwrap();
        let memberships =
            compute_memberships(assigner.as_ref(), &kernel, &x, &tessellation).unwrap();
        for i in 0..3 {
            let row = &memberships[i * 2..(i + 1) * 2];
            assert_rel_eq(row[0] + row[1], 1.0, 1e-12);
        }
        // The middle observation leans toward centre 0 (d² 0.2025 vs 0.3025).
        assert!(memberships[2] > 0.5 && memberships[2] < 0.7);
        assert!(occupied_under_membership(&memberships, 2));
        // A cell with zero mass everywhere fails the soft guard.
        let all_zero_col = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        assert!(!occupied_under_membership(&all_zero_col, 2));
    }

    /// A membership kernel emitting invalid weights surfaces as the Extension
    /// channel (never a NaN chain).
    #[test]
    fn invalid_membership_weights_surface_extension_error() {
        #[derive(Debug)]
        struct NegativeKernel;
        impl crate::extensions::membership::MembershipKernel for NegativeKernel {
            fn weights(&self, _keys: &[f64], weights: &mut [f64]) {
                weights.fill(-1.0);
            }
        }
        let x = Data::from_rows(&[[0.0], [1.0]]).unwrap();
        let tessellation = Tessellation {
            centres: vec![0.0, 1.0],
            dims: vec![0],
            mus: vec![0.0, 0.0],
        };
        let assigner = crate::extensions::distance::default_assigner(vec![
            crate::engine::data::Metric::Euclidean,
        ]);
        let err =
            compute_memberships(assigner.as_ref(), &NegativeKernel, &x, &tessellation).unwrap_err();
        assert!(matches!(
            err,
            crate::engine::error::AddiVortesError::Extension { .. }
        ));
    }

    /// Membership keys are the same comparison keys the hard path minimises:
    /// the argmin over `membership_keys` equals `assign_cells`, so the τ → 0
    /// limit of the soft weights is the hard assignment by construction.
    #[test]
    fn membership_keys_argmin_equals_hard_assignment() {
        let x = Data::from_rows(&[
            [0.1, 0.4],
            [-0.3, 0.2],
            [0.5, -0.5],
            [0.0, 0.0],
            [-0.2, -0.4],
        ])
        .unwrap();
        let tessellation = Tessellation {
            centres: vec![0.0, 0.0, 0.4, -0.4, -0.3, 0.3],
            dims: vec![0, 1],
            mus: vec![0.0; 3],
        };
        let assigner = crate::extensions::distance::default_assigner(
            vec![crate::engine::data::Metric::Euclidean; 2],
        );
        let keys = assigner.membership_keys(&x, &tessellation).unwrap();
        let hard = assigner.assign_cells(&x, &tessellation).unwrap();
        let b = tessellation.n_cells();
        for (i, &cell) in hard.iter().enumerate() {
            let row = &keys[i * b..(i + 1) * b];
            let argmin = row
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0;
            assert_eq!(argmin, cell, "row {i}");
        }
    }

    #[test]
    fn gaussian_marginal_reexport_used_by_oracle_exists() {
        // Guard against the oracle silently drifting from the shipped terms.
        let terms = crate::extensions::cell_model::gaussian_marginal_terms(
            [(2.0, 0.4)].into_iter(),
            0.5,
            0.02,
        );
        assert!(terms.is_finite());
    }
}
