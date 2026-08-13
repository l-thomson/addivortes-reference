//! **Exploration, not a shelf entry.** A working prototype of the two
//! non-Bayesian ensembles people ask for when they meet AddiVortes: the
//! XGBoost analogue (greedy second-order boosting over tessellations) and the
//! random-forest analogue (bagged tessellations, greedy or purely random),
//! benchmarked head-to-head against the shipped MCMC fit.
//!
//! ```sh
//! cargo run --release --example greedy_prototype
//! ```
//!
//! It exists to answer a design question with numbers rather than opinion, so
//! read `docs/design/greedy-tessellations.md` alongside it. Nothing here
//! touches the sampler, the golden chain, or the public API: the file compiles
//! against the published surface only, exactly like the extension templates.
//!
//! # The finding, up front
//!
//! Boosted trees work because the greedy step is *exhaustive*: sort each
//! feature once, sweep the prefix sums, and you have the exact best
//! axis-aligned split over **both** the feature and the threshold. Voronoi
//! structure has no such enumeration — a centre is a point in a continuous
//! subspace — so the greedy step here has to be a sampled search, and the only
//! real decision is *what to spend that sampling budget on*.
//!
//! Port XGBoost literally and you spend it all on centre positions
//! (`candidates`) while the subspace is drawn once at random and never
//! reconsidered — the "search over features" half silently disappears. That
//! version loses to the shipped MCMC by a wide margin. Spend the same budget
//! the other way round — a tiny candidate pool, many independent subspaces
//! tried per round (`subspace_tries`) — and it matches the MCMC in about a
//! third of the wall clock. Which covariates a tessellation lives in turns out
//! to be worth far more than where its centres sit.
//!
//! What *does* carry over exactly is the gain algebra. Under the usual
//! second-order objective
//!
//! ```text
//! L = Σ_i [ g_i μ_{c(i)} + ½ h_i μ_{c(i)}² ] + γ·b + ½λ Σ_k μ_k²
//! ```
//!
//! a cell's optimal payload is `μ_k = −G_k / (H_k + λ)` and its contribution
//! is `−½ G_k² / (H_k + λ)`, identically to XGBoost. The Voronoi-specific part
//! is which observations move when a centre is added, and there the geometry
//! is unusually kind: adding a centre never changes the distance from a point
//! to any existing centre, so the observations that move are exactly
//!
//! ```text
//! { i : d(x_i, c_new)² < best_key_i }
//! ```
//!
//! and every one of them lands in the new cell. Adding a centre is therefore a
//! *many-parents, one-child* split whose gain is one O(n) pass given the
//! cached nearest-centre distances — the same `best_keys` the shipped
//! `AssignmentCache` already maintains for the MCMC path.

use std::time::Instant;

use addivortes::{AddiVortesConfig, Data, Tessellation, mathsfn};
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Draw helpers (mirrors of the engine's, so the prototype needs no rand API)
// ---------------------------------------------------------------------------

/// One uniform f64 in `[0, 1)`.
fn uniform(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// One uniform index in `0..n` (n ≥ 1).
fn uniform_index(n: usize, rng: &mut ChaCha8Rng) -> usize {
    ((uniform(rng) * n as f64) as usize).min(n - 1)
}

// ---------------------------------------------------------------------------
// Standardisation
// ---------------------------------------------------------------------------

/// Per-column centre/scale. Voronoi cells are defined by a distance, so unlike
/// a tree — which is invariant to any monotone per-column transform — the
/// geometry here depends on the units. The MCMC path handles this in the
/// engine's scaler; the prototype does it up front.
struct Standardiser {
    mean: Vec<f64>,
    sd: Vec<f64>,
}

impl Standardiser {
    fn fit(x: &Data) -> Self {
        let (n, p) = (x.n_rows(), x.n_cols());
        let mut mean = vec![0.0; p];
        let mut sd = vec![0.0; p];
        for i in 0..n {
            for (c, value) in x.row(i).iter().enumerate() {
                mean[c] += value / n as f64;
            }
        }
        for i in 0..n {
            for (c, value) in x.row(i).iter().enumerate() {
                let d = value - mean[c];
                sd[c] += d * d;
            }
        }
        for s in &mut sd {
            *s = (*s / n as f64).sqrt();
            // A constant column has no scale to divide by; leave it at 1 so it
            // simply contributes nothing to any distance.
            if *s <= 0.0 {
                *s = 1.0;
            }
        }
        Self { mean, sd }
    }

    fn apply(&self, x: &Data) -> Data {
        let (n, p) = (x.n_rows(), x.n_cols());
        let mut values = Vec::with_capacity(n * p);
        for i in 0..n {
            for (c, value) in x.row(i).iter().enumerate() {
                values.push((value - self.mean[c]) / self.sd[c]);
            }
        }
        Data::new(values, n, p).expect("standardised copy keeps the input shape")
    }
}

// ---------------------------------------------------------------------------
// The greedy grower: one tessellation against a (g, h) working problem
// ---------------------------------------------------------------------------

/// How the next centre is chosen from the sampled candidate pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selection {
    /// Take the candidate with the largest gain (the XGBoost step).
    BestGain,
    /// Take a candidate at random, subject only to the occupancy guard (the
    /// extra-trees step: no optimisation at all).
    Random,
}

/// Hyperparameters of one greedy tessellation.
#[derive(Debug, Clone)]
struct GrowConfig {
    /// Ridge on the cell payload — XGBoost's `lambda`.
    lambda: f64,
    /// Per-cell complexity charge — XGBoost's `gamma`. A candidate has to beat
    /// it to be added, so it is the stopping rule.
    gamma: f64,
    /// Hard cap on cells (`max_depth`'s counterpart; a flat partition counts
    /// cells, not depth).
    max_cells: usize,
    /// Candidate centres sampled per tessellation. The knob that replaces
    /// exhaustive split enumeration: compute for fit quality.
    candidates: usize,
    /// Cap on the size of the random subspace a tessellation lives in
    /// (`colsample_bytree`). The paper's dimension prior plays this role in
    /// the MCMC.
    max_dims: usize,
    /// How many independent subspaces to grow in before keeping the best.
    ///
    /// Without this the subspace is drawn once and never revisited, which is a
    /// real handicap next to the MCMC: the sampler's add/remove-dimension moves
    /// are accepted on the likelihood, so it *searches* subspaces even though
    /// it proposes them at random. On a problem with noise columns, a
    /// once-and-for-all random draw spends most rounds fitting structure in
    /// subspaces that carry no signal.
    subspace_tries: usize,
    /// Minimum observations a cell must retain (`min_child_weight`).
    min_cell: usize,
    /// Shrinkage folded into the payload before it is written out.
    eta: f64,
    selection: Selection,
}

/// Grow one tessellation: try `subspace_tries` independent subspaces and keep
/// whichever achieved the most total gain.
///
/// `rows` are the (sub)sample the structure is fitted on; `g`/`h` are indexed
/// by global row. Returns `None` when nothing could be fitted — the caller
/// simply skips the round.
fn grow(
    z: &Data,
    rows: &[usize],
    g: &[f64],
    h: &[f64],
    cfg: &GrowConfig,
    rng: &mut ChaCha8Rng,
) -> Option<Tessellation> {
    if rows.len() < 2 * cfg.min_cell {
        return None;
    }
    let mut winner: Option<(f64, Tessellation)> = None;
    for _ in 0..cfg.subspace_tries.max(1) {
        let Some((gain, tessellation)) = grow_in_subspace(z, rows, g, h, cfg, rng) else {
            continue;
        };
        if winner.as_ref().is_none_or(|(best, _)| gain > *best) {
            winner = Some((gain, tessellation));
        }
    }
    winner.map(|(_, tessellation)| tessellation)
}

/// Grow greedily inside one freshly drawn subspace, returning the tessellation
/// and the total gain its structure achieved (zero under `Selection::Random`,
/// which does not evaluate gains at all).
fn grow_in_subspace(
    z: &Data,
    rows: &[usize],
    g: &[f64],
    h: &[f64],
    cfg: &GrowConfig,
    rng: &mut ChaCha8Rng,
) -> Option<(f64, Tessellation)> {
    // --- the random subspace ------------------------------------------------
    let n_dims = 1 + uniform_index(cfg.max_dims.min(z.n_cols()), rng);
    let mut dims: Vec<usize> = Vec::with_capacity(n_dims);
    while dims.len() < n_dims {
        let candidate = uniform_index(z.n_cols(), rng);
        if !dims.contains(&candidate) {
            dims.push(candidate);
        }
    }
    let d = dims.len();

    // --- the candidate pool, drawn from observed rows -----------------------
    //
    // The counterpart of "split at an observed value": candidate centres are
    // data points, so they sit where the data is and never strand a cell in
    // empty space.
    // Rejection-sampled without replacement, but with a bounded attempt count:
    // `rows` is a bootstrap resample under bagging, so it holds far fewer
    // distinct rows than its length and an unbounded "keep drawing until
    // distinct" loop would spin forever once the pool is exhausted. Ending up
    // with fewer candidates than asked for is fine; hanging is not.
    let target = cfg.candidates.min(rows.len());
    let mut candidate_rows: Vec<usize> = Vec::with_capacity(target);
    for _ in 0..8 * target {
        if candidate_rows.len() == target {
            break;
        }
        let pick = rows[uniform_index(rows.len(), rng)];
        if !candidate_rows.contains(&pick) {
            candidate_rows.push(pick);
        }
    }
    let n_candidates = candidate_rows.len();
    if n_candidates < 2 {
        return None;
    }

    // Distances from every fitting row to every candidate, computed once
    // (row-major, rows × candidates). Every later gain evaluation is then a
    // scan of this table against the running `best` vector: the greedy pass
    // costs O(rows · candidates), not O(rows · candidates · dims), per centre.
    let mut dist2 = vec![0.0_f64; rows.len() * n_candidates];
    for (ri, &row) in rows.iter().enumerate() {
        let point = z.row(row);
        for (ci, &candidate) in candidate_rows.iter().enumerate() {
            let centre = z.row(candidate);
            let mut total = 0.0;
            for &dim in &dims {
                let delta = point[dim] - centre[dim];
                total += delta * delta;
            }
            dist2[ri * n_candidates + ci] = total;
        }
    }

    // --- seed the single-cell tessellation ----------------------------------
    //
    // With one centre every observation is in cell 0 whatever the coordinates,
    // so the seed only sets the geometry the first real split works against.
    let mut chosen = vec![0_usize];
    let mut assign = vec![0_usize; rows.len()];
    let mut best: Vec<f64> = (0..rows.len()).map(|ri| dist2[ri * n_candidates]).collect();
    let mut cell_g = vec![rows.iter().map(|&r| g[r]).sum::<f64>()];
    let mut cell_h = vec![rows.iter().map(|&r| h[r]).sum::<f64>()];
    let mut cell_n = vec![rows.len()];

    // --- add centres while one pays for itself ------------------------------
    let mut scratch_g = Vec::new();
    let mut scratch_h = Vec::new();
    let mut scratch_n = Vec::new();
    let mut total_gain = 0.0;
    while chosen.len() < cfg.max_cells {
        let mut winner: Option<(usize, f64)> = None;
        for ci in 0..n_candidates {
            if chosen.contains(&ci) {
                continue;
            }
            let b = chosen.len();
            scratch_g.clear();
            scratch_g.resize(b, 0.0);
            scratch_h.clear();
            scratch_h.resize(b, 0.0);
            scratch_n.clear();
            scratch_n.resize(b, 0_usize);

            // Everything strictly closer to the candidate than to its current
            // centre moves, and all of it lands in the new cell.
            let (mut moved_g, mut moved_h, mut moved_n) = (0.0, 0.0, 0_usize);
            for (ri, &row) in rows.iter().enumerate() {
                if dist2[ri * n_candidates + ci] < best[ri] {
                    let cell = assign[ri];
                    scratch_g[cell] += g[row];
                    scratch_h[cell] += h[row];
                    scratch_n[cell] += 1;
                    moved_g += g[row];
                    moved_h += h[row];
                    moved_n += 1;
                }
            }

            // Occupancy guard, the counterpart of the sampler's empty-cell
            // rejection: the new cell and every donor must stay viable.
            if moved_n < cfg.min_cell {
                continue;
            }
            if (0..b).any(|k| cell_n[k] - scratch_n[k] < cfg.min_cell) {
                continue;
            }

            if cfg.selection == Selection::Random {
                winner = Some((ci, 0.0));
                break;
            }

            let mut gain = quadratic(moved_g, moved_h, cfg.lambda);
            for k in 0..b {
                gain += quadratic(
                    cell_g[k] - scratch_g[k],
                    cell_h[k] - scratch_h[k],
                    cfg.lambda,
                ) - quadratic(cell_g[k], cell_h[k], cfg.lambda);
            }
            gain = 0.5 * gain - cfg.gamma;
            if gain > winner.map_or(0.0, |(_, best_gain)| best_gain) {
                winner = Some((ci, gain));
            }
        }

        let Some((ci, gain)) = winner else { break };
        total_gain += gain;

        // Commit: recompute the moved set once for the winner and fold it in.
        let new_cell = chosen.len();
        let (mut moved_g, mut moved_h, mut moved_n) = (0.0, 0.0, 0_usize);
        for (ri, &row) in rows.iter().enumerate() {
            let candidate_distance = dist2[ri * n_candidates + ci];
            if candidate_distance < best[ri] {
                let from = assign[ri];
                cell_g[from] -= g[row];
                cell_h[from] -= h[row];
                cell_n[from] -= 1;
                assign[ri] = new_cell;
                best[ri] = candidate_distance;
                moved_g += g[row];
                moved_h += h[row];
                moved_n += 1;
            }
        }
        cell_g.push(moved_g);
        cell_h.push(moved_h);
        cell_n.push(moved_n);
        chosen.push(ci);
    }

    if chosen.len() < 2 {
        return None;
    }

    // --- payloads: the closed-form optimum, shrunk --------------------------
    let mus: Vec<f64> = (0..chosen.len())
        .map(|k| -cfg.eta * cell_g[k] / (cell_h[k] + cfg.lambda))
        .collect();
    let mut centres = Vec::with_capacity(chosen.len() * d);
    for &ci in &chosen {
        let row = z.row(candidate_rows[ci]);
        for &dim in &dims {
            centres.push(row[dim]);
        }
    }
    Tessellation::new(centres, dims, mus)
        .ok()
        .map(|tessellation| (total_gain, tessellation))
}

/// `G² / (H + λ)`: the structure score of one cell, up to the shared ½.
fn quadratic(sum_g: f64, sum_h: f64, lambda: f64) -> f64 {
    sum_g * sum_g / (sum_h + lambda)
}

// ---------------------------------------------------------------------------
// The two ensembles
// ---------------------------------------------------------------------------

/// A fitted greedy ensemble: tessellations plus how they combine.
struct Ensemble {
    tessellations: Vec<Tessellation>,
    base: f64,
    /// `false` sums the members (boosting), `true` averages them (bagging).
    average: bool,
}

/// Add one tessellation's cell payload to each row's running total: assign by
/// nearest centre over the tessellation's own dims, then accumulate.
fn accumulate(tessellation: &Tessellation, z: &Data, out: &mut [f64]) {
    let dims = tessellation.dims();
    let centres = tessellation.centres();
    let mus = tessellation.mus();
    let d = dims.len();
    for (i, value) in out.iter_mut().enumerate() {
        let point = z.row(i);
        let mut best = f64::INFINITY;
        let mut best_cell = 0;
        for cell in 0..tessellation.n_cells() {
            let centre = &centres[cell * d..(cell + 1) * d];
            let mut total = 0.0;
            for (slot, &dim) in dims.iter().enumerate() {
                let delta = point[dim] - centre[slot];
                total += delta * delta;
            }
            if total < best {
                best = total;
                best_cell = cell;
            }
        }
        *value += mus[best_cell];
    }
}

impl Ensemble {
    fn predict(&self, z: &Data) -> Vec<f64> {
        let mut out = vec![self.base; z.n_rows()];
        for tessellation in &self.tessellations {
            accumulate(tessellation, z, &mut out);
        }
        if self.average && !self.tessellations.is_empty() {
            let m = self.tessellations.len() as f64;
            for value in &mut out {
                *value = (*value - self.base) / m + self.base;
            }
        }
        out
    }

    /// Held-out RMSE after each successive member, in one pass over the
    /// ensemble. Boosting only: the members are cumulative, so the curve costs
    /// the same as a single prediction. (Recomputing a full prediction per
    /// round instead would be quadratic in the number of rounds — which
    /// dominates the fit itself once rounds run into the thousands.)
    fn learning_curve(&self, z: &Data, y: &[f64]) -> Vec<f64> {
        debug_assert!(!self.average, "cumulative curve is a boosting notion");
        let mut running = vec![self.base; z.n_rows()];
        let mut curve = Vec::with_capacity(self.tessellations.len());
        for tessellation in &self.tessellations {
            accumulate(tessellation, z, &mut running);
            curve.push(rmse(&running, y));
        }
        curve
    }
}

/// Greedy second-order boosting over tessellations: the XGBoost pattern with
/// the exhaustive split search replaced by a candidate-centre search.
///
/// The loss is squared error, so `g = F − y` and `h = 1`. Every other loss
/// XGBoost supports drops in by changing only those two lines — the gain
/// algebra, the grower, and the payload formula are loss-agnostic.
fn boost(
    z: &Data,
    y: &[f64],
    rounds: usize,
    row_subsample: f64,
    cfg: &GrowConfig,
    seed: u64,
) -> Ensemble {
    let n = z.n_rows();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let base = y.iter().sum::<f64>() / n as f64;
    let mut fit = vec![base; n];
    let mut ensemble = Ensemble {
        tessellations: Vec::new(),
        base,
        average: false,
    };

    let mut g = vec![0.0; n];
    let h = vec![1.0; n];
    for _ in 0..rounds {
        for i in 0..n {
            g[i] = fit[i] - y[i];
        }
        let rows: Vec<usize> = (0..n)
            .filter(|_| uniform(&mut rng) < row_subsample)
            .collect();
        let Some(tessellation) = grow(z, &rows, &g, &h, cfg, &mut rng) else {
            continue;
        };
        // Fold the new member into the running fit over *all* rows, not just
        // the subsample it was grown on.
        accumulate(&tessellation, z, &mut fit);
        ensemble.tessellations.push(tessellation);
    }
    ensemble
}

/// Bagged tessellations: the random-forest pattern. Each member fits the
/// response directly on a bootstrap resample (`g = −y`, `h = 1`, so a cell's
/// payload is the cell mean and the gain is variance reduction), and members
/// are averaged rather than summed.
fn forest(z: &Data, y: &[f64], trees: usize, cfg: &GrowConfig, seed: u64) -> Ensemble {
    let n = z.n_rows();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let g: Vec<f64> = y.iter().map(|value| -value).collect();
    let h = vec![1.0; n];
    let mut tessellations = Vec::with_capacity(trees);
    for _ in 0..trees {
        let rows: Vec<usize> = (0..n).map(|_| uniform_index(n, &mut rng)).collect();
        if let Some(tessellation) = grow(z, &rows, &g, &h, cfg, &mut rng) {
            tessellations.push(tessellation);
        }
    }
    Ensemble {
        tessellations,
        base: 0.0,
        average: true,
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

/// Friedman #1: five signal columns (two of them only through an interaction)
/// and `p − 5` pure noise columns. The standard sanity benchmark for this
/// family of models, and the one the AddiVortes paper reports.
fn friedman(n: usize, p: usize, sigma: f64, seed: u64) -> (Data, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..p).map(|_| uniform(&mut rng)).collect();
        let noise: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
        let centred = row[2] - 0.5;
        y.push(
            10.0 * mathsfn::sin(std::f64::consts::PI * row[0] * row[1])
                + 20.0 * centred * centred
                + 10.0 * row[3]
                + 5.0 * row[4]
                + sigma * noise,
        );
        values.extend_from_slice(&row);
    }
    (
        Data::new(values, n, p).expect("generated shape is consistent"),
        y,
    )
}

fn rmse(predicted: &[f64], actual: &[f64]) -> f64 {
    let sum: f64 = predicted
        .iter()
        .zip(actual)
        .map(|(p, a)| (p - a) * (p - a))
        .sum();
    (sum / actual.len() as f64).sqrt()
}

/// The round at which held-out error bottoms out, and that error: boosting's
/// early-stopping decision, which the MCMC has no counterpart for (and no need
/// of — the priors do that job).
fn best_round(curve: &[f64]) -> (usize, f64) {
    let mut best = (0_usize, f64::INFINITY);
    for (index, &error) in curve.iter().enumerate() {
        if error < best.1 {
            best = (index + 1, error);
        }
    }
    best
}

/// The boosting configuration a sweep settled on. The headline is the split of
/// the search budget: a *small* candidate pool and *many* subspace tries. See
/// the module docs and `docs/design/greedy-tessellations.md` for why that is
/// the interesting part.
fn tuned() -> GrowConfig {
    GrowConfig {
        lambda: 1.0,
        gamma: 0.0,
        max_cells: 4,
        candidates: 5,
        max_dims: 4,
        subspace_tries: 8,
        min_cell: 10,
        eta: 0.05,
        selection: Selection::BestGain,
    }
}

/// The configuration a literal port of XGBoost lands on: search centre
/// positions hard, take the subspace as drawn. Same compute, much worse fit.
fn literal_port() -> GrowConfig {
    GrowConfig {
        max_cells: 8,
        candidates: 40,
        subspace_tries: 1,
        eta: 0.1,
        ..tuned()
    }
}

fn main() -> addivortes::Result<()> {
    // Three independent draws: train, a validation set that owns the
    // early-stopping decision, and a test set that is only ever reported.
    // Hyperparameters were chosen by a sweep, so the numbers still carry
    // selection optimism — but not the circular kind where the reported set
    // also picks the stopping round.
    let (n_train, n_eval, p, sigma) = (500, 1000, 10, 1.0);
    let (x_train, y_train) = friedman(n_train, p, sigma, 11);
    let (x_valid, y_valid) = friedman(n_eval, p, sigma, 12);
    let (x_test, y_test) = friedman(n_eval, p, sigma, 13);

    let standardiser = Standardiser::fit(&x_train);
    let z_train = standardiser.apply(&x_train);
    let z_valid = standardiser.apply(&x_valid);
    let z_test = standardiser.apply(&x_test);

    println!("Friedman #1: n_train {n_train}, n_test {n_eval}, p {p} (5 signal), sigma {sigma}");
    println!("Test RMSE {sigma:.2} is the irreducible floor.\n");
    println!("{:<38} {:>9} {:>9}  notes", "model", "test RMSE", "fit (s)");

    // --- the shipped Bayesian fit ------------------------------------------
    let started = Instant::now();
    let mcmc = AddiVortesConfig::new(42)
        .with_m(200)
        .with_burn_in(200)
        .with_draws(500)
        .fit(&x_train, &y_train)?;
    let mcmc_seconds = started.elapsed().as_secs_f64();
    let mcmc_rmse = rmse(&mcmc.predict(&x_test)?, &y_test);
    let intervals = mcmc.prediction_interval(&x_test, 0.9)?;
    let covered = intervals
        .iter()
        .zip(&y_test)
        .filter(|(interval, actual)| interval.lower <= **actual && **actual <= interval.upper)
        .count();
    println!(
        "{:<38} {mcmc_rmse:>9.3} {mcmc_seconds:>9.1}  90% PI covers {:.1}% of test rows",
        "AddiVortes (MCMC, m=200)",
        100.0 * covered as f64 / y_test.len() as f64
    );

    // --- the XGBoost analogue, ported literally ----------------------------
    let rounds = 3000;
    let started = Instant::now();
    let naive = boost(&z_train, &y_train, rounds, 0.8, &literal_port(), 7);
    let naive_seconds = started.elapsed().as_secs_f64();
    let (naive_stop, _) = best_round(&naive.learning_curve(&z_valid, &y_valid));
    let naive_rmse = naive.learning_curve(&z_test, &y_test)[naive_stop - 1];
    println!(
        "{:<38} {naive_rmse:>9.3} {naive_seconds:>9.1}  40 candidates, 1 subspace, stop at {naive_stop}",
        "greedy boosting (literal XGBoost port)"
    );

    // --- the XGBoost analogue, budget re-pointed ---------------------------
    let started = Instant::now();
    let boosted = boost(&z_train, &y_train, rounds, 0.8, &tuned(), 7);
    let boost_seconds = started.elapsed().as_secs_f64();
    let (stop, _) = best_round(&boosted.learning_curve(&z_valid, &y_valid));
    let boost_rmse = boosted.learning_curve(&z_test, &y_test)[stop - 1];
    println!(
        "{:<38} {boost_rmse:>9.3} {boost_seconds:>9.1}  5 candidates, 8 subspaces, stop at {stop}",
        "greedy boosting (budget re-pointed)"
    );

    // --- where the search budget belongs -----------------------------------
    //
    // The point of the prototype in one table: at roughly matched compute,
    // spending the budget on subspaces beats spending it on centres.
    println!("\nsearch budget, at 3000 rounds (test RMSE / fit seconds):");
    println!(
        "{:>12} {:>16} {:>16}",
        "candidates", "1 subspace", "8 subspaces"
    );
    for candidates in [5usize, 20, 40] {
        let mut cells = String::new();
        for subspace_tries in [1usize, 8] {
            let cfg = GrowConfig {
                candidates,
                subspace_tries,
                ..literal_port()
            };
            let started = Instant::now();
            let fitted = boost(&z_train, &y_train, rounds, 0.8, &cfg, 7);
            let seconds = started.elapsed().as_secs_f64();
            let (round, _) = best_round(&fitted.learning_curve(&z_valid, &y_valid));
            let error = fitted.learning_curve(&z_test, &y_test)[round - 1];
            cells.push_str(&format!("{:>16}", format!("{error:.3} / {seconds:.1}s")));
        }
        println!("{candidates:>12} {cells}");
    }

    // --- the random-forest analogue ----------------------------------------
    println!();
    let forest_cfg = GrowConfig {
        lambda: 1e-6,
        gamma: 0.0,
        max_cells: 60,
        candidates: 200,
        max_dims: 6,
        subspace_tries: 8,
        min_cell: 5,
        eta: 1.0,
        selection: Selection::BestGain,
    };
    for (label, selection) in [
        ("bagged greedy (RF analogue)", Selection::BestGain),
        ("bagged random (extra-trees analogue)", Selection::Random),
    ] {
        let cfg = GrowConfig {
            selection,
            // Random selection does not score anything, so trying several
            // subspaces would just be picking one arbitrarily.
            subspace_tries: if selection == Selection::Random {
                1
            } else {
                forest_cfg.subspace_tries
            },
            ..forest_cfg.clone()
        };
        let started = Instant::now();
        let bagged = forest(&z_train, &y_train, 200, &cfg, 5);
        let seconds = started.elapsed().as_secs_f64();
        let error = rmse(&bagged.predict(&z_test), &y_test);
        let mean_cells = bagged
            .tessellations
            .iter()
            .map(Tessellation::n_cells)
            .sum::<usize>() as f64
            / bagged.tessellations.len().max(1) as f64;
        println!(
            "{label:<38} {error:>9.3} {seconds:>9.1}  200 members, {mean_cells:.0} cells each (cap 60)"
        );
    }

    println!(
        "\nThe MCMC row is the only one carrying calibrated uncertainty; every\n\
         greedy row is a point estimate and nothing else."
    );
    Ok(())
}
