//! The linear-in-cell Gaussian family, as one shelf entry: cells hold a coefficient vector β ∈ ℝ^q over a
//! per-observation basis row z(x) rather than a scalar, with the conjugate
//! prior β ~ N(0, σ_β² I_q). The sufficient statistic is the per-cell block
//! pair (ZᵀWZ, ZᵀWr) and every score/redraw is a q×q solve through the
//! pinned [`mathsfn::cholesky`]: block-diagonal by construction (hard
//! membership keeps cells independent; the block ≡ dense cross-path oracle
//! holds whenever both paths exist).
//!
//! The basis is not restricted to the tessellation's active dimensions:
//! the caller records whatever basis row it likes per observation
//! ([`LinearCellStats::record_row`]). The scalar [`CellStats::record`] path
//! is the q = 1 intercept-only basis (`z ≡ [1]`), which reproduces the
//! scalar Gaussian family to ≤ 1e-12 on marginal terms and draws
//! (pinned by this module's tests).

use crate::engine::mathsfn;
use crate::extensions::cell_model::{CellModel, CellStats};

/// The linear-in-cell sufficient statistic: the per-cell block pair
/// `ZᵀWZ` (q×q, row-major) and `ZᵀWr` (q), plus the accumulated weight (the
/// occupancy question). Sized lazily on the first recorded row; `merge` and
/// `remove` are plain matrix adds/subtracts, so the statistic stays
/// order-free and additive (the conformance sufficiency contract).
#[derive(Debug, Default, Clone)]
pub struct LinearCellStats {
    /// Σ wᵢ: the occupancy mass (dimensionless weight).
    weight: f64,
    /// `ZᵀWZ`, q×q row-major (scaled-space basis units squared).
    ztwz: Vec<f64>,
    /// `ZᵀWr`, length q (scaled-space basis × response units).
    ztwr: Vec<f64>,
}

impl LinearCellStats {
    /// Absorb one observation: basis row `z` (length q, scaled space),
    /// working value `value`, accumulation weight `weight`. Every row in one
    /// statistic must share q (debug-asserted).
    pub fn record_row(&mut self, z: &[f64], value: f64, weight: f64) {
        let q = z.len();
        if self.ztwr.is_empty() {
            self.ztwz = vec![0.0; q * q];
            self.ztwr = vec![0.0; q];
        }
        debug_assert_eq!(self.ztwr.len(), q, "basis dimension must be constant");
        self.weight += weight;
        for r in 0..q {
            self.ztwr[r] += weight * z[r] * value;
            for c in 0..q {
                self.ztwz[r * q + c] += weight * z[r] * z[c];
            }
        }
    }

    /// The basis dimension q of this statistic (0 while empty; a count).
    pub fn q(&self) -> usize {
        self.ztwr.len()
    }
}

impl CellStats for LinearCellStats {
    /// The scalar path: the q = 1 intercept-only basis `z ≡ [1]`, exactly
    /// the weighted scalar Gaussian statistic.
    fn record(&mut self, value: f64, weight: f64) {
        self.record_row(&[1.0], value, weight);
    }
    /// The basis path: accumulate this observation's (ZᵀWZ, ZᵀWr)
    /// contribution against its own basis row.
    fn record_basis(&mut self, z: &[f64], value: f64, weight: f64) {
        self.record_row(z, value, weight);
    }
    fn merge(&mut self, other: &Self) {
        if other.ztwr.is_empty() {
            return;
        }
        if self.ztwr.is_empty() {
            *self = other.clone();
            return;
        }
        debug_assert_eq!(self.ztwr.len(), other.ztwr.len());
        self.weight += other.weight;
        for (a, b) in self.ztwz.iter_mut().zip(&other.ztwz) {
            *a += b;
        }
        for (a, b) in self.ztwr.iter_mut().zip(&other.ztwr) {
            *a += b;
        }
    }
    fn remove(&mut self, other: &Self) {
        if other.ztwr.is_empty() {
            return;
        }
        debug_assert_eq!(self.ztwr.len(), other.ztwr.len());
        self.weight -= other.weight;
        for (a, b) in self.ztwz.iter_mut().zip(&other.ztwz) {
            *a -= b;
        }
        for (a, b) in self.ztwr.iter_mut().zip(&other.ztwr) {
            *a -= b;
        }
    }
    fn reset(&mut self) {
        self.weight = 0.0;
        self.ztwz.clear();
        self.ztwr.clear();
    }
    fn occupied(&self) -> bool {
        self.weight > 0.0
    }
}

/// The linear-in-cell Gaussian conjugate model: per cell, the
/// posterior precision `A = ZᵀWZ/σ² + I/σ_β²` and mean `A⁻¹ ZᵀWr/σ²`, both
/// through the pinned q×q Cholesky. The q = 1 intercept-only configuration
/// reproduces the scalar Gaussian family (≤ 1e-12 on marginal terms and
/// draws).
#[derive(Debug, Clone, PartialEq)]
pub struct LinearGaussianModel {
    sigma_beta_sq: f64,
    q: usize,
}

impl LinearGaussianModel {
    /// A linear cell model with coefficient-prior variance σ_β² (scaled
    /// space) over a q-dimensional basis. `q = 1` with the intercept basis
    /// is the scalar Gaussian family.
    pub fn new(sigma_beta_sq: f64, q: usize) -> Self {
        debug_assert!(sigma_beta_sq > 0.0 && q >= 1);
        Self { sigma_beta_sq, q }
    }

    /// The basis dimension q (a count).
    pub fn q(&self) -> usize {
        self.q
    }

    /// Per-cell posterior pieces: the lower Cholesky factor of
    /// `A = ZᵀWZ/σ² + I/σ_β²` and the posterior mean `A⁻¹ b`, `b = ZᵀWr/σ²`
    /// (scaled space). Empty statistics give the prior (A = I/σ_β², mean 0).
    fn posterior(&self, cell: &LinearCellStats, sigma_sq: f64) -> (Vec<f64>, Vec<f64>, f64) {
        let q = self.q;
        let mut a = vec![0.0_f64; q * q];
        let mut b = vec![0.0_f64; q];
        if !cell.ztwr.is_empty() {
            debug_assert_eq!(cell.q(), q, "statistic dimension must match the model");
            for r in 0..q {
                b[r] = cell.ztwr[r] / sigma_sq;
                for c in 0..q {
                    a[r * q + c] = cell.ztwz[r * q + c] / sigma_sq;
                }
            }
        }
        for r in 0..q {
            a[r * q + r] += 1.0 / self.sigma_beta_sq;
        }
        let spd = mathsfn::cholesky(&mut a, q);
        assert!(spd, "A = ZᵀWZ/σ² + I/σ_β² is SPD by construction");
        // b'A⁻¹b before b is overwritten by the solve.
        let mut mean = b.clone();
        mathsfn::cholesky_solve(&a, q, &mut mean);
        let quadratic: f64 = b.iter().zip(&mean).map(|(bi, ui)| bi * ui).sum();
        (a, mean, quadratic)
    }

    /// Draw one coefficient vector per cell (β ~ N(A⁻¹b, A⁻¹)) in
    /// ascending cell index, the q standard normals per cell drawn in
    /// ascending coefficient index before the transform (scaled space). The
    /// basis-payload sibling of the trait's scalar `draw_cell_values`.
    pub fn draw_cell_coefficients(
        &self,
        stats: &[LinearCellStats],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> Vec<Vec<f64>> {
        let q = self.q;
        let mut cells = Vec::with_capacity(stats.len());
        for cell in stats {
            let (l, mean, _) = self.posterior(cell, sigma_sq);
            let mut z: Vec<f64> = (0..q)
                .map(|_| rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng))
                .collect();
            // Lᵀv = z gives Cov(v) = (LLᵀ)⁻¹ = A⁻¹.
            mathsfn::cholesky_solve_transposed(&l, q, &mut z);
            let beta: Vec<f64> = mean.iter().zip(&z).map(|(m, v)| m + v).collect();
            cells.push(beta);
        }
        cells
    }
}

impl CellModel for LinearGaussianModel {
    type Stats = LinearCellStats;
    type Error = std::convert::Infallible;

    /// The integrated marginal over β per cell (complete, structure-varying
    /// part): `0.5·(bᵀA⁻¹b − ln det(σ_β² A))` with `A = ZᵀWZ/σ² + I/σ_β²`,
    /// `b = ZᵀWr/σ²`; the (2πσ²)^{−n/2}·exp(−rᵀWr/2σ²) factors are
    /// structure-invariant and dropped. At q = 1 this is algebraically the
    /// scalar family's `0.5·ln(σ²/(nσ_μ²+σ²)) + σ_μ²S²/(2σ²(nσ_μ²+σ²))`.
    fn log_marginal_terms(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
    ) -> std::result::Result<f64, Self::Error> {
        let q = self.q;
        let mut total = 0.0_f64;
        for cell in stats {
            let (l, _, quadratic) = self.posterior(cell, sigma_sq);
            let mut log_det = q as f64 * mathsfn::ln(self.sigma_beta_sq);
            for r in 0..q {
                log_det += 2.0 * mathsfn::ln(l[r * q + r]);
            }
            total += 0.5 * (quadratic - log_det);
        }
        Ok(total)
    }

    /// The scalar draw: defined for the q = 1 (intercept-only) basis, where
    /// the coefficient is the cell value (scaled space); q > 1 payloads are
    /// vectors and use [`draw_cell_coefficients`](Self::draw_cell_coefficients)
    /// (the sampler's scalar-payload path carries q = 1 only).
    fn draw_cell_values(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<Vec<f64>, Self::Error> {
        assert_eq!(
            self.q, 1,
            "scalar cell values exist only for the q = 1 intercept basis; \
             use draw_cell_coefficients for basis payloads"
        );
        Ok(self
            .draw_cell_coefficients(stats, sigma_sq, rng)
            .into_iter()
            .map(|beta| beta[0])
            .collect())
    }

    fn cell_basis(&self) -> bool {
        self.q > 1
    }

    fn payload_width(&self) -> usize {
        self.q
    }

    /// The basis payload: one β ∈ ℝ^q per cell, flattened row-major in
    /// ascending cell index (the q standard normals per cell drawn in ascending
    /// coefficient index, matching [`draw_cell_coefficients`](Self::draw_cell_coefficients),
    /// which this delegates to).
    fn draw_cell_payload(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<Vec<f64>, Self::Error> {
        Ok(self
            .draw_cell_coefficients(stats, sigma_sq, rng)
            .into_iter()
            .flatten()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::extensions::cell_model::{GaussianCellModel, gaussian_marginal_terms};
    use crate::test_support::{assert_abs_eq, assert_rel_eq};

    /// Scalar-equivalence proof, half one: the q = 1 intercept basis
    /// reproduces the scalar path to ≤ 1e-12 on marginal terms.
    #[test]
    fn q1_marginal_terms_match_the_scalar_family() {
        let sigma_mu_sq = 0.02;
        let sigma_sq = 0.3;
        let residuals = [0.12, -0.05, 0.31, 0.07, -0.22];
        let weights = [1.0, 0.5, 2.0, 1.5, 1.0];
        let assignment = [0usize, 1, 0, 1, 0];

        let linear = LinearGaussianModel::new(sigma_mu_sq, 1);
        let mut stats = vec![LinearCellStats::default(); 2];
        let mut pairs = [(0.0_f64, 0.0_f64); 2];
        for i in 0..residuals.len() {
            stats[assignment[i]].record(residuals[i], weights[i]);
            pairs[assignment[i]].0 += weights[i];
            pairs[assignment[i]].1 += weights[i] * residuals[i];
        }
        let claimed = linear.log_marginal_terms(&stats, sigma_sq).unwrap();
        let scalar = gaussian_marginal_terms(pairs.iter().copied(), sigma_sq, sigma_mu_sq);
        assert_rel_eq(claimed, scalar, 1e-12);
    }

    /// Scalar-equivalence proof, half two: q = 1 draws match the scalar
    /// family's draws to ≤ 1e-12 under the same RNG (one standard normal per
    /// cell either way).
    #[test]
    fn q1_draws_match_the_scalar_family() {
        let sigma_mu_sq = 0.02;
        let sigma_sq = 0.3;
        let residuals = [0.12, -0.05, 0.31, 0.07, -0.22];
        let assignment = [0usize, 1, 0, 1, 0];

        let linear = LinearGaussianModel::new(sigma_mu_sq, 1);
        let scalar = GaussianCellModel::new(sigma_mu_sq);
        let mut linear_stats = vec![LinearCellStats::default(); 2];
        let mut scalar_stats = vec![crate::extensions::cell_model::GaussianCellStats::default(); 2];
        for i in 0..residuals.len() {
            linear_stats[assignment[i]].record(residuals[i], 1.0);
            scalar_stats[assignment[i]].record(residuals[i], 1.0);
        }
        let mut rng_a = ChaCha8Rng::from_seed([31; 32]);
        let mut rng_b = ChaCha8Rng::from_seed([31; 32]);
        let linear_draws = linear
            .draw_cell_values(&linear_stats, sigma_sq, &mut rng_a)
            .unwrap();
        let scalar_draws = scalar
            .draw_cell_values(&scalar_stats, sigma_sq, &mut rng_b)
            .unwrap();
        for (a, b) in linear_draws.iter().zip(&scalar_draws) {
            assert_rel_eq(*a, *b, 1e-12);
        }
    }

    /// The q = 2 marginal against an independently-computed reference
    /// (hand-solved normal equations: A = ZᵀWZ/σ² + I/σ_β², term =
    /// 0.5(bᵀA⁻¹b − ln det(σ_β²A))).
    #[test]
    fn q2_marginal_matches_reference_fixture() {
        let model = LinearGaussianModel::new(0.05, 2);
        let z = [[1.0, 0.2], [1.0, -0.4], [1.0, 0.1], [1.0, 0.5]];
        let w = [1.0, 0.5, 2.0, 1.5];
        let r = [0.12, -0.05, 0.31, 0.07];
        let mut cell = LinearCellStats::default();
        for i in 0..4 {
            cell.record_row(&z[i], r[i], w[i]);
        }
        // Accumulated blocks (hand): ZᵀWZ = [[5, 0.95], [0.95, 0.515]],
        // ZᵀWr = [0.82, 0.1485].
        assert_abs_eq(cell.ztwz[0], 5.0, 1e-12);
        assert_abs_eq(cell.ztwz[1], 0.95, 1e-12);
        assert_abs_eq(cell.ztwr[0], 0.82, 1e-12);
        assert_abs_eq(cell.ztwr[1], 0.1485, 1e-12);
        let claimed = model
            .log_marginal_terms(std::slice::from_ref(&cell), 0.3)
            .unwrap();
        assert_rel_eq(claimed, -0.234_462_917_652_298, 1e-11);
    }

    /// Empty statistics draw the coefficient prior N(0, σ_β² I), what
    /// conjugacy means at zero data (checked on the sample moments).
    #[test]
    fn empty_statistic_draws_the_coefficient_prior() {
        let model = LinearGaussianModel::new(0.05, 2);
        let stats = vec![LinearCellStats::default()];
        let mut rng = ChaCha8Rng::from_seed([7; 32]);
        let n_draws = 100_000;
        let mut sum = [0.0_f64; 2];
        let mut sum_sq = [0.0_f64; 2];
        let mut cross = 0.0_f64;
        for _ in 0..n_draws {
            let beta = &model.draw_cell_coefficients(&stats, 0.3, &mut rng)[0];
            for c in 0..2 {
                sum[c] += beta[c];
                sum_sq[c] += beta[c] * beta[c];
            }
            cross += beta[0] * beta[1];
        }
        for c in 0..2 {
            assert_abs_eq(sum[c] / n_draws as f64, 0.0, 0.005);
            assert_abs_eq(sum_sq[c] / n_draws as f64, 0.05, 0.002);
        }
        assert_abs_eq(cross / n_draws as f64, 0.0, 0.002);
    }
}
