//! The inverse-χ² variance cell family, as one shelf entry (H
//! paper §3.2): cells hold a variance factor s² with the conjugate
//! scaled-inverse-χ²(ν′, λ′) prior, priced by the closed-form integrated
//! likelihood of H paper Eq. 10 and redrawn from the Eq. 9 full conditional.
//! This is the second conjugate family on the shelf: the working
//! values are squared scale-free residuals `ẽ²ᵢ = e²ᵢ / s²₋ₗ(xᵢ)` with
//! per-observation likelihood `ẽᵢ ~ N(0, s²)`, and the ensemble composes
//! multiplicatively.

use crate::engine::error::{Result, require_positive_finite};
use crate::engine::mathsfn;
use crate::extensions::cell_model::{CellModel, CellStats};
use crate::extensions::scale::sigma_sq_gamma_params;

/// The inverse-χ² sufficient statistic under hard assignment: integer
/// count + running sum of the squared working values `Σ ẽ²ᵢ` (H paper
/// Eq. 10's sufficient statistic).
#[derive(Debug, Default, Clone)]
pub struct InvChiSqStats {
    n: u64,
    sum_sq: f64,
}

impl InvChiSqStats {
    /// Observation count n_k as f64 (integer-valued; a count).
    pub fn count(&self) -> f64 {
        self.n as f64
    }
    /// Accumulated squared working-value sum `Σ ẽ²ᵢ` (scaled space,
    /// scale-free residual units).
    pub fn sum_sq(&self) -> f64 {
        self.sum_sq
    }
}

impl CellStats for InvChiSqStats {
    fn record(&mut self, value: f64, weight: f64) {
        debug_assert!(
            weight == 1.0,
            "InvChiSqStats is a hard-assignment statistic (the variance \
             ensemble runs under hard membership)"
        );
        debug_assert!(value >= 0.0, "squared working values are non-negative");
        let _ = weight;
        self.n += 1;
        self.sum_sq += value;
    }
    fn merge(&mut self, other: &Self) {
        self.n += other.n;
        self.sum_sq += other.sum_sq;
    }
    fn remove(&mut self, other: &Self) {
        debug_assert!(self.n >= other.n);
        self.n -= other.n;
        self.sum_sq -= other.sum_sq;
    }
    fn reset(&mut self) {
        self.n = 0;
        self.sum_sq = 0.0;
    }
    fn occupied(&self) -> bool {
        self.n > 0
    }
}

/// The inverse-χ² conjugate variance cell model (H paper §3.2). The cell
/// "value" is the variance factor s² (strictly positive); the trait's
/// `sigma_sq` argument is ignored: the family's noise level is the cell
/// value itself (the mean side's σ² is pinned to 1 in the H configuration and
/// the per-observation variance enters as the ensemble product).
#[derive(Debug, Clone, PartialEq)]
pub struct InvChiSqCellModel {
    nu: f64,
    lambda: f64,
}

impl InvChiSqCellModel {
    /// An inverse-χ² cell model with prior s² ~ χ⁻²(ν′, λ′) per cell
    /// (scaled space; ν′/λ′ from the H paper §3.3 calibration,
    /// [`h_variance_prior`](crate::extensions::scale::h_variance_prior)).
    /// Fails with
    /// [`AddiVortesError::InvalidHyperparameter`](crate::AddiVortesError::InvalidHyperparameter)
    /// unless both are finite and strictly positive.
    pub fn new(nu: f64, lambda: f64) -> Result<Self> {
        Ok(Self {
            nu: require_positive_finite("nu", nu)?,
            lambda: require_positive_finite("lambda", lambda)?,
        })
    }
}

impl CellModel for InvChiSqCellModel {
    type Stats = InvChiSqStats;
    type Error = std::convert::Infallible;

    /// H paper Eq. 10, complete per cell (log scale), the `(2π)^{−n/2}`
    /// factor dropped (Σ n_k is structure-invariant so it cancels in every
    /// acceptance ratio).
    ///
    /// Two deliberate departures from the equation as typeset: the paper
    /// writes the sums over all `n` where the per-cell `n_k` is meant (its
    /// Eq. 8 uses `n_kl` correctly), and it omits the `lnΓ(ν′/2)` prior
    /// normaliser, which is cell-count-dependent and so does *not* cancel in
    /// the acceptance ratio. Both are kept right here:
    ///
    /// ```text
    /// Σ_k [ (ν′/2)·ln(ν′λ′/2) − lnΓ(ν′/2)
    ///       + lnΓ((ν′+n_k)/2) − ((ν′+n_k)/2)·ln((ν′λ′+S_k)/2) ]
    /// ```
    ///
    /// The complete per-cell prior normalising term is what makes the
    /// cell-count-dependent constants of AC/RC emerge automatically, the
    /// same discipline as the Gaussian family's ±0.5·ln σ².
    fn log_marginal_terms(
        &self,
        stats: &[Self::Stats],
        _sigma_sq: f64,
    ) -> std::result::Result<f64, Self::Error> {
        let half_nu = 0.5 * self.nu;
        let prior_term = half_nu * mathsfn::ln(half_nu * self.lambda) - mathsfn::lgamma(half_nu);
        let mut total = 0.0_f64;
        for cell in stats {
            let half_post = 0.5 * (self.nu + cell.count());
            total += prior_term + mathsfn::lgamma(half_post)
                - half_post * mathsfn::ln(0.5 * (self.nu * self.lambda + cell.sum_sq()));
        }
        Ok(total)
    }

    /// H paper Eq. 9: s² | · ~ (ν′λ′ + S_k)/χ²_{ν′+n_k}, realised as
    /// 1/Gamma((ν′+n_k)/2, 2/(ν′λ′+S_k)), the same parameterisation as the
    /// global σ² Gibbs draw. An empty statistic draws the prior (n_k = 0,
    /// S_k = 0), which is what conjugacy means at zero data.
    fn draw_cell_values(
        &self,
        stats: &[Self::Stats],
        _sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<Vec<f64>, Self::Error> {
        let mut values = Vec::with_capacity(stats.len());
        for cell in stats {
            let (shape, scale) =
                sigma_sq_gamma_params(self.nu, self.lambda, cell.sum_sq(), cell.count() as usize);
            let gamma = rand_distr::Gamma::new(shape, scale)
                .expect("shape and scale are positive by construction");
            let precision: f64 = rand_distr::Distribution::sample(&gamma, rng);
            values.push(1.0 / precision);
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::test_support::assert_rel_eq;

    #[test]
    fn marginal_terms_match_the_hand_derived_eq_10() {
        // ν′ = 4, λ′ = 0.5; one cell with n = 3, S = 1.2:
        // (2)·ln(1) − lnΓ(2) + lnΓ(3.5) − 3.5·ln(1.6).
        let model = InvChiSqCellModel::new(4.0, 0.5).unwrap();
        let mut stats = InvChiSqStats::default();
        for v in [0.5, 0.3, 0.4] {
            stats.record(v, 1.0);
        }
        let ln = crate::engine::mathsfn::ln;
        let lg = crate::engine::mathsfn::lgamma;
        let hand = 2.0 * ln(2.0 * 0.5) - lg(2.0) + lg(3.5) - 3.5 * ln(0.5 * (2.0 + 1.2));
        let claimed = model
            .log_marginal_terms(std::slice::from_ref(&stats), 1.0)
            .unwrap();
        assert_rel_eq(claimed, hand, 1e-14);
    }

    #[test]
    fn empty_statistic_draws_the_prior() {
        // n = 0, S = 0: s² = ν′λ′/χ²_{ν′}. Check the sample mean against the
        // prior mean ν′λ′/(ν′−2) over many draws (ν′ = 8, λ′ = 0.4:
        // mean = 3.2/6).
        let model = InvChiSqCellModel::new(8.0, 0.4).unwrap();
        let stats = vec![InvChiSqStats::default()];
        let mut rng = ChaCha8Rng::from_seed([5; 32]);
        let n_draws = 200_000;
        let mut total = 0.0;
        for _ in 0..n_draws {
            total += model.draw_cell_values(&stats, 1.0, &mut rng).unwrap()[0];
        }
        let mean = total / n_draws as f64;
        assert!(
            (mean - 8.0 * 0.4 / 6.0).abs() < 0.01,
            "prior mean {mean} far from 8·0.4/6 ≈ 0.5333"
        );
    }

    #[test]
    fn posterior_draw_concentrates_on_the_data_scale() {
        // Large n at constant ẽ² = 0.09: the posterior mean of s² approaches
        // 0.09 (the ML variance of N(0, s²) data with Σẽ²/n = 0.09).
        let model = InvChiSqCellModel::new(4.0, 1.0).unwrap();
        let mut stats = InvChiSqStats::default();
        for _ in 0..5000 {
            stats.record(0.09, 1.0);
        }
        let mut rng = ChaCha8Rng::from_seed([9; 32]);
        let stats = vec![stats];
        let mut total = 0.0;
        let n_draws = 20_000;
        for _ in 0..n_draws {
            total += model.draw_cell_values(&stats, 1.0, &mut rng).unwrap()[0];
        }
        let mean = total / n_draws as f64;
        assert!(
            (mean - 0.09).abs() < 0.005,
            "posterior mean {mean} far from the data scale 0.09"
        );
    }
}
