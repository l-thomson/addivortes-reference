//! The Gaussian cell model, as one shelf entry: the
//! hard-assignment Gaussian sufficient statistic ([`GaussianCellStats`]), the
//! Gaussian conjugate cell model ([`GaussianCellModel`]): with the global σ²
//! Gibbs draw (`scale::GlobalSigma`) it is exactly the paper's published
//! sampler, and the fit-time default.

use crate::extensions::cell_model::{CellModel, CellStats, gaussian_marginal_terms, mu_posterior};

/// The built-in Gaussian sufficient statistic under **hard** assignment:
/// integer count + running residual sum (kept integer-counted so the default
/// path is bit-identical to the pre-seam sampler; a fractional-membership
/// statistic is a separate impl, e.g.
/// [`WeightedGaussianStats`](crate::extensions::cell_model::WeightedGaussianStats)).
#[derive(Debug, Default, Clone)]
pub struct GaussianCellStats {
    n: u64,
    sum: f64,
}

impl GaussianCellStats {
    /// Effective observation count n_k as f64 (integer-valued here).
    pub fn count(&self) -> f64 {
        self.n as f64
    }
    /// Accumulated working-value sum S_k (**scaled space**).
    pub fn sum(&self) -> f64 {
        self.sum
    }
}

impl CellStats for GaussianCellStats {
    fn record(&mut self, value: f64, weight: f64) {
        debug_assert!(
            weight == 1.0,
            "GaussianCellStats is the hard-assignment statistic; fractional \
             membership needs WeightedGaussianStats"
        );
        let _ = weight;
        self.n += 1;
        self.sum += value;
    }
    fn merge(&mut self, other: &Self) {
        self.n += other.n;
        self.sum += other.sum;
    }
    fn remove(&mut self, other: &Self) {
        debug_assert!(self.n >= other.n);
        self.n -= other.n;
        self.sum -= other.sum;
    }
    fn reset(&mut self) {
        self.n = 0;
        self.sum = 0.0;
    }
    fn occupied(&self) -> bool {
        self.n > 0
    }
}

/// The built-in Gaussian conjugate cell model: the complete
/// per-cell marginal-likelihood term and the conjugate Normal μ draw,
/// **bit-identical** to the pre-seam sampler (the golden chain freezes this).
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianCellModel {
    sigma_mu_sq: f64,
}

impl GaussianCellModel {
    /// A Gaussian cell model with prior cell-value variance σ_μ²
    /// (**scaled space**; σ_μ = 0.5/(k√m), the paper's μ prior).
    pub fn new(sigma_mu_sq: f64) -> Self {
        debug_assert!(sigma_mu_sq > 0.0);
        Self { sigma_mu_sq }
    }
}

impl CellModel for GaussianCellModel {
    type Stats = GaussianCellStats;
    type Error = std::convert::Infallible;

    fn log_marginal_terms(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
    ) -> std::result::Result<f64, Self::Error> {
        Ok(gaussian_marginal_terms(
            stats.iter().map(|s| (s.count(), s.sum())),
            sigma_sq,
            self.sigma_mu_sq,
        ))
    }

    fn draw_cell_values(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<Vec<f64>, Self::Error> {
        let mut values = Vec::with_capacity(stats.len());
        for cell in stats {
            let (mean, variance) =
                mu_posterior(cell.count(), cell.sum(), sigma_sq, self.sigma_mu_sq);
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
            values.push(mean + variance.sqrt() * z);
        }
        Ok(values)
    }
}
