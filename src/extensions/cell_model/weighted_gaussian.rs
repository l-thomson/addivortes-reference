//! The precision-weighted Gaussian shelf entry: weighted sufficient
//! statistic + the same conjugate arithmetic, for per-observation precision
//! weighting under hard membership only (H-AddiVortes Eq. 5–7,
//! Pólya-Gamma-style weights). Never feed fractional memberships through
//! per-cell diagonal statistics: soft membership needs the joint
//! within-tessellation draw of the dense path.

use crate::engine::error::{Result, require_positive_finite};
use crate::extensions::cell_model::{CellModel, CellStats, gaussian_marginal_terms, mu_posterior};

/// The precision-weighted Gaussian statistic: weighted count `Σ wᵢ` and
/// weighted sum `Σ wᵢ·rᵢ`. At weights ≡ 1 it is bit-identical to
/// [`GaussianCellStats`](crate::extensions::cell_model::GaussianCellStats) through the
/// shared conjugate formulas (the seam-boundary guarantee, proven
/// on the full chain by the seam tests). This is the statistic
/// precision-weighted models feed (weight = 1/s²(xᵢ), σ² ≡ 1; the
/// H-AddiVortes/HBART reduction), hard membership only: fractional
/// membership weights make the posterior precision non-diagonal, which
/// per-cell statistics cannot express (the same reason SBART's soft gates
/// need a joint draw; Linero & Yang 2018); soft membership rides the dense
/// path.
#[derive(Debug, Default, Clone)]
pub struct WeightedGaussianStats {
    pub(crate) weight: f64,
    pub(crate) sum: f64,
}

impl CellStats for WeightedGaussianStats {
    fn record(&mut self, value: f64, weight: f64) {
        self.weight += weight;
        self.sum += weight * value;
    }
    fn merge(&mut self, other: &Self) {
        self.weight += other.weight;
        self.sum += other.sum;
    }
    fn remove(&mut self, other: &Self) {
        self.weight -= other.weight;
        self.sum -= other.sum;
    }
    fn reset(&mut self) {
        self.weight = 0.0;
        self.sum = 0.0;
    }
    fn occupied(&self) -> bool {
        self.weight > 0.0
    }
}

/// The precision-weighted Gaussian conjugate model: identical conjugate
/// arithmetic to [`GaussianCellModel`](crate::extensions::cell_model::GaussianCellModel)
/// but over [`WeightedGaussianStats`]: weighted counts `Σw` and weighted
/// sums `Σw·r` flow through the same formulas (at weights ≡ 1 the whole chain
/// is bit-identical to the built-in Gaussian, proven by the sampler's seam
/// tests). This is the cell model precision-weighted extensions pair with a
/// weight-producing [`ResponseModel`](crate::extensions::response::ResponseModel) or a
/// precision-supplying [`ScaleModel`](crate::extensions::scale::ScaleModel)
/// (weight = 1/s²(xᵢ), σ² ≡ 1; the H-AddiVortes/HBART reduction), and it is
/// correct under hard membership only: never feed fractional
/// memberships through per-cell diagonal statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedGaussianModel {
    sigma_mu_sq: f64,
}

impl WeightedGaussianModel {
    /// A weighted Gaussian cell model with prior cell-value variance σ_μ²
    /// (scaled space). Fails with
    /// [`AddiVortesError::InvalidHyperparameter`](crate::AddiVortesError::InvalidHyperparameter)
    /// unless `sigma_mu_sq` is finite and strictly positive.
    pub fn new(sigma_mu_sq: f64) -> Result<Self> {
        Ok(Self {
            sigma_mu_sq: require_positive_finite("sigma_mu_sq", sigma_mu_sq)?,
        })
    }
}

impl CellModel for WeightedGaussianModel {
    type Stats = WeightedGaussianStats;
    type Error = std::convert::Infallible;

    fn log_marginal_terms(
        &self,
        stats: &[Self::Stats],
        sigma_sq: f64,
    ) -> std::result::Result<f64, Self::Error> {
        Ok(gaussian_marginal_terms(
            stats.iter().map(|s| (s.weight, s.sum)),
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
            let (mean, variance) = mu_posterior(cell.weight, cell.sum, sigma_sq, self.sigma_mu_sq);
            let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng);
            values.push(mean + variance.sqrt() * z);
        }
        Ok(values)
    }
}
