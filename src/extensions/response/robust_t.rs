//! The robust-t response family: Student-t errors via the classical
//! scale-mixture augmentation. With
//! `εᵢ | λᵢ ~ N(0, σ²/λᵢ)` and `λᵢ ~ Gamma(ν′/2, rate ν′/2)`, the marginal
//! error is exactly `ε ~ t_ν′(0, σ)`, so once per sweep [`RobustTStep`]
//! redraws every mixture precision from its full conditional
//! `λᵢ | eᵢ, σ² ~ Gamma((ν′+1)/2, rate (ν′ + eᵢ²/σ²)/2)` and hands the
//! engine the Gaussian working form `yᵢ | μ ~ N(μ, σ²/λᵢ)`. Observations in
//! the tails earn small λᵢ and lose their pull on the fit, that is the
//! robustness working.
//!
//! The pairing rules of the seam are what [`ResponseFamily::RobustT`]
//! assembles for you: a weight-aware mean family
//! ([`WeightedGaussianModel`]) and a σ² draw consistent with the weights,
//! [`WeightedGlobalSigma`], the precision-weighted sibling of
//! [`GlobalSigma`], whose Gibbs draw prices the weighted RSS
//! `σ² | rest ~ IG((ν+n)/2, (νλ + Σᵢ wᵢeᵢ²)/2)` by reading this sweep's
//! augmentation weights from [`ScaleCtx::response_weights`].
//!
//! Validated by the robust-t Geweke leg of the acceptance battery
//! (`tests/calibration_acceptance.rs`), like every other kernel-touching entry.
//!
//! [`ResponseFamily::RobustT`]: crate::engine::model::ResponseFamily::RobustT
//! [`WeightedGaussianModel`]: crate::extensions::cell_model::WeightedGaussianModel
//! [`GlobalSigma`]: crate::extensions::scale::GlobalSigma
//! [`ScaleCtx::response_weights`]: crate::extensions::scale::ScaleCtx::response_weights

use crate::extensions::response::ResponseModel;

/// The robust-t scale-mixture step: once per sweep, redraw the
/// per-observation mixture precisions from their full conditional and pass
/// the observed response through unchanged (`working = y`,
/// `weights = λ`). Stateless between sweeps: the λ are redrawn fresh each
/// time, so it composes with
/// [`Sampler::set_response`](crate::Sampler::set_response) like the probit
/// step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustTStep {
    df: f64,
}

impl RobustTStep {
    /// A robust-t step with ν′ = `df` error degrees of freedom
    /// (dimensionless count; must be finite and strictly positive,
    /// debug-asserted here, validated at fit time by the family assembly).
    /// Small ν′ (3–8) is the robust regime; ν′ → ∞ recovers Gaussian
    /// errors.
    pub fn new(df: f64) -> Self {
        debug_assert!(df.is_finite() && df > 0.0);
        Self { df }
    }
}

impl ResponseModel for RobustTStep {
    type Error = std::convert::Infallible;

    fn augment(
        &mut self,
        y: &[f64],
        fit: &[f64],
        sigma_sq: f64,
        rng: &mut dyn rand_core::Rng,
        working: &mut [f64],
        weights: &mut [f64],
    ) -> std::result::Result<(), Self::Error> {
        // λᵢ | eᵢ, σ² ~ Gamma(shape (ν′+1)/2, rate (ν′ + eᵢ²/σ²)/2), taken
        // at the previous sweep's σ² (the pinned hook order runs this hook
        // first; the scale update that follows sees these weights through
        // the context).
        let shape = 0.5 * (self.df + 1.0);
        for i in 0..working.len() {
            let residual = y[i] - fit[i];
            let scale = 2.0 / (self.df + residual * residual / sigma_sq);
            let gamma = rand_distr::Gamma::new(shape, scale)
                .expect("shape and scale are positive by construction");
            weights[i] = rand_distr::Distribution::sample(&gamma, rng);
            working[i] = y[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::scale::{ScaleCtx, ScaleModel, WeightedGlobalSigma};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    /// The λ full conditional has mean (ν′+1)/(ν′ + e²/σ²): near-unit for
    /// on-fit observations, shrinking toward 0 as the residual grows: the
    /// robustness mechanism, checked on long-run averages.
    #[test]
    fn tail_observations_are_downweighted() {
        let df = 5.0;
        let sigma_sq = 1.0;
        let y = [0.0, 4.0];
        let fit = [0.0, 0.0];
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut sums = [0.0_f64; 2];
        let n = 4000;
        for _ in 0..n {
            let mut step = RobustTStep::new(df);
            let mut working = [0.0_f64; 2];
            let mut weights = [0.0_f64; 2];
            step.augment(&y, &fit, sigma_sq, &mut rng, &mut working, &mut weights)
                .unwrap();
            assert_eq!(working, y, "the working response is the observed one");
            assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));
            sums[0] += weights[0];
            sums[1] += weights[1];
        }
        let mean_on_fit = sums[0] / n as f64;
        let mean_outlier = sums[1] / n as f64;
        // Exact conditional means: (5+1)/(5+0) = 1.2 and (5+1)/(5+16) ≈ 0.2857.
        assert!((mean_on_fit - 1.2).abs() < 0.05, "got {mean_on_fit}");
        assert!(
            (mean_outlier - 6.0 / 21.0).abs() < 0.02,
            "got {mean_outlier}"
        );
    }

    /// Under unit weights the weighted draw must equal the unweighted shelf
    /// draw bit for bit (same RNG stream, same Gamma parameters).
    #[test]
    fn unit_weights_match_global_sigma_exactly() {
        let n = 16;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let x = crate::engine::data::Data::new(xs, n, 1).unwrap();
        let y: Vec<f64> = (0..n).map(|i| 0.4 * (i as f64 / n as f64) - 0.2).collect();
        let fit = vec![0.05_f64; n];
        let move_set = crate::extensions::moves::default_move_set().unwrap();
        let assigner = crate::extensions::distance::default_assigner(vec![
            crate::engine::data::Metric::Euclidean,
        ]);
        let coord_dists: Vec<std::sync::Arc<dyn crate::extensions::coord::CoordinateDistribution>> =
            vec![std::sync::Arc::new(
                crate::extensions::coord::EuclideanNormal::new(0.8),
            )];
        let weights_enc = [1.0_f64];
        let unit = vec![1.0_f64; n];
        let make_ctx = |response_weights| ScaleCtx {
            y: &y,
            fit: &fit,
            x: &x,
            move_set: &move_set,
            assigner: assigner.as_ref(),
            omega: 0.5,
            lambda_c: 3.0,
            nu: 6.0,
            lambda: 0.02,
            p_enc: 1,
            coord_dists: &coord_dists,
            weights_enc: &weights_enc,
            response_weights,
        };

        let mut weighted = WeightedGlobalSigma::new(6.0, 0.02);
        let mut rng_a = ChaCha8Rng::seed_from_u64(3);
        ScaleModel::update(&mut weighted, &make_ctx(Some(unit.as_slice())), &mut rng_a).unwrap();

        let mut unweighted = crate::extensions::scale::GlobalSigma::new(6.0, 0.02);
        let mut rng_b = ChaCha8Rng::seed_from_u64(3);
        ScaleModel::update(&mut unweighted, &make_ctx(None), &mut rng_b).unwrap();

        assert_eq!(weighted.sigma_sq(), unweighted.sigma_sq());
    }

    /// Down-weighting an outlier must shrink the weighted RSS and therefore
    /// (same RNG stream) the σ² draw, relative to unit weights.
    #[test]
    fn downweighted_outlier_shrinks_the_draw() {
        let n = 16;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let x = crate::engine::data::Data::new(xs, n, 1).unwrap();
        let mut y: Vec<f64> = vec![0.0; n];
        y[7] = 2.0; // one gross outlier
        let fit = vec![0.0_f64; n];
        let move_set = crate::extensions::moves::default_move_set().unwrap();
        let assigner = crate::extensions::distance::default_assigner(vec![
            crate::engine::data::Metric::Euclidean,
        ]);
        let coord_dists: Vec<std::sync::Arc<dyn crate::extensions::coord::CoordinateDistribution>> =
            vec![std::sync::Arc::new(
                crate::extensions::coord::EuclideanNormal::new(0.8),
            )];
        let weights_enc = [1.0_f64];
        let mut soft = vec![1.0_f64; n];
        soft[7] = 0.05;
        let unit = vec![1.0_f64; n];
        let make_ctx = |response_weights| ScaleCtx {
            y: &y,
            fit: &fit,
            x: &x,
            move_set: &move_set,
            assigner: assigner.as_ref(),
            omega: 0.5,
            lambda_c: 3.0,
            nu: 6.0,
            lambda: 0.02,
            p_enc: 1,
            coord_dists: &coord_dists,
            weights_enc: &weights_enc,
            response_weights,
        };

        let mut a = WeightedGlobalSigma::new(6.0, 0.02);
        let mut rng_a = ChaCha8Rng::seed_from_u64(9);
        ScaleModel::update(&mut a, &make_ctx(Some(soft.as_slice())), &mut rng_a).unwrap();
        let mut b = WeightedGlobalSigma::new(6.0, 0.02);
        let mut rng_b = ChaCha8Rng::seed_from_u64(9);
        ScaleModel::update(&mut b, &make_ctx(Some(unit.as_slice())), &mut rng_b).unwrap();
        assert!(a.sigma_sq() < b.sigma_sq());
    }
}
