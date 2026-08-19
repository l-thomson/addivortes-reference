//! The fitted model (`FittedAddiVortes`), the posterior container
//! (`PosteriorSamples`), and the prediction surface.

use std::sync::Arc;

use crate::engine::config::AddiVortesConfig;
use crate::engine::data::{self, Data, Warning};
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::sampler::{Draw, Sampler};
use crate::engine::scaler::FittedScaler;
use crate::engine::tessellation::Tessellation;
use crate::extensions::basis::CellBasis;
use crate::extensions::distance::CellAssigner;
use crate::extensions::membership::MembershipKernel;

/// Pure container of **scaled-space** posterior draws: no
/// X-taking methods; every numeric accessor is in the sampler's scaled
/// coordinate system.
///
/// With the `serde` feature, deserialisation validates the container (at
/// least one draw, a constant ensemble size, strictly positive finite σ²
/// draws, finite tessellation values). A corrupt payload is rejected, never
/// a panic or a NaN prediction.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PosteriorSamplesParts")
)]
pub struct PosteriorSamples {
    /// σ² per kept draw (scaled space).
    sigma_sqs: Vec<f64>,
    /// m tessellations per kept draw (scaled space).
    draws: Vec<Vec<Tessellation>>,
}

impl PosteriorSamples {
    /// Number of kept posterior draws.
    pub fn n_draws(&self) -> usize {
        self.sigma_sqs.len()
    }

    /// The σ² draws (**scaled space**: multiply the square root by the
    /// response range for the response-scale error SD, or use
    /// [`FittedAddiVortes::sigma`]).
    pub fn sigma_sq(&self) -> &[f64] {
        &self.sigma_sqs
    }

    /// The m tessellations of draw `draw` (**scaled space**).
    ///
    /// # Panics
    ///
    /// Panics if `draw >= n_draws()` (programming error, like slice indexing).
    pub fn tessellations(&self, draw: usize) -> &[Tessellation] {
        &self.draws[draw]
    }

    /// Iterate the draws in order (borrowing).
    pub fn iter_draws(&self) -> Draws<'_> {
        Draws {
            samples: self,
            index: 0,
        }
    }

    /// Decompose into `(sigma_sqs, draws)` without copying (**scaled space**).
    pub fn into_parts(self) -> (Vec<f64>, Vec<Vec<Tessellation>>) {
        (self.sigma_sqs, self.draws)
    }
}

/// Serde shadow of [`PosteriorSamples`]: deserialisation lands here first,
/// then through the validating `TryFrom` below.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct PosteriorSamplesParts {
    sigma_sqs: Vec<f64>,
    draws: Vec<Vec<Tessellation>>,
}

#[cfg(feature = "serde")]
impl TryFrom<PosteriorSamplesParts> for PosteriorSamples {
    type Error = crate::engine::error::SavedModelError;

    fn try_from(parts: PosteriorSamplesParts) -> std::result::Result<Self, Self::Error> {
        let bad = |reason: String| Err(crate::engine::error::SavedModelError(reason));
        if parts.sigma_sqs.len() != parts.draws.len() {
            return bad("one sigma-squared value per draw".into());
        }
        if parts.draws.is_empty() {
            return bad("a posterior needs at least one draw".into());
        }
        if let Some(s) = parts
            .sigma_sqs
            .iter()
            .find(|s| !(s.is_finite() && **s > 0.0))
        {
            return bad(format!(
                "sigma-squared draws must be finite and positive, got {s}"
            ));
        }
        let m = parts.draws[0].len();
        if m == 0 {
            return bad("each draw needs at least one tessellation".into());
        }
        for (index, draw) in parts.draws.iter().enumerate() {
            if draw.len() != m {
                return bad(format!(
                    "draw {index} has {found} tessellations but draw 0 has {m}",
                    found = draw.len()
                ));
            }
            // Tessellation deserialisation already enforced the structural
            // invariants; predictions additionally require finite values.
            for tessellation in draw {
                if tessellation.centres().iter().any(|v| !v.is_finite())
                    || tessellation.mus().iter().any(|v| !v.is_finite())
                {
                    return bad(format!("draw {index} contains a non-finite tessellation"));
                }
            }
        }
        Ok(Self {
            sigma_sqs: parts.sigma_sqs,
            draws: parts.draws,
        })
    }
}

/// Borrowing iterator over posterior draws.
#[derive(Debug)]
pub struct Draws<'a> {
    samples: &'a PosteriorSamples,
    index: usize,
}

impl<'a> Iterator for Draws<'a> {
    type Item = Draw<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.samples.n_draws() {
            return None;
        }
        let draw = Draw {
            sigma_sq: self.samples.sigma_sqs[self.index],
            tessellations: &self.samples.draws[self.index],
        };
        self.index += 1;
        Some(draw)
    }
}

/// A central credible interval for one prediction (passive record; both ends
/// **response scale**).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CredibleInterval {
    /// Lower end (response scale).
    pub lower: f64,
    /// Upper end (response scale).
    pub upper: f64,
}

/// A central posterior-predictive interval for one new observation at a
/// prediction input (passive record; both ends **response scale**). Always at
/// least as wide as [`CredibleInterval`] at the same level: it folds each
/// draw's error SD into that draw's fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionInterval {
    /// Lower end (response scale).
    pub lower: f64,
    /// Upper end (response scale).
    pub upper: f64,
}

/// Quantile predictions: row-major `n_rows × n_probs` (observation-major:
/// `values()[row * n_probs + probability_index]`), **response scale**.
/// An empty prediction input yields a zero-row result (valid by design).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantilePredictions {
    values: Vec<f64>,
    probs: Vec<f64>,
    n_rows: usize,
}

impl QuantilePredictions {
    /// The requested probabilities, in the caller's order.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Number of predicted observations (rows).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Row-major `n_rows × probs().len()` quantile values (**response scale**);
    /// `values()[row * probs().len() + j]` is the `probs()[j]` quantile for
    /// observation `row`.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Decompose into `(values, probs, n_rows)` without copying (values are
    /// **response scale**, row-major).
    pub fn into_parts(self) -> (Vec<f64>, Vec<f64>, usize) {
        (self.values, self.probs, self.n_rows)
    }
}

/// The response family of a model (the predict-side half;
/// one fitted type with a family payload): selects
/// the link and the per-variant predictive distribution, so classification
/// models predict on their own scale instead of hand-composing.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResponseFamily {
    /// The paper's Gaussian regression: identity link, predictions on the
    /// **response scale**, predictive distribution N(fitᵈ, σᵈ²) per draw.
    #[default]
    Gaussian,
    /// Binary-AddiVortes (Albert–Chib probit): {0, 1} labels in, predictions
    /// on the **probability scale** (P(Y = 1 | x) = Φ(Fᵈ(x)) per draw on
    /// the σ ≡ 1 latent scale), with the Bernoulli mixture as the predictive
    /// distribution.
    BinaryProbit,
    /// Robust regression with Student-t errors (the scale-mixture
    /// augmentation): identity link, predictions on the **response scale**,
    /// predictive distribution fitᵈ + σᵈ·t_ν′ per draw. Fit-side the family
    /// assembles the Student-t augmentation step, the weight-aware mean
    /// family and the precision-weighted σ² draw together (the
    /// crate-internal pairing rule).
    RobustT {
        /// Error degrees of freedom ν′ (a dimensionless count; must be
        /// finite and strictly positive; validated at fit and at load).
        /// Small values (3–8) are the robust regime; ν′ → ∞ recovers
        /// Gaussian errors.
        df: f64,
    },
}

/// A fitted AddiVortes model: owns the posterior, the scaler, the (consumed)
/// configuration, and the assigner. Fully self-contained; the training data
/// is not retained.
///
/// # Save and load (`serde` feature)
///
/// A model fitted with built-in extension points only round-trips through any serde
/// format: predictions from the loaded model are bit-identical to the
/// original's. Binary formats are exact by construction; for JSON, enable
/// `serde_json`'s `float_roundtrip` feature: its default float parser can be
/// one ULP off, which silently breaks bit-identical predictions. A model
/// carrying a custom extension point (move set, coordinate law, assigner,
/// inclusion, cell model, kernel step or scale model) refuses to serialise,
/// since trait objects have no portable form; persist its
/// [`into_parts`](FittedAddiVortes::into_parts) values instead and rebuild in
/// code. Loading validates everything (a corrupt payload is an
/// error, never a panic) and reconstructs the built-in per-column assigner
/// from the saved scaler's encoded metrics.
#[derive(Debug, Clone)]
pub struct FittedAddiVortes {
    posterior: PosteriorSamples,
    scaler: FittedScaler,
    config: AddiVortesConfig,
    assigner: Arc<dyn CellAssigner>,
    /// Soft-membership kernel; `None` = hard assignment at predict.
    membership: Option<Arc<dyn MembershipKernel>>,
    /// Cell basis; `None` = scalar cell payloads at predict.
    basis: Option<Arc<dyn CellBasis>>,
    /// True when any component axis was set explicitly at fit; the
    /// serialisation refusal reads it.
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    custom_components: bool,
    warnings: Vec<Warning>,
    in_sample_rmse: f64,
}

impl FittedAddiVortes {
    /// Posterior-mean predictions for raw-scale input `x`, on the family's
    /// own scale: **response scale** for
    /// [`ResponseFamily::Gaussian`], **probability scale**
    /// (P(Y = 1 | x) through the probit link) for
    /// [`ResponseFamily::BinaryProbit`]. Runs the predict boundary
    /// checks, applies the fitted encoding (unseen categorical levels
    /// error), and averages every kept draw's linked prediction.
    #[must_use = "predictions are returned, not stored"]
    pub fn predict(&self, x: &Data) -> Result<Vec<f64>> {
        let per_draw = self.predictions_by_draw(x)?;
        let n = x.n_rows();
        let n_draws = self.posterior.n_draws() as f64;
        let mut means = vec![0.0_f64; n];
        for draw in &per_draw {
            for (mean, value) in means.iter_mut().zip(draw) {
                *mean += value;
            }
        }
        for mean in &mut means {
            *mean /= n_draws;
            // finite-in ⇒ finite-out contract: μs are asserted
            // finite at draw time and unscaling is affine over a finite range,
            // so a non-finite prediction is impossible for validated input.
            assert!(
                mean.is_finite(),
                "predictions must be finite for validated input"
            );
        }
        Ok(means)
    }

    /// Per-observation posterior quantiles at `probs` (each must be finite and
    /// inside (0, 1)); **response scale**, linear interpolation between order
    /// statistics. Empty `x` yields a zero-row result.
    #[must_use = "predictions are returned, not stored"]
    pub fn predict_quantiles(&self, x: &Data, probs: &[f64]) -> Result<QuantilePredictions> {
        if probs.is_empty() {
            return Err(AddiVortesError::InvalidQuantileProb { value: f64::NAN });
        }
        for &p in probs {
            if !(p.is_finite() && p > 0.0 && p < 1.0) {
                return Err(AddiVortesError::InvalidQuantileProb { value: p });
            }
        }
        let per_draw = self.predictions_by_draw(x)?;
        let n = x.n_rows();
        let n_draws = per_draw.len();
        let mut values = Vec::with_capacity(n * probs.len());
        let mut sorted = vec![0.0_f64; n_draws];
        for row in 0..n {
            for (slot, draw) in sorted.iter_mut().zip(&per_draw) {
                *slot = draw[row];
            }
            sorted.sort_by(f64::total_cmp);
            for &p in probs {
                values.push(quantile_sorted(&sorted, p));
            }
        }
        Ok(QuantilePredictions {
            values,
            probs: probs.to_vec(),
            n_rows: n,
        })
    }

    /// Central credible interval at `level` (e.g. 0.9 → the 5% and 95%
    /// posterior quantiles): sugar over [`predict_quantiles`]; **response
    /// scale**. `level` must be finite and inside (0, 1).
    ///
    /// [`predict_quantiles`]: FittedAddiVortes::predict_quantiles
    #[must_use = "predictions are returned, not stored"]
    pub fn credible_interval(&self, x: &Data, level: f64) -> Result<Vec<CredibleInterval>> {
        if !(level.is_finite() && level > 0.0 && level < 1.0) {
            return Err(AddiVortesError::InvalidQuantileProb { value: level });
        }
        let tail = 0.5 * (1.0 - level);
        let quantiles = self.predict_quantiles(x, &[tail, 1.0 - tail])?;
        let (values, _, n_rows) = quantiles.into_parts();
        Ok((0..n_rows)
            .map(|row| CredibleInterval {
                lower: values[row * 2],
                upper: values[row * 2 + 1],
            })
            .collect())
    }

    /// Central posterior-predictive interval at `level` for a new
    /// observation at each row of `x` (**response scale**). The predictive
    /// law is the equal-weight mixture over kept draws of N(fitᵈ(x), σᵈ²)
    /// (each draw's ensemble fit and error SD); the interval ends are its
    /// (1−level)/2 and (1+level)/2 quantiles, solved by deterministic
    /// bisection on the mixture CDF.
    ///
    /// [`credible_interval`] at the same level is the interval for the mean
    /// fit only (no noise term) and is contained in this one. `level` must be
    /// finite and inside (0, 1).
    ///
    /// [`credible_interval`]: FittedAddiVortes::credible_interval
    #[must_use = "predictions are returned, not stored"]
    pub fn prediction_interval(&self, x: &Data, level: f64) -> Result<Vec<PredictionInterval>> {
        if !(level.is_finite() && level > 0.0 && level < 1.0) {
            return Err(AddiVortesError::InvalidQuantileProb { value: level });
        }
        let per_draw = self.predictions_by_draw(x)?;
        let sigmas = self.sigma();
        let tail = 0.5 * (1.0 - level);
        let n = x.n_rows();
        let mut intervals = Vec::with_capacity(n);
        let mut fits = vec![0.0_f64; per_draw.len()];
        for row in 0..n {
            for (fit, draw) in fits.iter_mut().zip(&per_draw) {
                *fit = draw[row];
            }
            let (lower, upper) = match self.config.family {
                ResponseFamily::Gaussian => (
                    predictive_quantile(&fits, &sigmas, tail),
                    predictive_quantile(&fits, &sigmas, 1.0 - tail),
                ),
                // The per-variant predictive: the Bernoulli
                // mixture over per-draw probabilities; its quantiles are
                // the labels 0/1 (P(Y = 0) = mean(1 − p_d)).
                ResponseFamily::BinaryProbit => {
                    let p_zero = fits.iter().map(|p| 1.0 - p).sum::<f64>() / fits.len() as f64;
                    let quantile = |q: f64| if q <= p_zero { 0.0 } else { 1.0 };
                    (quantile(tail), quantile(1.0 - tail))
                }
                // Robust-t: the equal-weight t mixture Σ_d t_ν′(fitᵈ, σᵈ)/D,
                // the same bisection as the Gaussian arm, with Student-t
                // component CDFs (and a self-widening bracket: t tails are
                // polynomial, so no fixed SD pad can pin them).
                ResponseFamily::RobustT { df } => (
                    t_predictive_quantile(&fits, &sigmas, df, tail),
                    t_predictive_quantile(&fits, &sigmas, df, 1.0 - tail),
                ),
            };
            intervals.push(PredictionInterval { lower, upper });
        }
        Ok(intervals)
    }

    /// The posterior error-SD draws on the **response scale**: √σ² rescaled by
    /// the training response range (the y-scaling is affine). Under
    /// [`ResponseFamily::BinaryProbit`] the scale is pinned to the σ ≡ 1
    /// latent scale, so every entry is the constant 1 (the {0, 1} response
    /// range is 1).
    pub fn sigma(&self) -> Vec<f64> {
        let range = self.scaler.y_max() - self.scaler.y_min();
        self.posterior
            .sigma_sq()
            .iter()
            .map(|s| s.sqrt() * range)
            .collect()
    }

    /// Typed fit-time warnings (e.g. p > n), never printed.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// The kept posterior draws (**scaled space**).
    pub fn posterior(&self) -> &PosteriorSamples {
        &self.posterior
    }

    /// The fitted scaler (raw ↔ scaled mappings and the encoding).
    pub fn scaler(&self) -> &FittedScaler {
        &self.scaler
    }

    /// The configuration this model was fitted with.
    pub fn config(&self) -> &AddiVortesConfig {
        &self.config
    }

    /// **Response-scale** RMSE of the posterior-mean prediction against the
    /// raw training response.
    pub fn in_sample_rmse(&self) -> f64 {
        self.in_sample_rmse
    }

    /// Posterior variable-inclusion proportions, one per caller-visible
    /// (pre-encoding) column. Dimensionless shares in [0, 1] summing to 1:
    /// the fraction of all active tessellation dimensions across the kept
    /// draws that map to each column. One-hot groups aggregate onto their
    /// source column, mirroring the sampler's per-sweep usage counts.
    ///
    /// The BART-style covariate-importance summary: a column the ensemble
    /// never partitions on scores 0; under the uniform inclusion default an
    /// uninformative fit spreads mass evenly.
    pub fn variable_inclusion_proportions(&self) -> Vec<f64> {
        let col_map = self.scaler.col_map();
        let mut counts = vec![0u64; self.scaler.n_raw_cols()];
        let mut total = 0u64;
        for draw in self.posterior.iter_draws() {
            for tessellation in draw.tessellations {
                for &dim in tessellation.dims() {
                    counts[col_map[dim]] += 1;
                    total += 1;
                }
            }
        }
        // Unreachable via fit (≥ 1 draw × m ≥ 1 tessellations × ≥ 1 dim), but
        // never divide by zero.
        if total == 0 {
            return vec![0.0; counts.len()];
        }
        counts.iter().map(|&c| c as f64 / total as f64).collect()
    }

    /// Per-draw predictions for raw-scale input `x`, draw-major
    /// (`n_draws() × x.n_rows()`), each on the family's own scale:
    /// **response scale** for [`ResponseFamily::Gaussian`] and
    /// [`ResponseFamily::RobustT`], **probability scale** for
    /// [`ResponseFamily::BinaryProbit`]. The full, unreduced posterior of
    /// the fit. [`predict`](FittedAddiVortes::predict) is this matrix's
    /// per-row mean; posterior-predictive workflows (PPC plots, LOO/WAIC,
    /// per-draw partial dependence) consume the draw axis directly.
    #[must_use = "predictions are returned, not stored"]
    pub fn predict_draws(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        self.predictions_by_draw(x)
    }

    /// Pointwise log-likelihood ln p(yᵢ | draw d) for raw-scale input `x`
    /// against observed responses `y` (**response scale**), draw-major
    /// (`n_draws() × x.n_rows()`), the matrix PSIS-LOO and WAIC
    /// estimators consume. Per draw the density is the family's own
    /// predictive law: N(fitᵈ(xᵢ), σᵈ²) for
    /// [`ResponseFamily::Gaussian`], the location-scale Student-t
    /// fitᵈ + σᵈ·t_ν′ for [`ResponseFamily::RobustT`], and
    /// Bernoulli(pᵈ(xᵢ)) for [`ResponseFamily::BinaryProbit`], with pᵈ
    /// clamped into [ε, 1 − ε] (ε = f64 machine epsilon) so a saturated
    /// probability yields a large finite value rather than −∞.
    ///
    /// `y` must be finite, row-matched to `x`, and {0, 1}-valued under
    /// [`ResponseFamily::BinaryProbit`] (the fit boundary's own rule).
    #[must_use = "log-likelihoods are returned, not stored"]
    pub fn log_likelihood(&self, x: &Data, y: &[f64]) -> Result<Vec<Vec<f64>>> {
        if y.len() != x.n_rows() {
            return Err(AddiVortesError::RowCountMismatch {
                y_len: y.len(),
                x_rows: x.n_rows(),
            });
        }
        if let Some(row) = y.iter().position(|v| !v.is_finite()) {
            return Err(AddiVortesError::NonFiniteResponse { row });
        }
        if self.config.family == ResponseFamily::BinaryProbit {
            if let Some(bad) = y.iter().find(|v| **v != 0.0 && **v != 1.0) {
                return Err(AddiVortesError::InvalidHyperparameter {
                    name: "response_family".into(),
                    reason: format!("BinaryProbit needs a {{0, 1}} response; found {bad}"),
                });
            }
        }
        let per_draw = self.predictions_by_draw(x)?;
        let sigmas = self.sigma();
        Ok(per_draw
            .iter()
            .zip(&sigmas)
            .map(|(fits, &sigma)| {
                y.iter()
                    .zip(fits)
                    .map(|(&yi, &fit)| match self.config.family {
                        ResponseFamily::Gaussian => gaussian_log_density(yi, fit, sigma),
                        ResponseFamily::RobustT { df } => student_t_log_density(yi, fit, sigma, df),
                        ResponseFamily::BinaryProbit => bernoulli_log_density(yi, fit),
                    })
                    .collect()
            })
            .collect())
    }

    /// Decompose into the posterior and the scaler.
    pub fn into_parts(self) -> (PosteriorSamples, FittedScaler) {
        (self.posterior, self.scaler)
    }

    /// Response-scale predictions per kept draw (draw-major), after the
    /// predict boundary checks and encoding.
    fn predictions_by_draw(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        let raw_metrics = self
            .config
            .metrics
            .clone()
            .unwrap_or_else(|| vec![crate::Metric::Euclidean; self.scaler.n_raw_cols()]);
        data::validate_predict(x, self.scaler.n_raw_cols(), &raw_metrics)?;
        let x_enc = self.scaler.apply_x(x)?;
        let n = x_enc.n_rows();

        let mut per_draw = Vec::with_capacity(self.posterior.n_draws());
        for draw in self.posterior.iter_draws() {
            let mut sums = vec![0.0_f64; n];
            for tessellation in draw.tessellations {
                match &self.membership {
                    // Hard membership: nearest-centre assignment. The cell's
                    // contribution is its scalar μ, or the basis inner product
                    // z(xᵢ)·β_k when a cell basis is configured.
                    None => {
                        let assignment = self.assigner.assign_cells(&x_enc, tessellation)?;
                        match &self.basis {
                            None => {
                                for (sum, &cell) in sums.iter_mut().zip(&assignment) {
                                    *sum += tessellation.mus()[cell];
                                }
                            }
                            Some(basis) => {
                                let q = basis.q();
                                let mut z = vec![0.0_f64; q];
                                for (i, (sum, &cell)) in
                                    sums.iter_mut().zip(&assignment).enumerate()
                                {
                                    basis.row(x_enc.row(i), &mut z);
                                    *sum += tessellation
                                        .cell_payload(cell)
                                        .iter()
                                        .zip(&z)
                                        .map(|(beta, zi)| beta * zi)
                                        .sum::<f64>();
                                }
                            }
                        }
                    }
                    // Soft membership: predict-time assignment
                    // dispatches through the same assigner + membership kernel
                    // as fitting, giving membership-weighted
                    // cell outputs.
                    Some(kernel) => {
                        let membership = crate::engine::backfit::compute_memberships(
                            self.assigner.as_ref(),
                            kernel.as_ref(),
                            &x_enc,
                            tessellation,
                        )?;
                        let b = tessellation.n_cells();
                        for (i, sum) in sums.iter_mut().enumerate() {
                            let phi = &membership[i * b..(i + 1) * b];
                            *sum += phi
                                .iter()
                                .zip(tessellation.mus())
                                .map(|(p, mu)| p * mu)
                                .sum::<f64>();
                        }
                    }
                }
            }
            // The family link (predict side): the identity-link
            // families (Gaussian, robust-t) unscale the ensemble sum to the
            // response scale; BinaryProbit maps the σ ≡ 1 latent-scale fit
            // through Φ to the probability scale.
            match self.config.family {
                ResponseFamily::Gaussian | ResponseFamily::RobustT { .. } => {
                    for value in &mut sums {
                        *value = self.scaler.unscale_y_value(*value);
                    }
                }
                ResponseFamily::BinaryProbit => {
                    for value in &mut sums {
                        *value = normal_cdf(*value);
                    }
                }
            }
            per_draw.push(sums);
        }
        Ok(per_draw)
    }
}

/// Standard normal CDF, Φ(z) = erfc(−z/√2)/2. Precision is kept in both
/// tails, and the pinned `mathsfn` path keeps the result bit-reproducible
/// across platforms.
fn normal_cdf(z: f64) -> f64 {
    0.5 * crate::engine::mathsfn::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Log-density at `y` of N(`fit`, `sigma`²) (all **response scale**). A zero
/// `sigma` (unreachable from a real fit, whose σ² draws are strictly
/// positive) degenerates to the point mass at `fit` (±∞ log-density),
/// mirroring [`mixture_cdf`]'s unit-step convention.
fn gaussian_log_density(y: f64, fit: f64, sigma: f64) -> f64 {
    if sigma > 0.0 {
        let z = (y - fit) / sigma;
        -0.5 * crate::engine::mathsfn::ln(2.0 * std::f64::consts::PI)
            - crate::engine::mathsfn::ln(sigma)
            - 0.5 * z * z
    } else if y == fit {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    }
}

/// Log-density at `y` of the location-scale Student-t `fit` + `sigma`·t_ν′
/// (all **response scale**; `df` = ν′). Zero `sigma` degenerates to the
/// point mass, mirroring [`gaussian_log_density`].
fn student_t_log_density(y: f64, fit: f64, sigma: f64, df: f64) -> f64 {
    if sigma > 0.0 {
        let z = (y - fit) / sigma;
        crate::engine::mathsfn::lgamma(0.5 * (df + 1.0))
            - crate::engine::mathsfn::lgamma(0.5 * df)
            - 0.5 * crate::engine::mathsfn::ln(df * std::f64::consts::PI)
            - crate::engine::mathsfn::ln(sigma)
            - 0.5 * (df + 1.0) * crate::engine::mathsfn::ln_1p(z * z / df)
    } else if y == fit {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    }
}

/// Bernoulli log-density of label `y` ∈ {0, 1} under success probability
/// `p` (**probability scale**), with `p` clamped into [ε, 1 − ε]
/// (ε = f64 machine epsilon) so saturated probabilities stay finite.
fn bernoulli_log_density(y: f64, p: f64) -> f64 {
    let p = p.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    if y == 1.0 {
        crate::engine::mathsfn::ln(p)
    } else {
        crate::engine::mathsfn::ln_1p(-p)
    }
}

/// CDF at `t` of the equal-weight normal mixture Σ_d N(fit_d, σ_d²)/D
/// (response scale). A zero σ_d (unreachable from a real fit, whose σ² draws
/// are strictly positive) contributes its point mass as a unit step rather
/// than a NaN.
fn mixture_cdf(fits: &[f64], sigmas: &[f64], t: f64) -> f64 {
    let mut sum = 0.0_f64;
    for (&fit, &sigma) in fits.iter().zip(sigmas) {
        sum += if sigma > 0.0 {
            normal_cdf((t - fit) / sigma)
        } else {
            f64::from(t >= fit)
        };
    }
    sum / fits.len() as f64
}

/// CDF at `t` of the equal-weight Student-t mixture Σ_d (fit_d + σ_d·T_ν′)/D
/// (response scale): the robust-t predictive. Zero σ_d contributes a unit
/// step, mirroring [`mixture_cdf`].
fn t_mixture_cdf(fits: &[f64], sigmas: &[f64], df: f64, t: f64) -> f64 {
    let mut sum = 0.0_f64;
    for (&fit, &sigma) in fits.iter().zip(sigmas) {
        sum += if sigma > 0.0 {
            student_t_cdf((t - fit) / sigma, df)
        } else {
            f64::from(t >= fit)
        };
    }
    sum / fits.len() as f64
}

/// Quantile of the robust-t predictive mixture: the same deterministic
/// bisection as [`predictive_quantile`], with one difference forced by the
/// polynomial t tails: no fixed SD pad can push the CDF to exactly 0/1, so
/// the bracket starts at the Gaussian pad and doubles itself outward
/// until it straddles `p` (each side independently; the doubling is
/// deterministic, so the result still is).
fn t_predictive_quantile(fits: &[f64], sigmas: &[f64], df: f64, p: f64) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sigma_max = 0.0_f64;
    for (&fit, &sigma) in fits.iter().zip(sigmas) {
        lo = lo.min(fit);
        hi = hi.max(fit);
        sigma_max = sigma_max.max(sigma);
    }
    let pad = 39.0 * sigma_max.max(f64::MIN_POSITIVE);
    let (mut lo, mut hi) = (lo - pad, hi + pad);
    let mut width = pad;
    for _ in 0..1024 {
        if t_mixture_cdf(fits, sigmas, df, lo) < p {
            break;
        }
        lo -= width;
        width *= 2.0;
    }
    let mut width = pad;
    for _ in 0..1024 {
        if t_mixture_cdf(fits, sigmas, df, hi) > p {
            break;
        }
        hi += width;
        width *= 2.0;
    }
    for _ in 0..128 {
        let mid = 0.5 * (lo + hi);
        if t_mixture_cdf(fits, sigmas, df, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Standard Student-t CDF with ν′ = `df` degrees of freedom, through the
/// regularised incomplete beta: F(x) = 1 − I_A(ν′/2, 1/2)/2 for x > 0 with
/// A = ν′/(ν′ + x²), and by symmetry below zero. Routed through `mathsfn`
/// end to end (probability, bit-reproducible across platforms).
fn student_t_cdf(x: f64, df: f64) -> f64 {
    if x == 0.0 {
        return 0.5;
    }
    let a = df / (df + x * x);
    let tail = 0.5 * incomplete_beta(0.5 * df, 0.5, a);
    if x > 0.0 { 1.0 - tail } else { tail }
}

/// The regularised incomplete beta I_x(a, b) (a probability): the
/// continued-fraction evaluation (modified Lentz), converging for the
/// convergent side of the split at x = (a+1)/(a+b+2) and reflected via
/// I_x(a, b) = 1 − I_{1−x}(b, a) on the other. Transcendentals through
/// `mathsfn` (determinism contract).
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x));
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_prefactor = crate::engine::mathsfn::lgamma(a + b)
        - crate::engine::mathsfn::lgamma(a)
        - crate::engine::mathsfn::lgamma(b)
        + a * crate::engine::mathsfn::ln(x)
        + b * crate::engine::mathsfn::ln_1p(-x);
    let prefactor = crate::engine::mathsfn::exp(ln_prefactor);
    if x < (a + 1.0) / (a + b + 2.0) {
        prefactor * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - prefactor * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// The incomplete-beta continued fraction (Numerical Recipes `betacf`,
/// modified Lentz): a dimensionless factor consumed by
/// [`incomplete_beta`]. Deterministic float arithmetic only.
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    const EPS: f64 = 3e-16;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Quantile of the predictive mixture by deterministic bisection on its
/// monotone CDF (the same approach as the χ² quantile in `scale`); `p` in
/// (0, 1), result on the response scale. The bracket pads the extreme fits by
/// 39 SDs: Φ(−39) underflows to exactly 0, so the CDF at the bracket ends
/// lies beyond any representable `p`.
fn predictive_quantile(fits: &[f64], sigmas: &[f64], p: f64) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sigma_max = 0.0_f64;
    for (&fit, &sigma) in fits.iter().zip(sigmas) {
        lo = lo.min(fit);
        hi = hi.max(fit);
        sigma_max = sigma_max.max(sigma);
    }
    let pad = 39.0 * sigma_max;
    let (mut lo, mut hi) = (lo - pad, hi + pad);
    for _ in 0..128 {
        let mid = 0.5 * (lo + hi);
        if mixture_cdf(fits, sigmas, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Linear-interpolation quantile of an ascending-sorted slice (the classic
/// "type 7" definition: h = p(n−1), interpolate neighbours).
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let fraction = h - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * fraction
}

/// The fit convenience loop: validate → sampler → burn-in discard →
/// thinned collection → self-contained fitted model. Every component choice on the
/// config (move set, coordinate laws, assigner, inclusion, deep seam) is
/// honoured by the [`Sampler`] constructor.
pub(crate) fn fit(config: AddiVortesConfig, x: &Data, y: &[f64]) -> Result<FittedAddiVortes> {
    config.validate()?;
    let sampler = Sampler::new(config, x, y)?;
    fit_sampler(sampler, x, y)
}

/// The shared burn-in/thinning collection loop over a constructed sampler.
pub(crate) fn fit_sampler(mut sampler: Sampler, x: &Data, y: &[f64]) -> Result<FittedAddiVortes> {
    for _ in 0..sampler.config().burn_in {
        sampler.step()?;
    }
    let n_draws = sampler.config().n_draws;
    let thinning = sampler.config().thinning;
    let mut sigma_sqs = Vec::with_capacity(n_draws);
    let mut draws = Vec::with_capacity(n_draws);
    for _ in 0..n_draws {
        // Keep the last of each `thinning`-sized block.
        for _ in 0..thinning - 1 {
            sampler.step()?;
        }
        let draw = sampler.step()?;
        sigma_sqs.push(draw.sigma_sq);
        draws.push(draw.tessellations.to_vec());
    }
    FittedAddiVortes::from_parts(sampler, x, y, sigma_sqs, draws)
}

impl FittedAddiVortes {
    /// The "keep" verb: package draws collected over a caller-driven
    /// [`Sampler`] loop as a self-contained fitted model. Crate-internal:
    /// this is the path a model file takes when its sweep needs work between
    /// steps that `fit` cannot express (an augmentation drawn in the model
    /// file, an outer Gibbs block), so the burn-in/thinning loop lives with
    /// the caller and only the packaging is shared. `fit_sampler` itself
    /// finishes through here, so the two paths cannot drift.
    ///
    /// `sigma_sqs` and `draws` are the kept sweeps, in order, exactly as
    /// [`Sampler::step`] returned them; `x`/`y` are the training data, used
    /// once for the in-sample RMSE and not retained.
    pub(crate) fn from_parts(
        sampler: Sampler,
        x: &Data,
        y: &[f64],
        sigma_sqs: Vec<f64>,
        draws: Vec<Vec<Tessellation>>,
    ) -> Result<Self> {
        if draws.is_empty() {
            return Err(AddiVortesError::InvalidHyperparameter {
                name: "draws".into(),
                reason: "a fitted model needs at least one kept draw".into(),
            });
        }
        if sigma_sqs.len() != draws.len() {
            return Err(AddiVortesError::InvalidHyperparameter {
                name: "draws".into(),
                reason: format!(
                    "{} sigma-squared values for {} kept draws",
                    sigma_sqs.len(),
                    draws.len()
                ),
            });
        }
        let posterior = PosteriorSamples { sigma_sqs, draws };
        let parts = sampler.into_fitted_parts();

        let mut model = FittedAddiVortes {
            posterior,
            scaler: parts.scaler,
            config: parts.config,
            assigner: parts.assigner,
            membership: parts.membership,
            basis: parts.basis,
            custom_components: parts.custom_components,
            warnings: parts.warnings,
            in_sample_rmse: 0.0,
        };
        // Response-scale RMSE of the posterior mean on the training data.
        let predictions = model.predict(x)?;
        let mut sum_sq = 0.0_f64;
        for (prediction, &observed) in predictions.iter().zip(y) {
            let residual = prediction - observed;
            sum_sq += residual * residual;
        }
        model.in_sample_rmse = (sum_sq / y.len() as f64).sqrt();
        Ok(model)
    }
}

/// Save/load of the fitted model (`serde` feature): a versioned on-disk form
/// holding the config's plain fields plus the validated value types.
#[cfg(feature = "serde")]
mod persist {
    use std::sync::Arc;

    use super::{FittedAddiVortes, PosteriorSamples};
    use crate::engine::config::AddiVortesConfig;
    use crate::engine::data::{Metric, Warning};
    use crate::engine::error::SavedModelError;
    use crate::engine::scaler::FittedScaler;

    /// The on-disk form. `format` is bumped on any layout change; a loader
    /// rejects formats it does not know.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct SavedModel {
        format: u32,
        seed: u64,
        m: usize,
        nu: f64,
        q: f64,
        k: f64,
        sigma_c: f64,
        omega: f64,
        lambda_c: f64,
        burn_in: usize,
        n_draws: usize,
        thinning: usize,
        metrics: Option<Vec<Metric>>,
        /// Format 2: the response family; absent in format-1
        /// payloads, which default to Gaussian.
        #[serde(default)]
        family: super::ResponseFamily,
        posterior: PosteriorSamples,
        scaler: FittedScaler,
        warnings: Vec<Warning>,
        in_sample_rmse: f64,
    }

    const FORMAT: u32 = 2;

    impl serde::Serialize for FittedAddiVortes {
        fn serialize<S: serde::Serializer>(
            &self,
            serializer: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            // Destructured with **no** `..` rest pattern, deliberately: `SavedModel`
            // mirrors only the plain fields, so an extension point missing from the refusal
            // below serialises "successfully" and is silently rebuilt as the
            // *default* component on load — the reloaded model then disagrees with
            // the one that was fitted, with no error anywhere. That drift happened
            // twice (`count_priors`, `basis`). Naming every field here turns the
            // next occurrence into a compile error: a new config field cannot be
            // added without classifying it as portable (mirror it into `SavedModel`)
            // or not (add it to `has_custom_extension`).
            let AddiVortesConfig {
                seed,
                m,
                nu,
                q,
                k,
                sigma_c,
                omega,
                lambda_c,
                burn_in,
                n_draws,
                thinning,
                metrics,
                family,
                cell_prior_sd,
            } = &self.config;

            // Not portable: an arbitrary trait object has no serialisable form.
            // The component flag travels on the fitted model itself; the
            // cell-prior dial is refused on the same channel (a crate-internal
            // fit-side hook only a model file can set).
            let has_custom_extension = self.custom_components || cell_prior_sd.is_some();
            if has_custom_extension {
                return Err(serde::ser::Error::custom(
                    "a model fitted with custom extension points cannot be serialised \
                     (trait objects have no portable form); persist its into_parts() \
                     values and rebuild in code instead",
                ));
            }
            SavedModel {
                format: FORMAT,
                family: *family,
                seed: *seed,
                m: *m,
                nu: *nu,
                q: *q,
                k: *k,
                sigma_c: *sigma_c,
                omega: *omega,
                lambda_c: *lambda_c,
                burn_in: *burn_in,
                n_draws: *n_draws,
                thinning: *thinning,
                metrics: metrics.clone(),
                posterior: self.posterior.clone(),
                scaler: self.scaler.clone(),
                warnings: self.warnings.clone(),
                in_sample_rmse: self.in_sample_rmse,
            }
            .serialize(serializer)
        }
    }

    impl<'de> serde::Deserialize<'de> for FittedAddiVortes {
        fn deserialize<D: serde::Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            let saved = SavedModel::deserialize(deserializer)?;
            rebuild(saved).map_err(serde::de::Error::custom)
        }
    }

    /// Cross-validate the parts against each other (each part validated
    /// itself already) and reassemble a predict-ready model.
    fn rebuild(saved: SavedModel) -> std::result::Result<FittedAddiVortes, SavedModelError> {
        let bad = |reason: String| Err(SavedModelError(reason));
        // Format 2 added the family payload; format-1 payloads
        // still load, their family defaulting to Gaussian.
        if saved.format != 1 && saved.format != FORMAT {
            return bad(format!(
                "format {found} is not supported (this version reads formats 1 and {FORMAT})",
                found = saved.format
            ));
        }
        if !(saved.in_sample_rmse.is_finite() && saved.in_sample_rmse >= 0.0) {
            return bad("in-sample RMSE must be finite and non-negative".into());
        }
        // Family parameters are validated like every other payload field,
        // using the same bound the fit boundary enforces.
        if let super::ResponseFamily::RobustT { df } = saved.family {
            if !(df.is_finite() && df > 0.0) {
                return bad(format!("RobustT needs finite df > 0; found {df}"));
            }
        }

        // The posterior must fit the scaler's encoded layout, and the plain
        // config fields must describe the posterior they accompany.
        let p_enc = saved.scaler.n_encoded_cols();
        for draw_index in 0..saved.posterior.n_draws() {
            for tessellation in saved.posterior.tessellations(draw_index) {
                if let Some(&dim) = tessellation.dims().iter().find(|d| **d >= p_enc) {
                    return bad(format!(
                        "draw {draw_index} uses covariate {dim} but the scaler encodes \
                         {p_enc} columns"
                    ));
                }
            }
        }
        if saved.posterior.tessellations(0).len() != saved.m {
            return bad(format!(
                "the posterior holds {found} tessellations per draw but m = {m}",
                found = saved.posterior.tessellations(0).len(),
                m = saved.m
            ));
        }
        if saved.posterior.n_draws() != saved.n_draws {
            return bad(format!(
                "the posterior holds {found} draws but the configuration says {expected}",
                found = saved.posterior.n_draws(),
                expected = saved.n_draws
            ));
        }

        // The raw metric list must agree with the encoding the scaler
        // actually performed (levels ⇔ Categorical, encoded metrics for the
        // rest; None means all-Euclidean).
        let n_raw = saved.scaler.n_raw_cols();
        let mut derived: Vec<Metric> = Vec::with_capacity(n_raw);
        let mut encoded_index = 0usize;
        for col in 0..n_raw {
            match saved.scaler.levels(col) {
                Some(levels) => {
                    derived.push(Metric::Categorical);
                    encoded_index += levels.len();
                }
                None => {
                    derived.push(saved.scaler.metrics()[encoded_index]);
                    encoded_index += 1;
                }
            }
        }
        let coherent = match &saved.metrics {
            Some(metrics) => *metrics == derived,
            None => derived.iter().all(|m| *m == Metric::Euclidean),
        };
        if !coherent {
            return bad("the metric list does not match the scaler's encoding".into());
        }

        let mut config = AddiVortesConfig::new(saved.seed)
            .with_m(saved.m)
            .with_nu(saved.nu)
            .with_q(saved.q)
            .with_k(saved.k)
            .with_sigma_c(saved.sigma_c)
            .with_omega(saved.omega)
            .with_lambda_c(saved.lambda_c)
            .with_burn_in(saved.burn_in)
            .with_draws(saved.n_draws)
            .with_thinning(saved.thinning)
            .with_response_family(saved.family);
        config.metrics = saved.metrics;
        if let Err(e) = config.validate() {
            return bad(format!("saved hyperparameters are invalid: {e}"));
        }

        // Built-in-component models only ever carry the default per-column
        // assigner over the encoded metrics; rebuild it from the scaler.
        let assigner: Arc<dyn crate::extensions::distance::CellAssigner> =
            crate::extensions::distance::default_assigner(saved.scaler.metrics().to_vec());
        Ok(FittedAddiVortes {
            posterior: saved.posterior,
            scaler: saved.scaler,
            config,
            assigner,
            // Only default-component models serialise, so all three reset.
            membership: None,
            basis: None,
            custom_components: false,
            warnings: saved.warnings,
            in_sample_rmse: saved.in_sample_rmse,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_abs_eq;
    use crate::{AddiVortesConfig, Metric};

    fn training_data() -> (Data, Vec<f64>) {
        let n = 30;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v + 0.5).collect();
        (Data::new(xs, n, 1).unwrap(), y)
    }

    fn quick_config() -> AddiVortesConfig {
        AddiVortesConfig::new(31)
            .with_m(10)
            .with_burn_in(30)
            .with_draws(40)
    }

    #[test]
    fn happy_path_fit_predict() {
        // The README happy path, verbatim shape.
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.n_rows());
        assert!(predictions.iter().all(|p| p.is_finite()));
        assert!(
            model.in_sample_rmse() < 0.4,
            "rmse {}",
            model.in_sample_rmse()
        );
    }

    #[test]
    fn predict_on_training_reproduces_in_sample_rmse() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();
        let rmse = (predictions
            .iter()
            .zip(&y)
            .map(|(p, o)| (p - o) * (p - o))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();
        assert_eq!(rmse.to_bits(), model.in_sample_rmse().to_bits());
    }

    #[test]
    fn fitted_model_is_self_contained() {
        let predictions_before;
        let model;
        {
            let (x, y) = training_data();
            model = quick_config().fit(&x, &y).unwrap();
            predictions_before = model.predict(&x).unwrap();
        } // x and y dropped here
        let (x, _) = training_data();
        assert_eq!(model.predict(&x).unwrap(), predictions_before);
    }

    #[test]
    fn credible_interval_is_quantile_sugar() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let intervals = model.credible_interval(&x, 0.9).unwrap();
        // The sugar's own tail probabilities (0.5·(1−0.9) is one ULP off the
        // literal 0.05; the identity is checked against what the sugar computes).
        let tail = 0.5 * (1.0 - 0.9);
        let quantiles = model.predict_quantiles(&x, &[tail, 1.0 - tail]).unwrap();
        assert_eq!(intervals.len(), x.n_rows());
        for (row, interval) in intervals.iter().enumerate() {
            assert_eq!(interval.lower, quantiles.values()[row * 2]);
            assert_eq!(interval.upper, quantiles.values()[row * 2 + 1]);
            assert!(interval.lower <= interval.upper);
        }
    }

    #[test]
    fn prediction_interval_single_draw_matches_the_normal_quantile() {
        // With exactly one kept draw the predictive mixture is one normal
        // N(fit, σ²), so the interval ends are fit + σ·z at the tail
        // quantiles. Hand-derived oracle: Φ⁻¹(0.05) = −1.644853626951472
        // (scipy.stats.norm.ppf).
        let (x, y) = training_data();
        let model = quick_config().with_draws(1).fit(&x, &y).unwrap();
        let fits = model.predict(&x).unwrap(); // mean over one draw = the draw
        let sigma = model.sigma()[0];
        let z = 1.644_853_626_951_472_1_f64;
        let intervals = model.prediction_interval(&x, 0.9).unwrap();
        for (interval, fit) in intervals.iter().zip(&fits) {
            assert_abs_eq(interval.lower, fit - z * sigma, 1e-9);
            assert_abs_eq(interval.upper, fit + z * sigma, 1e-9);
        }
    }

    #[test]
    fn prediction_interval_contains_the_credible_interval() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let predictive = model.prediction_interval(&x, 0.9).unwrap();
        let credible = model.credible_interval(&x, 0.9).unwrap();
        assert_eq!(predictive.len(), x.n_rows());
        let mut covered = 0usize;
        for ((p, c), observed) in predictive.iter().zip(&credible).zip(&y) {
            // Strict: every σ draw is positive, so the noise term widens both ends.
            assert!(p.lower < c.lower && p.upper > c.upper);
            if (p.lower..=p.upper).contains(observed) {
                covered += 1;
            }
        }
        // Coverage sanity on the training data (deterministic given the seed).
        assert!(
            covered * 10 >= y.len() * 8,
            "0.9 prediction interval covered only {covered}/{} training points",
            y.len()
        );
    }

    #[test]
    fn prediction_interval_level_validation_and_empty_input() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
            assert!(matches!(
                model.prediction_interval(&x, bad),
                Err(AddiVortesError::InvalidQuantileProb { .. })
            ));
        }
        let empty_one_col = Data::new(vec![], 0, 1).unwrap();
        assert_eq!(
            model.prediction_interval(&empty_one_col, 0.9).unwrap(),
            vec![]
        );
    }

    #[test]
    fn quantile_probability_validation() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
            let err = model.predict_quantiles(&x, &[bad]).unwrap_err();
            match err {
                AddiVortesError::InvalidQuantileProb { .. } => {}
                other => panic!("expected InvalidQuantileProb, got {other:?}"),
            }
        }
        assert!(model.credible_interval(&x, 1.0).is_err());
        // Empty probs also error (nothing to compute).
        assert!(model.predict_quantiles(&x, &[]).is_err());
        // Empty X is fine: zero-row result.
        let empty = Data::from_rows::<&[f64]>(&[]).unwrap();
        let err = model.predict_quantiles(&empty, &[0.5]);
        // 0 columns vs 1 expected → FeatureCountMismatch (predict order).
        assert!(matches!(
            err,
            Err(AddiVortesError::FeatureCountMismatch { .. })
        ));
        let empty_one_col = Data::new(vec![], 0, 1).unwrap();
        let q = model.predict_quantiles(&empty_one_col, &[0.5]).unwrap();
        assert_eq!(q.n_rows(), 0);
        assert!(q.values().is_empty());
    }

    #[test]
    fn predict_draws_shape_and_mean_matches_predict() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let draws = model.predict_draws(&x).unwrap();
        assert_eq!(draws.len(), model.posterior().n_draws());
        assert!(draws.iter().all(|row| row.len() == x.n_rows()));
        // predict is exactly the per-row mean of predict_draws (same
        // accumulation order → bit-identical).
        let mean = model.predict(&x).unwrap();
        let n_draws = draws.len() as f64;
        for (row, expected) in mean.iter().enumerate() {
            let mut sum = 0.0_f64;
            for draw in &draws {
                sum += draw[row];
            }
            assert_eq!((sum / n_draws).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn log_likelihood_gaussian_matches_closed_form() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let ll = model.log_likelihood(&x, &y).unwrap();
        assert_eq!(ll.len(), model.posterior().n_draws());
        assert!(ll.iter().all(|row| row.len() == x.n_rows()));
        let draws = model.predict_draws(&x).unwrap();
        let sigmas = model.sigma();
        // Spot-check draw 0, row 0 against the N(fit, σ²) log-density.
        let z = (y[0] - draws[0][0]) / sigmas[0];
        let expected = -0.5 * crate::engine::mathsfn::ln(2.0 * std::f64::consts::PI)
            - crate::engine::mathsfn::ln(sigmas[0])
            - 0.5 * z * z;
        assert_abs_eq(ll[0][0], expected, 1e-12);
        assert!(ll.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn log_likelihood_validates_response() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        assert!(matches!(
            model.log_likelihood(&x, &y[1..]),
            Err(AddiVortesError::RowCountMismatch { .. })
        ));
        let mut bad = y.clone();
        bad[3] = f64::NAN;
        assert!(matches!(
            model.log_likelihood(&x, &bad),
            Err(AddiVortesError::NonFiniteResponse { row: 3 })
        ));
    }

    #[test]
    fn log_likelihood_probit_is_bernoulli_and_rejects_non_labels() {
        let (x, _) = training_data();
        let labels: Vec<f64> = (0..x.n_rows()).map(|i| f64::from(i % 2 == 0)).collect();
        let model = quick_config()
            .with_response_family(ResponseFamily::BinaryProbit)
            .fit(&x, &labels)
            .unwrap();
        let ll = model.log_likelihood(&x, &labels).unwrap();
        let probs = model.predict_draws(&x).unwrap();
        // ln p for label 1, ln(1 − p) for label 0, clamped: check draw 0.
        for (i, &label) in labels.iter().enumerate() {
            let p = probs[0][i].clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            let expected = if label == 1.0 {
                crate::engine::mathsfn::ln(p)
            } else {
                crate::engine::mathsfn::ln_1p(-p)
            };
            assert_eq!(ll[0][i].to_bits(), expected.to_bits());
        }
        // Non-{0,1} response is the fit boundary's own error.
        let bad: Vec<f64> = labels.iter().map(|v| v + 0.5).collect();
        assert!(matches!(
            model.log_likelihood(&x, &bad),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "response_family"
        ));
    }

    #[test]
    fn log_likelihood_robust_t_matches_closed_form() {
        let (x, y) = training_data();
        let model = quick_config()
            .with_response_family(ResponseFamily::RobustT { df: 4.0 })
            .fit(&x, &y)
            .unwrap();
        let ll = model.log_likelihood(&x, &y).unwrap();
        let draws = model.predict_draws(&x).unwrap();
        let sigmas = model.sigma();
        let df = 4.0_f64;
        let z = (y[0] - draws[0][0]) / sigmas[0];
        let expected = crate::engine::mathsfn::lgamma(0.5 * (df + 1.0))
            - crate::engine::mathsfn::lgamma(0.5 * df)
            - 0.5 * crate::engine::mathsfn::ln(df * std::f64::consts::PI)
            - crate::engine::mathsfn::ln(sigmas[0])
            - 0.5 * (df + 1.0) * crate::engine::mathsfn::ln_1p(z * z / df);
        assert_abs_eq(ll[0][0], expected, 1e-12);
    }

    #[test]
    fn public_error_paths_via_the_api() {
        let (x, y) = training_data();

        // ω ≥ p at the fit boundary (p = 1 skips it; use p = 2 data).
        let x2 = Data::from_rows(&[[0.0, 1.0], [0.4, 0.2], [1.0, 0.6], [0.7, 0.9]]).unwrap();
        let y2 = vec![1.0, 2.0, 3.0, 4.0];
        let err = AddiVortesConfig::new(1).fit(&x2, &y2).unwrap_err(); // default ω = 3 ≥ 2
        assert!(
            matches!(err, AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "omega")
        );

        // Constant column.
        let constant = Data::from_rows(&[[1.0], [1.0], [1.0]]).unwrap();
        let err = quick_config().fit(&constant, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, AddiVortesError::DegenerateFeature { col: 0 }));

        // NaN feature.
        let nan = Data::from_rows(&[[0.0], [f64::NAN], [1.0]]).unwrap();
        let err = quick_config().fit(&nan, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(
            err,
            AddiVortesError::NonFiniteFeature { row: 1, col: 0 }
        ));

        // Feature-count mismatch at predict.
        let model = quick_config().fit(&x, &y).unwrap();
        let wide = Data::from_rows(&[[0.0, 1.0]]).unwrap();
        let err = model.predict(&wide).unwrap_err();
        assert!(matches!(
            err,
            AddiVortesError::FeatureCountMismatch {
                expected: 1,
                found: 2
            }
        ));

        // Unseen category at predict.
        let xc =
            Data::from_rows(&[[0.0, 1.0], [0.3, 2.0], [0.6, 1.0], [0.8, 2.0], [1.0, 1.0]]).unwrap();
        let yc = vec![0.1, 1.0, 0.4, 1.3, 0.8];
        let model = quick_config()
            .with_omega(1.5)
            .with_metrics(vec![Metric::Euclidean, Metric::Categorical])
            .fit(&xc, &yc)
            .unwrap();
        let unseen = Data::from_rows(&[[0.5, 9.0]]).unwrap();
        let err = model.predict(&unseen).unwrap_err();
        assert!(matches!(err, AddiVortesError::UnseenCategory { col: 1, value } if value == 9.0));
    }

    #[test]
    fn validate_rejects_bad_hyperparameters() {
        let cases: Vec<(AddiVortesConfig, &str)> = vec![
            (AddiVortesConfig::new(1).with_m(0), "m"),
            (AddiVortesConfig::new(1).with_nu(0.0), "nu"),
            (AddiVortesConfig::new(1).with_q(1.0), "q"),
            (AddiVortesConfig::new(1).with_k(-1.0), "k"),
            (AddiVortesConfig::new(1).with_sigma_c(f64::NAN), "sigma_c"),
            (AddiVortesConfig::new(1).with_omega(0.0), "omega"),
            (
                AddiVortesConfig::new(1).with_lambda_c(f64::INFINITY),
                "lambda_c",
            ),
            (AddiVortesConfig::new(1).with_draws(0), "draws"),
            (AddiVortesConfig::new(1).with_thinning(0), "thinning"),
        ];
        for (config, field) in cases {
            let err = config.validate().unwrap_err();
            match err {
                AddiVortesError::InvalidHyperparameter { name, .. } => assert_eq!(name, field),
                other => panic!("expected InvalidHyperparameter, got {other:?}"),
            }
        }
        assert!(AddiVortesConfig::new(1).validate().is_ok());
        // ω > 0 is enough data-free: ω between 0 and 1 is legal (spec: ω>0,
        // not ω≥1); the ω < p check is data-dependent.
        assert!(AddiVortesConfig::new(1).with_omega(0.5).validate().is_ok());
    }

    #[test]
    fn inclusion_weight_validation_and_expansion() {
        use crate::engine::builder::SamplerBuilder;
        use crate::extensions::inclusion::WeightedInclusion;
        let config = || quick_config().with_omega(1.5);
        let x2 = Data::from_rows(&[[0.0, 1.0], [0.4, 0.2], [1.0, 0.6], [0.7, 0.9]]).unwrap();
        let y2 = vec![1.0, 2.0, 3.0, 4.0];

        // Non-positive weight: InvalidHyperparameter at the fit boundary
        // (the first place the model meets the engine).
        let err = SamplerBuilder::new(config())
            .with_inclusion(WeightedInclusion::new(vec![1.0, 0.0]))
            .fit(&x2, &y2)
            .unwrap_err();
        assert!(matches!(
            err,
            AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "inclusion_weights"
        ));

        // Wrong length: InvalidHyperparameter at the fit boundary.
        let err = SamplerBuilder::new(config())
            .with_inclusion(WeightedInclusion::new(vec![1.0, 2.0, 3.0]))
            .fit(&x2, &y2)
            .unwrap_err();
        assert!(matches!(
            err,
            AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "inclusion_weights"
        ));

        // One-hot expansion: a categorical column shares one weight across its
        // group (fit succeeds with p_raw = 2 weights despite 4 encoded columns).
        let xc = Data::from_rows(&[
            [0.0, 1.0],
            [0.3, 2.0],
            [0.6, 3.0],
            [0.8, 2.0],
            [1.0, 1.0],
            [0.2, 3.0],
        ])
        .unwrap();
        let yc = vec![0.1, 1.0, 0.4, 1.3, 0.8, 0.2];
        let model = SamplerBuilder::new(
            config().with_metrics(vec![Metric::Euclidean, Metric::Categorical]),
        )
        .with_inclusion(WeightedInclusion::new(vec![1.0, 4.0]))
        .fit(&xc, &yc)
        .unwrap();
        assert_eq!(model.scaler().n_encoded_cols(), 4); // 1 + 3 one-hot
    }

    #[test]
    fn variable_inclusion_proportions_p1_is_all_mass_on_the_only_column() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        assert_eq!(model.variable_inclusion_proportions(), vec![1.0]);
    }

    #[test]
    fn variable_inclusion_proportions_aggregate_one_hot_groups() {
        // Column 0 Euclidean, column 1 categorical with 3 levels → encoded
        // layout [E, onehot, onehot, onehot], col map [0, 1, 1, 1] (encoding
        // expands in place).
        let xc = Data::from_rows(&[
            [0.0, 1.0],
            [0.3, 2.0],
            [0.6, 3.0],
            [0.8, 2.0],
            [1.0, 1.0],
            [0.2, 3.0],
        ])
        .unwrap();
        let yc = vec![0.1, 1.0, 0.4, 1.3, 0.8, 0.2];
        let model = quick_config()
            .with_omega(1.5)
            .with_metrics(vec![Metric::Euclidean, Metric::Categorical])
            .fit(&xc, &yc)
            .unwrap();

        let proportions = model.variable_inclusion_proportions();
        assert_eq!(proportions.len(), 2); // raw columns, not the 4 encoded
        assert_abs_eq(proportions.iter().sum::<f64>(), 1.0, 1e-12);
        assert!(proportions.iter().all(|p| (0.0..=1.0).contains(p)));

        // Hand recount through the public posterior with the documented
        // in-place expansion (encoded 0 → raw 0, encoded 1..4 → raw 1).
        let col_map = [0usize, 1, 1, 1];
        let mut counts = [0u64; 2];
        let mut total = 0u64;
        for draw in model.posterior().iter_draws() {
            for tessellation in draw.tessellations {
                for &dim in tessellation.dims() {
                    counts[col_map[dim]] += 1;
                    total += 1;
                }
            }
        }
        for (proportion, count) in proportions.iter().zip(counts) {
            assert_eq!(
                proportion.to_bits(),
                (count as f64 / total as f64).to_bits()
            );
        }
    }

    #[test]
    fn prepared_column_fits_end_to_end() {
        // Column 1 is caller-prepared (already on a [−0.5, 0.5]-commensurate
        // scale): the full fit → predict loop runs, and the fitted scaler
        // passed the column through untouched.
        let x = Data::from_rows(&[
            [0.0, -0.5],
            [0.2, -0.1],
            [0.4, 0.3],
            [0.6, -0.3],
            [0.8, 0.1],
            [1.0, 0.5],
        ])
        .unwrap();
        let y = vec![0.1, 0.5, 1.2, 0.6, 0.9, 1.8];
        let model = quick_config()
            .with_omega(1.5)
            .with_metrics(vec![Metric::Euclidean, Metric::Prepared])
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            model.scaler().metrics(),
            &[Metric::Euclidean, Metric::Prepared]
        );
        let predictions = model.predict(&x).unwrap();
        assert!(predictions.iter().all(|p| p.is_finite()));
        assert_eq!(model.variable_inclusion_proportions().len(), 2);
    }

    /// The Student-t CDF against its closed forms: Cauchy (ν′ = 1),
    /// ν′ = 2, the printed t-table quantiles, symmetry, and the Gaussian
    /// limit.
    #[test]
    fn student_t_cdf_matches_closed_forms() {
        // ν′ = 1 (Cauchy): F(x) = 1/2 + atan(x)/π.
        for x in [-3.0_f64, -0.7, 0.4, 2.0] {
            let expected = 0.5 + x.atan() / std::f64::consts::PI;
            assert_abs_eq(student_t_cdf(x, 1.0), expected, 1e-12);
        }
        // ν′ = 2: F(x) = 1/2 + x / (2√(2 + x²)).
        for x in [-2.5_f64, -1.0, 1.0, 3.0] {
            let expected = 0.5 + x / (2.0 * (2.0 + x * x).sqrt());
            assert_abs_eq(student_t_cdf(x, 2.0), expected, 1e-12);
        }
        // The t table: F(3.182446305284263, 3) = 0.975 and
        // F(2.5705818366147395, 5) = 0.975.
        assert_abs_eq(student_t_cdf(3.182446305284263, 3.0), 0.975, 1e-10);
        assert_abs_eq(student_t_cdf(2.5705818366147395, 5.0), 0.975, 1e-10);
        // Symmetry and the centre.
        assert_abs_eq(
            student_t_cdf(-1.3, 7.0) + student_t_cdf(1.3, 7.0),
            1.0,
            1e-13,
        );
        assert_abs_eq(student_t_cdf(0.0, 4.0), 0.5, 0.0);
        // ν′ → ∞ recovers the normal CDF.
        for x in [-2.0, -0.5, 1.0, 2.5] {
            assert_abs_eq(student_t_cdf(x, 1.0e7), normal_cdf(x), 1e-6);
        }
    }

    /// The t-mixture quantile inverts the t-mixture CDF, and a single
    /// standard component reproduces the t table.
    #[test]
    fn t_predictive_quantile_inverts_the_mixture() {
        let fits = [0.0];
        let sigmas = [1.0];
        assert_abs_eq(
            t_predictive_quantile(&fits, &sigmas, 3.0, 0.975),
            3.182446305284263,
            1e-9,
        );
        // A genuine mixture: round-trip CDF(quantile(p)) = p.
        let fits = [-0.4, 0.2, 1.1];
        let sigmas = [0.5, 1.5, 0.9];
        for p in [0.01, 0.2, 0.5, 0.9, 0.999] {
            let q = t_predictive_quantile(&fits, &sigmas, 4.0, p);
            assert_abs_eq(t_mixture_cdf(&fits, &sigmas, 4.0, q), p, 1e-12);
        }
    }

    /// The robust-t family end to end: fits, predicts on the response
    /// scale, and resists a gross outlier that visibly drags the Gaussian
    /// fit (both chains share a seed, so the comparison is deterministic).
    #[test]
    fn robust_t_family_resists_an_outlier() {
        let n = 30;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let mut y: Vec<f64> = xs.clone();
        y[15] = 25.0; // one gross outlier on a clean linear trend
        let x = Data::new(xs.clone(), n, 1).unwrap();

        let robust = quick_config()
            .with_response_family(ResponseFamily::RobustT { df: 4.0 })
            .fit(&x, &y)
            .unwrap();
        let gaussian = quick_config().fit(&x, &y).unwrap();

        // Probe the fit away from the outlier's x: the robust prediction
        // must sit far closer to the uncontaminated trend.
        let probe_rows = [3usize, 8, 22, 27];
        let robust_predictions = robust.predict(&x).unwrap();
        let gaussian_predictions = gaussian.predict(&x).unwrap();
        let error = |predictions: &[f64]| -> f64 {
            probe_rows
                .iter()
                .map(|&r| (predictions[r] - xs[r]).abs())
                .sum()
        };
        let (robust_err, gaussian_err) = (error(&robust_predictions), error(&gaussian_predictions));
        assert!(
            robust_err < 0.5 * gaussian_err,
            "robust {robust_err} vs gaussian {gaussian_err}"
        );

        // The per-variant predictive: finite, ordered intervals.
        let intervals = robust.prediction_interval(&x, 0.9).unwrap();
        for interval in &intervals {
            assert!(interval.lower.is_finite() && interval.upper.is_finite());
            assert!(interval.lower < interval.upper);
        }
    }

    /// The fit boundary rejects unusable degrees of freedom.
    #[test]
    fn robust_t_rejects_bad_df() {
        let (x, y) = training_data();
        for df in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            let err = quick_config()
                .with_response_family(ResponseFamily::RobustT { df })
                .fit(&x, &y)
                .unwrap_err();
            assert!(matches!(err, AddiVortesError::InvalidHyperparameter { .. }));
        }
    }

    #[test]
    fn sigma_is_response_scale() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let range = model.scaler().y_max() - model.scaler().y_min();
        let sigma = model.sigma();
        assert_eq!(sigma.len(), model.posterior().n_draws());
        for (s, s_sq) in sigma.iter().zip(model.posterior().sigma_sq()) {
            assert_abs_eq(*s, s_sq.sqrt() * range, 1e-15);
        }
    }

    #[test]
    fn posterior_container_accessors_agree() {
        let (x, y) = training_data();
        let model = quick_config().fit(&x, &y).unwrap();
        let posterior = model.posterior();
        assert_eq!(posterior.n_draws(), 40);
        let collected: Vec<f64> = posterior.iter_draws().map(|d| d.sigma_sq).collect();
        assert_eq!(collected, posterior.sigma_sq());
        assert_eq!(posterior.iter_draws().count(), 40);
        assert_eq!(posterior.tessellations(0).len(), 10); // m
        let clone = model.posterior().clone();
        let (sigma_sqs, draws) = clone.into_parts();
        assert_eq!(sigma_sqs.len(), 40);
        assert_eq!(draws.len(), 40);
    }

    #[test]
    fn thinning_keeps_every_kth_sweep() {
        // A thinning-t fit's draws must equal every t-th sweep of an identical
        // unthinned chain (same seed ⇒ same underlying sweep sequence).
        let (x, y) = training_data();
        let thinned = quick_config()
            .with_burn_in(5)
            .with_draws(6)
            .with_thinning(3)
            .fit(&x, &y)
            .unwrap();
        let dense = quick_config()
            .with_burn_in(5)
            .with_draws(18)
            .with_thinning(1)
            .fit(&x, &y)
            .unwrap();
        let dense_sigma = dense.posterior().sigma_sq();
        let expected: Vec<f64> = (0..6).map(|i| dense_sigma[i * 3 + 2]).collect();
        assert_eq!(thinned.posterior().sigma_sq(), expected.as_slice());
    }

    #[cfg(feature = "serde")]
    mod serde_round_trip {
        use super::*;

        fn categorical_model() -> (FittedAddiVortes, Data) {
            let x = Data::from_rows(&[
                [0.0, 1.0],
                [0.3, 2.0],
                [0.6, 3.0],
                [0.8, 2.0],
                [1.0, 1.0],
                [0.2, 3.0],
            ])
            .unwrap();
            let y = vec![0.1, 1.0, 0.4, 1.3, 0.8, 0.2];
            let model = quick_config()
                .with_omega(1.5)
                .with_metrics(vec![Metric::Euclidean, Metric::Categorical])
                .fit(&x, &y)
                .unwrap();
            (model, x)
        }

        #[test]
        fn round_trip_is_bit_identical() {
            let (model, x) = categorical_model();
            let json = serde_json::to_string(&model).unwrap();
            let loaded: FittedAddiVortes = serde_json::from_str(&json).unwrap();

            assert_eq!(loaded.config(), model.config());
            assert_eq!(loaded.scaler(), model.scaler());
            assert_eq!(loaded.posterior(), model.posterior());
            assert_eq!(loaded.warnings(), model.warnings());
            assert_eq!(
                loaded.in_sample_rmse().to_bits(),
                model.in_sample_rmse().to_bits()
            );
            // The reconstructed assigner predicts identically, bit for bit.
            let (a, b) = (loaded.predict(&x).unwrap(), model.predict(&x).unwrap());
            assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_eq!(
                loaded.prediction_interval(&x, 0.9).unwrap(),
                model.prediction_interval(&x, 0.9).unwrap()
            );
            assert_eq!(
                loaded.variable_inclusion_proportions(),
                model.variable_inclusion_proportions()
            );
        }

        /// Format 2: a BinaryProbit model round-trips with
        /// bit-identical probability-scale predictions and keeps its family.
        #[test]
        fn format_2_round_trips_the_family_payload() {
            let (x, _) = training_data();
            let labels: Vec<f64> = (0..x.n_rows()).map(|i| f64::from(i % 3 != 0)).collect();
            let model = quick_config()
                .with_response_family(crate::ResponseFamily::BinaryProbit)
                .fit(&x, &labels)
                .unwrap();
            let json = serde_json::to_string(&model).unwrap();
            let loaded: FittedAddiVortes = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded.config().family, crate::ResponseFamily::BinaryProbit);
            let original = model.predict(&x).unwrap();
            let reloaded = loaded.predict(&x).unwrap();
            for (a, b) in original.iter().zip(&reloaded) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }

        /// The robust-t family payload round-trips (format 2): the family
        /// (df included) survives, and predictions and t-mixture intervals
        /// are bit-identical.
        #[test]
        fn robust_t_round_trips_the_family_payload() {
            let (x, y) = training_data();
            let model = quick_config()
                .with_response_family(crate::ResponseFamily::RobustT { df: 5.0 })
                .fit(&x, &y)
                .unwrap();
            let json = serde_json::to_string(&model).unwrap();
            let loaded: FittedAddiVortes = serde_json::from_str(&json).unwrap();
            assert_eq!(
                loaded.config().family,
                crate::ResponseFamily::RobustT { df: 5.0 }
            );
            let original = model.predict(&x).unwrap();
            let reloaded = loaded.predict(&x).unwrap();
            for (a, b) in original.iter().zip(&reloaded) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            assert_eq!(
                loaded.prediction_interval(&x, 0.9).unwrap(),
                model.prediction_interval(&x, 0.9).unwrap()
            );
        }

        /// A tampered family parameter is rejected at load, like every other
        /// corrupt payload field.
        #[test]
        fn corrupt_family_df_is_rejected() {
            let (x, y) = training_data();
            let model = quick_config()
                .with_response_family(crate::ResponseFamily::RobustT { df: 5.0 })
                .fit(&x, &y)
                .unwrap();
            let mut v: serde_json::Value =
                serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
            v["family"] = serde_json::json!({ "RobustT": { "df": -1.0 } });
            let result: std::result::Result<FittedAddiVortes, _> = serde_json::from_value(v);
            assert!(result.is_err(), "df ≤ 0 must be rejected at load");
        }

        /// Format-1 payloads (no family field) still load, defaulting to the
        /// Gaussian family, with bit-identical predictions.
        #[test]
        fn format_1_payloads_still_load() {
            let (x, y) = training_data();
            let model = quick_config().fit(&x, &y).unwrap();
            let mut value = serde_json::to_value(&model).unwrap();
            // Rewrite the payload into its format-1 shape: version 1, no
            // family key.
            value["format"] = serde_json::json!(1);
            value.as_object_mut().unwrap().remove("family");
            let loaded: FittedAddiVortes = serde_json::from_value(value).unwrap();
            assert_eq!(loaded.config().family, crate::ResponseFamily::Gaussian);
            let original = model.predict(&x).unwrap();
            let reloaded = loaded.predict(&x).unwrap();
            for (a, b) in original.iter().zip(&reloaded) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }

        /// Every non-portable component must refuse to serialise, **one at a time**.
        ///
        /// Setting several components at once would prove nothing: any single one of
        /// them trips the refusal, so an extension point missing from the list still passes.
        /// Only perturbing one extension point per case can catch an omission — and two were
        /// omitted (`count_priors`, `basis`), which meant a model fitted with a
        /// custom count prior saved happily and reloaded as the *paper's* prior,
        /// with no error anywhere.
        ///
        /// `basis` is absent below because it cannot be fitted on its own: the
        /// sampler rejects a basis without a matching basis cell model, so any
        /// fitted model carrying one also carries a custom `cell_model` and is
        /// refused through that. Its guard is the exhaustive destructure in
        /// `Serialize`, which will not compile if the component is dropped.
        #[test]
        fn every_custom_extension_refuses_to_serialise_on_its_own() {
            use crate::extensions::cell_model::GaussianCellModel;
            use crate::extensions::coord::EuclideanNormal;
            use crate::extensions::count_priors::ShiftedPoissonBinomial;
            use crate::extensions::distance::Manhattan;
            use crate::extensions::inclusion::UniformInclusion;
            use crate::extensions::membership::SoftmaxKernel;
            use crate::extensions::moves::MoveSetBuilder;
            use crate::extensions::response::ResponseModel;
            use crate::extensions::scale::PinnedSigma;

            // A do-nothing response model, so this case isolates *the component being set*
            // rather than a shelf entry's semantics. (`RobustTStep` cannot be used
            // here: on its own it feeds fractional weights to the default
            // hard-assignment cell statistic. The `with_response_family` route
            // assembles the matching trio; the raw seam does not.)
            #[derive(Debug, Clone)]
            struct NoOpStep;
            impl ResponseModel for NoOpStep {
                type Error = std::convert::Infallible;
                fn augment(
                    &mut self,
                    y: &[f64],
                    _fit: &[f64],
                    _sigma_sq: f64,
                    _rng: &mut dyn rand_core::Rng,
                    working: &mut [f64],
                    weights: &mut [f64],
                ) -> std::result::Result<(), Self::Error> {
                    working.copy_from_slice(y);
                    weights.fill(1.0);
                    Ok(())
                }
            }

            use crate::engine::builder::SamplerBuilder;
            let (x, y) = training_data();
            let builder = || SamplerBuilder::new(quick_config());
            let cases: Vec<(&str, SamplerBuilder)> = vec![
                (
                    "move_set",
                    builder().with_move_set(MoveSetBuilder::stone_gosling().build().unwrap()),
                ),
                (
                    "coords",
                    builder().with_coords(vec![Arc::new(EuclideanNormal::new(0.8).unwrap())]),
                ),
                ("assigner", builder().with_distance(Manhattan)),
                (
                    "inclusion",
                    builder().with_inclusion(UniformInclusion::new(1)),
                ),
                (
                    "cell_model",
                    builder().with_cell_model(GaussianCellModel::new(0.01).unwrap()),
                ),
                ("response_model", builder().with_response_model(NoOpStep)),
                (
                    "scale_model",
                    builder().with_scale_model(PinnedSigma::unit()),
                ),
                (
                    "membership",
                    builder().with_membership(SoftmaxKernel::new(0.1).unwrap()),
                ),
                (
                    "count_priors",
                    builder().with_count_priors(ShiftedPoissonBinomial),
                ),
            ];

            // The baseline must still serialise, or the cases below prove nothing.
            let plain = quick_config().fit(&x, &y).unwrap();
            serde_json::to_string(&plain).expect("a default-component model still serialises");

            for (component, builder) in cases {
                let model = builder
                    .fit(&x, &y)
                    .unwrap_or_else(|e| panic!("{component}: {e}"));
                match serde_json::to_string(&model) {
                    Ok(_) => panic!(
                        "`{component}` serialised instead of being refused: it is missing from \
                         `has_custom_extension`, so this model would silently reload with the \
                         default component"
                    ),
                    Err(err) => assert!(
                        err.to_string().contains("custom extension points"),
                        "`{component}` was refused with the wrong error: {err}"
                    ),
                }
            }
        }

        #[test]
        fn corrupt_payloads_are_rejected_not_panics() {
            let (model, _) = categorical_model();
            let pristine = serde_json::to_value(&model).unwrap();
            let reject = |mutate: &dyn Fn(&mut serde_json::Value), fragment: &str| {
                let mut value = pristine.clone();
                mutate(&mut value);
                let err = serde_json::from_value::<FittedAddiVortes>(value).unwrap_err();
                assert!(err.to_string().contains(fragment), "{err}");
            };
            reject(&|v| v["format"] = 3.into(), "not supported");
            reject(
                &|v| v["posterior"]["sigma_sqs"][0] = (-1.0).into(),
                "finite and positive",
            );
            reject(
                &|v| v["posterior"]["draws"][0][0]["dims"][0] = 999.into(),
                "covariate 999",
            );
            reject(&|v| v["scaler"]["col_map"][0] = 1.into(), "column map");
            reject(&|v| v["m"] = 999.into(), "tessellations per draw");
            reject(
                &|v| v["metrics"] = serde_json::Value::Null,
                "does not match the scaler",
            );
        }

        #[test]
        fn tessellation_deserialisation_validates_structure() {
            let t = Tessellation::new(vec![0.1, 0.2], vec![1], vec![0.3, 0.4]).unwrap();
            let json = serde_json::to_string(&t).unwrap();
            assert_eq!(serde_json::from_str::<Tessellation>(&json).unwrap(), t);
            let err = serde_json::from_str::<Tessellation>(r#"{"centres":[],"dims":[0],"mus":[]}"#)
                .unwrap_err();
            assert!(err.to_string().contains("at least one cell"), "{err}");
        }
    }

    #[test]
    fn config_partial_eq_is_plain_value_equality() {
        let a = AddiVortesConfig::new(5).with_m(3);
        let b = AddiVortesConfig::new(5).with_m(3);
        assert_eq!(a, b);
        assert_ne!(a, b.clone().with_m(4));
        assert_ne!(a, b.with_cell_prior_sd(0.2));
    }
}
