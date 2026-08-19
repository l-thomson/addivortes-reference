//! The H-AddiVortes variance ensemble, as one shelf entry (H
//! paper §3.2–3.3): the conditional variance `s²(x) = ∏_{l=1}^{m′} h(x; T′_l,
//! M′_l)` modelled by a second instance of the shared backfit block
//! (inverse-χ² cells composed multiplicatively), running inside
//! [`ScaleModel::update`] as an ordinary scale supplier. The mean side pairs
//! it with [`WeightedGaussianModel`](crate::extensions::cell_model::WeightedGaussianModel)
//! (precision weighting, H paper Eq. 5–7): this model reports σ² ≡ 1 and
//! supplies per-observation precisions `1/s²(xᵢ)`, which the conductor
//! composes into the conjugate accumulation.
//!
//! The two ensembles share moves, coordinate laws, assigner, and count priors
//! by construction: everything structural arrives through the
//! [`ScaleCtx`] each sweep; nothing is re-implemented here.

use crate::engine::backfit::{Composition, EnsembleUnit};
use crate::engine::error::{
    AddiVortesError, Result, require_at_least_one, require_positive_finite,
};
use crate::engine::mathsfn;
use crate::engine::tessellation::Tessellation;
use crate::extensions::cell_model::{CellModel, InvChiSqCellModel};
use crate::extensions::distance::AssignmentCache;
use crate::extensions::erasure::KernelOf;
use crate::extensions::moves::ModelCtx;
use crate::extensions::scale::{ScaleCtx, ScaleModel};

/// The H paper §3.3 prior calibration: given the homoscedastic σ² ~ χ⁻²(ν, λ)
/// prior and the variance-ensemble size m′, the per-cell prior s² ~ χ⁻²(ν′, λ′)
/// that matches the prior mean of the product is
///
/// ```text
/// λ′ = λ^(1/m′),        ν′ = 2 / (1 − (1 − 2/ν)^(1/m′)).
/// ```
///
/// Returns `(ν′, λ′)`: dimensionless prior parameters, λ′ in scaled space
/// (λ's own coordinate system). Fails with
/// [`AddiVortesError::InvalidHyperparameter`] unless ν > 2 (the
/// homoscedastic prior mean must exist), λ is finite and strictly positive,
/// and m′ ≥ 1.
pub fn h_variance_prior(nu: f64, lambda: f64, m_prime: usize) -> Result<(f64, f64)> {
    if !(nu.is_finite() && nu > 2.0) {
        return Err(AddiVortesError::InvalidHyperparameter {
            name: "nu".into(),
            reason: format!("the H §3.3 calibration needs ν > 2 (finite prior mean), got {nu}"),
        });
    }
    let lambda = require_positive_finite("lambda", lambda)?;
    let m_prime = require_at_least_one("m_prime", m_prime)?;
    let inv_m = 1.0 / m_prime as f64;
    let lambda_prime = mathsfn::powf(lambda, inv_m);
    let nu_prime = 2.0 / (1.0 - mathsfn::powf(1.0 - 2.0 / nu, inv_m));
    Ok((nu_prime, lambda_prime))
}

/// The variance ensemble's lazily-initialised sampling state (the design is
/// first seen at the first `update`).
#[derive(Debug)]
struct HState {
    ensemble: EnsembleUnit,
    /// Running product `S_i = ∏_l h_l(i)` (scaled space).
    fit: Vec<f64>,
    /// `1/S_i`: the per-observation precisions the conductor composes.
    precisions: Vec<f64>,
    /// Scratch: squared mean-residuals `e²_i` (the ensemble's working
    /// response).
    e_sq: Vec<f64>,
    /// The (ν′, λ′) actually in force (resolved at initialisation).
    prior: (f64, f64),
}

/// The H-AddiVortes variance ensemble (H paper §3.2): a product of m′
/// inverse-χ² tessellations supplying per-observation precisions through the
/// scale seam.
///
/// The mean side needs
/// [`WeightedGaussianModel`](crate::extensions::cell_model::WeightedGaussianModel)
/// — the hard-assignment Gaussian statistic rejects the non-unit weights these
/// precisions produce, on purpose. **The engine pairs it for you**, exactly as it
/// does for a `RobustT` response: `with_scale_model(HVariance::new(m)?)` is
/// enough. Setting your own `with_cell_model` overrides that, as everywhere.
///
/// By default (ν′, λ′) are resolved at the first sweep from the engine's own
/// homoscedastic calibration (ν, the data-calibrated λ) through the §3.3
/// matching ([`h_variance_prior`]); pin them explicitly with
/// [`with_prior`](HVariance::with_prior) (the statistical gates do). The
/// variance-cell family is an ordinary cell-model [`CellModel`], replaceable via
/// [`with_cell_model`](HVariance::with_cell_model).
///
/// Chain state: m′ single-cell tessellations with h ≡ 1 (so s²(x) ≡ 1),
/// deterministic: initialisation consumes no RNG, like the mean ensemble's.
#[derive(Debug)]
pub struct HVariance {
    m_prime: usize,
    /// Explicit (ν′, λ′) override; `None` = calibrate from the context.
    prior: Option<(f64, f64)>,
    /// Replacement variance-cell kernel; `None` = inverse-χ² built from the
    /// resolved (ν′, λ′). Held as a factory so cloning (fresh-per-fit config
    /// semantics) mints a fresh kernel.
    kernel: Option<std::sync::Arc<dyn crate::extensions::erasure::CellModelFactory>>,
    state: Option<HState>,
}

impl HVariance {
    /// A variance ensemble of `m_prime` tessellations (H paper default 40),
    /// (ν′, λ′) calibrated from the engine's homoscedastic prior at the first
    /// sweep (§3.3). Fails with [`AddiVortesError::InvalidHyperparameter`]
    /// unless `m_prime ≥ 1`.
    pub fn new(m_prime: usize) -> Result<Self> {
        Ok(Self {
            m_prime: require_at_least_one("m_prime", m_prime)?,
            prior: None,
            kernel: None,
            state: None,
        })
    }

    /// Pin (ν′, λ′) explicitly instead of calibrating from the context
    /// (scaled space for λ′; the statistical gates pin so the generating
    /// and fitted priors coincide). Fails with
    /// [`AddiVortesError::InvalidHyperparameter`] unless both are finite
    /// and strictly positive.
    pub fn with_prior(mut self, nu_prime: f64, lambda_prime: f64) -> Result<Self> {
        self.prior = Some((
            require_positive_finite("nu_prime", nu_prime)?,
            require_positive_finite("lambda_prime", lambda_prime)?,
        ));
        Ok(self)
    }

    /// Replace the variance-cell family (the cell-model point: the cells are an
    /// ordinary [`CellModel`]; last wins). The model owns its own prior
    /// parameters, so any pinned/calibrated (ν′, λ′) only affect the default
    /// inverse-χ² family. Must be `Clone` (the fresh-per-fit factory
    /// semantics of config-level selection).
    #[must_use]
    pub fn with_cell_model<M: CellModel + Clone + 'static>(mut self, model: M) -> Self {
        self.kernel = Some(std::sync::Arc::new(model));
        self
    }

    /// The current per-observation variance values `s²(xᵢ)` (scaled
    /// space, ascending index); `None` before the first sweep. The H
    /// diagnostics (`diagnostics::h_evidence`) read posterior draws of this.
    pub fn s_sq_values(&self) -> Option<&[f64]> {
        self.state.as_ref().map(|state| state.fit.as_slice())
    }

    /// The variance ensemble's current tessellations (scaled space; cell
    /// values are variance factors); `None` before the first sweep.
    pub fn tessellations(&self) -> Option<&[Tessellation]> {
        self.state
            .as_ref()
            .map(|state| state.ensemble.tessellations.as_slice())
    }

    /// The (ν′, λ′) actually in force (dimensionless prior parameters, λ′ in
    /// scaled-space response units); `None` before the first sweep unless
    /// pinned.
    pub fn prior_in_force(&self) -> Option<(f64, f64)> {
        self.prior
            .or_else(|| self.state.as_ref().map(|state| state.prior))
    }

    fn init_state(&mut self, ctx: &ScaleCtx<'_>) -> Result<()> {
        let n = ctx.x().n_rows();
        let prior = match self.prior {
            Some(prior) => prior,
            None => {
                if ctx.nu() <= 2.0 {
                    return Err(AddiVortesError::InvalidHyperparameter {
                        name: "nu".into(),
                        reason: "the H §3.3 calibration needs ν > 2 (finite prior mean); \
                                 pin (ν′, λ′) via HVariance::with_prior instead"
                            .into(),
                    });
                }
                // A zero-residual response calibrates λ to exactly 0, which
                // would pin every variance cell at 0 (λ′ = 0).
                if ctx.calibrated_lambda() <= 0.0 {
                    return Err(AddiVortesError::DegenerateResidual {});
                }
                h_variance_prior(ctx.nu(), ctx.calibrated_lambda(), self.m_prime)?
            }
        };
        let init = Tessellation {
            centres: vec![0.0],
            dims: vec![0],
            mus: vec![1.0],
        };
        let tessellations = vec![init; self.m_prime];
        let assignments = vec![AssignmentCache::new(vec![0usize; n], Vec::new()); self.m_prime];
        let kernel = match &self.kernel {
            Some(factory) => factory.kernel(),
            None => Box::new(KernelOf(InvChiSqCellModel::new(prior.0, prior.1)?))
                as Box<dyn crate::extensions::erasure::ErasedCellKernel>,
        };
        self.state = Some(HState {
            ensemble: EnsembleUnit::new(
                tessellations,
                assignments,
                kernel,
                Composition::Multiplicative,
            ),
            fit: vec![1.0; n],
            precisions: vec![1.0; n],
            e_sq: vec![0.0; n],
            prior,
        });
        Ok(())
    }
}

impl HVariance {
    /// Test-only (Geweke successive-conditional simulator): set the variance
    /// ensemble's tessellation state to an externally-drawn value, recomputing
    /// assignments, the running product and the precisions. Consumes no RNG.
    /// Requires the prior to be pinned (`with_prior`): the gates pin it so
    /// the generating and fitted priors coincide.
    #[cfg(test)]
    pub(crate) fn set_state_for_tests(
        &mut self,
        tessellations: Vec<Tessellation>,
        x: &crate::engine::data::Data,
        assigner: &dyn crate::extensions::distance::CellAssigner,
    ) -> Result<()> {
        debug_assert_eq!(tessellations.len(), self.m_prime);
        let prior = self.prior.expect("pin (ν′, λ′) before injecting state");
        let n = x.n_rows();
        let mut assignments = Vec::with_capacity(tessellations.len());
        let mut fit = vec![1.0_f64; n];
        for tessellation in &tessellations {
            let assignment = assigner.assign_cells(x, tessellation)?;
            for i in 0..n {
                fit[i] *= tessellation.mus[assignment[i]];
            }
            assignments.push(AssignmentCache::new(assignment, Vec::new()));
        }
        let precisions: Vec<f64> = fit.iter().map(|s_sq| 1.0 / s_sq).collect();
        let kernel = match &self.kernel {
            Some(factory) => factory.kernel(),
            None => Box::new(KernelOf(InvChiSqCellModel::new(prior.0, prior.1)?))
                as Box<dyn crate::extensions::erasure::ErasedCellKernel>,
        };
        self.state = Some(HState {
            ensemble: EnsembleUnit::new(
                tessellations,
                assignments,
                kernel,
                Composition::Multiplicative,
            ),
            fit,
            precisions,
            e_sq: vec![0.0; n],
            prior,
        });
        Ok(())
    }
}

impl ScaleModel for HVariance {
    type Error = AddiVortesError;

    /// One variance-ensemble Gibbs pass (H paper §3.2): squared
    /// mean-residuals `e²ᵢ = (yᵢ − Fᵢ)²` become the ensemble's working
    /// response; each of the m′ tessellations runs the shared backfit block
    /// (structural MH + conjugate s² redraw) under the multiplicative
    /// composition; the precisions `1/S_i` are refreshed for the conductor.
    fn update(
        &mut self,
        ctx: &ScaleCtx<'_>,
        rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        if self.state.is_none() {
            self.init_state(ctx)?;
        }
        let state = self.state.as_mut().expect("initialised above");
        let (y, fit) = (ctx.y(), ctx.fit());
        for i in 0..y.len() {
            let e = y[i] - fit[i];
            state.e_sq[i] = e * e;
        }
        // The variance ensemble's own move context: the shared structural
        // machinery (coordinate laws, inclusion weights, count priors via
        // ω/λ_c) with σ²/σ_μ² irrelevant to the InvChiSq kernel (its noise
        // level is the cell value itself); 1.0 keeps the ModelCtx valid.
        let mctx = ModelCtx::new(
            1.0,
            ctx.omega(),
            ctx.lambda_c(),
            1.0,
            ctx.p_enc(),
            ctx.coord_dists(),
            ctx.inclusion_weights(),
        );
        for l in 0..self.m_prime {
            state.ensemble.backfit_one(
                l,
                ctx.x(),
                &state.e_sq,
                &mut state.fit,
                None,
                &mctx,
                ctx.move_set(),
                ctx.assigner(),
                rng,
            )?;
        }
        for (precision, s_sq) in state.precisions.iter_mut().zip(&state.fit) {
            *precision = 1.0 / s_sq;
        }
        Ok(())
    }

    /// σ² ≡ 1 (scaled space): the noise level lives entirely in the
    /// per-observation precisions `1/s²(xᵢ)`.
    fn sigma_sq(&self) -> f64 {
        1.0
    }

    fn precisions(&self) -> Option<&[f64]> {
        self.state.as_ref().map(|state| state.precisions.as_slice())
    }

    /// The whole point of this model: σ²(x) varies, so every observation
    /// carries its own precision. Declared here because `precisions()` above is
    /// still `None` until the first `update` builds the ensemble, and the engine
    /// must know *before* it assembles the mean cells.
    fn heteroscedastic(&self) -> bool {
        true
    }
}

/// Cloning resets the sampling state (the fresh-per-fit factory semantics of
/// config-level selection): the spec (m′, any pinned prior, any replacement
/// cell family) is copied; the lazily-initialised ensemble is not.
impl Clone for HVariance {
    fn clone(&self) -> Self {
        Self {
            m_prime: self.m_prime,
            prior: self.prior,
            kernel: self.kernel.clone(),
            state: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_rel_eq;
    use crate::{AddiVortesConfig, Data};

    /// `HVariance` selected on its own must fit.
    ///
    /// It did not: the model publishes per-observation precisions, which make
    /// the accumulation weights fractional, and the engine still assembled the
    /// *hard-assignment* `GaussianCellStats` — which asserts its weights are 1.
    /// The fit panicked partway through the first sweep. The crate's own H
    /// sampler paired `WeightedGaussianModel` by hand and no public API said a
    /// caller had to, so the only heteroscedastic entry on the scale shelf was
    /// unusable through `with_scale_model`.
    #[test]
    fn h_variance_fits_without_hand_pairing_a_weighted_cell_model() {
        let values: Vec<f64> = (0..40)
            .flat_map(|i| [i as f64 / 40.0, ((i % 5) as f64) / 5.0])
            .collect();
        let x = Data::new(values, 40, 2).unwrap();
        // Residual scatter, or the σ² prior calibrates λ to zero.
        let y: Vec<f64> = (0..40)
            .map(|i| {
                let t = i as f64 / 40.0;
                2.0 * t - 0.5 + 0.3 * ((i * 7 % 11) as f64 / 11.0 - 0.5)
            })
            .collect();

        crate::engine::builder::SamplerBuilder::new(
            AddiVortesConfig::new(1)
                .with_m(2)
                .with_omega(1.0)
                .with_burn_in(2)
                .with_draws(2),
        )
        .with_scale_model(HVariance::new(5).unwrap())
        .fit(&x, &y)
        .expect("a heteroscedastic scale model fits on its own");
    }

    /// A response with no residual variation (an exact linear function of
    /// the features) calibrates λ to exactly 0, which the §3.3 matching would
    /// carry into λ′ = 0: every variance cell pinned at zero. The fit must
    /// refuse at the boundary rather than run that chain.
    #[test]
    fn zero_residual_data_is_a_degenerate_residual_error() {
        let n = 50;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&v| 2.0 * v).collect();
        let x = Data::new(xs, n, 1).unwrap();
        let err = crate::engine::builder::SamplerBuilder::new(
            AddiVortesConfig::new(7)
                .with_m(5)
                .with_omega(0.5)
                .with_burn_in(5)
                .with_draws(5),
        )
        .with_scale_model(HVariance::new(10).unwrap())
        .fit(&x, &ys)
        .unwrap_err();
        assert_eq!(err, AddiVortesError::DegenerateResidual {});
    }

    /// The constructor arguments are checked in every build profile.
    #[test]
    fn constructors_reject_out_of_domain_arguments() {
        assert!(matches!(
            HVariance::new(0),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "m_prime"
        ));
        assert!(matches!(
            HVariance::new(3).unwrap().with_prior(0.0, 0.5),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "nu_prime"
        ));
        assert!(matches!(
            HVariance::new(3).unwrap().with_prior(4.0, f64::NAN),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "lambda_prime"
        ));
        assert!(matches!(
            h_variance_prior(2.0, 0.02, 4),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "nu"
        ));
        assert!(matches!(
            h_variance_prior(6.0, 0.0, 4),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "lambda"
        ));
        assert!(matches!(
            h_variance_prior(6.0, 0.02, 0),
            Err(AddiVortesError::InvalidHyperparameter { ref name, .. }) if name == "m_prime"
        ));
    }

    /// The claim the engine reads at assembly. `precisions()` cannot answer it:
    /// it is `None` until the first `update` builds the ensemble.
    #[test]
    fn h_variance_declares_itself_heteroscedastic_before_any_update() {
        let model = HVariance::new(5).unwrap();
        assert!(
            model.precisions().is_none(),
            "no state before the first update"
        );
        assert!(
            ScaleModel::heteroscedastic(&model),
            "the declaration must not depend on the state existing yet"
        );
    }

    /// The §3.3 calibration against independently-computed reference values
    /// (ν = 6, the paper's default; λ = 0.02, a representative data-calibrated
    /// value — the paper parameterises the σ² prior by (ν, q) = (6, 0.85) and
    /// derives λ from the data, so λ has no published default; m′ = 40, the
    /// H paper's default), plus the matching
    /// identities the derivation must satisfy: λ′^m′ = λ and
    /// (ν′/(ν′−2))^m′ = ν/(ν−2).
    #[test]
    fn h_prior_calibration_matches_the_paper_matching() {
        let (nu_prime, lambda_prime) = h_variance_prior(6.0, 0.02, 40).unwrap();
        assert_rel_eq(lambda_prime, 0.906_829_730_118_553_8, 1e-12);
        assert_rel_eq(nu_prime, 198.305_966_425_172_73, 1e-10);
        assert_rel_eq(mathsfn::powf(lambda_prime, 40.0), 0.02, 1e-12);
        assert_rel_eq(
            mathsfn::powf(nu_prime / (nu_prime - 2.0), 40.0),
            6.0 / (6.0 - 2.0),
            1e-10,
        );
        // m′ = 1 must be the identity.
        let (nu_prime, lambda_prime) = h_variance_prior(6.0, 0.02, 1).unwrap();
        assert_rel_eq(nu_prime, 6.0, 1e-12);
        assert_rel_eq(lambda_prime, 0.02, 1e-12);
    }

    /// The multiplicative fit bookkeeping: after any number of sweeps the
    /// running product S_i must equal the product of the tessellations' cell
    /// values at each row, and the reported precisions must be 1/S_i.
    #[test]
    fn running_product_matches_recomputed_product() {
        use rand_chacha::ChaCha8Rng;
        use rand_core::SeedableRng;

        let n = 40;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let x = crate::engine::data::Data::new(xs, n, 1).unwrap();
        let y: Vec<f64> = (0..n).map(|i| 0.3 * (i as f64 / n as f64) - 0.15).collect();
        let fit = vec![0.0_f64; n];
        let move_set = crate::extensions::moves::default_move_set().unwrap();
        let assigner = crate::extensions::distance::default_assigner(vec![
            crate::engine::data::Metric::Euclidean,
        ]);
        let coord_dists: Vec<std::sync::Arc<dyn crate::extensions::coord::CoordinateDistribution>> =
            vec![std::sync::Arc::new(
                crate::extensions::coord::EuclideanNormal::new(0.8).unwrap(),
            )];
        let weights = [1.0_f64];
        let ctx = ScaleCtx {
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
            weights_enc: &weights,
            response_weights: None,
        };
        let mut model = HVariance::new(4).unwrap().with_prior(8.0, 0.3).unwrap();
        let mut rng = ChaCha8Rng::from_seed([21; 32]);
        for _ in 0..25 {
            ScaleModel::update(&mut model, &ctx, &mut rng).unwrap();
        }
        let s_sq = model.s_sq_values().unwrap().to_vec();
        let tessellations = model.tessellations().unwrap();
        for (row, &running) in s_sq.iter().enumerate() {
            let mut product = 1.0_f64;
            for tessellation in tessellations {
                let assignment = assigner.assign_cells(&x, tessellation).unwrap();
                product *= tessellation.mus()[assignment[row]];
            }
            crate::test_support::assert_rel_eq(running, product, 1e-9);
        }
        let precisions = ScaleModel::precisions(&model).unwrap();
        for (precision, s_sq) in precisions.iter().zip(&s_sq) {
            assert_eq!(precision.to_bits(), (1.0 / s_sq).to_bits());
        }
    }
}
