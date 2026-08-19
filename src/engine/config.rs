//! Model configuration (`AddiVortesConfig`): mandatory-seed constructor,
//! consuming `with_*` setters (never panic, never clamp), data-free
//! `validate()`, and the `fit()` convenience loop.

use std::sync::Arc;

use crate::engine::data::{Data, Metric};
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::model::FittedAddiVortes;
use crate::engine::model::ResponseFamily;
use crate::extensions::basis::CellBasis;
use crate::extensions::cell_model::CellModel;
use crate::extensions::coord::CoordinateDistribution;
use crate::extensions::count_priors::CountPriors;
use crate::extensions::distance::{CellAssigner, PairwiseDistance};
use crate::extensions::erasure::{CellModelFactory, ResponseModelFactory, ScaleModelFactory};
use crate::extensions::inclusion::{ErasedInclusionModel, InclusionModel};
use crate::extensions::membership::MembershipKernel;
use crate::extensions::moves::MoveSet;
use crate::extensions::response::ResponseModel;
use crate::extensions::scale::ScaleModel;

/// Configuration for an AddiVortes fit: mandatory seed, hyperparameters at the
/// paper's defaults (λ_c excepted — see the field note), swappable extension points at
/// their built-in defaults.
///
/// No `Default` impl on purpose: the seed is mandatory (reproducibility
/// contract).
/// Setters are consuming and never panic or clamp; all checking happens in
/// [`validate`](AddiVortesConfig::validate) (called by `fit`).
#[derive(Debug, Clone)]
pub struct AddiVortesConfig {
    /// Chain seed (expanded to the ChaCha8 key via splitmix64).
    pub(crate) seed: u64,
    /// Ensemble size m (paper default 200).
    pub(crate) m: usize,
    /// σ² prior degrees of freedom ν (paper default 6).
    pub(crate) nu: f64,
    /// σ² prior calibration quantile q (paper default 0.85).
    pub(crate) q: f64,
    /// μ prior spread parameter k (paper default 3): σ_μ = 0.5/(k√m).
    pub(crate) k: f64,
    /// Centre-coordinate prior/proposal spread σ_c (paper default 0.8).
    pub(crate) sigma_c: f64,
    /// Dimension-count prior parameter ω (paper default 3).
    pub(crate) omega: f64,
    /// Centre-count prior parameter λ_c (default 5, chosen by benchmark
    /// calibration of the exact shifted-Poisson prior; the paper reports 25).
    pub(crate) lambda_c: f64,
    /// Burn-in sweeps discarded by `fit` (default 200).
    pub(crate) burn_in: usize,
    /// Posterior draws kept by `fit` (default 1000).
    pub(crate) n_draws: usize,
    /// Thinning interval for `fit` (default 1 = keep every sweep).
    pub(crate) thinning: usize,
    /// Per-raw-column metrics; `None` = all Euclidean.
    pub(crate) metrics: Option<Vec<Metric>>,
    /// Move set; `None` = the paper set.
    pub(crate) move_set: Option<Arc<MoveSet>>,
    /// Per-raw-column coordinate laws; `None` = the defaults
    /// (`EuclideanNormal`, `WrappedNormal` on spherical columns).
    pub(crate) coords: Option<Vec<Arc<dyn CoordinateDistribution>>>,
    /// Custom cell assigner; `None` = built-in `ColumnMetrics`.
    pub(crate) assigner: Option<Arc<dyn CellAssigner>>,
    /// Soft-membership kernel; `None` = hard
    /// membership (the golden-pinned diagonal path).
    pub(crate) membership: Option<Arc<dyn MembershipKernel>>,
    /// Response family (the predict-side half):
    /// selects the link and the per-variant predictive distribution.
    pub(crate) family: ResponseFamily,
    /// Inclusion model; `None` = `UniformInclusion`.
    pub(crate) inclusion: Option<Arc<dyn ErasedInclusionModel>>,
    /// Cell model (deep seam); `None` = `GaussianCellModel`.
    pub(crate) cell_model: Option<Arc<dyn CellModelFactory>>,
    /// Cell basis; `None` = the scalar payload. Required by, and only
    /// valid for, a cell model whose `cell_basis()` is true.
    pub(crate) basis: Option<Arc<dyn CellBasis>>,
    /// Kernel step (deep seam); `None` = no augmentation.
    pub(crate) response_model: Option<Arc<dyn ResponseModelFactory>>,
    /// Scale model (deep seam); `None` = `GlobalSigma`.
    pub(crate) scale_model: Option<Arc<dyn ScaleModelFactory>>,
    /// Count priors; `None` = `ShiftedPoissonBinomial`, the paper's
    /// shifted Poisson / shifted Binomial pair.
    pub(crate) count_priors: Option<Arc<dyn CountPriors>>,
    /// Cell-value prior SD σ_μ, set directly (scaled space); `None` = the
    /// k-rule σ_μ = 0.5/(k√m), or the family's own rule (BinaryProbit widens
    /// to 3/(k√m)). Crate-internal: the dial a model file turns when its
    /// derivation prescribes a σ_μ the k-rule cannot express.
    pub(crate) cell_prior_sd: Option<f64>,
}

impl AddiVortesConfig {
    /// A configuration with the shipped defaults (the paper's, except
    /// λ_c = 5) and the mandatory chain seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            m: 200,
            nu: 6.0,
            q: 0.85,
            k: 3.0,
            sigma_c: 0.8,
            omega: 3.0,
            lambda_c: 5.0,
            burn_in: 200,
            n_draws: 1000,
            thinning: 1,
            metrics: None,
            move_set: None,
            coords: None,
            assigner: None,
            membership: None,
            family: ResponseFamily::Gaussian,
            inclusion: None,
            cell_model: None,
            basis: None,
            response_model: None,
            scale_model: None,
            count_priors: None,
            cell_prior_sd: None,
        }
    }

    /// Replace the chain seed (last wins).
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Ensemble size m.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// σ² prior degrees of freedom ν.
    #[must_use]
    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// σ² prior calibration quantile q (Pr(σ < σ̂) = q).
    #[must_use]
    pub fn with_q(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    /// μ prior spread parameter k (σ_μ = 0.5/(k√m)).
    #[must_use]
    pub fn with_k(mut self, k: f64) -> Self {
        self.k = k;
        self
    }

    /// Centre-coordinate prior/proposal spread σ_c.
    #[must_use]
    pub fn with_sigma_c(mut self, sigma_c: f64) -> Self {
        self.sigma_c = sigma_c;
        self
    }

    /// Dimension-count prior parameter ω (must satisfy ω < p at fit).
    #[must_use]
    pub fn with_omega(mut self, omega: f64) -> Self {
        self.omega = omega;
        self
    }

    /// Centre-count prior parameter λ_c (default 5; the paper reports 25,
    /// reproduced with `with_lambda_c(25.0)`).
    #[must_use]
    pub fn with_lambda_c(mut self, lambda_c: f64) -> Self {
        self.lambda_c = lambda_c;
        self
    }

    /// Burn-in sweeps discarded by `fit`.
    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Posterior draws kept by `fit`.
    #[must_use]
    pub fn with_draws(mut self, n_draws: usize) -> Self {
        self.n_draws = n_draws;
        self
    }

    /// Thinning interval for `fit` (keep every `thinning`-th sweep).
    #[must_use]
    pub fn with_thinning(mut self, thinning: usize) -> Self {
        self.thinning = thinning;
        self
    }

    /// Per-column metrics for the raw (pre-encoding) columns; the default
    /// is all-Euclidean. Last wins.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Vec<Metric>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// A custom move set (last wins; the default is the paper's
    /// six moves, `MoveSetBuilder::stone_gosling()`).
    #[must_use]
    pub fn with_move_set(mut self, move_set: MoveSet) -> Self {
        // Single-threaded sharing between config and sampler; Send + Sync is
        // never required of move sets (the j-loop is never parallelised
        // within a chain).
        #[allow(clippy::arc_with_non_send_sync)]
        let move_set = Arc::new(move_set);
        self.move_set = Some(move_set);
        self
    }

    /// Custom centre-coordinate laws (last wins), one per
    /// caller-visible pre-encoding column, expanded alongside one-hot
    /// encoding exactly like `Vec<Metric>` (a one-hot group shares one law).
    /// The default pairs `EuclideanNormal` with Euclidean columns and
    /// `WrappedNormal` with spherical ones. Each law is used as both the
    /// centre prior and the within-move proposal; the two must coincide (the
    /// trait contract).
    #[must_use]
    pub fn with_coords(mut self, coords: Vec<Arc<dyn CoordinateDistribution>>) -> Self {
        self.coords = Some(coords);
        self
    }

    /// A custom assignment geometry (last wins; the default is
    /// the compound `ColumnMetrics`). Sugar over
    /// [`with_assigner`](AddiVortesConfig::with_assigner) for the common
    /// case, a plain [`PairwiseDistance`]. `Vec<Metric>` still governs
    /// scaling and validation; a custom metric changes assignment geometry
    /// only.
    #[must_use]
    pub fn with_distance<D: PairwiseDistance + 'static>(mut self, distance: D) -> Self {
        self.assigner = Some(Arc::new(distance));
        self
    }

    /// A custom cell assigner (last wins). The batch-level entry:
    /// implement [`with_distance`](AddiVortesConfig::with_distance)'s
    /// `PairwiseDistance` instead unless you need to own the whole n×nC loop.
    #[must_use]
    pub fn with_assigner(mut self, assigner: Arc<dyn CellAssigner>) -> Self {
        self.assigner = Some(assigner);
        self
    }

    /// The response family (predict side; last wins; the default
    /// is [`ResponseFamily::Gaussian`], the paper's model). Selecting
    /// [`ResponseFamily::BinaryProbit`] turns the fit into Binary-AddiVortes:
    /// the response must be {0, 1} labels, the Albert–Chib augmentation and
    /// the pinned unit scale attach automatically (unless a custom kernel
    /// step / scale model overrides them), the cell-value prior widens to
    /// the ±3 latent range (σ_μ = 3/(k√m)), and every prediction entry
    /// speaks the probability scale through the probit link. Selecting
    /// [`ResponseFamily::RobustT`] keeps the identity link and the response
    /// scale but swaps the error law for Student-t of the given `df`: the
    /// scale-mixture augmentation, the weight-aware mean family and the
    /// precision-weighted σ² draw attach automatically, and prediction
    /// intervals come from the t-mixture predictive.
    #[must_use]
    pub fn with_response_family(mut self, family: ResponseFamily) -> Self {
        self.family = family;
        self
    }

    /// The configured response family (what
    /// [`with_response_family`](AddiVortesConfig::with_response_family) set,
    /// or the Gaussian default). The read-side bindings and loaders need to
    /// know which scale a fitted model's predictions speak.
    pub fn response_family(&self) -> ResponseFamily {
        self.family
    }

    /// Soft membership (last wins; the default is hard
    /// membership, the golden-pinned diagonal path): switch the mean
    /// ensemble onto the dense path with this membership kernel. The
    /// per-tessellation payload draws become joint b×b draws, the
    /// incremental assignment cache is bypassed (memberships are fully
    /// recomputed per proposal), and prediction weights cells through the
    /// same assigner + kernel as fitting. The configured assigner must
    /// provide dense per-cell keys (`CellAssigner::membership_keys`; every
    /// `PairwiseDistance` does).
    #[must_use]
    pub fn with_membership<K: MembershipKernel + 'static>(mut self, kernel: K) -> Self {
        self.membership = Some(Arc::new(kernel));
        self
    }

    /// A custom inclusion model (last wins; the default is
    /// `UniformInclusion`, exactly the paper's model). The model must be
    /// `Clone` (the sampler takes its own mutable copy).
    #[must_use]
    pub fn with_inclusion<M: InclusionModel + Clone + 'static>(mut self, model: M) -> Self {
        self.inclusion = Some(Arc::new(model));
        self
    }

    /// A custom cell model (deep seam; last wins; the default is
    /// `GaussianCellModel`). The model must be `Clone` (the config mints a
    /// fresh copy per fit); `Sampler::with_cell_model` takes non-Clone models.
    #[must_use]
    pub fn with_cell_model<M: CellModel + Clone + 'static>(mut self, model: M) -> Self {
        self.cell_model = Some(Arc::new(model));
        self
    }

    /// The cell basis (last wins): what the per-observation basis row
    /// z(x) is, for a payload whose cells are linear in covariates rather than
    /// constant. A cell's contribution to the fit becomes `z(xᵢ) · β_k`.
    ///
    /// Required by a basis payload (a [`CellModel`] whose `cell_basis()` is
    /// true, e.g. [`LinearGaussianModel`](crate::basis::LinearGaussianModel)),
    /// and rejected for a scalar one; its `q` must match the payload's width.
    /// All three are checked at `fit`.
    #[must_use]
    pub fn with_cell_basis<B: CellBasis + 'static>(mut self, basis: B) -> Self {
        self.basis = Some(Arc::new(basis));
        self
    }

    /// A data-augmentation kernel step (deep seam; last wins; the default
    /// is none: zero copies, zero RNG). Must be `Clone` (fresh copy per
    /// fit); `Sampler::with_response_model` takes non-Clone steps.
    #[must_use]
    pub fn with_response_model<K: ResponseModel + Clone + 'static>(mut self, step: K) -> Self {
        self.response_model = Some(Arc::new(step));
        self
    }

    /// A custom scale model (deep seam; last wins; the default is the
    /// global σ² Gibbs draw). Must be `Clone` (fresh copy per fit);
    /// `Sampler::with_scale_model` takes non-Clone models.
    #[must_use]
    pub fn with_scale_model<S: ScaleModel + Clone + 'static>(mut self, scale: S) -> Self {
        self.scale_model = Some(Arc::new(scale));
        self
    }

    /// Custom count priors (last wins; the default is
    /// [`ShiftedPoissonBinomial`](crate::extensions::count_priors::ShiftedPoissonBinomial),
    /// the paper's shifted Poisson(λ_c) cell count and shifted
    /// Binomial(p−1, ω/p) dimension count).
    ///
    /// Every move prices structures through these two hooks, so selecting a
    /// prior here changes the acceptance ratio of every count-changing move
    /// and **nothing else**: a custom count prior never touches a move. The
    /// default is bit-identical to the pre-hook pricing, so leaving this unset
    /// leaves the sampled chain exactly as it was.
    ///
    /// Validate a new prior with
    /// [`conformance::check_count_priors`](crate::conformance::check_count_priors)
    /// first; it is a local check, and only SBC can show the ratios describe a
    /// prior the sampler actually targets.
    #[must_use]
    pub fn with_count_priors<C: CountPriors + 'static>(mut self, priors: C) -> Self {
        self.count_priors = Some(Arc::new(priors));
        self
    }

    /// The cell-value prior SD σ_μ, directly (last wins; scaled space; the
    /// default is the k-rule σ_μ = 0.5/(k√m), and BinaryProbit's family
    /// wiring widens to 3/(k√m)). The crate-internal width dial for model
    /// files whose derivation prescribes σ_μ itself; when set it wins over
    /// both the k-rule and the family rule, and `k` no longer reaches σ_μ.
    #[must_use]
    pub(crate) fn with_cell_prior_sd(mut self, sigma_mu: f64) -> Self {
        self.cell_prior_sd = Some(sigma_mu);
        self
    }

    /// Fit `n_chains` independent chains of the same model (the
    /// multi-chain entry the convergence diagnostics consume:
    /// [`diagnostics::r_hat`](crate::diagnostics::r_hat) and friends take
    /// per-chain draw vectors). Chain 0 runs this configuration's own seed,
    /// so its output is bit-identical to a single [`fit`]: the
    /// derivation never perturbs the single-chain contract; chains 1.. use
    /// seeds derived from it by successive splitmix64 outputs (pinned, so
    /// "seed S, chain k" names exactly one chain).
    ///
    /// [`fit`]: AddiVortesConfig::fit
    pub fn fit_chains(
        &self,
        x: &Data,
        y: &[f64],
        n_chains: usize,
    ) -> Result<Vec<FittedAddiVortes>> {
        if n_chains == 0 {
            return Err(AddiVortesError::InvalidHyperparameter {
                name: "n_chains".into(),
                reason: "must be at least 1".into(),
            });
        }
        let mut state = self.seed;
        (0..n_chains)
            .map(|chain| {
                let mut config = self.clone();
                if chain > 0 {
                    config.seed = crate::engine::sampler::splitmix64(&mut state);
                }
                config.fit(x, y)
            })
            .collect()
    }

    /// Data-free validation of every hyperparameter (fit calls this first).
    /// Never clamps; every failure is `InvalidHyperparameter` with the exact
    /// field name. ω is required positive here; the ω < p check needs data
    /// and happens at the fit boundary.
    pub fn validate(&self) -> Result<()> {
        let bad = |name: &str, reason: String| {
            Err(AddiVortesError::InvalidHyperparameter {
                name: name.into(),
                reason,
            })
        };
        if self.m < 1 {
            return bad("m", "must be at least 1".into());
        }
        if !(self.nu.is_finite() && self.nu > 0.0) {
            return bad(
                "nu",
                format!("must be finite and positive, got {}", self.nu),
            );
        }
        if !(self.q.is_finite() && self.q > 0.0 && self.q < 1.0) {
            return bad(
                "q",
                format!("must be in the open interval (0, 1), got {}", self.q),
            );
        }
        if !(self.k.is_finite() && self.k > 0.0) {
            return bad("k", format!("must be finite and positive, got {}", self.k));
        }
        if !(self.sigma_c.is_finite() && self.sigma_c > 0.0) {
            return bad(
                "sigma_c",
                format!("must be finite and positive, got {}", self.sigma_c),
            );
        }
        if !(self.omega.is_finite() && self.omega > 0.0) {
            return bad(
                "omega",
                format!("must be finite and positive, got {}", self.omega),
            );
        }
        if !(self.lambda_c.is_finite() && self.lambda_c > 0.0) {
            return bad(
                "lambda_c",
                format!("must be finite and positive, got {}", self.lambda_c),
            );
        }
        if let Some(sd) = self.cell_prior_sd {
            if !(sd.is_finite() && sd > 0.0) {
                return bad(
                    "cell_prior_sd",
                    format!("must be finite and positive, got {sd}"),
                );
            }
        }
        if self.n_draws < 1 {
            return bad("draws", "must be at least 1".into());
        }
        if self.thinning < 1 {
            return bad("thinning", "must be at least 1".into());
        }
        if let Some(model) = &self.inclusion {
            if let Some(w) = model
                .weights()
                .iter()
                .find(|w| !w.is_finite() || **w <= 0.0)
            {
                return bad(
                    "inclusion_weights",
                    format!("every weight must be finite and positive, got {w}"),
                );
            }
        }
        Ok(())
    }

    /// Fit the model: validate the configuration, run the boundary checks, and
    /// drive the [`Sampler`](crate::Sampler) for `burn_in + draws × thinning`
    /// sweeps, keeping every `thinning`-th sweep after burn-in.
    ///
    /// Consumes the configuration and stores it in the fitted model
    /// (self-contained).
    pub fn fit(self, x: &Data, y: &[f64]) -> Result<FittedAddiVortes> {
        crate::engine::model::fit(self, x, y)
    }

    /// Like [`fit`](AddiVortesConfig::fit), with a caller-supplied move set.
    pub fn fit_with_move_set(
        self,
        x: &Data,
        y: &[f64],
        move_set: MoveSet,
    ) -> Result<FittedAddiVortes> {
        crate::engine::model::fit_with_move_set(self, x, y, move_set)
    }
}

/// Compare two optional `Arc`s by pointer identity (`Arc::ptr_eq`), the
/// same hand-rolled `PartialEq` precedent as `AddiVortesError::Extension`.
fn arc_opt_eq<T: ?Sized>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// Value equality on every plain field; every `Arc`-held component compares by
/// pointer identity (see `arc_opt_eq`).
///
/// `self` is destructured with **no** `..` rest pattern, deliberately. An omitted
/// field makes two configs that would fit *different models* compare equal, which
/// silently guts any test written as `assert_eq!(built, expected)`: it can no
/// longer fail on the very component it means to pin. `family`, `membership` and
/// `basis` were all missing once. Naming every field turns the next omission into
/// a compile error rather than a quietly toothless test suite.
impl PartialEq for AddiVortesConfig {
    fn eq(&self, other: &Self) -> bool {
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
            move_set,
            coords,
            assigner,
            membership,
            inclusion,
            cell_model,
            basis,
            response_model,
            scale_model,
            count_priors,
            cell_prior_sd,
        } = self;

        let coords_equal = match (coords, &other.coords) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y))
            }
            _ => false,
        };
        let arcs_equal = arc_opt_eq(assigner, &other.assigner)
            && arc_opt_eq(inclusion, &other.inclusion)
            && arc_opt_eq(move_set, &other.move_set)
            && arc_opt_eq(cell_model, &other.cell_model)
            && arc_opt_eq(response_model, &other.response_model)
            && arc_opt_eq(scale_model, &other.scale_model)
            && arc_opt_eq(count_priors, &other.count_priors)
            && arc_opt_eq(membership, &other.membership)
            && arc_opt_eq(basis, &other.basis)
            && coords_equal;
        arcs_equal
            && *seed == other.seed
            && *m == other.m
            && *nu == other.nu
            && *q == other.q
            && *k == other.k
            && *sigma_c == other.sigma_c
            && *omega == other.omega
            && *lambda_c == other.lambda_c
            && *burn_in == other.burn_in
            && *n_draws == other.n_draws
            && *thinning == other.thinning
            && *metrics == other.metrics
            && *family == other.family
            && *cell_prior_sd == other.cell_prior_sd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResponseFamily;
    use crate::extensions::basis::LinearBasis;
    use crate::extensions::cell_model::GaussianCellModel;
    use crate::extensions::coord::EuclideanNormal;
    use crate::extensions::count_priors::ShiftedPoissonBinomial;
    use crate::extensions::distance::Manhattan;
    use crate::extensions::inclusion::UniformInclusion;
    use crate::extensions::membership::SoftmaxKernel;
    use crate::extensions::moves::MoveSetBuilder;
    use crate::extensions::response::RobustTStep;
    use crate::extensions::scale::PinnedSigma;

    /// Every field must take part in `PartialEq`. A field left out of the `eq`
    /// makes two configs that fit *different models* compare equal, which turns
    /// every `assert_eq!(built, expected)` elsewhere into a test that cannot fail
    /// on that extension point. `family`, `membership` and `basis` were each omitted once.
    ///
    /// Each case perturbs exactly one field of an otherwise-identical config and
    /// demands inequality.
    #[test]
    fn every_field_participates_in_equality() {
        let base = || AddiVortesConfig::new(1);
        let cases: Vec<(&str, AddiVortesConfig)> = vec![
            ("seed", base().with_seed(2)),
            ("m", base().with_m(7)),
            ("nu", base().with_nu(9.0)),
            ("q", base().with_q(0.5)),
            ("k", base().with_k(1.5)),
            ("sigma_c", base().with_sigma_c(0.25)),
            ("omega", base().with_omega(1.5)),
            ("lambda_c", base().with_lambda_c(25.0)),
            ("burn_in", base().with_burn_in(3)),
            ("n_draws", base().with_draws(3)),
            ("thinning", base().with_thinning(3)),
            ("metrics", base().with_metrics(vec![Metric::Spherical])),
            (
                "family",
                base().with_response_family(ResponseFamily::BinaryProbit),
            ),
            (
                "move_set",
                base().with_move_set(MoveSetBuilder::stone_gosling().build().unwrap()),
            ),
            (
                "coords",
                base().with_coords(vec![Arc::new(EuclideanNormal::new(0.4).unwrap())]),
            ),
            ("assigner", base().with_distance(Manhattan)),
            (
                "membership",
                base().with_membership(SoftmaxKernel::new(0.5).unwrap()),
            ),
            ("inclusion", base().with_inclusion(UniformInclusion::new(2))),
            (
                "cell_model",
                base().with_cell_model(GaussianCellModel::new(0.1).unwrap()),
            ),
            ("basis", base().with_cell_basis(LinearBasis::new(vec![0]))),
            (
                "response_model",
                base().with_response_model(RobustTStep::new(4.0).unwrap()),
            ),
            ("scale_model", base().with_scale_model(PinnedSigma::unit())),
            (
                "count_priors",
                base().with_count_priors(ShiftedPoissonBinomial),
            ),
            ("cell_prior_sd", base().with_cell_prior_sd(0.25)),
        ];

        assert_eq!(base(), base(), "a config must equal itself");
        for (field, perturbed) in cases {
            assert_ne!(
                perturbed,
                base(),
                "changing `{field}` left the config comparing equal: it is missing from `PartialEq`"
            );
        }
    }

    /// The dial validates like every other hyperparameter: never clamped,
    /// rejected with the exact field name.
    #[test]
    fn cell_prior_sd_validates_like_any_hyperparameter() {
        assert!(
            AddiVortesConfig::new(1)
                .with_cell_prior_sd(0.25)
                .validate()
                .is_ok()
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = AddiVortesConfig::new(1)
                .with_cell_prior_sd(bad)
                .validate()
                .unwrap_err();
            assert!(matches!(
                err,
                AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "cell_prior_sd"
            ));
        }
    }
}
