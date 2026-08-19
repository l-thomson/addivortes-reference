//! Serde-serialisable configuration **spec** — the single, binding-agnostic
//! bridge between a data payload (a JSON object from any language binding) and
//! an [`AddiVortesConfig`].
//!
//! This is the crate's *pass-through* contract. A binding never enumerates
//! hyperparameters or shelf items in its own source: it hands over a
//! [`ConfigSpec`] built from a native map — a Python `dict`, an R `list` — and
//! the core maps it. Adding a shelf entry is therefore a **one-place edit
//! here**, and it becomes reachable from *every* binding on rebuild with no
//! binding-side change. Every project that instead hand-mirrors its shelf into
//! each binding drifts: it is not a question of discipline, it is a question of
//! nothing failing when you forget.
//!
//! Only the **shelf** crosses this boundary. Trait-authored components (a
//! researcher's own move, cell model or membership kernel) are native code and
//! cannot be reduced to data — they stay Rust-side, exactly as the extension
//! points intend.
//!
//! # Defaults have exactly one home
//!
//! Every optional field left unset is simply *not applied*, so the crate's own
//! default (and any future retune of it) is what takes effect. A binding never
//! copies a default value, which is the other half of how mirrors drift.
//!
//! # Late binding: why [`into_config`](ConfigSpec::into_config) takes the data
//!
//! Half the shelf's constructors take arguments that do not exist until the data
//! does — `UniformInclusion::new(p)` and `DartInclusion::new(_, p)` need the
//! covariate count; a coordinate law is needed per raw column. A binding always
//! holds `x` before it calls `fit`, so the spec resolves those against it rather
//! than making a user restate what the data already says.
//!
//! Two things are **not** resolvable even then, and are declared by the caller:
//!
//! - Gower's per-column `levels` — the engine cannot know how many levels you
//!   *intend*, only how many happen to appear in this sample.
//! - Mahalanobis' `precision` width and `LinearBasis`' `columns`, which index
//!   the **encoded** (post one-hot) design. That width is a product of fitting
//!   the scaler, so it is checked at `fit` and reported there.
//!
//! # What this spec deliberately does not expose
//!
//! **The cell-model point (`cell_model`) has no variant, on purpose.** Every shipped entry's
//! constructor argument is a value the *engine* derives: `GaussianCellModel`'s
//! σ_μ² comes from `k` and `m` (and is widened again for a probit fit), and
//! `InvChiSqCellModel`'s λ is the data-calibrated one. Exposing them as data
//! would invite a caller to type a number that silently overrides the engine's
//! own calibration — a wrong fit with no error. The one cell model a caller
//! genuinely chooses is the basis payload, and [`BasisSpec`] wires that for
//! them (deriving `q`, which therefore cannot be mismatched).
//!
//! For the same reason `scale` exposes only `pinned` and `h_variance`:
//! `GlobalSigma` and `WeightedGlobalSigma` take the engine's calibrated λ and
//! *are* what it attaches when the point is unset.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::engine::config::AddiVortesConfig;
use crate::engine::data::{Data, Metric};
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::model::ResponseFamily;
use crate::extensions::basis::{LinearBasis, LinearGaussianModel};
use crate::extensions::coord::{CoordinateDistribution, EuclideanNormal, WrappedNormal};
use crate::extensions::count_priors::ShiftedPoissonBinomial;
use crate::extensions::distance::{
    Cosine, Euclidean, Gower, GowerKind, Mahalanobis, Manhattan, Minkowski, Spherical,
};
use crate::extensions::inclusion::{DartInclusion, UniformInclusion, WeightedInclusion};
use crate::extensions::membership::SoftmaxKernel;
use crate::extensions::moves::{
    AddCentre, AddDimension, Change, MoveSetBuilder, ProposalMove, RemoveCentre, RemoveDimension,
    Swap,
};
use crate::extensions::scale::{HVariance, PinnedSigma};

/// Build the crate's standard `InvalidHyperparameter` error.
fn invalid(name: &str, reason: impl Into<String>) -> AddiVortesError {
    AddiVortesError::InvalidHyperparameter {
        name: name.into(),
        reason: reason.into(),
    }
}

/// Reject a non-positive or non-finite scalar under the spec's own dotted
/// field name (`scale.sigma_sq`, `coords.sigma_c`, …), so a binding's user
/// sees the key they set rather than the constructor argument behind it. The
/// shelf constructors check the same condition themselves; this runs first
/// purely for the message.
fn positive_finite(name: &str, value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(
            name,
            format!("must be finite and strictly positive, got {value}"),
        ));
    }
    Ok(value)
}

/// A complete, data-only description of an [`AddiVortesConfig`]: the mandatory
/// seed plus every shelf-selectable knob as an `Option` (unset = the crate
/// default). Deserialise one from a binding's native map, then call
/// [`into_config`](ConfigSpec::into_config).
///
/// `deny_unknown_fields` turns a typo'd key into a loud error rather than a
/// silently ignored setting — the pass-through's safety net. A misspelt
/// `lambda_C` that is quietly dropped is exactly the failure a config surface
/// exists to prevent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigSpec {
    /// Chain seed (mandatory — there is no default seed; the reproducibility
    /// contract requires the caller to state one).
    pub seed: u64,
    /// Ensemble size m.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub m: Option<usize>,
    /// Burn-in sweeps discarded by `fit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub burn_in: Option<usize>,
    /// Posterior draws kept by `fit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draws: Option<usize>,
    /// Thinning interval for `fit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinning: Option<usize>,
    /// σ² prior degrees of freedom ν.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nu: Option<f64>,
    /// σ² prior calibration quantile q.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q: Option<f64>,
    /// μ prior spread parameter k.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<f64>,
    /// Centre-count prior parameter λ_c.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lambda_c: Option<f64>,
    /// Dimension-count prior parameter ω.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub omega: Option<f64>,
    /// Centre-coordinate prior/proposal spread σ_c.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma_c: Option<f64>,
    /// Per-raw-column metrics by name (`euclidean` | `spherical` |
    /// `categorical` | `prepared`); unset = all-Euclidean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,
    /// Structural moves and their selection weights; unset = the
    /// paper's six-move set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moves: Option<Vec<MoveSpec>>,
    /// Per-raw-column centre-coordinate laws; unset = the per-metric
    /// defaults at `sigma_c`. One entry per raw column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coords: Option<Vec<CoordSpec>>,
    /// Assignment geometry; unset = the built-in compound metric.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance: Option<DistanceSpec>,
    /// Covariate-inclusion model; unset = uniform.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inclusion: Option<InclusionSpec>,
    /// Response family by name (`gaussian` | `binary_probit` | `robust_t`);
    /// unset = `gaussian`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_family: Option<String>,
    /// Student-t degrees of freedom — required by, and only valid for,
    /// `response_family = "robust_t"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_df: Option<f64>,
    /// Scale (σ²) model; unset = the engine's calibrated `GlobalSigma`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<ScaleSpec>,
    /// Cell/dimension count priors; unset = the paper's pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count_priors: Option<CountPriorsSpec>,
    /// Within-cell basis; unset = a scalar payload per cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<BasisSpec>,
    /// Soft-membership kernel; unset = hard membership.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub membership: Option<MembershipSpec>,
}

/// One entry of the move set: a built-in move and its selection weight.
///
/// A *custom* move is Rust — it is a trait implementation, and no data payload
/// can carry one. What a binding user can do from data is retune the paper's
/// set, or drop moves from it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveSpec {
    /// `add_centre` | `remove_centre` | `add_dimension` | `remove_dimension` |
    /// `change` | `swap`.
    pub name: String,
    /// Selection weight (positive; the set renormalises over valid moves).
    pub weight: f64,
}

impl MoveSpec {
    /// The built-in move this entry names.
    fn build(&self) -> Result<Box<dyn ProposalMove>> {
        Ok(match self.name.as_str() {
            "add_centre" => Box::new(AddCentre) as Box<dyn ProposalMove>,
            "remove_centre" => Box::new(RemoveCentre),
            "add_dimension" => Box::new(AddDimension),
            "remove_dimension" => Box::new(RemoveDimension),
            "change" => Box::new(Change),
            "swap" => Box::new(Swap),
            other => {
                return Err(invalid(
                    "moves",
                    format!(
                        "unknown move {other:?}: expected one of 'add_centre', \
                         'remove_centre', 'add_dimension', 'remove_dimension', \
                         'change', 'swap'"
                    ),
                ));
            }
        })
    }
}

/// A per-raw-column centre-coordinate law, tagged by `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoordSpec {
    /// Normal on the real line — the paper's law for a Euclidean column.
    EuclideanNormal {
        /// Spread σ_c (scaled space).
        sigma_c: f64,
    },
    /// Normal wrapped onto the circle, for an angular column.
    WrappedNormal {
        /// Spread σ_c (scaled space).
        sigma_c: f64,
    },
}

impl CoordSpec {
    fn build(&self) -> Result<Arc<dyn CoordinateDistribution>> {
        Ok(match *self {
            CoordSpec::EuclideanNormal { sigma_c } => Arc::new(EuclideanNormal::new(
                positive_finite("coords.sigma_c", sigma_c)?,
            )?)
                as Arc<dyn CoordinateDistribution>,
            CoordSpec::WrappedNormal { sigma_c } => Arc::new(WrappedNormal::new(positive_finite(
                "coords.sigma_c",
                sigma_c,
            )?)?),
        })
    }
}

/// The assignment-geometry shelf, tagged by `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DistanceSpec {
    /// Squared Euclidean — the paper's geometry.
    Euclidean,
    /// Manhattan (L1).
    Manhattan,
    /// Cosine distance from the mid-range origin of scaled space.
    Cosine,
    /// Great-circle geometry for an all-angular design.
    Spherical,
    /// Minkowski (L_p) of order `p ≥ 1`.
    Minkowski {
        /// The order p.
        p: f64,
    },
    /// Gower mixed numeric/categorical geometry, declared from the raw layout.
    Gower {
        /// One entry per raw column, in `metrics` order.
        columns: Vec<GowerColumnSpec>,
    },
    /// Mahalanobis geometry from a square precision matrix over the **encoded**
    /// design width (square, finite, symmetric, positive definite).
    Mahalanobis {
        /// Row-major square precision matrix.
        precision: Vec<Vec<f64>>,
    },
}

/// A Gower raw-column declaration, tagged by `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GowerColumnSpec {
    /// A numeric column.
    Numeric,
    /// A categorical column with `levels` distinct levels.
    ///
    /// You state the level count because the engine cannot infer intent from a
    /// sample: a level absent from *this* data is still a level.
    Categorical {
        /// The number of distinct levels the fit will see.
        levels: usize,
    },
}

impl From<GowerColumnSpec> for GowerKind {
    fn from(spec: GowerColumnSpec) -> Self {
        match spec {
            GowerColumnSpec::Numeric => GowerKind::Numeric,
            GowerColumnSpec::Categorical { levels } => GowerKind::Categorical { levels },
        }
    }
}

impl DistanceSpec {
    /// Apply this geometry to `config`, validating any structured payload with
    /// the crate's own rules and messages.
    fn apply(self, config: AddiVortesConfig) -> Result<AddiVortesConfig> {
        Ok(match self {
            DistanceSpec::Euclidean => config.with_distance(Euclidean),
            DistanceSpec::Manhattan => config.with_distance(Manhattan),
            DistanceSpec::Cosine => config.with_distance(Cosine),
            DistanceSpec::Spherical => config.with_distance(Spherical),
            DistanceSpec::Minkowski { p } => config.with_distance(Minkowski::new(p)?),
            DistanceSpec::Gower { columns } => {
                for (i, column) in columns.iter().enumerate() {
                    // A categorical column with no levels is not a column. Left
                    // unchecked it produces a non-finite key at assignment time,
                    // and the caller is told "distance is not finite" rather than
                    // what they actually got wrong.
                    if let GowerColumnSpec::Categorical { levels: 0 } = column {
                        return Err(invalid(
                            "gower.columns",
                            format!("column {i} is categorical with 0 levels: it needs at least 1"),
                        ));
                    }
                }
                let kinds: Vec<GowerKind> = columns.into_iter().map(GowerKind::from).collect();
                config.with_distance(Gower::new(kinds)?)
            }
            DistanceSpec::Mahalanobis { precision } => {
                let width = precision.len();
                for (i, row) in precision.iter().enumerate() {
                    if row.len() != width {
                        return Err(invalid(
                            "mahalanobis_precision",
                            format!(
                                "must be square: row {i} has {} entries, expected {width}",
                                row.len()
                            ),
                        ));
                    }
                }
                let flat: Vec<f64> = precision.into_iter().flatten().collect();
                config.with_distance(Mahalanobis::new(flat, width)?)
            }
        })
    }
}

/// The covariate-inclusion shelf, tagged by `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InclusionSpec {
    /// Equal weight on every covariate (the default).
    Uniform,
    /// Fixed weights, one per **raw** (pre-encoding) covariate.
    Weighted {
        /// One positive weight per raw column.
        weights: Vec<f64>,
    },
    /// DART: a Dirichlet prior over the inclusion weights, resampled each sweep.
    /// Smaller `alpha` concentrates mass on fewer covariates.
    Dart {
        /// Dirichlet concentration α (finite, positive).
        alpha: f64,
    },
}

impl InclusionSpec {
    /// Everything checkable without the data: α, and the weight *values* (but
    /// not how many of them there should be).
    fn validate(&self) -> Result<()> {
        match self {
            InclusionSpec::Uniform => Ok(()),
            InclusionSpec::Weighted { weights } => {
                for (i, &w) in weights.iter().enumerate() {
                    positive_finite(&format!("inclusion.weights[{i}]"), w)?;
                }
                Ok(())
            }
            InclusionSpec::Dart { alpha } => positive_finite("inclusion.alpha", *alpha).map(|_| ()),
        }
    }

    /// `p` is the raw covariate count, taken from the data.
    fn apply(self, config: AddiVortesConfig, p: usize) -> Result<AddiVortesConfig> {
        self.validate()?;
        Ok(match self {
            InclusionSpec::Uniform => config.with_inclusion(UniformInclusion::new(p)),
            InclusionSpec::Weighted { weights } => {
                if weights.len() != p {
                    return Err(invalid(
                        "inclusion.weights",
                        format!(
                            "expected one weight per covariate (p = {p}), got {}",
                            weights.len()
                        ),
                    ));
                }
                config.with_inclusion(WeightedInclusion::new(weights))
            }
            InclusionSpec::Dart { alpha } => config.with_inclusion(DartInclusion::new(alpha, p)?),
        })
    }
}

/// The σ² shelf, tagged by `type`.
///
/// `GlobalSigma` and `WeightedGlobalSigma` are absent deliberately: they take
/// the engine's *data-calibrated* λ and are exactly what it attaches when this
/// point is unset. Naming them here would only let a caller supply a worse λ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScaleSpec {
    /// σ² held fixed (the probit configuration pins it to 1).
    Pinned {
        /// The fixed σ² (scaled space; finite, positive).
        sigma_sq: f64,
    },
    /// Heteroscedastic variance: a second ensemble of `m_prime` tessellations
    /// over log σ²(x).
    HVariance {
        /// Ensemble size m′ (the H paper's default is 40).
        m_prime: usize,
        /// Pin the variance prior instead of calibrating it from the data at
        /// the first sweep. Both parts are required together.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nu_prime: Option<f64>,
        /// See `nu_prime` (scaled space).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lambda_prime: Option<f64>,
    },
}

impl ScaleSpec {
    fn apply(self, config: AddiVortesConfig) -> Result<AddiVortesConfig> {
        Ok(match self {
            ScaleSpec::Pinned { sigma_sq } => config.with_scale_model(PinnedSigma::new(
                positive_finite("scale.sigma_sq", sigma_sq)?,
            )?),
            ScaleSpec::HVariance {
                m_prime,
                nu_prime,
                lambda_prime,
            } => {
                if m_prime < 1 {
                    return Err(invalid("scale.m_prime", "must be at least 1"));
                }
                let model = HVariance::new(m_prime)?;
                let model = match (nu_prime, lambda_prime) {
                    (None, None) => model,
                    (Some(nu), Some(lambda)) => model.with_prior(
                        positive_finite("scale.nu_prime", nu)?,
                        positive_finite("scale.lambda_prime", lambda)?,
                    )?,
                    // Half a prior is not a prior: pinning one part and
                    // calibrating the other would silently mix two regimes.
                    _ => {
                        return Err(invalid(
                            "scale.nu_prime",
                            "nu_prime and lambda_prime must be given together (or both omitted, \
                             to calibrate the variance prior from the data)",
                        ));
                    }
                };
                config.with_scale_model(model)
            }
        })
    }
}

/// The count-prior shelf, tagged by `type`.
///
/// Only the paper's pair ships today, so selecting it is a no-op — the variant
/// exists so the vocabulary is complete and a second entry is a one-line
/// addition here rather than a change to every binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CountPriorsSpec {
    /// Shifted Poisson on cells, shifted Binomial on dimensions.
    ShiftedPoissonBinomial,
}

impl CountPriorsSpec {
    fn apply(self, config: AddiVortesConfig) -> AddiVortesConfig {
        match self {
            CountPriorsSpec::ShiftedPoissonBinomial => {
                config.with_count_priors(ShiftedPoissonBinomial)
            }
        }
    }
}

/// The within-cell basis shelf, tagged by `type`.
///
/// This variant sets **both** halves of the point: the basis and the matching
/// payload family. `q` is derived (`1 + columns.len()`), so the mismatch that
/// `fit` would otherwise have to catch cannot be expressed here at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BasisSpec {
    /// An intercept plus the named **encoded** (post one-hot) columns, so each
    /// cell carries a local linear fit rather than a constant.
    Linear {
        /// Encoded column indices; bounds-checked against the design at `fit`.
        columns: Vec<usize>,
        /// Coefficient prior variance σ_β² (scaled space; finite, positive).
        sigma_beta_sq: f64,
    },
}

impl BasisSpec {
    fn apply(self, config: AddiVortesConfig) -> Result<AddiVortesConfig> {
        Ok(match self {
            BasisSpec::Linear {
                columns,
                sigma_beta_sq,
            } => {
                let sigma_beta_sq = positive_finite("basis.sigma_beta_sq", sigma_beta_sq)?;
                let q = 1 + columns.len();
                config
                    .with_cell_basis(LinearBasis::new(columns))
                    .with_cell_model(LinearGaussianModel::new(sigma_beta_sq, q)?)
            }
        })
    }
}

/// The soft-membership shelf, tagged by `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MembershipSpec {
    /// Softmax over the negated membership keys, with temperature `tau`.
    /// Smaller `tau` approaches hard assignment.
    Softmax {
        /// Temperature τ (finite, positive).
        tau: f64,
    },
}

impl MembershipSpec {
    fn apply(self, config: AddiVortesConfig) -> Result<AddiVortesConfig> {
        Ok(match self {
            MembershipSpec::Softmax { tau } => {
                config.with_membership(SoftmaxKernel::new(positive_finite("membership.tau", tau)?)?)
            }
        })
    }
}

impl ConfigSpec {
    /// A spec with only the mandatory seed: every extension point and knob at the crate
    /// default.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            m: None,
            burn_in: None,
            draws: None,
            thinning: None,
            nu: None,
            q: None,
            k: None,
            lambda_c: None,
            omega: None,
            sigma_c: None,
            metrics: None,
            moves: None,
            coords: None,
            distance: None,
            inclusion: None,
            response_family: None,
            t_df: None,
            scale: None,
            count_priors: None,
            basis: None,
            membership: None,
        }
    }

    /// Check the spec **without any data**, so a binding can fail at the point
    /// the user writes the config rather than at `fit`.
    ///
    /// This is every check except the ones that need the covariate count: the
    /// scalar ranges, the shelf names, and every structured payload (a
    /// Minkowski `p < 1`, a non-symmetric Mahalanobis precision, a
    /// non-positive `tau`). The two it cannot make are the *lengths* of the
    /// covariate-sized settings — one coordinate law per column, one inclusion
    /// weight per column — which only `into_config` can check.
    ///
    /// It runs the same code path as [`into_config`](Self::into_config), with
    /// the covariate count absent, so the two cannot disagree about what counts
    /// as valid.
    pub fn validate(&self) -> Result<()> {
        self.clone().assemble(None).map(|_| ())
    }

    /// Build and validate the [`AddiVortesConfig`] this spec describes, against
    /// the data it will be fitted to.
    ///
    /// Every unset field falls through to the crate default. The data supplies
    /// only what the caller cannot state up front — today, the raw covariate
    /// count `p` that the inclusion and coordinate settings are sized by (see the
    /// module docs on late binding). The assembled config is `validate`d before
    /// it is returned, so a bad spec fails here rather than mid-fit.
    pub fn into_config(self, x: &Data) -> Result<AddiVortesConfig> {
        self.assemble(Some(x.n_cols()))
    }

    /// The one assembly path. `p` is the raw covariate count when the data is
    /// known; `None` means "data-free": the covariate-sized settings still have
    /// their *values* validated, but they are not applied and their lengths are
    /// not checked, because nothing yet knows what the length should be.
    fn assemble(self, p: Option<usize>) -> Result<AddiVortesConfig> {
        let mut config = AddiVortesConfig::new(self.seed);

        if let Some(v) = self.m {
            config = config.with_m(v);
        }
        if let Some(v) = self.burn_in {
            config = config.with_burn_in(v);
        }
        if let Some(v) = self.draws {
            config = config.with_draws(v);
        }
        if let Some(v) = self.thinning {
            config = config.with_thinning(v);
        }
        if let Some(v) = self.nu {
            config = config.with_nu(v);
        }
        if let Some(v) = self.q {
            config = config.with_q(v);
        }
        if let Some(v) = self.k {
            config = config.with_k(v);
        }
        if let Some(v) = self.lambda_c {
            config = config.with_lambda_c(v);
        }
        if let Some(v) = self.omega {
            config = config.with_omega(v);
        }
        if let Some(v) = self.sigma_c {
            config = config.with_sigma_c(v);
        }
        if let Some(names) = self.metrics {
            let parsed: Result<Vec<Metric>> = names.iter().map(|n| parse_metric(n)).collect();
            config = config.with_metrics(parsed?);
        }

        if let Some(moves) = self.moves {
            if moves.is_empty() {
                return Err(invalid("moves", "a move set needs at least one move"));
            }
            let mut builder = MoveSetBuilder::empty();
            for entry in &moves {
                builder = builder.with_move(
                    entry.build()?,
                    positive_finite("moves.weight", entry.weight)?,
                );
            }
            // `build` enforces the set's own invariants (unique names, mutual
            // pairing, positive weights) with its own messages.
            config = config.with_move_set(builder.build()?);
        }
        if let Some(coords) = self.coords {
            // The laws' own parameters are checked either way; only the *count*
            // needs the data (one law per raw column).
            let laws: Result<Vec<Arc<dyn CoordinateDistribution>>> =
                coords.iter().map(CoordSpec::build).collect();
            let laws = laws?;
            if let Some(p) = p {
                if laws.len() != p {
                    return Err(invalid(
                        "coords",
                        format!(
                            "expected one coordinate law per covariate (p = {p}), got {}",
                            laws.len()
                        ),
                    ));
                }
                config = config.with_coords(laws);
            }
        }
        if let Some(distance) = self.distance {
            config = distance.apply(config)?;
        }
        if let Some(inclusion) = self.inclusion {
            // As for `coords`: α and the weight values are checked either way,
            // the length only when the data says what it should be.
            inclusion.validate()?;
            if let Some(p) = p {
                config = inclusion.apply(config, p)?;
            }
        }
        let family = resolve_response_family(self.response_family.as_deref(), self.t_df)?;
        config = config.with_response_family(family);
        if let Some(scale) = self.scale {
            config = scale.apply(config)?;
        }
        if let Some(count_priors) = self.count_priors {
            config = count_priors.apply(config);
        }
        if let Some(basis) = self.basis {
            config = basis.apply(config)?;
        }
        if let Some(membership) = self.membership {
            config = membership.apply(config)?;
        }

        config.validate()?;
        Ok(config)
    }
}

/// Parse a metric name into a [`Metric`]; an unknown name is a loud error.
fn parse_metric(name: &str) -> Result<Metric> {
    match name {
        "euclidean" => Ok(Metric::Euclidean),
        "spherical" => Ok(Metric::Spherical),
        "categorical" => Ok(Metric::Categorical),
        "prepared" => Ok(Metric::Prepared),
        other => Err(invalid(
            "metrics",
            format!(
                "unknown metric {other:?}: expected one of 'euclidean', 'spherical', \
                 'categorical', 'prepared'"
            ),
        )),
    }
}

/// Resolve a `(response_family, t_df)` pair, enforcing the coupling rule: `t_df`
/// is required by — and only valid for — `robust_t`.
fn resolve_response_family(name: Option<&str>, t_df: Option<f64>) -> Result<ResponseFamily> {
    let coupling = || invalid("t_df", "t_df only applies to response_family='robust_t'");
    match name.unwrap_or("gaussian") {
        "gaussian" => {
            if t_df.is_some() {
                return Err(coupling());
            }
            Ok(ResponseFamily::Gaussian)
        }
        "binary_probit" => {
            if t_df.is_some() {
                return Err(coupling());
            }
            Ok(ResponseFamily::BinaryProbit)
        }
        "robust_t" => {
            let df = t_df.ok_or_else(|| {
                invalid(
                    "t_df",
                    "response_family='robust_t' requires t_df (the Student-t degrees of freedom)",
                )
            })?;
            Ok(ResponseFamily::RobustT {
                df: positive_finite("t_df", df)?,
            })
        }
        other => Err(invalid(
            "response_family",
            format!(
                "unknown response_family {other:?}: expected 'gaussian', 'binary_probit', \
                 or 'robust_t'"
            ),
        )),
    }
}

/// The canonical name of a resolved [`ResponseFamily`] — the read-side mirror of
/// `resolve_response_family`, used by a binding to report a fitted model's
/// family. The match is exhaustive on purpose: a new family must be named here
/// (and given a `resolve_response_family` arm), so the pass-through can never
/// silently report an "unknown" family.
pub fn response_family_name(family: ResponseFamily) -> &'static str {
    match family {
        ResponseFamily::Gaussian => "gaussian",
        ResponseFamily::BinaryProbit => "binary_probit",
        ResponseFamily::RobustT { .. } => "robust_t",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two raw columns, so `p = 2` and the covariate-sized settings have something
    /// to be sized against.
    fn data() -> Data {
        let values: Vec<f64> = (0..40)
            .flat_map(|i| [i as f64 / 40.0, ((i % 5) as f64) / 5.0])
            .collect();
        Data::new(values, 40, 2).unwrap()
    }

    /// A response with genuine residual scatter. It must not be an exact
    /// function of a covariate: the σ² prior is calibrated from the residual
    /// scale of a linear fit, so a noiseless response calibrates λ to zero, and
    /// `HVariance`'s §3.3 prior matching then divides into it.
    fn y() -> Vec<f64> {
        (0..40)
            .map(|i| {
                let t = i as f64 / 40.0;
                // A deterministic, non-linear wobble: no RNG, so the fixture
                // stays reproducible.
                2.0 * t - 0.5 + 0.3 * ((i * 7 % 11) as f64 / 11.0 - 0.5)
            })
            .collect()
    }

    fn from_json(json: &str) -> Result<AddiVortesConfig> {
        let spec: ConfigSpec = serde_json::from_str(json).expect("valid json");
        spec.into_config(&data())
    }

    fn err(json: &str) -> String {
        from_json(json).unwrap_err().to_string()
    }

    /// The comparisons in this module are only meaningful because `PartialEq`
    /// covers every field of the config. It did not always: `family`,
    /// `membership` and `basis` were omitted, which would have made
    /// `assert_eq!(spec.into_config(..), expected)` unable to fail on exactly
    /// the selections this spec adds.
    #[test]
    fn a_bare_seed_yields_the_crate_defaults() {
        assert_eq!(
            from_json(r#"{"seed": 7}"#).unwrap(),
            AddiVortesConfig::new(7)
        );
    }

    #[test]
    fn an_unset_field_does_not_override_its_default() {
        assert_eq!(
            from_json(r#"{"seed": 1, "lambda_c": 5.0}"#).unwrap(),
            AddiVortesConfig::new(1).with_lambda_c(5.0)
        );
    }

    #[test]
    fn scalar_knobs_round_trip() {
        let config = from_json(
            r#"{"seed": 2, "m": 50, "burn_in": 10, "draws": 20, "thinning": 2,
                "nu": 4.0, "q": 0.9, "k": 2.0, "lambda_c": 5.0, "omega": 1.5,
                "sigma_c": 0.5}"#,
        )
        .unwrap();
        let expected = AddiVortesConfig::new(2)
            .with_m(50)
            .with_burn_in(10)
            .with_draws(20)
            .with_thinning(2)
            .with_nu(4.0)
            .with_q(0.9)
            .with_k(2.0)
            .with_lambda_c(5.0)
            .with_omega(1.5)
            .with_sigma_c(0.5);
        assert_eq!(config, expected);
    }

    /// Every shelf entry the spec claims to reach must actually build a config
    /// **and fit**. Building alone would not prove much: a mispaired selection (a
    /// basis without its payload family, say) is only rejected at `fit`.
    #[test]
    fn every_spec_variant_builds_and_fits() {
        let cases = [
            // moves, retuned from data
            r#"{"seed": 1, "omega": 1.0, "moves": [
                  {"name": "add_centre", "weight": 0.3},
                  {"name": "remove_centre", "weight": 0.3},
                  {"name": "add_dimension", "weight": 0.1},
                  {"name": "remove_dimension", "weight": 0.1},
                  {"name": "change", "weight": 0.1},
                  {"name": "swap", "weight": 0.1}]}"#,
            // coords, one law per raw column
            r#"{"seed": 1, "omega": 1.0, "coords": [
                  {"type": "euclidean_normal", "sigma_c": 0.8},
                  {"type": "wrapped_normal", "sigma_c": 0.5}]}"#,
            // distance — every geometry
            r#"{"seed": 1, "omega": 1.0, "distance": {"type": "euclidean"}}"#,
            r#"{"seed": 1, "omega": 1.0, "distance": {"type": "manhattan"}}"#,
            r#"{"seed": 1, "omega": 1.0, "distance": {"type": "cosine"}}"#,
            r#"{"seed": 1, "omega": 1.0, "distance": {"type": "spherical"}}"#,
            r#"{"seed": 1, "omega": 1.0, "distance": {"type": "minkowski", "p": 3.0}}"#,
            r#"{"seed": 1, "omega": 1.0, "distance": {"type": "gower",
                  "columns": [{"type": "numeric"}, {"type": "numeric"}]}}"#,
            r#"{"seed": 1, "omega": 1.0, "distance": {"type": "mahalanobis",
                  "precision": [[2.0, -0.5], [-0.5, 1.0]]}}"#,
            // inclusion
            r#"{"seed": 1, "omega": 1.0, "inclusion": {"type": "uniform"}}"#,
            r#"{"seed": 1, "omega": 1.0, "inclusion": {"type": "weighted", "weights": [1.0, 3.0]}}"#,
            r#"{"seed": 1, "omega": 1.0, "inclusion": {"type": "dart", "alpha": 0.5}}"#,
            // response
            r#"{"seed": 1, "omega": 1.0, "response_family": "gaussian"}"#,
            r#"{"seed": 1, "omega": 1.0, "response_family": "robust_t", "t_df": 4.0}"#,
            // scale
            r#"{"seed": 1, "omega": 1.0, "scale": {"type": "pinned", "sigma_sq": 1.0}}"#,
            r#"{"seed": 1, "omega": 1.0, "scale": {"type": "h_variance", "m_prime": 5}}"#,
            r#"{"seed": 1, "omega": 1.0, "scale": {"type": "h_variance", "m_prime": 5,
                  "nu_prime": 3.0, "lambda_prime": 0.5}}"#,
            // count priors
            r#"{"seed": 1, "omega": 1.0, "count_priors": {"type": "shifted_poisson_binomial"}}"#,
            // basis (q is derived, so it cannot be mispaired)
            r#"{"seed": 1, "omega": 1.0, "basis": {"type": "linear", "columns": [0],
                  "sigma_beta_sq": 0.1}}"#,
            // membership
            r#"{"seed": 1, "omega": 1.0, "membership": {"type": "softmax", "tau": 0.2}}"#,
        ];

        let (x, y) = (data(), y());
        for json in cases {
            let spec: ConfigSpec = serde_json::from_str(json).expect("valid json");
            let config = spec
                .into_config(&x)
                .unwrap_or_else(|e| panic!("spec rejected: {e}\n{json}"))
                .with_m(2)
                .with_burn_in(2)
                .with_draws(2);
            config
                .fit(&x, &y)
                .unwrap_or_else(|e| panic!("fit failed: {e}\n{json}"));
        }
    }

    /// The basis spec sets both halves of the basis point and derives `q` from the
    /// column count, so the mismatch `fit` exists to catch cannot be written.
    #[test]
    fn the_basis_spec_derives_q_from_the_columns() {
        let from_spec = from_json(
            r#"{"seed": 1, "basis": {"type": "linear", "columns": [0, 1], "sigma_beta_sq": 0.1}}"#,
        )
        .unwrap();
        let by_hand = AddiVortesConfig::new(1)
            .with_cell_basis(LinearBasis::new(vec![0, 1]))
            .with_cell_model(LinearGaussianModel::new(0.1, 3).unwrap()); // q = 1 + 2
        // `Arc::ptr_eq` semantics mean the configs are not `==`; compare the
        // observable consequence instead.
        let (x, y) = (data(), y());
        let a = from_spec
            .with_m(2)
            .with_burn_in(2)
            .with_draws(2)
            .with_omega(1.0)
            .fit(&x, &y)
            .expect("spec-built basis fits");
        let b = by_hand
            .with_m(2)
            .with_burn_in(2)
            .with_draws(2)
            .with_omega(1.0)
            .fit(&x, &y)
            .expect("hand-built basis fits");
        assert_eq!(
            a.predict(&x).unwrap(),
            b.predict(&x).unwrap(),
            "the spec must build the same model as the hand-written pair"
        );
    }

    /// A bad DART α is an error naming the spec's own key.
    #[test]
    fn a_bad_dart_alpha_is_an_error_not_a_process_abort() {
        assert!(
            err(r#"{"seed": 1, "inclusion": {"type": "dart", "alpha": 0.0}}"#).contains("alpha")
        );
        assert!(
            err(r#"{"seed": 1, "inclusion": {"type": "dart", "alpha": -1.0}}"#).contains("alpha")
        );
    }

    /// Shelf scalars are rejected under the spec's dotted key, before the
    /// constructor behind them reports the bare argument name.
    #[test]
    fn shelf_scalars_are_rejected_under_the_spec_key() {
        assert!(
            err(r#"{"seed": 1, "membership": {"type": "softmax", "tau": 0.0}}"#).contains("tau")
        );
        assert!(
            err(r#"{"seed": 1, "scale": {"type": "pinned", "sigma_sq": -1.0}}"#)
                .contains("sigma_sq")
        );
        assert!(
            err(
                r#"{"seed": 1, "basis": {"type": "linear", "columns": [0], "sigma_beta_sq": 0.0}}"#
            )
            .contains("sigma_beta_sq")
        );
        assert!(
            err(
                r#"{"seed": 1, "coords": [{"type": "euclidean_normal", "sigma_c": 0.0},
                                      {"type": "euclidean_normal", "sigma_c": 1.0}]}"#
            )
            .contains("sigma_c")
        );
        assert!(
            err(r#"{"seed": 1, "response_family": "robust_t", "t_df": -2.0}"#).contains("t_df")
        );
    }

    #[test]
    fn invalid_values_reproduce_the_shelf_messages() {
        assert!(err(r#"{"seed": 1, "m": 0}"#).contains("`m`"));
        assert!(err(r#"{"seed": 1, "q": 2.0}"#).contains("`q`"));
        assert!(err(r#"{"seed": 1, "metrics": ["nope"]}"#).contains("metric"));
        assert!(err(r#"{"seed": 1, "response_family": "cauchy"}"#).contains("response_family"));
        assert!(err(r#"{"seed": 1, "response_family": "robust_t"}"#).contains("t_df"));
        assert!(err(r#"{"seed": 1, "response_family": "gaussian", "t_df": 4.0}"#).contains("t_df"));
        assert!(
            err(r#"{"seed": 1, "moves": [{"name": "teleport", "weight": 1.0}]}"#)
                .contains("teleport")
        );
        assert!(
            err(r#"{"seed": 1, "distance": {"type": "minkowski", "p": 0.5}}"#)
                .contains("minkowski_p")
        );
        assert!(
            err(r#"{"seed": 1, "distance": {"type": "mahalanobis",
                "precision": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}}"#)
            .contains("square")
        );
        assert!(
            err(r#"{"seed": 1, "distance": {"type": "mahalanobis",
                "precision": [[1.0, 2.0], [2.0, 1.0]]}}"#)
            .contains("positive definite")
        );
        assert!(
            err(r#"{"seed": 1, "distance": {"type": "mahalanobis",
                "precision": [[1.0, 0.5], [-0.5, 1.0]]}}"#)
            .contains("symmetric")
        );
    }

    /// The covariate-sized settings are checked against the data, not left to fail
    /// deep inside the sampler.
    #[test]
    fn covariate_sized_settings_are_checked_against_the_data() {
        let e = err(r#"{"seed": 1, "coords": [{"type": "euclidean_normal", "sigma_c": 0.8}]}"#);
        assert!(e.contains("p = 2"), "{e}");
        let e = err(r#"{"seed": 1, "inclusion": {"type": "weighted", "weights": [1.0]}}"#);
        assert!(e.contains("p = 2"), "{e}");
    }

    /// Pinning half the H-variance prior would silently mix a pinned regime with
    /// a calibrated one.
    #[test]
    fn half_an_h_variance_prior_is_refused() {
        let e =
            err(r#"{"seed": 1, "scale": {"type": "h_variance", "m_prime": 5, "nu_prime": 3.0}}"#);
        assert!(e.contains("together"), "{e}");
    }

    #[test]
    fn a_typod_key_is_loud_rather_than_ignored() {
        let e = serde_json::from_str::<ConfigSpec>(r#"{"seed": 1, "lambda_C": 5.0}"#).unwrap_err();
        assert!(e.to_string().contains("unknown field"), "{e}");
    }

    #[test]
    fn a_spec_serialises_without_its_unset_fields() {
        let mut spec = ConfigSpec::new(3);
        spec.m = Some(10);
        assert_eq!(
            serde_json::to_string(&spec).unwrap(),
            r#"{"seed":3,"m":10}"#
        );
    }

    #[test]
    fn a_spec_round_trips_through_json() {
        let json = r#"{"seed":5,"m":20,"distance":{"type":"minkowski","p":3.0},
                       "inclusion":{"type":"dart","alpha":0.5},
                       "membership":{"type":"softmax","tau":0.2}}"#;
        let spec: ConfigSpec = serde_json::from_str(json).unwrap();
        let round_tripped: ConfigSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(spec, round_tripped);
    }

    /// A binding builds the config at `fit` (that is when `p` exists) but the
    /// user writes it much earlier. `validate()` is what lets the error land at
    /// the point of the mistake — `Distance.minkowski(0.5)` must raise when it
    /// is written, not several lines later.
    #[test]
    fn validate_catches_the_data_free_mistakes_before_any_fit() {
        let bad = [
            r#"{"seed": 1, "m": 0}"#,
            r#"{"seed": 1, "q": 2.0}"#,
            r#"{"seed": 1, "metrics": ["nope"]}"#,
            r#"{"seed": 1, "distance": {"type": "minkowski", "p": 0.5}}"#,
            r#"{"seed": 1, "distance": {"type": "mahalanobis", "precision": [[1.0,2.0],[2.0,1.0]]}}"#,
            r#"{"seed": 1, "membership": {"type": "softmax", "tau": 0.0}}"#,
            r#"{"seed": 1, "inclusion": {"type": "dart", "alpha": -1.0}}"#,
            r#"{"seed": 1, "inclusion": {"type": "weighted", "weights": [1.0, -2.0]}}"#,
            r#"{"seed": 1, "coords": [{"type": "euclidean_normal", "sigma_c": 0.0}]}"#,
            r#"{"seed": 1, "scale": {"type": "pinned", "sigma_sq": -1.0}}"#,
            r#"{"seed": 1, "response_family": "cauchy"}"#,
            r#"{"seed": 1, "moves": [{"name": "teleport", "weight": 1.0}]}"#,
            r#"{"seed": 1, "basis": {"type": "linear", "columns": [0], "sigma_beta_sq": 0.0}}"#,
        ];
        for json in bad {
            let spec: ConfigSpec = serde_json::from_str(json).expect("valid json");
            assert!(
                spec.validate().is_err(),
                "validate() passed a spec it should have rejected: {json}"
            );
        }
    }

    /// `validate()` must not reject anything `into_config` would accept, or a
    /// binding would refuse a legitimate config at construction. In particular
    /// it must not fail the covariate-sized settings just because it cannot know
    /// `p` yet.
    #[test]
    fn validate_accepts_everything_into_config_accepts() {
        let x = data();
        let good = [
            r#"{"seed": 1}"#,
            // p-sized settings: valid here, and validate() cannot see the length.
            r#"{"seed": 1, "coords": [{"type": "euclidean_normal", "sigma_c": 0.8},
                                      {"type": "wrapped_normal", "sigma_c": 0.5}]}"#,
            r#"{"seed": 1, "inclusion": {"type": "weighted", "weights": [1.0, 3.0]}}"#,
            r#"{"seed": 1, "inclusion": {"type": "dart", "alpha": 0.5}}"#,
            r#"{"seed": 1, "distance": {"type": "minkowski", "p": 3.0}}"#,
            r#"{"seed": 1, "scale": {"type": "h_variance", "m_prime": 5}}"#,
            r#"{"seed": 1, "basis": {"type": "linear", "columns": [0], "sigma_beta_sq": 0.1}}"#,
            r#"{"seed": 1, "response_family": "robust_t", "t_df": 4.0}"#,
        ];
        for json in good {
            let spec: ConfigSpec = serde_json::from_str(json).expect("valid json");
            spec.validate()
                .unwrap_or_else(|e| panic!("validate() rejected a valid spec: {e}\n{json}"));
            spec.into_config(&x)
                .unwrap_or_else(|e| panic!("into_config rejected a valid spec: {e}\n{json}"));
        }
    }

    /// The one class `validate()` cannot catch, and must not pretend to: a
    /// covariate-sized setting of the wrong length. It passes data-free and fails
    /// once the data says how long it should have been.
    #[test]
    fn a_wrong_length_passes_validate_and_fails_at_into_config() {
        let json = r#"{"seed": 1, "inclusion": {"type": "weighted", "weights": [1.0]}}"#;
        let spec: ConfigSpec = serde_json::from_str(json).unwrap();
        assert!(
            spec.validate().is_ok(),
            "data-free validation cannot know p, so it must not guess"
        );
        let e = spec.into_config(&data()).unwrap_err().to_string();
        assert!(e.contains("p = 2"), "{e}");
    }

    /// The moves point is selected by *name*, not by an enum variant, so
    /// `ci/check-shelf-parity.py` cannot see it from rustdoc. This is the moves
    /// half of that gate.
    ///
    /// The reference is the built-in move set itself — `MoveSet::names()`, each
    /// move's own `name()` — not a list restated here. So a built-in that is
    /// renamed, added or dropped shows up as a mismatch instead of quietly
    /// becoming unreachable from Python and R.
    #[test]
    fn every_builtin_move_is_reachable_from_a_spec() {
        // The spec name each built-in answers to.
        let spec_names = [
            ("AddCentre", "add_centre"),
            ("RemoveCentre", "remove_centre"),
            ("AddDimension", "add_dimension"),
            ("RemoveDimension", "remove_dimension"),
            ("Change", "change"),
            ("Swap", "swap"),
        ];

        let builtin: Vec<&str> = MoveSetBuilder::stone_gosling()
            .build()
            .unwrap()
            .names()
            .collect();

        for name in &builtin {
            let (_, spec_name) = spec_names
                .iter()
                .find(|(move_name, _)| move_name == name)
                .unwrap_or_else(|| {
                    panic!(
                        "the built-in move `{name}` has no spec name, so no binding can select \
                         it: add an arm to `MoveSpec::build`"
                    )
                });
            let built = MoveSpec {
                name: (*spec_name).to_string(),
                weight: 1.0,
            }
            .build()
            .unwrap_or_else(|e| panic!("`{spec_name}` is not reachable from a spec: {e}"));
            assert_eq!(
                built.name(),
                *name,
                "the spec name `{spec_name}` built the wrong move"
            );
        }
        assert_eq!(builtin.len(), spec_names.len(), "stale spec-name table");
    }

    #[test]
    fn a_resolved_family_reports_its_name() {
        let config =
            from_json(r#"{"seed": 1, "response_family": "robust_t", "t_df": 4.0}"#).unwrap();
        assert_eq!(response_family_name(config.response_family()), "robust_t");
    }
}
