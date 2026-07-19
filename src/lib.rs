//! Bayesian additive Voronoi-tessellation regression: a Rust implementation
//! of the **AddiVortes** method (Stone & Gosling, 2025, JCGS 34(3):859–871).
//!
//! The model is a sum of `m` Voronoi tessellations: `Y = Σ g(x | T_j, M_j) + ε`
//! with `ε ~ N(0, σ²)`. Each tessellation partitions a random subspace of the
//! covariates into cells; each cell carries one output value; a Gibbs
//! backfitting sampler with six structural Metropolis–Hastings moves explores
//! the posterior.
//!
//! # Quick start
//!
//! ```
//! use addivortes::{AddiVortesConfig, Data};
//!
//! # fn main() -> addivortes::Result<()> {
//! // 30 observations of a noiseless curve, one covariate.
//! let n = 30;
//! let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
//! let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
//! let x = Data::new(xs, n, 1)?;
//!
//! let model = AddiVortesConfig::new(42) // the seed is mandatory
//!     .with_m(10)                       // small ensemble for the example
//!     .with_burn_in(20)
//!     .with_draws(30)
//!     .fit(&x, &y)?;
//! let predictions = model.predict(&x)?;
//! assert_eq!(predictions.len(), n);
//! # Ok(())
//! # }
//! ```
//!
//! # Reproducibility
//!
//! A chain is reproducible given the same seed, the same addivortes version,
//! and the same compilation target, built with this crate's default release
//! profile and no overriding RUSTFLAGS (in particular no -Ctarget-cpu=native
//! and no target-feature=+fma). Any change that alters the sampled chain for
//! a fixed seed bumps the 0.y minor version and regenerates the golden
//! vectors deliberately; patch releases guarantee bit-identical chains,
//! enforced by a golden-chain regression test in CI (Linux x86_64, macOS ARM,
//! Windows). Any major bump of rand, rand_core, rand_distr, or libm is
//! treated as chain-altering by definition.
//!
//! # Persistence
//!
//! With the `serde` cargo feature, a model fitted with built-in extension points
//! round-trips through any serde format with bit-identical predictions
//! (JSON needs `serde_json`'s `float_roundtrip` feature); loading validates
//! the payload, so a corrupt file is an error, never a panic. Models
//! carrying custom extension points refuse to serialise, persist their
//! [`into_parts`](FittedAddiVortes::into_parts) values instead. See the
//! [`FittedAddiVortes`] docs.
//!
//! # Extending
//!
//! The crate has ten extension points. Each has the same shape: a trait you
//! implement, a shelf of shipped implementations beside it, a copy-paste
//! template in `examples/`, and a one-command conformance check. Every default
//! is replaceable; the paper's method is one shelf entry.
//! The workflow is the same three steps on every point:
//!
//! 1. copy the point's template (`examples/template_<point>.rs`), rename it,
//!    and fill the marked blocks with your maths;
//! 2. run it: each template is its own conformance check, printing
//!    plain-language pass/fail verdicts in seconds
//!    (`cargo run --example template_moves`);
//! 3. only if the component is destined for real inference, run the
//!    statistical battery ([`calibration`]): the conformance checks catch
//!    mechanical faults, the battery catches wrong maths.
//!
//! Each point's module documents what you implement, what the engine
//! provides, what is on the shelf, which conformance check applies, and the
//! sources. Three rules keep a custom component's chains reproducible (the
//! conformance checks verify the mechanical ones):
//!
//! - route every transcendental in a structure ratio through [`mathsfn`]
//!   (`ln`, `exp`, …), never `f64::ln`: std float results are
//!   platform-dependent and break bit-exact reproducibility;
//! - take all randomness from the `&mut dyn rand_core::Rng` you are handed:
//!   hidden state or system entropy breaks the same-seed contract;
//! - look everything up by global covariate index (a tessellation's `dims`
//!   hold global indices), never by local slot position.
//!
//! When no point fits: first check the idea is not a
//! [`ResponseModel`](response::ResponseModel) augmentation in disguise (most
//! are); otherwise embed, driving [`Sampler::step`] from your own outer Gibbs
//! loop through [`Sampler::set_response`], so the engine becomes one
//! conditional inside your sampler and the novel block stays in your crate
//! (worked example: `examples/template_embed.rs`). An in-sweep hook is not
//! offered. Out of scope: non-Voronoi base learners, non-augmentable
//! likelihoods, within-chain parallelism.
//!
//! # Crate map
//!
//! `src/engine/` is the fixed machinery: the Gibbs backfitter, the
//! tessellation, the scaler/encoder, the fitted-model API. Extending the model
//! never means editing it. `src/extensions/` holds the ten points, one folder
//! each: the `.rs` file is the trait you implement, and the folder beside it is
//! the shelf.
//!
//! | Point (`src/extensions/`) | Trait | Shelf (one file per approach) | Default (paper) | Select via |
//! |---|---|---|---|---|
//! | `moves` | `ProposalMove` | `stone_gosling` (the six paper moves) | `stone_gosling` | `with_move_set` |
//! | `coord` | `CoordinateDistribution` | `normal`, `wrapped_normal` | `normal` | `with_coords` |
//! | `distance` | `PairwiseDistance`, `CellAssigner` | `euclidean`, `spherical`, `per_column`, `gower`, `manhattan`, `minkowski`, `cosine`, `mahalanobis` | `per_column` (all-Euclidean = `euclidean`) | `with_distance` |
//! | `inclusion` | `InclusionModel` | `uniform`, `weighted`, `dart` | `uniform` | `with_inclusion` |
//! | `cell_model` | `CellModel`, `CellStats` | `gaussian`, `weighted_gaussian`, `inv_chi_sq` | `gaussian` | `with_cell_model` |
//! | `response` | `ResponseModel` | `albert_chib`, `robust_t` | none (Gaussian) | `with_response_model` |
//! | `scale` | `ScaleModel` | `global_sigma`, `pinned`, `weighted_global_sigma`, `h_variance` | `global_sigma` | `with_scale_model` |
//! | `count_priors` | `CountPriors` | `shifted_poisson_binomial` | `shifted_poisson_binomial` | `with_count_priors` |
//! | `basis` | `CellBasis` (+ `CellModel`) | `linear` | constant cells | `with_cell_basis` (+ `with_cell_model`) |
//! | `membership` | `MembershipKernel` | `softmax` | hard membership | `with_membership` |
//!
//! Shelf files are named for the method they implement (`euclidean`,
//! `gaussian`, `stone_gosling`). The table above marks the paper's entry and
//! the fit-time default; each file's own docs repeat it.
//!
//! Templates live in `examples/` rather than beside their point, because an
//! example compiles against the public API only: exactly the constraint an
//! external extension faces.
//!
//! Validation sits in two modules: [`conformance`] (per-component `check_*`
//! suites you run against your own type) and [`calibration`] (the externalised
//! Geweke/SBC drivers, for components destined for real inference).
//!
//! # Fidelity to the paper
//!
//! This crate implements the method as published (see `NOTICE`).
//! Where the paper leaves behaviour unstated, this crate chooses strictness
//! and documents the choice at the implementation site: invalid
//! hyperparameters (ω ≥ p makes the stated dimension prior degenerate),
//! degenerate columns or response, and any NaN/Inf in X or y are hard
//! errors, never silently repaired. One default is not the paper's: λ_c ships
//! at 5 (the paper reports 25), chosen by benchmark calibration of this
//! implementation's exact shifted-Poisson cell-count prior. The paper's
//! setting: `AddiVortesConfig::with_lambda_c`.
#![warn(missing_docs)]

// The engine: fixed sampler machinery, not an extension point. Private module;
// its public types are re-exported at the crate root below.
mod engine;

pub mod calibration;
pub mod conformance;
pub mod diagnostics;

// The data-only config surface the language bindings pass through, so that a
// new shelf entry reaches Python and R with no binding-side edit.
#[cfg(feature = "serde")]
pub mod config_spec;

// The extension points: the ten swappable surfaces. Private module; each point
// is re-exported at the crate root below so import paths stay short.
mod extensions;

pub use engine::mathsfn;

pub use engine::config::AddiVortesConfig;
pub use engine::data::{Data, Metric, Warning};
pub use engine::error::{AddiVortesError, Result};
pub use engine::model::{
    CredibleInterval, Draws, FittedAddiVortes, PosteriorSamples, PredictionInterval,
    QuantilePredictions, ResponseFamily,
};
pub use engine::sampler::{Draw, OwnedDraw, Sampler};
pub use engine::scaler::FittedScaler;
pub use engine::tessellation::Tessellation;

// The extension points: unconditionally public,
// the whole surface is snapshotted by the public-api CI gate. Each point's
// module is re-exported here so its shelf keeps a short import path
// (`addivortes::cell_model::GaussianCellModel`).
pub use extensions::{basis, cell_model, count_priors, response, scale};

pub use extensions::distance::{
    AssignmentCache, AssignmentDelta, CellAssigner, ColumnMetrics, Cosine, Euclidean, Gower,
    GowerKind, Mahalanobis, Manhattan, Minkowski, PairwiseDistance, Spherical,
};

pub use extensions::coord::{CoordinateDistribution, EuclideanNormal, WrappedNormal};

pub use extensions::inclusion::{
    DartInclusion, InclusionModel, InclusionUsage, UniformInclusion, WeightedInclusion,
};

pub use extensions::membership::{MembershipKernel, SoftmaxKernel};

pub use extensions::count_priors::{CountPriors, ShiftedPoissonBinomial};

pub use extensions::moves::{
    AddCentre, AddDimension, Change, ModelCtx, MoveSet, MoveSetBuilder, Proposal, ProposalMove,
    RemoveCentre, RemoveDimension, Reverse, Swap,
};

#[cfg(test)]
mod gate_tests;
#[cfg(test)]
mod stat_gates;
#[cfg(test)]
mod test_support;
