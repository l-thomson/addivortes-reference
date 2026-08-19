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
//! The engine is sealed: the component seams (moves, distances, coordinate
//! laws, inclusion, cell models, response augmentation, scale models, count
//! priors, bases, membership) are crate-internal, reached only by the
//! crate's own model files. Callers select shipped behaviour as data: the
//! plain [`AddiVortesConfig`] for hyperparameters, metrics and the response
//! family, and (with the `serde` feature) [`config_spec::ConfigSpec`] for
//! every shelf-selectable component, the same surface the R and Python
//! bindings pass through.
//!
//! For research needs beyond the shelf, drive the loop directly: construct a
//! [`Sampler`], then alternate [`Sampler::set_response`] and
//! [`Sampler::step`] from your own outer Gibbs loop, so the engine becomes
//! one conditional inside your sampler and the novel block stays in your
//! crate (worked example: `examples/template_embed.rs`). The loop speaks the
//! caller's response scale and consumes none of your RNG. Two limits are
//! deliberate: the loop cannot rewire membership or cell internals, and an
//! in-sweep hook is not offered. Out of scope: non-Voronoi base learners,
//! non-augmentable likelihoods, within-chain parallelism.
//!
//! # Crate map
//!
//! `src/engine/` is the fixed machinery: the Gibbs backfitter, the
//! tessellation, the scaler/encoder, the fitted-model API. `src/extensions/`
//! holds the crate-internal component shelves, one folder per seam.
//! `src/models/` holds one readable file per shipped model, composed from
//! engine calls through the crate-internal wiring surface.
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

// The statistical batteries and per-component conformance checks: crate
// infrastructure for the shelf's own gates, compiled with them.
#[cfg(test)]
mod calibration;
#[cfg(test)]
mod conformance;

pub mod diagnostics;

// The data-only config surface the language bindings pass through, so that a
// new shelf entry reaches Python and R with no binding-side edit.
#[cfg(feature = "serde")]
pub mod config_spec;

// The component shelves: crate-internal seams, reached by the model files
// and the spec layer through `engine::builder`.
mod extensions;

// The model-files surface, one file per model. Crate-private until the models
// surface lands for real (no public-API change).
mod models;

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

#[cfg(test)]
mod calibration_acceptance;
#[cfg(test)]
mod gate_tests;
#[cfg(test)]
mod stat_gates;
#[cfg(test)]
mod test_support;
