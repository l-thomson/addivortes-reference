//! The component shelves: the crate-internal seams behind the engine.
//!
//! Each submodule is one seam and has the same shape: the `.rs` file holds
//! the trait, and the folder beside it is the *shelf*: one file per shipped
//! implementation. Model files and the spec layer wire shelf entries through
//! `engine::builder`; each entry keeps a conformance check in
//! `crate::conformance` (test builds).
//!
//! The shelf is inventory: entries keep their introspection accessors even
//! when only their gates read them, so the seams stay uniformly testable.
#![allow(dead_code)]
//!
//! | Point | Trait | Swap it when |
//! |---|---|---|
//! | `moves` | `ProposalMove` | the sampler should explore tessellation space differently |
//! | `coord` | `CoordinateDistribution` | centres should be proposed/priced from another law |
//! | `distance` | `PairwiseDistance` | "nearness" should mean something else |
//! | `inclusion` | `InclusionModel` | some covariates should be favoured |
//! | `cell_model` | `CellModel` / `CellStats` | cells should hold something other than a Gaussian mean |
//! | `response` | `ResponseModel` | the response is not Gaussian |
//! | `scale` | `ScaleModel` | the noise level is not one global constant |
//! | `count_priors` | `CountPriors` | the cell/dimension count priors should differ |
//! | `basis` | (reuses `CellModel`) | cell outputs should be linear in covariates |
//! | `membership` | `MembershipKernel` | assignment to cells should be soft |
//!
//! The engine (`crate::engine`) is fixed, extending the model never means
//! editing it.

pub mod basis;
pub mod cell_model;
pub mod coord;
pub mod count_priors;
pub mod distance;
pub mod inclusion;
pub mod membership;
pub mod moves;
pub mod response;
pub mod scale;

// Internal plumbing shared by the points above: not an extension point.
pub(crate) mod erasure;
