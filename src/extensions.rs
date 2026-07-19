//! The extension points: everything a researcher can swap.
//!
//! Each submodule is one extension point and has the same shape: the `.rs`
//! file holds the trait you implement, and the folder beside it is the *shelf*:
//! one file per shipped implementation. A copy-paste starting point for each
//! lives in `examples/template_<name>.rs`, and each has a conformance check in
//! [`crate::conformance`].
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
