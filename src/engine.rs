//! The engine: the sampler machinery that is *not* an extension point.
//!
//! Everything here is fixed infrastructure: the Gauss–Seidel backfit loop, the
//! tessellation representation, the scaler/encoder, the fitted-model and data
//! types. Contributors extending the model do not edit this module; they
//! implement a trait under `crate::extensions` instead.
//!
//! The module is private: its public types are re-exported from the crate root.

pub mod backfit;
pub mod builder;
pub mod column;
pub mod config;
pub mod data;
pub mod error;
pub mod mathsfn;
pub mod model;
pub mod sampler;
pub mod scaler;
pub mod tessellation;
