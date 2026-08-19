//! The model-files surface, one file per model.
//!
//! Each file owns everything derivation-bearing about its model (data rules,
//! augmentation, scale rule, link) and composes the sweep out of engine
//! calls. Crate-private until the models surface lands for real: no
//! public-API change, so the `ci/public-api.txt` snapshot is untouched.
#![allow(dead_code)]

pub(crate) mod binary;
pub(crate) mod gaussian;
pub(crate) mod h;
pub(crate) mod soft;
