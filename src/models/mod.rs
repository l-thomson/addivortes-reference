//! OSS-1 spike: the model-files surface, one file per model.
//!
//! Each file owns everything derivation-bearing about its model — data
//! rules, augmentation, scale rule, link — and composes the sweep out of
//! engine calls. Crate-private for the spike (no public-API change; the
//! `ci/public-api.txt` snapshot is untouched); exercised by its own tests.
//! Becomes `pub` when the models surface lands for real.
#![allow(dead_code)]

pub(crate) mod binary;
pub(crate) mod gaussian;
