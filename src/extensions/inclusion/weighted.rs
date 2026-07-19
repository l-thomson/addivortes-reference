//! Fixed non-uniform inclusion weights.

use crate::extensions::inclusion::{InclusionModel, InclusionUsage};

/// Fixed, non-uniform inclusion weights: no adaptation, no
/// correction needed, and the test-bearing instantiation for the weighted
/// oracles. Weight validity (strictly positive, finite, right length) is
/// enforced by the config.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedInclusion {
    weights: Vec<f64>,
}

impl WeightedInclusion {
    /// Fixed weights, one per pre-encoding covariate.
    #[allow(dead_code)] // public API surface; selected via the config setter
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }
}

impl InclusionModel for WeightedInclusion {
    type Error = std::convert::Infallible;

    fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn update(
        &mut self,
        _usage: &InclusionUsage,
        _rng: &mut dyn rand_core::Rng,
    ) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}
