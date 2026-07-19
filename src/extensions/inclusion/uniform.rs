//! The uniform inclusion model: uniform preference over covariates, exactly
//! the paper's model, and the fit-time default.

use crate::extensions::inclusion::{InclusionModel, InclusionUsage};

/// The default: uniform preference over covariates, exactly the
/// paper's model. `update` is a no-op that consumes no RNG, so the default
/// chain is bit-independent of the extension machinery.
///
/// Weights are all exactly `1.0`: the weight-generic move ratios detect the
/// all-equal case and reduce exactly (not approximately) to the paper's
/// constants.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformInclusion {
    weights: Vec<f64>,
}

impl UniformInclusion {
    /// Uniform weights over `n_covariates` pre-encoding columns.
    pub fn new(n_covariates: usize) -> Self {
        Self {
            weights: vec![1.0; n_covariates],
        }
    }
}

impl InclusionModel for UniformInclusion {
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
