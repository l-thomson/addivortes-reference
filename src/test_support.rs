//! Test-only helpers for comparing floating-point numbers.
//!
//! Floating-point results rarely come out exactly equal after a calculation,
//! so tests check that two values are close enough rather than identical.

/// Panic unless `actual` and `expected` are within `tolerance` of each other.
///
/// Use this near zero, where a relative comparison is meaningless.
pub(crate) fn assert_abs_eq(actual: f64, expected: f64, tolerance: f64) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance,
        "assertion failed: |{actual} - {expected}| = {difference} > {tolerance}"
    );
}

/// Panic unless `actual` and `expected` agree to within a relative `tolerance`
/// (their difference divided by the larger magnitude of the two).
///
/// Use this for quantities that scale with the data (sums over n, likelihoods).
/// For values at or near zero use [`assert_abs_eq`] instead; this helper panics
/// if both values are zero, on purpose.
#[allow(dead_code)] // consumed by the scaling / oracle tests
pub(crate) fn assert_rel_eq(actual: f64, expected: f64, tolerance: f64) {
    let magnitude = actual.abs().max(expected.abs());
    assert!(
        magnitude > 0.0,
        "assert_rel_eq({actual}, {expected}): both zero: use assert_abs_eq near zero"
    );
    let relative = (actual - expected).abs() / magnitude;
    assert!(
        relative <= tolerance,
        "assertion failed: |{actual} - {expected}| / {magnitude} = {relative} > {tolerance}"
    );
}

/// A test-side [`ScaleModel`](crate::extensions::scale::ScaleModel) wrapper sharing
/// an [`HVariance`](crate::extensions::scale::HVariance) with the test driver
/// (which reads `s²(xᵢ)` and the variance tessellations between sweeps):
/// `update` delegates; the precisions are cached locally so `precisions()`
/// can hand out a plain slice.
#[derive(Debug)]
pub(crate) struct SharedHVariance {
    pub(crate) inner: std::sync::Arc<std::sync::Mutex<crate::extensions::scale::HVariance>>,
    pub(crate) cache: Vec<f64>,
}

impl crate::extensions::scale::ScaleModel for SharedHVariance {
    type Error = crate::engine::error::AddiVortesError;
    fn update(
        &mut self,
        ctx: &crate::extensions::scale::ScaleCtx<'_>,
        rng: &mut dyn rand_core::Rng,
    ) -> Result<(), Self::Error> {
        let mut inner = self.inner.lock().expect("test lock");
        crate::extensions::scale::ScaleModel::update(&mut *inner, ctx, rng)?;
        self.cache.clear();
        self.cache.extend_from_slice(
            crate::extensions::scale::ScaleModel::precisions(&*inner).expect("updated above"),
        );
        Ok(())
    }
    fn sigma_sq(&self) -> f64 {
        1.0
    }
    fn precisions(&self) -> Option<&[f64]> {
        if self.cache.is_empty() {
            None
        } else {
            Some(&self.cache)
        }
    }
}
