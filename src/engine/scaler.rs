//! Scaling, one-hot encoding (`FittedScaler`), and the one-time prior
//! calibration: σ̂ via a hand-rolled OLS/Cholesky solve, λ via the χ² quantile,
//! σ_μ² from (k, m), all in scaled space.
//!
//! Coordinate systems: callers see raw space; the sampler
//! sees scaled space (y in [−0.5, 0.5], Euclidean X columns min-max scaled to
//! [−0.5, 0.5], spherical columns untouched on [−π, π], categorical columns
//! one-hot expanded then scaled like Euclidean, i.e. to exactly ±0.5).

use crate::engine::column::semantics_for;
use crate::engine::data::{Data, Metric};
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::mathsfn;

/// The fitted scaling + encoding state, stored on the fitted model and applied
/// to every prediction input. All accessors document their
/// coordinate system.
///
/// With the `serde` feature, deserialisation validates the whole encoding
/// layout (lengths, the column map, level lists, scaling ranges): a corrupt
/// or hand-edited payload is rejected, never a panic or a NaN prediction.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "FittedScalerParts")
)]
pub struct FittedScaler {
    y_min: f64,
    y_max: f64,
    /// Per encoded column; meaningful for Euclidean-treated columns only
    /// (spherical entries hold the fixed domain ends −π/π and are never applied).
    x_min: Vec<f64>,
    x_max: Vec<f64>,
    /// Per encoded column (categorical columns already expanded to their
    /// one-hot groups, which are Euclidean here).
    metrics: Vec<Metric>,
    /// Per pre-encoding column: the sorted distinct level values of a
    /// categorical column, `None` for non-categorical columns.
    levels: Vec<Option<Vec<f64>>>,
    /// Encoded column → pre-encoding (caller) column.
    col_map: Vec<usize>,
    n_raw_cols: usize,
}

impl FittedScaler {
    /// The identity scaler behind the pinned-prior battery constructor
    /// (`Sampler::pinned_prior`): the caller's data is already in the
    /// sampler's scaled coordinate system, so y maps [−0.5, 0.5] onto itself
    /// and no column is rescaled or encoded.
    pub(crate) fn identity(p: usize, metrics: Vec<Metric>) -> Self {
        debug_assert_eq!(metrics.len(), p);
        let (x_min, x_max): (Vec<f64>, Vec<f64>) = metrics
            .iter()
            .map(|metric| match metric {
                Metric::Spherical => (-std::f64::consts::PI, std::f64::consts::PI),
                _ => (-0.5, 0.5),
            })
            .unzip();
        Self {
            y_min: -0.5,
            y_max: 0.5,
            x_min,
            x_max,
            metrics,
            levels: vec![None; p],
            col_map: (0..p).collect(),
            n_raw_cols: p,
        }
    }

    /// Fit the scaler on validated raw input and return
    /// `(scaler, encoded scaled X, scaled y)` in the sampler's coordinate system.
    ///
    /// Callers must have run `validate_fit` first; this function still hard-errors
    /// (never divides by zero) on what it alone can see: a constant Euclidean
    /// column and a single-level categorical column both raise
    /// `DegenerateFeature` with the pre-encoding column index.
    pub(crate) fn fit(x: &Data, y: &[f64], metrics: &[Metric]) -> Result<(Self, Data, Vec<f64>)> {
        debug_assert_eq!(metrics.len(), x.n_cols());
        debug_assert!(y.len() == x.n_rows() && y.len() >= 2);

        // y range (validate_fit rejected constant y).
        let (y_min, y_max) = min_max(y.iter().copied());
        let y_scaled: Vec<f64> = y
            .iter()
            .map(|&v| (v - y_min) / (y_max - y_min) - 0.5)
            .collect();

        // Pass 1: per raw column, the distinct sorted levels for categoricals.
        let n = x.n_rows();
        let mut levels: Vec<Option<Vec<f64>>> = Vec::with_capacity(x.n_cols());
        for (col, metric) in metrics.iter().enumerate() {
            if *metric == Metric::Categorical {
                let mut seen: Vec<f64> = Vec::new();
                for r in 0..n {
                    let v = x.row(r)[col];
                    if !seen.contains(&v) {
                        seen.push(v);
                    }
                }
                seen.sort_by(f64::total_cmp);
                if seen.len() < 2 {
                    // A single-level categorical encodes to a constant column,
                    // the same no-information condition as a constant Euclidean.
                    return Err(AddiVortesError::DegenerateFeature { col });
                }
                levels.push(Some(seen));
            } else {
                levels.push(None);
            }
        }

        // Pass 2: build the encoded (pre-scaling) matrix column layout.
        let mut expanded_metrics: Vec<Metric> = Vec::new();
        let mut col_map: Vec<usize> = Vec::new();
        for (col, metric) in metrics.iter().enumerate() {
            match metric {
                Metric::Categorical => {
                    let l = levels[col].as_ref().expect("categorical has levels").len();
                    for _ in 0..l {
                        expanded_metrics.push(Metric::Euclidean);
                        col_map.push(col);
                    }
                }
                m => {
                    expanded_metrics.push(*m);
                    col_map.push(col);
                }
            }
        }
        let p_enc = expanded_metrics.len();
        let mut encoded = vec![0.0_f64; n * p_enc];
        for r in 0..n {
            let raw = x.row(r);
            let mut e = 0;
            for (col, metric) in metrics.iter().enumerate() {
                match metric {
                    Metric::Categorical => {
                        let lv = levels[col].as_ref().expect("categorical has levels");
                        for level in lv {
                            encoded[r * p_enc + e] = if raw[col] == *level { 1.0 } else { 0.0 };
                            e += 1;
                        }
                    }
                    _ => {
                        encoded[r * p_enc + e] = raw[col];
                        e += 1;
                    }
                }
            }
        }

        // Pass 3: per encoded column, the column's semantics own its scaling
        // range and per-value transform (angle columns: fixed [−π, π], identity;
        // real columns including one-hot: min–max onto [−0.5, 0.5]). A `None`
        // range is a degenerate (constant) column; report the caller-visible
        // index (validate_fit catches raw Euclidean ones; this also guards any
        // future encoded source).
        let mut x_min = vec![0.0_f64; p_enc];
        let mut x_max = vec![0.0_f64; p_enc];
        for e in 0..p_enc {
            let semantics = semantics_for(&expanded_metrics[e]);
            let column: Vec<f64> = (0..n).map(|r| encoded[r * p_enc + e]).collect();
            let (lo, hi) = semantics
                .fit_range(&column)
                .ok_or(AddiVortesError::DegenerateFeature { col: col_map[e] })?;
            x_min[e] = lo;
            x_max[e] = hi;
            for r in 0..n {
                let v = &mut encoded[r * p_enc + e];
                *v = semantics.scale(*v, lo, hi);
            }
        }

        let scaler = Self {
            y_min,
            y_max,
            x_min,
            x_max,
            metrics: expanded_metrics,
            levels,
            col_map,
            n_raw_cols: x.n_cols(),
        };
        let encoded = Data::new(encoded, n, p_enc).expect("shape correct by construction");
        Ok((scaler, encoded, y_scaled))
    }

    /// Apply the fitted encoding + scaling to raw prediction input (already
    /// boundary-validated). Errors with `UnseenCategory` on a categorical value
    /// that was not present during fit. Euclidean values outside the training
    /// range scale to values outside [−0.5, 0.5], deliberately not clamped.
    #[allow(dead_code)] // consumed by predict(); fully tested here
    pub(crate) fn apply_x(&self, x: &Data) -> Result<Data> {
        debug_assert_eq!(x.n_cols(), self.n_raw_cols);
        let n = x.n_rows();
        let p_enc = self.metrics.len();
        let mut out = vec![0.0_f64; n * p_enc];
        for r in 0..n {
            let raw = x.row(r);
            let mut e = 0;
            for (col, maybe_levels) in self.levels.iter().enumerate() {
                match maybe_levels {
                    Some(levels) => {
                        let value = raw[col];
                        let hit = levels
                            .iter()
                            .position(|l| *l == value)
                            .ok_or(AddiVortesError::UnseenCategory { col, value })?;
                        for (i, _) in levels.iter().enumerate() {
                            let one_hot = if i == hit { 1.0 } else { 0.0 };
                            out[r * p_enc + e] = self.scale_encoded(e, one_hot);
                            e += 1;
                        }
                    }
                    None => {
                        out[r * p_enc + e] = self.scale_encoded(e, raw[col]);
                        e += 1;
                    }
                }
            }
        }
        Ok(Data::new(out, n, p_enc).expect("shape correct by construction"))
    }

    /// Scale one value for encoded column `e` through the column's semantics
    /// (angle columns: identity; real columns: affine onto [−0.5, 0.5]).
    fn scale_encoded(&self, e: usize, v: f64) -> f64 {
        semantics_for(&self.metrics[e]).scale(v, self.x_min[e], self.x_max[e])
    }

    /// Map a response value from raw space to scaled space ([−0.5, 0.5] over the
    /// training range).
    #[allow(dead_code)] // consumed by predict(); fully tested here
    pub(crate) fn scale_y_value(&self, v: f64) -> f64 {
        (v - self.y_min) / (self.y_max - self.y_min) - 0.5
    }

    /// Map a response value from scaled space back to raw (caller) space.
    #[allow(dead_code)] // consumed by predict(); fully tested here
    pub(crate) fn unscale_y_value(&self, v: f64) -> f64 {
        (v + 0.5) * (self.y_max - self.y_min) + self.y_min
    }

    /// Minimum of the training response (raw space).
    pub fn y_min(&self) -> f64 {
        self.y_min
    }

    /// Maximum of the training response (raw space).
    pub fn y_max(&self) -> f64 {
        self.y_max
    }

    /// Per-encoded-column training minima (raw space; spherical entries hold
    /// the fixed domain end −π and are never applied).
    pub fn x_min(&self) -> &[f64] {
        &self.x_min
    }

    /// Per-encoded-column training maxima (raw space; spherical entries hold
    /// the fixed domain end π and are never applied).
    pub fn x_max(&self) -> &[f64] {
        &self.x_max
    }

    /// Per-encoded-column metrics: categorical columns appear here as their
    /// one-hot groups (Euclidean); order matches the encoded matrix.
    pub fn metrics(&self) -> &[Metric] {
        &self.metrics
    }

    /// The sorted distinct level values seen at fit for pre-encoding column
    /// `col` (raw space), or `None` if that column is not categorical.
    pub fn levels(&self, col: usize) -> Option<&[f64]> {
        self.levels.get(col).and_then(|l| l.as_deref())
    }

    /// Number of caller-visible (pre-encoding) columns.
    pub fn n_raw_cols(&self) -> usize {
        self.n_raw_cols
    }

    /// Number of encoded columns (after one-hot expansion).
    pub fn n_encoded_cols(&self) -> usize {
        self.metrics.len()
    }

    /// Encoded column → pre-encoding column (one-hot groups share their source
    /// column). Drives inclusion-weight expansion and usage aggregation.
    pub(crate) fn col_map(&self) -> &[usize] {
        &self.col_map
    }
}

/// Serde shadow of [`FittedScaler`]: deserialisation lands here first, then
/// through the validating `TryFrom` below.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct FittedScalerParts {
    y_min: f64,
    y_max: f64,
    x_min: Vec<f64>,
    x_max: Vec<f64>,
    metrics: Vec<Metric>,
    levels: Vec<Option<Vec<f64>>>,
    col_map: Vec<usize>,
    n_raw_cols: usize,
}

#[cfg(feature = "serde")]
impl TryFrom<FittedScalerParts> for FittedScaler {
    type Error = crate::engine::error::SavedModelError;

    /// Every invariant `fit` establishes is re-checked, so the loaded scaler
    /// can never panic or emit non-finite values from `apply_x`/`scale_y_value`:
    /// consistent encoded lengths, a column map that matches the level lists'
    /// expansion, a usable y range, per-column scaling ranges, and sorted
    /// distinct finite level values.
    fn try_from(parts: FittedScalerParts) -> std::result::Result<Self, Self::Error> {
        let bad = |reason: String| Err(crate::engine::error::SavedModelError(reason));
        let p_enc = parts.metrics.len();
        if parts.x_min.len() != p_enc || parts.x_max.len() != p_enc || parts.col_map.len() != p_enc
        {
            return bad("scaler length mismatch across encoded columns".into());
        }
        if parts.levels.len() != parts.n_raw_cols {
            return bad("scaler has one level list per raw column".into());
        }
        if !(parts.y_min.is_finite() && parts.y_max.is_finite() && parts.y_min < parts.y_max) {
            return bad("scaler y range must be finite with y_min < y_max".into());
        }
        // The column map must be exactly the in-place expansion the level
        // lists describe (a categorical column contributes one encoded column
        // per level, everything else one).
        let mut expected_col_map = Vec::with_capacity(p_enc);
        for (col, maybe_levels) in parts.levels.iter().enumerate() {
            let width = match maybe_levels {
                Some(levels) => {
                    if levels.len() < 2 {
                        return bad(format!("column {col} has fewer than two levels"));
                    }
                    if levels.iter().any(|l| !l.is_finite()) {
                        return bad(format!("column {col} has a non-finite level"));
                    }
                    if levels
                        .windows(2)
                        .any(|pair| pair[0].total_cmp(&pair[1]) != std::cmp::Ordering::Less)
                    {
                        return bad(format!("column {col} levels are not sorted and distinct"));
                    }
                    levels.len()
                }
                None => 1,
            };
            expected_col_map.extend(std::iter::repeat_n(col, width));
        }
        if expected_col_map != parts.col_map {
            return bad("scaler column map does not match the level lists".into());
        }
        for (e, metric) in parts.metrics.iter().enumerate() {
            if *metric == Metric::Categorical {
                return bad(format!(
                    "encoded column {e} is Categorical: encoding expands categoricals \
                     to Euclidean one-hot columns"
                ));
            }
            if !(parts.x_min[e].is_finite() && parts.x_max[e].is_finite()) {
                return bad(format!("encoded column {e} has a non-finite fit range"));
            }
            // Euclidean scaling divides by (max − min).
            if *metric == Metric::Euclidean && parts.x_min[e] >= parts.x_max[e] {
                return bad(format!(
                    "encoded Euclidean column {e} needs x_min < x_max to scale"
                ));
            }
        }
        Ok(Self {
            y_min: parts.y_min,
            y_max: parts.y_max,
            x_min: parts.x_min,
            x_max: parts.x_max,
            metrics: parts.metrics,
            levels: parts.levels,
            col_map: parts.col_map,
            n_raw_cols: parts.n_raw_cols,
        })
    }
}

/// (min, max) of a non-empty sequence, ascending-index accumulation.
fn min_max(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in values {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// One-time prior calibration (the paper's prior defaults)
// ---------------------------------------------------------------------------

/// σ_μ² = (0.5 / (k√m))² (the paper's μ prior), scaled space.
pub(crate) fn sigma_mu_sq(k: f64, m: usize) -> f64 {
    let s = 0.5 / (k * (m as f64).sqrt());
    s * s
}

/// σ̂ for the σ² prior: OLS residual SD on the intercept-augmented scaled,
/// encoded design when n > p + 1 and the system is well conditioned; otherwise
/// the sample SD of scaled y: an explicit fallback, never silent NaN.
pub(crate) fn sigma_hat(x_scaled: &Data, y_scaled: &[f64]) -> f64 {
    ols_residual_sd(x_scaled, y_scaled).unwrap_or_else(|| sample_sd(y_scaled))
}

/// OLS residual SD via normal equations + hand-rolled Cholesky on the
/// (p+1)×(p+1) system (no linear-algebra dependency; one-time cost). Returns
/// `None` when n ≤ p + 1 or the system is rank-deficient/ill-conditioned.
pub(crate) fn ols_residual_sd(x: &Data, y: &[f64]) -> Option<f64> {
    let n = x.n_rows();
    let p1 = x.n_cols() + 1; // intercept column + p features
    if n <= p1 {
        return None;
    }

    // A = DᵀD, b = Dᵀy with D = [1 | X]; ascending-index accumulation (pinned).
    let design = |r: usize, j: usize| -> f64 { if j == 0 { 1.0 } else { x.row(r)[j - 1] } };
    let mut a = vec![0.0_f64; p1 * p1];
    let mut b = vec![0.0_f64; p1];
    for (r, &y_r) in y.iter().enumerate().take(n) {
        for i in 0..p1 {
            let di = design(r, i);
            b[i] += di * y_r;
            for j in i..p1 {
                a[i * p1 + j] += di * design(r, j);
            }
        }
    }
    for i in 0..p1 {
        for j in 0..i {
            a[i * p1 + j] = a[j * p1 + i];
        }
    }

    // Cholesky A = LLᵀ; a near-zero pivot (relative to the largest diagonal)
    // signals rank deficiency → fallback, never a divide-by-near-zero.
    let max_diag = (0..p1).fold(0.0_f64, |acc, i| acc.max(a[i * p1 + i].abs()));
    let mut l = vec![0.0_f64; p1 * p1];
    for i in 0..p1 {
        for j in 0..=i {
            let mut sum = a[i * p1 + j];
            for k in 0..j {
                sum -= l[i * p1 + k] * l[j * p1 + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-10 * max_diag {
                    return None;
                }
                l[i * p1 + i] = sum.sqrt();
            } else {
                l[i * p1 + j] = sum / l[j * p1 + j];
            }
        }
    }

    // Solve LLᵀ β = b.
    let mut beta = b;
    for i in 0..p1 {
        for k in 0..i {
            beta[i] -= l[i * p1 + k] * beta[k];
        }
        beta[i] /= l[i * p1 + i];
    }
    for i in (0..p1).rev() {
        for k in (i + 1)..p1 {
            beta[i] -= l[k * p1 + i] * beta[k];
        }
        beta[i] /= l[i * p1 + i];
    }

    let mut rss = 0.0_f64;
    for (r, &y_r) in y.iter().enumerate().take(n) {
        let mut fitted = 0.0_f64;
        for (j, coefficient) in beta.iter().enumerate() {
            fitted += coefficient * design(r, j);
        }
        let residual = y_r - fitted;
        rss += residual * residual;
    }
    let sd = (rss / (n - p1) as f64).sqrt();
    sd.is_finite().then_some(sd)
}

/// Sample standard deviation (divisor n − 1), ascending-index accumulation.
pub(crate) fn sample_sd(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let ss = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
    (ss / (n - 1.0)).sqrt()
}

/// λ of the σ² ~ inverse-χ²(ν, λ) prior, calibrated so Pr(σ < σ̂) = q
/// (as in the paper): with σ² = νλ/X, X ~ χ²_ν, the condition rearranges to
/// λ = σ̂² · F⁻¹_{χ²_ν}(1 − q) / ν.
pub(crate) fn calibrate_lambda(nu: f64, q: f64, sigma_hat: f64) -> f64 {
    debug_assert!(nu > 0.0 && q > 0.0 && q < 1.0);
    sigma_hat * sigma_hat * chi2_quantile(1.0 - q, nu) / nu
}

/// χ²_ν quantile: the x with P(ν/2, x/2) = p, found by deterministic bisection
/// on the regularised incomplete gamma (monotone; the 200 halvings below are
/// well past the ~60 that exhaust f64 precision, no Newton fragility).
pub(crate) fn chi2_quantile(p: f64, nu: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0 && nu > 0.0);
    let a = 0.5 * nu;
    let target = p;
    // Bracket: grow hi until the CDF exceeds the target.
    let mut hi = nu.max(1.0);
    while gamma_p(a, 0.5 * hi) < target {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_p(a, 0.5 * mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Regularised lower incomplete gamma P(a, x): series for x < a + 1, Lentz
/// continued fraction for the complement otherwise (Numerical Recipes gser/gcf
/// structure, evaluated through the pinned `mathsfn`/libm path).
pub(crate) fn gamma_p(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && x >= 0.0);
    if x == 0.0 {
        return 0.0;
    }
    let log_prefactor = a * mathsfn::ln(x) - x - mathsfn::lgamma(a);
    if x < a + 1.0 {
        // Series: P(a,x) = e^{-x} x^a / Γ(a) · Σ_{k≥0} x^k / (a(a+1)…(a+k)).
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut denominator = a;
        for _ in 0..500 {
            denominator += 1.0;
            term *= x / denominator;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * mathsfn::exp(log_prefactor)
    } else {
        // Continued fraction for Q(a,x) (modified Lentz).
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - mathsfn::exp(log_prefactor) * h
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::{Rng, SeedableRng};

    use super::*;
    use crate::test_support::{assert_abs_eq, assert_rel_eq};

    fn uniform(rng: &mut ChaCha8Rng, lo: f64, hi: f64) -> f64 {
        // 53-bit uniform in [0,1), plenty for test fixtures.
        let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }

    // ---- scaling round trips ----

    #[test]
    fn y_scaling_round_trips_on_random_data() {
        let mut rng = ChaCha8Rng::from_seed([11; 32]);
        let y: Vec<f64> = (0..50).map(|_| uniform(&mut rng, -3.0, 40.0)).collect();
        let x = Data::new(
            (0..50).map(|_| uniform(&mut rng, 0.0, 1.0)).collect(),
            50,
            1,
        )
        .unwrap();
        let (scaler, _, y_scaled) = FittedScaler::fit(&x, &y, &[Metric::Euclidean]).unwrap();

        let (lo, hi) = super::min_max(y_scaled.iter().copied());
        assert_abs_eq(lo, -0.5, 1e-12);
        assert_abs_eq(hi, 0.5, 1e-12);
        for (raw, scaled) in y.iter().zip(&y_scaled) {
            assert_abs_eq(scaler.unscale_y_value(*scaled), *raw, 1e-12);
            assert_abs_eq(scaler.scale_y_value(*raw), *scaled, 0.0);
        }
    }

    #[test]
    fn x_scaling_maps_training_range_to_half_unit_interval() {
        let x = Data::from_rows(&[[2.0, -1.0], [4.0, 0.0], [6.0, 3.0]]).unwrap();
        let y = vec![1.0, 2.0, 3.0];
        let (scaler, xs, _) =
            FittedScaler::fit(&x, &y, &[Metric::Euclidean, Metric::Euclidean]).unwrap();
        assert_eq!(scaler.x_min(), &[2.0, -1.0]);
        assert_eq!(scaler.x_max(), &[6.0, 3.0]);
        assert_eq!(xs.row(0), &[-0.5, -0.5]);
        assert_eq!(xs.row(2), &[0.5, 0.5]);
        assert_abs_eq(xs.row(1)[0], 0.0, 1e-15); // midpoint of [2, 6]
        assert_abs_eq(xs.row(1)[1], -0.25, 1e-15); // 0 in [-1, 3]
    }

    #[test]
    fn spherical_columns_bypass_scaling() {
        let x = Data::from_rows(&[[1.0, -3.0], [2.0, 0.5], [3.0, 3.0]]).unwrap();
        let y = vec![1.0, 2.0, 3.0];
        let metrics = [Metric::Euclidean, Metric::Spherical];
        let (scaler, xs, _) = FittedScaler::fit(&x, &y, &metrics).unwrap();
        // The angle column comes through untouched.
        for r in 0..3 {
            assert_eq!(xs.row(r)[1], x.row(r)[1]);
        }
        assert_eq!(scaler.metrics(), &metrics);
    }

    #[test]
    fn prepared_columns_bypass_scaling_and_permit_constants() {
        // Column 1 is caller-prepared: identity pass-through (fit and apply),
        // observed range stored but never applied; a constant prepared column
        // is legitimate (the caller owns its preparation).
        let x = Data::from_rows(&[[1.0, 0.25], [2.0, -0.4], [3.0, 0.25]]).unwrap();
        let y = vec![1.0, 2.0, 3.0];
        let metrics = [Metric::Euclidean, Metric::Prepared];
        let (scaler, xs, _) = FittedScaler::fit(&x, &y, &metrics).unwrap();
        for r in 0..3 {
            assert_eq!(xs.row(r)[1], x.row(r)[1]);
        }
        assert_eq!(scaler.metrics(), &metrics);
        assert_eq!((scaler.x_min()[1], scaler.x_max()[1]), (-0.4, 0.25));
        assert_eq!(scaler.apply_x(&x).unwrap(), xs);

        let constant = Data::from_rows(&[[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]]).unwrap();
        let (_, xs, _) = FittedScaler::fit(&constant, &y, &metrics).unwrap();
        for r in 0..3 {
            assert_eq!(xs.row(r)[1], 7.0);
        }
    }

    // ---- encoding ----

    #[test]
    fn one_hot_encoding_expands_metrics_and_round_trips() {
        // Column 1 is categorical with levels {2, 5, 9} (deliberately unsorted in
        // the data); column 0 Euclidean, column 2 spherical.
        let x = Data::from_rows(&[
            [1.0, 5.0, 0.1],
            [2.0, 2.0, 0.2],
            [3.0, 9.0, 0.3],
            [4.0, 2.0, 0.4],
        ])
        .unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let metrics = [Metric::Euclidean, Metric::Categorical, Metric::Spherical];
        let (scaler, xs, _) = FittedScaler::fit(&x, &y, &metrics).unwrap();

        assert_eq!(scaler.n_raw_cols(), 3);
        assert_eq!(scaler.n_encoded_cols(), 5);
        assert_eq!(scaler.levels(1), Some(&[2.0, 5.0, 9.0][..]));
        assert_eq!(scaler.levels(0), None);
        assert_eq!(
            scaler.metrics(),
            &[
                Metric::Euclidean,
                Metric::Euclidean,
                Metric::Euclidean,
                Metric::Euclidean,
                Metric::Spherical
            ]
        );
        assert_eq!(scaler.col_map(), &[0, 1, 1, 1, 2]);

        // Row 0 has level 5 → one-hot (0, 1, 0) → scaled (−0.5, +0.5, −0.5).
        assert_eq!(&xs.row(0)[1..4], &[-0.5, 0.5, -0.5]);
        // Rows 1 and 3 share level 2 → identical encoded group values.
        assert_eq!(&xs.row(1)[1..4], &xs.row(3)[1..4]);

        // apply_x on the training input reproduces the fit-time encoding exactly.
        let reapplied = scaler.apply_x(&x).unwrap();
        assert_eq!(reapplied, xs);
    }

    #[test]
    fn unseen_category_errors_at_apply() {
        let x = Data::from_rows(&[[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]]).unwrap();
        let y = vec![1.0, 2.0, 3.0];
        let metrics = [Metric::Euclidean, Metric::Categorical];
        let (scaler, _, _) = FittedScaler::fit(&x, &y, &metrics).unwrap();
        let unseen = Data::from_rows(&[[0.5, 3.0]]).unwrap();
        let err = scaler.apply_x(&unseen).unwrap_err();
        assert_eq!(err, AddiVortesError::UnseenCategory { col: 1, value: 3.0 });
    }

    #[test]
    fn single_level_categorical_is_degenerate() {
        let x = Data::from_rows(&[[0.0, 7.0], [1.0, 7.0], [2.0, 7.0]]).unwrap();
        let y = vec![1.0, 2.0, 3.0];
        let err = FittedScaler::fit(&x, &y, &[Metric::Euclidean, Metric::Categorical]).unwrap_err();
        assert_eq!(err, AddiVortesError::DegenerateFeature { col: 1 });
    }

    // ---- prior calibration ----

    #[test]
    fn sigma_mu_sq_matches_closed_form() {
        // k = 3, m = 200 (paper defaults): σ_μ = 0.5 / (3√200).
        let sigma_mu = 0.5 / (3.0 * 200.0_f64.sqrt());
        assert_rel_eq(sigma_mu_sq(3.0, 200), sigma_mu * sigma_mu, 1e-15);
    }

    #[test]
    fn ols_residual_sd_matches_hand_computed_value() {
        // Fixture solved independently (numpy lstsq): β = (0, 0.8, 0.2),
        // RSS = 0.04, residual SD = √(0.04 / (6 − 3)) = 0.2/√3.
        let x = Data::from_rows(&[
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 6.0],
            [6.0, 5.0],
        ])
        .unwrap();
        let y = vec![1.1, 1.9, 3.2, 3.8, 5.3, 5.7];
        let sd = ols_residual_sd(&x, &y).expect("well-conditioned system");
        assert_rel_eq(sd, 0.2 / 3.0_f64.sqrt(), 1e-10);
    }

    #[test]
    fn ols_falls_back_on_rank_deficiency_and_small_n() {
        // Duplicated column → rank-deficient → None.
        let x =
            Data::from_rows(&[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]).unwrap();
        let y = vec![1.0, 2.5, 2.9, 4.2, 5.1];
        assert!(ols_residual_sd(&x, &y).is_none());
        // n ≤ p + 1 → None.
        let small = Data::from_rows(&[[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]).unwrap();
        assert!(ols_residual_sd(&small, &[1.0, 2.0, 3.0]).is_none());
        // …and sigma_hat then equals sd(y_scaled).
        let y_small = [1.0, 2.0, 4.0];
        assert_rel_eq(sigma_hat(&small, &y_small), sample_sd(&y_small), 0.0);
        assert_rel_eq(sample_sd(&y_small), (7.0_f64 / 3.0).sqrt(), 1e-15);
    }

    #[test]
    fn chi2_quantile_matches_reference_values() {
        // Reference values computed independently (scipy.stats.chi2.ppf).
        assert_rel_eq(chi2_quantile(0.15, 6.0), 2.661_273_176_146_905, 1e-10);
        assert_rel_eq(chi2_quantile(0.10, 3.0), 0.584_374_374_155_183_5, 1e-10);
    }

    #[test]
    fn lambda_calibration_matches_hand_computed_values() {
        // λ = σ̂² · χ²⁻¹(1−q, ν)/ν, reference values via scipy.
        assert_rel_eq(
            calibrate_lambda(6.0, 0.85, 0.2),
            1.774_182_117_431_271e-2,
            1e-10,
        );
        assert_rel_eq(
            calibrate_lambda(3.0, 0.90, 0.05),
            4.869_786_451_293_196e-4,
            1e-10,
        );
    }

    #[test]
    fn gamma_p_basic_identities() {
        // P(a, 0) = 0; P(1, x) = 1 − e^{−x} (exponential CDF).
        assert_abs_eq(gamma_p(3.0, 0.0), 0.0, 0.0);
        for &x in &[0.1, 1.0, 5.0] {
            assert_rel_eq(gamma_p(1.0, x), 1.0 - mathsfn::exp(-x), 1e-12);
        }
        // Median of χ²₂ is 2 ln 2: P(1, ln 2) = 0.5 exactly.
        assert_rel_eq(gamma_p(1.0, std::f64::consts::LN_2), 0.5, 1e-12);
    }
}
