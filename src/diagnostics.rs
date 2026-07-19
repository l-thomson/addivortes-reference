//! Statistical diagnostics: the public H-AddiVortes model-evaluation metrics
//! (H paper §3.4: the e-statistic, the H-evidence plot data, and the
//! predictive-QQ PIT values) plus the two-sample Kolmogorov–Smirnov
//! machinery and SBC rank computation the calibration gate batteries drive,
//! public alongside the `calibration` drivers. Deterministic pure functions;
//! the stochastic tests live in `stat_gates` (calibration gate / release CI
//! legs).
//!
//! Scope note: [`predictive_qq`] uses a local heteroscedastic-Gaussian
//! predictive, the right form for the H entry, not a general per-variant
//! predictive surface.

use crate::engine::mathsfn;
use crate::extensions::moves::uniform_index;

// ---------------------------------------------------------------------------
// H model-evaluation metrics (H paper §3.4), public
// ---------------------------------------------------------------------------

/// The e-statistic (energy distance, H paper Eq. 11) between two univariate
/// samples `u` and `v` (dimensionless; the samples' own scale):
///
/// ```text
/// e = (2/n₁n₂) ΣΣ |uᵢ − vⱼ| − (1/n₁²) ΣΣ |uᵢ − uⱼ| − (1/n₂²) ΣΣ |vᵢ − vⱼ|
/// ```
///
/// Zero when the samples share a distribution (in expectation); larger means
/// greater discrepancy. The H paper's calibration use compares predictive
/// PIT values ([`predictive_qq`]) against uniform draws on [0, 1].
pub fn e_statistic(u: &[f64], v: &[f64]) -> f64 {
    assert!(
        !u.is_empty() && !v.is_empty(),
        "both samples must be non-empty"
    );
    let (n1, n2) = (u.len() as f64, v.len() as f64);
    let mut cross = 0.0_f64;
    for &a in u {
        for &b in v {
            cross += (a - b).abs();
        }
    }
    let mut within_u = 0.0_f64;
    for &a in u {
        for &b in u {
            within_u += (a - b).abs();
        }
    }
    let mut within_v = 0.0_f64;
    for &a in v {
        for &b in v {
            within_v += (a - b).abs();
        }
    }
    2.0 * cross / (n1 * n2) - within_u / (n1 * n1) - within_v / (n2 * n2)
}

/// One observation's H-evidence summary (H paper §3.4): the posterior point
/// estimate of its error SD `s(xᵢ)` and a central credible interval, all on
/// the scale of the draws handed to [`h_evidence`].
#[derive(Debug, Clone, PartialEq)]
pub struct HEvidencePoint {
    /// The observation's row index in the caller's design.
    pub observation: usize,
    /// Posterior mean of `s(xᵢ)` (the draws' own scale).
    pub s_hat: f64,
    /// Lower end of the central credible interval (the draws' own scale).
    pub lower: f64,
    /// Upper end of the central credible interval (the draws' own scale).
    pub upper: f64,
}

/// H-evidence plot data (H paper §3.4): per observation, the posterior mean
/// of `s(xᵢ)` and the central `level` credible interval, sorted ascending
/// by the point estimate (the paper's plotting order; heteroscedasticity
/// shows as intervals separating from a horizontal homoscedastic line).
///
/// `s_draws` is one slice per posterior draw, each of length n, of the error
/// SD `s(xᵢ)` (e.g. `HVariance::s_sq_values` square-rooted, collected per
/// kept sweep; any scale, the output inherits it). `level` is the central
/// interval mass, in (0, 1).
pub fn h_evidence(s_draws: &[Vec<f64>], level: f64) -> Vec<HEvidencePoint> {
    assert!(!s_draws.is_empty(), "at least one posterior draw");
    assert!(level > 0.0 && level < 1.0, "level must lie in (0, 1)");
    let n = s_draws[0].len();
    assert!(
        s_draws.iter().all(|draw| draw.len() == n),
        "every draw covers the same observations"
    );
    let tail = 0.5 * (1.0 - level);
    let mut points = Vec::with_capacity(n);
    let mut buffer = Vec::with_capacity(s_draws.len());
    for observation in 0..n {
        buffer.clear();
        buffer.extend(s_draws.iter().map(|draw| draw[observation]));
        let s_hat = buffer.iter().sum::<f64>() / buffer.len() as f64;
        buffer.sort_by(f64::total_cmp);
        points.push(HEvidencePoint {
            observation,
            s_hat,
            lower: quantile_sorted(&buffer, tail),
            upper: quantile_sorted(&buffer, 1.0 - tail),
        });
    }
    points.sort_by(|a, b| a.s_hat.total_cmp(&b.s_hat));
    points
}

/// Predictive-QQ PIT values (H paper §3.4, which calls them "percentiles"
/// though they lie in [0, 1]) under the heteroscedastic
/// Gaussian predictive `Yᵢ | draw d ~ N(f_d(xᵢ), s_d(xᵢ)²)`: each
/// observation's probability-integral-transform value
///
/// ```text
/// PITᵢ = (1/D) Σ_d Φ((yᵢ − f_d(xᵢ)) / s_d(xᵢ)),
/// ```
///
/// returned sorted ascending (probability scale); plot against the uniform
/// quantiles `(i − ½)/n`, or feed to [`e_statistic`] against uniform draws.
/// A well-calibrated model gives an approximately straight line.
///
/// `y`, `fit_draws` and `s_draws` share one scale (caller's choice);
/// `fit_draws`/`s_draws` are one slice per posterior draw, each of length n
/// (`s` is the error SD, strictly positive).
pub fn predictive_qq(y: &[f64], fit_draws: &[Vec<f64>], s_draws: &[Vec<f64>]) -> Vec<f64> {
    let n = y.len();
    assert!(
        !fit_draws.is_empty() && fit_draws.len() == s_draws.len(),
        "one fit slice and one s slice per posterior draw"
    );
    assert!(
        fit_draws
            .iter()
            .zip(s_draws)
            .all(|(fit, s)| fit.len() == n && s.len() == n),
        "every draw covers the same observations as y"
    );
    let draws = fit_draws.len() as f64;
    let mut pit: Vec<f64> = (0..n)
        .map(|i| {
            let total: f64 = fit_draws
                .iter()
                .zip(s_draws)
                .map(|(fit, s)| {
                    debug_assert!(s[i] > 0.0);
                    normal_cdf((y[i] - fit[i]) / s[i])
                })
                .sum();
            total / draws
        })
        .collect();
    pit.sort_by(f64::total_cmp);
    pit
}

/// Standard normal CDF via the complementary error function (the pinned
/// far-tail-stable path): Φ(z) = erfc(−z/√2)/2. Probability scale.
fn normal_cdf(z: f64) -> f64 {
    0.5 * mathsfn::erfc(-z / std::f64::consts::SQRT_2)
}

/// Linear-interpolation quantile of an ascending-sorted slice ("type 7", the
/// same convention as the fitted model's intervals). The draws' own scale.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let fraction = h - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * fraction
}

// ---------------------------------------------------------------------------
// Convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter, Bürkner 2021)
// ---------------------------------------------------------------------------

/// Rank-normalised split-R̂ (Vehtari et al. 2021, the paper's final
/// recommendation): the maximum of the split-R̂ of the rank-normalised draws
/// (location) and of the rank-normalised folded draws `|θ − median|`
/// (scale). Dimensionless; ≈ 1 at convergence (the paper's threshold is
/// R̂ < 1.01). Needs ≥ 2 chains of equal length ≥ 4 (asserted); pinned by a
/// fixture cross-check against an independently written reference
/// implementation of the published formulas.
pub fn r_hat(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    let bulk = basic_r_hat(&rank_normalise(&split));
    let folded = basic_r_hat(&rank_normalise(&fold(&split)));
    bulk.max(folded)
}

/// Bulk effective sample size (Vehtari et al. 2021): the combined-chain
/// autocorrelation ESS of the rank-normalised split chains: a count of
/// effectively independent draws for centre-of-distribution summaries.
/// Same input contract and fixture pinning as [`r_hat`].
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    basic_ess(&rank_normalise(&split_chains(chains)))
}

/// Tail effective sample size (Vehtari et al. 2021): the minimum ESS of the
/// 5% and 95% quantile indicators over the split chains: a count of
/// effectively independent draws for tail summaries (interval ends). Same
/// input contract and fixture pinning as [`r_hat`].
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    let split = split_chains(chains);
    let mut pooled: Vec<f64> = split.iter().flatten().copied().collect();
    pooled.sort_by(f64::total_cmp);
    let q05 = quantile_sorted(&pooled, 0.05);
    let q95 = quantile_sorted(&pooled, 0.95);
    let indicator = |threshold: f64| -> Vec<Vec<f64>> {
        split
            .iter()
            .map(|chain| chain.iter().map(|&v| f64::from(v <= threshold)).collect())
            .collect()
    };
    basic_ess(&indicator(q05)).min(basic_ess(&indicator(q95)))
}

/// Halve every chain (the "split" of split-R̂: intra-chain drift shows up as
/// between-half disagreement). Odd lengths drop the last draw.
fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    assert!(
        chains.len() >= 2 && chains.iter().all(|c| c.len() == chains[0].len()),
        "need ≥ 2 chains of equal length"
    );
    assert!(chains[0].len() >= 4, "need ≥ 4 draws per chain");
    let half = chains[0].len() / 2;
    let mut out = Vec::with_capacity(chains.len() * 2);
    for chain in chains {
        out.push(chain[..half].to_vec());
        out.push(chain[half..2 * half].to_vec());
    }
    out
}

/// Pooled fractional ranks (average rank for ties) mapped through the
/// normal quantile: z = Φ⁻¹((r − 3/8)/(S + 1/4)) (Blom's offsets, per the
/// paper).
fn rank_normalise(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut pooled: Vec<(f64, usize, usize)> = chains
        .iter()
        .enumerate()
        .flat_map(|(ci, chain)| chain.iter().enumerate().map(move |(i, &v)| (v, ci, i)))
        .collect();
    pooled.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total = pooled.len();
    let mut out: Vec<Vec<f64>> = chains.iter().map(|c| vec![0.0; c.len()]).collect();
    let mut j = 0;
    while j < total {
        let mut k = j;
        while k + 1 < total && pooled[k + 1].0 == pooled[j].0 {
            k += 1;
        }
        let average_rank = (j + k) as f64 / 2.0 + 1.0;
        let z = normal_quantile((average_rank - 0.375) / (total as f64 + 0.25));
        for &(_, ci, i) in &pooled[j..=k] {
            out[ci][i] = z;
        }
        j = k + 1;
    }
    out
}

/// Fold around the pooled median: `|θ − median|` (the scale-sensitive half
/// of the rank-normalised R̂).
fn fold(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    pooled.sort_by(f64::total_cmp);
    let total = pooled.len();
    let median = if total % 2 == 1 {
        pooled[total / 2]
    } else {
        0.5 * (pooled[total / 2 - 1] + pooled[total / 2])
    };
    chains
        .iter()
        .map(|chain| chain.iter().map(|&v| (v - median).abs()).collect())
        .collect()
}

/// The classical split-R̂ core: √(var̂⁺/W) with W the mean within-chain
/// variance and var̂⁺ = (n−1)/n·W + B/n.
fn basic_r_hat(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len() as f64;
    let n = chains[0].len() as f64;
    let means: Vec<f64> = chains
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / n)
        .collect();
    let grand = means.iter().sum::<f64>() / m;
    let b = n / (m - 1.0)
        * means
            .iter()
            .map(|mu| (mu - grand) * (mu - grand))
            .sum::<f64>();
    let w = chains
        .iter()
        .zip(&means)
        .map(|(chain, mu)| chain.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1.0))
        .sum::<f64>()
        / m;
    let var_plus = (n - 1.0) / n * w + b / n;
    (var_plus / w).sqrt()
}

/// Combined-chain autocorrelation ESS with Geyer's initial positive monotone
/// sequence over lag pairs (Stan's formulation): τ̂ = −1 + 2·Σ P̂_k with
/// P̂_k = ρ̂_{2k} + ρ̂_{2k+1}, truncated at the first non-positive pair and
/// forced non-increasing; ESS = mn/τ̂.
fn basic_ess(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len() as f64;
    let n_draws = chains[0].len();
    let n = n_draws as f64;
    let means: Vec<f64> = chains
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / n)
        .collect();
    let variances: Vec<f64> = chains
        .iter()
        .zip(&means)
        .map(|(chain, mu)| chain.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1.0))
        .collect();
    let w = variances.iter().sum::<f64>() / m;
    let grand = means.iter().sum::<f64>() / m;
    let b = n / (m - 1.0)
        * means
            .iter()
            .map(|mu| (mu - grand) * (mu - grand))
            .sum::<f64>();
    let var_plus = (n - 1.0) / n * w + b / n;
    // NaN-catching guard: only a strictly positive pooled variance divides.
    if var_plus.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return f64::NAN;
    }
    let autocovariance = |chain: &[f64], mu: f64, t: usize| -> f64 {
        (0..n_draws - t)
            .map(|i| (chain[i] - mu) * (chain[i + t] - mu))
            .sum::<f64>()
            / n
    };
    let rho = |t: usize| -> f64 {
        let mean_acov = chains
            .iter()
            .zip(&means)
            .map(|(chain, &mu)| autocovariance(chain, mu, t))
            .sum::<f64>()
            / m;
        1.0 - (w - mean_acov) / var_plus
    };
    let mut tau_half = 0.0_f64;
    let mut previous = f64::INFINITY;
    let mut k = 0usize;
    while 2 * k + 1 < n_draws {
        let pair = rho(2 * k) + rho(2 * k + 1);
        if pair <= 0.0 {
            break;
        }
        let pair = pair.min(previous);
        previous = pair;
        tau_half += pair;
        k += 1;
    }
    let tau = (2.0 * tau_half - 1.0).max(1e-300);
    m * n / tau
}

/// Standard normal quantile Φ⁻¹(p) by deterministic bisection on the pinned
/// erfc-based CDF (the same approach as the χ² quantile: no Newton
/// fragility, ~90 halvings to machine precision). `p` in (0, 1); the result
/// is a z-score (standard-normal units).
fn normal_quantile(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    let (mut lo, mut hi) = (-40.0_f64, 40.0_f64);
    for _ in 0..90 {
        let mid = 0.5 * (lo + hi);
        if normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// Gate-battery machinery (crate-internal)
// ---------------------------------------------------------------------------

/// Two-sample Kolmogorov–Smirnov statistic `D = sup |F_a − F_b|`, computed
/// exactly over the pooled sample (tie-aware: both ECDFs are evaluated at every
/// pooled value, so discrete statistics are handled correctly; the classical
/// critical values are then conservative, which errs toward fewer false
/// reds).
///
/// Inputs need not be sorted; non-finite values are a caller bug
/// (`debug_assert`ed).
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> f64 {
    debug_assert!(!a.is_empty() && !b.is_empty());
    debug_assert!(a.iter().chain(b).all(|v| v.is_finite()));
    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_by(f64::total_cmp);
    b_sorted.sort_by(f64::total_cmp);

    let (n_a, n_b) = (a_sorted.len() as f64, b_sorted.len() as f64);
    let (mut i, mut j) = (0usize, 0usize);
    let mut d_max = 0.0_f64;
    while i < a_sorted.len() && j < b_sorted.len() {
        let value = a_sorted[i].min(b_sorted[j]);
        // Step both ECDFs past every observation equal to `value` (ties).
        while i < a_sorted.len() && a_sorted[i] == value {
            i += 1;
        }
        while j < b_sorted.len() && b_sorted[j] == value {
            j += 1;
        }
        let d = (i as f64 / n_a - j as f64 / n_b).abs();
        d_max = d_max.max(d);
    }
    d_max
}

/// Asymptotic two-sample KS critical value at significance `alpha` (a
/// probability); the result is dimensionless, like the KS statistic it
/// bounds:
/// `c(α)·√((n_a + n_b)/(n_a·n_b))` with `c(α) = √(−ln(α/2)/2)`
/// (Smirnov). Valid for continuous statistics; conservative for discrete ones
/// (see [`ks_two_sample`]).
pub fn ks_critical_value(alpha: f64, n_a: usize, n_b: usize) -> f64 {
    debug_assert!(alpha > 0.0 && alpha < 1.0);
    let c = (-crate::engine::mathsfn::ln(alpha / 2.0) / 2.0).sqrt();
    c * ((n_a + n_b) as f64 / (n_a as f64 * n_b as f64)).sqrt()
}

/// SBC rank of `true_value` among `draws` (Talts et al. 2018): the number of
/// posterior draws strictly below the prior-drawn true value, with ties broken
/// uniformly (required for discrete quantities, such as cell counts and
/// dimension counts, whose ranks are otherwise non-uniform under the null).
/// The result
/// lies in `0..=draws.len()`.
pub fn sbc_rank(true_value: f64, draws: &[f64], rng: &mut dyn rand_core::Rng) -> usize {
    let below = draws.iter().filter(|d| **d < true_value).count();
    let equal = draws.iter().filter(|d| **d == true_value).count();
    if equal == 0 {
        below
    } else {
        below + uniform_index(equal + 1, rng)
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::test_support::assert_abs_eq;

    // CI leg: fast-PR (deterministic).

    #[test]
    fn ks_statistic_matches_hand_values() {
        // Disjoint samples: D = 1.
        assert_abs_eq(ks_two_sample(&[1.0, 2.0], &[3.0, 4.0]), 1.0, 0.0);
        // Identical samples: D = 0.
        assert_abs_eq(ks_two_sample(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0, 0.0);
        // Hand case: a = {1, 3}, b = {2, 3, 4}. After value 1: F_a=1/2, F_b=0
        // → D=1/2; after 2: 1/2 vs 1/3 → 1/6; after 3 (tie): 1 vs 2/3 → 1/3.
        assert_abs_eq(ks_two_sample(&[1.0, 3.0], &[2.0, 3.0, 4.0]), 0.5, 1e-15);
    }

    #[test]
    fn ks_critical_value_matches_reference() {
        // c(0.05) = 1.3581, c(0.01) = 1.6276 (Smirnov table, 4 dp).
        let c05 = ks_critical_value(0.05, 1, 1) / (2.0_f64).sqrt();
        let c01 = ks_critical_value(0.01, 1, 1) / (2.0_f64).sqrt();
        assert_abs_eq(c05, 1.3581, 1e-4);
        assert_abs_eq(c01, 1.6276, 1e-4);
    }

    // ---- H model-evaluation metrics (H paper §3.4) ----

    #[test]
    fn e_statistic_matches_hand_values() {
        // U = {0, 1}, V = {0.5}: cross = 2/2·(0.5 + 0.5) = 1; within-U =
        // (0 + 1 + 1 + 0)/4 = 0.5; within-V = 0 → e = 0.5.
        assert_abs_eq(e_statistic(&[0.0, 1.0], &[0.5]), 0.5, 1e-15);
        // Identical samples: e = 0 exactly.
        assert_abs_eq(e_statistic(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0, 1e-15);
        // Separated samples: U = {0, 1}, V = {10, 11}: cross = 2·40/4 = 20,
        // within terms 0.5 each → e = 19.
        assert_abs_eq(e_statistic(&[0.0, 1.0], &[10.0, 11.0]), 19.0, 1e-12);
    }

    #[test]
    fn h_evidence_matches_hand_quantiles() {
        // 3 draws, 2 observations. Obs 0 draws {1, 2, 3}: mean 2, 90%
        // interval [1.1, 2.9] (type-7 at p = 0.05/0.95 over 3 points);
        // obs 1 draws {5, 4, 6}: mean 5, interval [4.1, 5.9]. Sorted by ŝ.
        let draws = vec![vec![1.0, 5.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let points = h_evidence(&draws, 0.9);
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].observation, 0);
        assert_abs_eq(points[0].s_hat, 2.0, 1e-15);
        assert_abs_eq(points[0].lower, 1.1, 1e-12);
        assert_abs_eq(points[0].upper, 2.9, 1e-12);
        assert_eq!(points[1].observation, 1);
        assert_abs_eq(points[1].s_hat, 5.0, 1e-15);
        assert_abs_eq(points[1].lower, 4.1, 1e-12);
        assert_abs_eq(points[1].upper, 5.9, 1e-12);
    }

    #[test]
    fn predictive_qq_matches_normal_cdf_reference() {
        // One draw, f = 0, s = 1: PIT_i = Φ(y_i). Reference values:
        // Φ(0) = 0.5, Φ(1.959964) ≈ 0.975, Φ(−1.644854) ≈ 0.05.
        let y = [0.0, 1.959_963_984_540_054, -1.644_853_626_951_472_2];
        let pit = predictive_qq(&y, &[vec![0.0; 3]], &[vec![1.0; 3]]);
        assert_abs_eq(pit[0], 0.05, 1e-9);
        assert_abs_eq(pit[1], 0.5, 1e-12);
        assert_abs_eq(pit[2], 0.975, 1e-9);
    }

    /// The calibration reading the H paper builds on the metrics: data drawn
    /// from the predictive itself gives near-uniform PIT values (small
    /// e-statistic against the uniform grid); a model claiming half the true
    /// SD gives a clearly larger one.
    #[test]
    fn predictive_qq_and_e_statistic_grade_calibration() {
        let mut rng = ChaCha8Rng::from_seed([3; 32]);
        let n = 400;
        let normal = |rng: &mut ChaCha8Rng| -> f64 {
            rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng)
        };
        let fit: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) - 0.5).collect();
        let s: Vec<f64> = (0..n).map(|i| 0.2 + i as f64 / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| fit[i] + s[i] * normal(&mut rng)).collect();
        let uniform_grid: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();

        let calibrated = predictive_qq(&y, std::slice::from_ref(&fit), std::slice::from_ref(&s));
        let overconfident_s: Vec<f64> = s.iter().map(|v| 0.5 * v).collect();
        let overconfident = predictive_qq(&y, std::slice::from_ref(&fit), &[overconfident_s]);

        let e_good = e_statistic(&calibrated, &uniform_grid);
        let e_bad = e_statistic(&overconfident, &uniform_grid);
        assert!(
            e_good < 0.02,
            "calibrated PIT should be near-uniform, e = {e_good}"
        );
        assert!(
            e_bad > 5.0 * e_good,
            "halved-SD PIT should be clearly worse: e_good = {e_good}, e_bad = {e_bad}"
        );
    }

    #[test]
    fn sbc_rank_counts_and_tie_breaks() {
        let mut rng = ChaCha8Rng::from_seed([1; 32]);
        let draws = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(sbc_rank(0.05, &draws, &mut rng), 0);
        assert_eq!(sbc_rank(0.25, &draws, &mut rng), 2);
        assert_eq!(sbc_rank(0.9, &draws, &mut rng), 4);
        // Ties: rank must fall in [below, below + equal]; over many tie-broken
        // draws every admissible value appears.
        let tied = [1.0, 1.0, 1.0, 2.0];
        let mut seen = [false; 4];
        for _ in 0..200 {
            let r = sbc_rank(1.0, &tied, &mut rng);
            assert!(r <= 3);
            seen[r] = true;
        }
        assert_eq!(seen, [true, true, true, true]);
    }
}

#[cfg(test)]
mod convergence_tests {
    use super::*;
    use crate::test_support::{assert_abs_eq, assert_rel_eq};

    /// The reference fixture: 4 chains × 100 AR(0.3) draws from a pinned
    /// LCG with Box–Muller, chain 3 optionally mean-shifted by 0.8,
    /// identical to the independently written Python reference
    /// implementation of the Vehtari et al. (2021) formulas whose outputs
    /// pin the assertions.
    fn fixture(shift_last: bool) -> Vec<Vec<f64>> {
        let mut chains = Vec::with_capacity(4);
        for c in 0..4u64 {
            let mut state = c + 1;
            let mut uniform = move || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let mut previous = 0.0_f64;
            let mut draws = Vec::with_capacity(100);
            for _ in 0..100 {
                let (u1, u2) = (uniform(), uniform());
                let z = (-2.0 * crate::engine::mathsfn::ln(1.0 - u1)).sqrt()
                    * crate::engine::mathsfn::cos(2.0 * std::f64::consts::PI * u2);
                previous = 0.3 * previous + z;
                draws.push(previous + if shift_last && c == 3 { 0.8 } else { 0.0 });
            }
            chains.push(draws);
        }
        chains
    }

    /// Fixture cross-check: expected values computed independently from
    /// the published formulas.
    #[test]
    fn r_hat_and_ess_match_the_reference_fixture() {
        let shifted = fixture(true);
        assert_rel_eq(r_hat(&shifted), 1.095_436_538_018_134, 1e-9);
        assert_rel_eq(ess_bulk(&shifted), 29.872_097_259_990_166, 1e-6);
        assert_rel_eq(ess_tail(&shifted), 180.146_429_183_079_16, 1e-6);
        let clean = fixture(false);
        assert_rel_eq(r_hat(&clean), 1.013_537_721_785_757_7, 1e-9);
        assert_rel_eq(ess_bulk(&clean), 227.632_800_054_154_76, 1e-6);
        assert_rel_eq(ess_tail(&clean), 346.715_518_407_877_37, 1e-6);
    }

    /// Behavioural sanity: a mean shift inflates R̂ and collapses bulk-ESS
    /// relative to the same chains without it.
    #[test]
    fn shift_inflates_r_hat_and_collapses_ess() {
        let (shifted, clean) = (fixture(true), fixture(false));
        assert!(r_hat(&shifted) > 1.05 && r_hat(&clean) < 1.02);
        assert!(ess_bulk(&shifted) < 0.2 * ess_bulk(&clean));
    }

    #[test]
    fn normal_quantile_inverts_the_cdf() {
        assert_abs_eq(normal_quantile(0.5), 0.0, 1e-12);
        assert_abs_eq(normal_quantile(0.975), 1.959_963_984_540_054, 1e-9);
        assert_abs_eq(normal_quantile(0.05), -1.644_853_626_951_472_2, 1e-9);
    }
}
