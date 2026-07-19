//! Deterministic transcendental maths.
//!
//! Every transcendental in the crate routes through these thin wrappers over
//! [`libm`], never through `f64`'s inherent methods (`f64::ln`, `f64::exp`, …).
//! Rust's standard library is permitted by its own specification to produce
//! platform-dependent bits for transcendentals, which would make the bit-exact
//! golden-chain regression test (`tests/golden_chain.rs`) unsound. `libm` produces
//! identical bits on every target. The rule is deliberately mechanical and is
//! lint-enforced (`clippy.toml` `disallowed-methods`): call `mathsfn::*`,
//! never `f64::*`.

// Not every wrapper has an in-crate caller; this module is the complete
// transcendental surface, kept whole for extension authors.
#![allow(dead_code)]

/// Natural logarithm (base e).
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Natural logarithm of `1 + x`, accurate for small `x`.
pub fn ln_1p(x: f64) -> f64 {
    libm::log1p(x)
}

/// Exponential function, eˣ.
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `exp(x) - 1`, accurate for small `x`.
pub fn exp_m1(x: f64) -> f64 {
    libm::expm1(x)
}

/// Sine, `x` in radians.
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine, `x` in radians.
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Arc-cosine, returned in radians. Defined on the domain [-1, 1];
/// returns `NaN` for inputs outside that range.
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

/// `base` raised to the power `exponent`.
pub fn powf(base: f64, exponent: f64) -> f64 {
    libm::pow(base, exponent)
}

/// Complementary error function, erfc(x) = 1 − erf(x).
///
/// Used by the posterior-predictive quantile solve (the normal CDF is
/// Φ(z) = erfc(−z/√2)/2, which keeps precision in the far tails where
/// 1 − Φ(z) would underflow), never in per-move acceptance ratios.
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// Natural logarithm of the absolute value of the gamma function, ln|Γ(x)|.
/// Used only by the one-time prior calibration (incomplete-gamma series in
/// `scale`), never in per-move acceptance ratios (those telescope to
/// elementary logs).
pub fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// `base` raised to an integer power, by exponentiation-by-squaring.
///
/// Uses only `f64` multiplication (and one division for negative exponents), so the
/// result is bit-deterministic; `f64::powi`'s precision is unspecified by std.
pub fn powi(base: f64, exponent: i32) -> f64 {
    let mut base = if exponent < 0 { 1.0 / base } else { base };
    let mut exp = exponent.unsigned_abs();
    let mut acc = 1.0;
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

/// In-place lower Cholesky factorisation of a symmetric positive-definite
/// q×q matrix (row-major; only the lower triangle is read and written; the
/// strict upper triangle is left untouched). Returns `false` (without
/// modifying further) when a pivot is not strictly positive and finite
/// (the matrix is not numerically SPD).
///
/// The pinned small-matrix factorisation of the conjugate cell solves
/// (the basis point block solves; the dense membership path): plain `f64` arithmetic
/// (`*`, `/`, and the correctly-rounded `sqrt`) in a fixed ascending order,
/// so results are bit-deterministic on every target (the crate-level reproducibility contract).
pub fn cholesky(a: &mut [f64], q: usize) -> bool {
    debug_assert_eq!(a.len(), q * q);
    for i in 0..q {
        for j in 0..=i {
            let mut sum = a[i * q + j];
            for k in 0..j {
                sum -= a[i * q + k] * a[j * q + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return false;
                }
                a[i * q + i] = sum.sqrt();
            } else {
                a[i * q + j] = sum / a[j * q + j];
            }
        }
    }
    true
}

/// Solve `L Lᵀ x = b` in place given the lower Cholesky factor from
/// [`cholesky`] (forward then back substitution, fixed ascending/descending
/// order, bit-deterministic). `b` holds `x` on return (the caller's own
/// units throughout).
pub fn cholesky_solve(l: &[f64], q: usize, b: &mut [f64]) {
    debug_assert_eq!(l.len(), q * q);
    debug_assert_eq!(b.len(), q);
    for i in 0..q {
        for k in 0..i {
            b[i] -= l[i * q + k] * b[k];
        }
        b[i] /= l[i * q + i];
    }
    for i in (0..q).rev() {
        for k in (i + 1)..q {
            b[i] -= l[k * q + i] * b[k];
        }
        b[i] /= l[i * q + i];
    }
}

/// Back-substitution solve of `Lᵀ x = b` in place given the lower Cholesky
/// factor from [`cholesky`]: the half-solve that turns a standard-normal
/// vector into a draw with covariance `(L Lᵀ)⁻¹` (the caller's own units).
pub fn cholesky_solve_transposed(l: &[f64], q: usize, b: &mut [f64]) {
    debug_assert_eq!(l.len(), q * q);
    debug_assert_eq!(b.len(), q);
    for i in (0..q).rev() {
        for k in (i + 1)..q {
            b[i] -= l[k * q + i] * b[k];
        }
        b[i] /= l[i * q + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_abs_eq;

    #[test]
    fn cholesky_factors_and_solves_known_system() {
        // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, √2]]; A x = [1, 1] →
        // x = [1/8, 1/4] (inverse [[3, −2], [−2, 4]]/8).
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        assert!(cholesky(&mut a, 2));
        assert_abs_eq(a[0], 2.0, 0.0);
        assert_abs_eq(a[2], 1.0, 0.0);
        assert_abs_eq(a[3], 2.0_f64.sqrt(), 1e-15);
        let mut b = vec![1.0, 1.0];
        cholesky_solve(&a, 2, &mut b);
        assert_abs_eq(b[0], 0.125, 1e-15);
        assert_abs_eq(b[1], 0.25, 1e-15);
        // Lᵀ x = [2, √2] → x = [1/2, 1] (backward pass only).
        let mut c = vec![2.0, 2.0_f64.sqrt()];
        cholesky_solve_transposed(&a, 2, &mut c);
        assert_abs_eq(c[1], 1.0, 1e-15);
        assert_abs_eq(c[0], 0.5, 1e-15);
        // A non-SPD matrix is refused, never NaN.
        let mut bad = vec![1.0, 2.0, 2.0, 1.0];
        assert!(!cholesky(&mut bad, 2));
    }

    #[test]
    fn ln_of_known_values() {
        // ln(1) = 0
        assert_abs_eq(ln(1.0), 0.0, 1e-12);
        // ln(e) = 1
        assert_abs_eq(ln(std::f64::consts::E), 1.0, 1e-12);
    }

    #[test]
    fn exp_of_known_values() {
        // exp(0) = 1
        assert_abs_eq(exp(0.0), 1.0, 1e-12);
        // exp(1) = e
        assert_abs_eq(exp(1.0), std::f64::consts::E, 1e-12);
    }

    #[test]
    fn ln_1p_of_known_values() {
        // ln(1 + 0) = 0
        assert_abs_eq(ln_1p(0.0), 0.0, 1e-12);
        // ln(1 + (e - 1)) = 1
        assert_abs_eq(ln_1p(std::f64::consts::E - 1.0), 1.0, 1e-12);
        // near zero, ln(1 + x) ≈ x; a naive ln(1.0 + 1e-18) would collapse to 0 exactly
        assert_abs_eq(ln_1p(1e-18), 1e-18, 1e-30);
    }

    #[test]
    fn exp_m1_of_known_values() {
        // exp(0) - 1 = 0
        assert_abs_eq(exp_m1(0.0), 0.0, 1e-12);
        // exp(1) - 1 = e - 1
        assert_abs_eq(exp_m1(1.0), std::f64::consts::E - 1.0, 1e-12);
        // near zero, exp(x) - 1 ≈ x
        assert_abs_eq(exp_m1(1e-18), 1e-18, 1e-30);
    }

    #[test]
    fn sin_cos_of_known_values() {
        assert_abs_eq(sin(0.0), 0.0, 1e-12);
        assert_abs_eq(sin(std::f64::consts::FRAC_PI_2), 1.0, 1e-12);
        assert_abs_eq(cos(0.0), 1.0, 1e-12);
        assert_abs_eq(cos(std::f64::consts::PI), -1.0, 1e-12);
    }

    #[test]
    fn erfc_of_known_values() {
        // erfc(0) = 1; symmetry erfc(−x) = 2 − erfc(x).
        assert_abs_eq(erfc(0.0), 1.0, 0.0);
        for &x in &[0.3, 1.0, 2.5] {
            assert_abs_eq(erfc(-x), 2.0 - erfc(x), 1e-15);
        }
        // Reference value computed independently (scipy.special.erfc).
        assert_abs_eq(erfc(1.0), 0.157_299_207_050_285_13, 1e-15);
        // Far tail underflows cleanly to 0 (used to bracket quantile solves).
        assert_abs_eq(erfc(30.0), 0.0, 0.0);
    }

    #[test]
    fn lgamma_of_known_values() {
        // Γ(1) = 1, Γ(2) = 1 → lgamma = 0
        assert_abs_eq(lgamma(1.0), 0.0, 1e-12);
        assert_abs_eq(lgamma(2.0), 0.0, 1e-12);
        // Γ(5) = 24
        assert_abs_eq(lgamma(5.0), ln(24.0), 1e-12);
        // Γ(1/2) = √π
        assert_abs_eq(lgamma(0.5), 0.5 * ln(std::f64::consts::PI), 1e-12);
    }

    #[test]
    fn powi_of_known_values() {
        // 2^10 = 1024, exactly representable so the comparison is exact
        assert_abs_eq(powi(2.0, 10), 1024.0, 0.0);
        // x^0 = 1 for any x
        assert_abs_eq(powi(123.456, 0), 1.0, 0.0);
        // 2^-2 = 0.25 exactly
        assert_abs_eq(powi(2.0, -2), 0.25, 0.0);
        // (-3)^3 = -27 exactly
        assert_abs_eq(powi(-3.0, 3), -27.0, 0.0);
        // i32::MIN must not overflow the sign flip (unsigned_abs handles it)
        assert_abs_eq(powi(1.0, i32::MIN), 1.0, 0.0);
    }

    #[test]
    fn acos_of_known_values() {
        // acos(1) = 0
        assert_abs_eq(acos(1.0), 0.0, 1e-12);
        // acos(0) = pi/2
        assert_abs_eq(acos(0.0), std::f64::consts::FRAC_PI_2, 1e-12);
        // acos(-1) = pi
        assert_abs_eq(acos(-1.0), std::f64::consts::PI, 1e-12);
    }

    #[test]
    fn powf_of_known_values() {
        // 2^3 = 8
        assert_abs_eq(powf(2.0, 3.0), 8.0, 1e-12);
        // 9^0.5 = 3 (square root)
        assert_abs_eq(powf(9.0, 0.5), 3.0, 1e-12);
    }
}
