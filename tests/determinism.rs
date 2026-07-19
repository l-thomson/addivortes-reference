//! Compiled-configuration determinism assertions (the reproducibility contract).
//!
//! This is an integration test on purpose: integration tests are built with the
//! crate's dev-dependencies present, so cargo's feature unification is at its most
//! permissive here. If any crate in the unified build enabled `num-traits/std`
//! (e.g. via `rand_distr/std_math`), `std`'s platform-dependent transcendentals
//! would silently shadow `libm` inside `num-traits`: a property `cargo tree`
//! cannot prove for the artifact that actually runs. So we assert it behaviourally:
//! a `rand_distr` draw must equal, bit for bit, the same draw recomputed through
//! the explicit `libm` path.

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use rand_distr::{Distribution, LogNormal, Normal};

/// `LogNormal(μ, σ).sample(rng)` is documented and implemented as
/// `exp(Normal(μ, σ).sample(rng))`; with `num-traits` on its libm path, that `exp`
/// is `libm::exp`. Feed two clones of one RNG through the two routes and demand
/// bit equality across many draws: if `num-traits/std` were enabled anywhere in
/// the unified build, the platform `exp` would disagree with `libm::exp` in the
/// low bits on some draw with overwhelming probability.
#[test]
fn rand_distr_draws_route_through_libm() {
    let seed = [7u8; 32];
    let mut via_lognormal = ChaCha8Rng::from_seed(seed);
    let mut via_normal = ChaCha8Rng::from_seed(seed);

    let lognormal = LogNormal::new(0.0_f64, 1.0).expect("valid parameters");
    let normal = Normal::new(0.0_f64, 1.0).expect("valid parameters");

    for draw in 0..4096 {
        let sampled: f64 = lognormal.sample(&mut via_lognormal);
        let z: f64 = normal.sample(&mut via_normal);
        let recomputed = libm::exp(z);
        assert_eq!(
            sampled.to_bits(),
            recomputed.to_bits(),
            "draw {draw}: LogNormal sample {sampled:e} != libm::exp(z) {recomputed:e}: \
             is num-traits/std enabled somewhere in the build graph?"
        );
    }
}

/// Same-process self-consistency: two identically-seeded runs of the same
/// stochastic computation must agree bit for bit within one execution
/// (catches hash-order or uninitialised-state non-determinism).
#[test]
fn identically_seeded_runs_are_bit_identical() {
    let run = || -> Vec<u64> {
        let mut rng = ChaCha8Rng::from_seed([3u8; 32]);
        let normal = Normal::new(1.5_f64, 2.5).expect("valid parameters");
        (0..256)
            .map(|_| {
                let x: f64 = normal.sample(&mut rng);
                x.to_bits()
            })
            .collect()
    };
    assert_eq!(run(), run());
}
