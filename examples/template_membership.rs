//! Template for a custom membership-blending recipe: copy
//! this file, rename the kernel, fill the marked block, and run:
//!
//! ```sh
//! cargo run --example template_membership
//! ```
//!
//! Soft membership is mostly a selection: `with_membership(kernel)`
//! switches the mean ensemble onto the dense path (joint b×b payload draws,
//! membership-weighted fits, predict-time weighting), all engine code held
//! to the cross-path oracle. What you author is one small trait: turn one
//! observation's per-cell distance keys (the distance assigner's own
//! comparison keys, squared-distance scale) into unnormalised membership
//! weights. The engine owns row normalisation and the soft empty-cell guard.
//!
//! The contract (see the `MembershipKernel` rustdoc): pure, deterministic,
//! every weight finite and ≥ 0, at least one > 0 per row. If your recipe
//! exponentiates, be shift-stable: subtract the row's best key first, as
//! the shelf `SoftmaxKernel` does.
//!
//! No single formulation is endorsed, so `check_membership_kernel` proves the
//! mechanical contract and nothing more. A recipe destined for real inference
//! is validated through the Geweke/SBC battery exactly like the shelf kernel
//! (the soft-membership leg in `tests/calibration_acceptance.rs` is the worked
//! example).

use addivortes::{AddiVortesConfig, Data, MembershipKernel, conformance};

/// An inverse-quadratic (Cauchy-style) blend: weight 1/(1 + d²/τ). Heavier
/// tails than softmax (distant cells keep more say) and strictly positive
/// everywhere, so the "at least one > 0" clause holds by construction.
#[derive(Debug)]
struct InverseQuadratic {
    /// Blending temperature τ > 0 (squared-distance scale): smaller is
    /// closer to hard assignment.
    tau: f64,
}

impl MembershipKernel for InverseQuadratic {
    fn weights(&self, keys: &[f64], weights: &mut [f64]) {
        // ----- your blending recipe here ------------------------------------
        // `keys[c]` is this observation's comparison key against cell c
        // (squared-distance scale). Write one unnormalised weight per cell;
        // the engine normalises the row and enforces the soft empty-cell
        // guard. Pure and deterministic; no RNG exists on this extension point.
        for (w, &key) in weights.iter_mut().zip(keys) {
            *w = 1.0 / (1.0 + key / self.tau);
        }
        // --------------------------------------------------------------------
    }
}

fn main() -> addivortes::Result<()> {
    let kernel = InverseQuadratic { tau: 0.1 };

    // 1. The one-command check on a fixture row: finiteness, non-negativity,
    //    surviving mass, purity, and the portability digest. The distant-row
    //    check is the one to watch if your recipe exponentiates.
    let keys = [0.0, 0.04, 0.5, 3.0];
    if !conformance::report(&conformance::check_membership_kernel(&kernel, &keys)) {
        std::process::exit(1);
    }

    // 2. Fit-time selection: one line on the config switches the ensemble
    //    onto the dense path. Compare against the hard-assignment fit.
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
    let x = Data::new(xs.clone(), n, 1)?;
    let config = || {
        AddiVortesConfig::new(42)
            .with_m(10)
            .with_burn_in(20)
            .with_draws(30)
    };
    let hard = config().fit(&x, &y)?;
    let soft = config()
        .with_membership(InverseQuadratic { tau: 0.1 })
        .fit(&x, &y)?;
    println!(
        "template_membership: hard RMSE {:.4}, inverse-quadratic RMSE {:.4}",
        hard.in_sample_rmse(),
        soft.in_sample_rmse()
    );
    println!(
        "next step for a real recipe: a Geweke/SBC battery leg like the \
         soft-membership leg in tests/calibration_acceptance.rs"
    );
    Ok(())
}
