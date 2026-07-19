//! Template for a custom assignment geometry: copy this file,
//! rename the metric, fill the marked block, and run:
//!
//! ```sh
//! cargo run --example template_distance
//! ```
//!
//! A distance approach is one method: the pairwise comparison key between an
//! observation and a candidate centre. The blanket `CellAssigner` impl owns
//! the batch loop, tie-breaking, and the always-on finiteness check; you
//! never write those. The contract (see the `PairwiseDistance` rustdoc):
//! pure, deterministic, and a strictly monotone key (magnitude never used, so
//! squared/other increasing transforms are valid).

use addivortes::{AddiVortesConfig, Data, PairwiseDistance, Tessellation, conformance};

/// Manhattan (L1) assignment geometry: the sum of absolute coordinate
/// differences over the active dimensions.
#[derive(Debug)]
struct Manhattan;

impl PairwiseDistance for Manhattan {
    fn distance(&self, x_row: &[f64], centre_row: &[f64], active_dims: &[usize]) -> f64 {
        // ----- your metric here ---------------------------------------------
        // Both rows are full p-length rows in scaled space; the caller
        // synthesises the centre row equal to the observation outside
        // `active_dims`, so a joint metric may read the full context.
        active_dims
            .iter()
            .map(|&d| (x_row[d] - centre_row[d]).abs())
            .sum()
        // --------------------------------------------------------------------
    }
}

fn main() -> addivortes::Result<()> {
    // 1. The one-command conformance check on a small fixture: metric-level correctness
    //    (self-minimality, fast-path claim, portability digest) plus the
    //    batch-level checks (determinism, incremental-reassign consistency).
    //    The digest line is how you verify bit-portability: run this on every
    //    platform you target and compare the digests.
    let x = Data::from_rows(&[[-0.4, 0.2], [0.4, -0.1], [0.05, 0.9]])?;
    let tessellation = Tessellation::new(vec![-0.5, 0.5], vec![0], vec![0.0, 0.0])?;
    let mut results = conformance::check_distance(&Manhattan, &x, &tessellation);
    results.extend(conformance::check_assigner(&Manhattan, &x, &tessellation));
    if !conformance::report(&results) {
        std::process::exit(1);
    }

    // 2. Fit-time selection: the metric is one line on the config,
    //    identical DX to every other extension point.
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
    let x = Data::new(xs, n, 1)?;
    let model = AddiVortesConfig::new(42)
        .with_m(10)
        .with_burn_in(20)
        .with_draws(30)
        .with_distance(Manhattan)
        .fit(&x, &y)?;
    println!(
        "template_distance: all checks passed; Manhattan fit RMSE {:.4}",
        model.in_sample_rmse()
    );
    Ok(())
}
