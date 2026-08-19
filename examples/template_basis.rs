//! Template for basis-valued cells: copy this file, choose
//! your basis, fill the marked blocks, and run:
//!
//! ```sh
//! cargo run --example template_basis
//! ```
//!
//! The basis point changes what a cell outputs: a coefficient vector β ∈ ℝ^q over a
//! per-observation basis row z(x), instead of one constant, so a cell's
//! contribution to the fit is `z(xᵢ) · β_k` rather than `μ_k`. Two halves go
//! on the config: the payload (`LinearGaussianModel` + `LinearCellStats`:
//! per-cell (ZᵀWZ, ZᵀWr) blocks, conjugate q×q solves through the pinned
//! Cholesky) and the basis that says what z(x) *is* (`LinearBasis`: an
//! intercept plus the columns you name). Unlike the other extension points there is
//! usually nothing to implement: you choose q, the coefficient-prior variance
//! σ_β², and the columns.
//!
//! The two must agree: a basis payload requires a basis, a scalar payload
//! refuses one, and `basis.q()` must equal the payload's width. All three are
//! errors at `fit`. `LinearGaussianModel` at q = 1 *is* the scalar Gaussian
//! family (the intercept basis `z ≡ [1]`), so it takes no basis and reproduces
//! the default chain exactly.

use addivortes::basis::{LinearBasis, LinearGaussianModel};
use addivortes::{AddiVortesConfig, Data, conformance};

fn main() -> addivortes::Result<()> {
    // ----- your basis here ---------------------------------------------------
    // The encoded columns entering the basis, on top of the always-present
    // intercept. Here: column 0, so z(x) = [1, x₀] and q = 2.
    let basis_columns = vec![0_usize];
    let q = 1 + basis_columns.len();
    let sigma_beta_sq = 0.05; // coefficient-prior variance (scaled space)
    // --------------------------------------------------------------------------

    // 1. Check the BASIS: this is the half of the basis point you actually write. z(x)
    //    must fill every entry, be finite, be pure, and — the one with real
    //    teeth — must not read a column beyond the `max_column()` it declares.
    //
    //    The engine bounds-checks that claim against the design exactly once and
    //    then trusts it, so a basis that reads column 7 while declaring `Some(2)`
    //    passes `fit` on a 3-column design and then indexes out of range in the
    //    hot loop. The check finds out what your basis really reads, by perturbing
    //    each column and watching z(x), rather than taking your word for it.
    let basis = LinearBasis::new(basis_columns.clone());
    let design_rows: Vec<Vec<f64>> = vec![
        vec![0.10, -0.25, 0.40],
        vec![-0.30, 0.15, -0.20],
        vec![0.35, 0.05, 0.10],
    ];
    if !conformance::report(&conformance::check_cell_basis(&basis, &design_rows)) {
        std::process::exit(1);
    }

    // 2. Check the PAYLOAD model: sufficiency over `record_basis`, the
    //    marginal-vs-Monte-Carlo Bayes factor, and coefficient-draw SBC along a
    //    probe direction — the basis sibling of `check_cell_model`. Generic over
    //    any `CellModel`, so a payload family of your own is checked exactly like
    //    the shelf's.
    use rand_core::SeedableRng;
    let n = 24;
    let covariate: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
    let basis_rows: Vec<Vec<f64>> = covariate.iter().map(|&v| vec![1.0, v]).collect();
    let observations: Vec<f64> = covariate.iter().map(|&v| 0.3 * v - 0.1).collect();
    let weights = vec![1.0; n];
    let sigma_sq = 0.2;
    let model = LinearGaussianModel::new(sigma_beta_sq, q)?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let results = conformance::check_basis_cell_model(
        &model,
        &basis_rows,
        &observations,
        &weights,
        sigma_sq,
        &mut rng,
    );
    if !conformance::report(&results) {
        std::process::exit(1);
    }

    // 3. Fit-time selection: the payload and the basis, two lines on the config.
    //    The surface below is linear in x₀ with a slope that flips with x₁ —
    //    constant cells must spend cells chasing the slope; linear cells fit it
    //    with a coefficient.
    let rows = 200;
    let mut xs = Vec::with_capacity(rows * 2);
    let mut y = Vec::with_capacity(rows);
    for i in 0..rows {
        let a = i as f64 / (rows - 1) as f64;
        let b = ((i * 7) % rows) as f64 / (rows - 1) as f64;
        xs.push(a);
        xs.push(b);
        y.push(if b > 0.5 { 3.0 * a } else { -3.0 * a } + 0.5 * b);
    }
    let x = Data::new(xs, rows, 2)?;

    let common = || {
        AddiVortesConfig::new(42)
            .with_m(20)
            .with_omega(1.0)
            .with_burn_in(120)
            .with_draws(120)
    };
    let constant_cells = common().fit(&x, &y)?;
    let linear_cells = common()
        .with_cell_model(LinearGaussianModel::new(sigma_beta_sq, q)?)
        .with_cell_basis(LinearBasis::new(basis_columns))
        .fit(&x, &y)?;

    println!(
        "template_basis: all checks passed; RMSE constant cells {:.4} vs linear cells {:.4}",
        constant_cells.in_sample_rmse(),
        linear_cells.in_sample_rmse()
    );
    Ok(())
}
