//! **Cell basis**, *"cell outputs should be linear in covariates, not constant."*
//!
//! Two halves. The payload is a
//! [`CellModel`](crate::extensions::cell_model::CellModel) whose cells carry a
//! coefficient vector β ∈ ℝ^q instead of one constant ([`LinearGaussianModel`]:
//! per-cell (ZᵀWZ, ZᵀWr) blocks with conjugate q×q solves through the pinned
//! Cholesky). The basis is the [`CellBasis`] that says what the per-observation
//! row z(x) *is*; the engine builds it once per fit and hands it to the payload.
//! A cell's contribution to the fit is then `z(xᵢ) · β_k` rather than `μ_k`.
//!
//! Block-diagonal by construction: hard membership keeps cells independent, so
//! every move is unchanged and no dense path is needed.
//!
//! Shelf: [`LinearBasis`] (an intercept plus the declared covariates). The basis
//! is *not* restricted to the tessellation's active dimensions: it is a fixed set
//! of columns, so q is constant across tessellations and survives every
//! dimension move. Template: `examples/template_basis.rs`. Conformance
//! checks: `conformance::check_cell_basis` (the basis itself) and
//! `check_basis_cell_model` (the payload family; generic over `CellModel`,
//! so a payload of your own is checked exactly like the shelf's).
//!
//! Pairing rules, all errors at `fit`, never a panic: a basis payload
//! requires a basis; a scalar payload refuses one; `basis.q()` must equal the
//! payload's width; and a basis does not compose with soft membership (the
//! dense path owns its own joint draw). [`LinearGaussianModel`] at q = 1 *is*
//! the scalar Gaussian family (the intercept basis `z ≡ [1]`), so it takes no
//! basis and reproduces the default chain.
//!
//! Sources: classic Bayesian linear conjugacy (Lindley & Smith 1972); the
//! tree-ensemble precedents are Bayesian treed models (Chipman, George &
//! McCulloch 2002) and linear-leaf BART, MOTR-BART (Prado, Moral & Parnell
//! 2021); the `cell_basis()` seam name nods to the `RequiresBasis` design of
//! the stochtree BART library.

mod linear;

pub use linear::{LinearCellStats, LinearGaussianModel};

/// The cell-basis point: what the per-observation basis row z(x) is.
///
/// The engine evaluates this once per fit over the **scaled** design and once
/// per row at predict time, so it must be pure and deterministic. [`q`](Self::q)
/// is fixed for the life of the fit and must equal the payload model's own
/// width (validated at `fit`).
pub trait CellBasis: std::fmt::Debug + Send + Sync {
    /// The basis dimension q: values written per row (a count, `>= 1`).
    fn q(&self) -> usize;

    /// Write z(x) for one scaled observation row into `out` (length
    /// [`q`](Self::q)). Called with `out` already sized; the implementation
    /// must fill every entry.
    fn row(&self, x_row: &[f64], out: &mut [f64]);

    /// The largest **encoded** column index this basis reads, or `None` if it
    /// reads none (a pure intercept). The engine bounds-checks it against the
    /// fitted design once, so an out-of-range column is a clean error at `fit`
    /// instead of an index panic in the hot loop.
    fn max_column(&self) -> Option<usize>;
}

/// The linear basis (the shelf entry): `z(x) = [1, x[c₁], …, x[c_{q−1}]]` over
/// the declared **encoded** columns, so `q = 1 + columns.len()`. The intercept
/// is always present, which is what makes `columns = []` (q = 1) reproduce the
/// scalar families exactly.
///
/// Columns index the encoded design (post one-hot), like every other global
/// column index in the crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearBasis {
    columns: Vec<usize>,
}

impl LinearBasis {
    /// A linear basis over `columns` of the encoded design, plus an intercept.
    pub fn new(columns: Vec<usize>) -> Self {
        Self { columns }
    }

    /// The encoded columns entering the basis (the intercept is implicit).
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }
}

impl CellBasis for LinearBasis {
    fn q(&self) -> usize {
        1 + self.columns.len()
    }

    fn row(&self, x_row: &[f64], out: &mut [f64]) {
        out[0] = 1.0;
        for (slot, &column) in out[1..].iter_mut().zip(&self.columns) {
            *slot = x_row[column];
        }
    }

    fn max_column(&self) -> Option<usize> {
        self.columns.iter().copied().max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_basis_is_intercept_plus_named_columns() {
        let basis = LinearBasis::new(vec![2, 0]);
        assert_eq!(basis.q(), 3);
        let mut z = vec![0.0; 3];
        basis.row(&[0.5, -0.25, 0.125], &mut z);
        assert_eq!(z, vec![1.0, 0.125, 0.5]);
    }

    #[test]
    fn empty_columns_give_the_intercept_only_basis() {
        let basis = LinearBasis::new(vec![]);
        assert_eq!(basis.q(), 1);
        let mut z = vec![0.0; 1];
        basis.row(&[0.5, -0.25], &mut z);
        assert_eq!(z, vec![1.0]);
    }
}
