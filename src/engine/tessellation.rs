//! The Voronoi tessellation (`Tessellation`): centre coordinates, active
//! dimensions, and per-cell output values.

use crate::engine::error::{AddiVortesError, Result};

/// One tessellation T_j of the additive ensemble: `n_cells` centres in the
/// subspace spanned by `dims`, plus the per-cell payload.
///
/// The payload is `q` values per cell (`mus`, row-major: cell `c`'s
/// coefficients are `mus[c * q .. (c + 1) * q]`). The scalar families are
/// `q = 1`, where the payload is the familiar one output value μ per cell;
/// the basis families carry a coefficient vector β ∈ ℝ^q. `q` is
/// **derived**, not stored: the cell count comes from the centres, so
/// `q = mus.len() / n_cells()`. An old `q = 1` payload therefore stays valid
/// with no format change.
///
/// All coordinates live in scaled space (the sampler's coordinate
/// system); all indices are 0-based.
///
/// With the `serde` feature, deserialisation routes through
/// [`Tessellation::new`], so a hand-edited payload cannot bypass the
/// structural invariants (coordinate values stay uninspected, exactly like
/// the constructor).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "TessellationParts")
)]
pub struct Tessellation {
    /// Row-major centre coordinates: centre `c`'s coordinate along `dims[di]`
    /// is `centres[c * dims.len() + di]`.
    pub(crate) centres: Vec<f64>,
    /// Active covariate indices: 0-based global column indices of the
    /// encoded matrix.
    pub(crate) dims: Vec<usize>,
    /// Per-cell payload (scaled space), row-major over cells: cell `c` holds
    /// `mus[c * q .. (c + 1) * q]`, where `q = mus.len() / n_cells()`.
    pub(crate) mus: Vec<f64>,
}

impl Tessellation {
    /// Validating constructor (also the serde validator and the entry point for
    /// extension authors): requires at least one cell, at least one active
    /// dimension, distinct dimension indices, `centres.len()` a whole number of
    /// `dims.len()`-dimensional centres, and a payload that divides evenly over
    /// those cells (`mus.len() == n_cells * q` for some `q >= 1`).
    ///
    /// Scalar families pass one μ per cell (`q = 1`); basis families
    /// pass `q` coefficients per cell, row-major.
    ///
    /// Coordinate values are not inspected here; the sampler's release-mode
    /// invariants own numeric health.
    pub fn new(centres: Vec<f64>, dims: Vec<usize>, mus: Vec<f64>) -> Result<Self> {
        if mus.is_empty() {
            return Err(AddiVortesError::InvalidDataShape {
                reason: "tessellation must have at least one cell".into(),
            });
        }
        if dims.is_empty() {
            return Err(AddiVortesError::InvalidDataShape {
                reason: "tessellation must have at least one active dimension".into(),
            });
        }
        for (i, dim) in dims.iter().enumerate() {
            if dims[..i].contains(dim) {
                return Err(AddiVortesError::InvalidDataShape {
                    reason: format!("tessellation dims contain covariate {dim} twice"),
                });
            }
        }
        let d = dims.len();
        if centres.len() % d != 0 || centres.is_empty() {
            return Err(AddiVortesError::InvalidDataShape {
                reason: format!(
                    "{found} centre coordinates cannot form whole cells of {d} dimensions",
                    found = centres.len(),
                ),
            });
        }
        let n_cells = centres.len() / d;
        if mus.len() % n_cells != 0 {
            return Err(AddiVortesError::InvalidDataShape {
                reason: format!(
                    "{payload} payload values cannot divide evenly over {n_cells} cells",
                    payload = mus.len(),
                ),
            });
        }
        Ok(Self { centres, dims, mus })
    }

    /// Row-major centre coordinates (scaled space): centre `c` along `dims[di]`
    /// is `centres()[c * dims().len() + di]`.
    pub fn centres(&self) -> &[f64] {
        &self.centres
    }

    /// Active covariate indices (0-based, global columns of the encoded matrix).
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// The per-cell payload (scaled space), row-major: cell `c` holds
    /// `mus()[c * q() .. (c + 1) * q()]`. For the scalar families (`q == 1`)
    /// this is one output value μ per cell.
    pub fn mus(&self) -> &[f64] {
        &self.mus
    }

    /// Number of cells (= number of centres).
    pub fn n_cells(&self) -> usize {
        self.centres.len() / self.dims.len()
    }

    /// Payload width: values per cell. `1` for the scalar families; the basis
    /// dimension q for the basis families.
    pub fn q(&self) -> usize {
        self.mus.len() / self.n_cells()
    }

    /// Cell `c`'s payload block (length [`q`](Self::q), **scaled space**): its
    /// output value μ for the scalar families, its coefficient vector β for a
    /// basis payload.
    pub fn cell_payload(&self, c: usize) -> &[f64] {
        let q = self.q();
        &self.mus[c * q..(c + 1) * q]
    }

    /// Borrowed row-major view of the centres (internal hot-path helper).
    pub(crate) fn centres_view(&self) -> Centres<'_> {
        Centres {
            values: &self.centres,
            d: self.dims.len(),
        }
    }
}

/// Serde shadow of [`Tessellation`]: deserialisation lands here first, then
/// goes through the validating constructor.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct TessellationParts {
    centres: Vec<f64>,
    dims: Vec<usize>,
    mus: Vec<f64>,
}

#[cfg(feature = "serde")]
impl TryFrom<TessellationParts> for Tessellation {
    type Error = AddiVortesError;

    fn try_from(parts: TessellationParts) -> Result<Self> {
        Tessellation::new(parts.centres, parts.dims, parts.mus)
    }
}

/// Borrowed row-major view over centre coordinates (internal; the public
/// surface is `Tessellation::centres()`).
pub(crate) struct Centres<'a> {
    values: &'a [f64],
    d: usize,
}

impl Centres<'_> {
    /// Centre `c` as its `d` coordinates along the tessellation's `dims`.
    pub(crate) fn row(&self, c: usize) -> &[f64] {
        &self.values[c * self.d..(c + 1) * self.d]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_construction_round_trips() {
        let t = Tessellation::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![2, 0],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(t.n_cells(), 3);
        assert_eq!(t.dims(), &[2, 0]);
        assert_eq!(t.mus(), &[1.0, 2.0, 3.0]);
        assert_eq!(t.centres_view().row(1), &[0.3, 0.4]);
        assert_eq!(t.centres(), &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn rejects_empty_cells_and_dims() {
        let err = Tessellation::new(vec![], vec![0], vec![]).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidDataShape {
                reason: "tessellation must have at least one cell".into()
            }
        );
        let err = Tessellation::new(vec![], vec![], vec![1.0]).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidDataShape {
                reason: "tessellation must have at least one active dimension".into()
            }
        );
    }

    #[test]
    fn rejects_duplicate_dims_and_length_mismatch() {
        let err = Tessellation::new(vec![0.0; 4], vec![1, 1], vec![1.0, 2.0]).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidDataShape {
                reason: "tessellation dims contain covariate 1 twice".into()
            }
        );
        let err = Tessellation::new(vec![0.0; 5], vec![0, 1], vec![1.0, 2.0]).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidDataShape {
                reason: "5 centre coordinates cannot form whole cells of 2 dimensions".into()
            }
        );
        // Three cells (6 centres / 2 dims) cannot carry a 2-value payload.
        let err = Tessellation::new(vec![0.0; 6], vec![0, 1], vec![1.0, 2.0]).unwrap_err();
        assert_eq!(
            err,
            AddiVortesError::InvalidDataShape {
                reason: "2 payload values cannot divide evenly over 3 cells".into()
            }
        );
    }

    #[test]
    fn scalar_payload_is_q_one() {
        let t = Tessellation::new(vec![0.0; 6], vec![0, 1], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(t.n_cells(), 3);
        assert_eq!(t.q(), 1);
        assert_eq!(t.cell_payload(1), &[2.0]);
    }

    #[test]
    fn basis_payload_carries_q_coefficients_per_cell() {
        // 2 cells × 2 dims = 4 centres; q = 2 ⇒ 4 payload values.
        let t = Tessellation::new(
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0, 1],
            vec![1.0, 10.0, 2.0, 20.0],
        )
        .unwrap();
        assert_eq!(t.n_cells(), 2);
        assert_eq!(t.q(), 2);
        assert_eq!(t.cell_payload(0), &[1.0, 10.0]);
        assert_eq!(t.cell_payload(1), &[2.0, 20.0]);
    }
}
