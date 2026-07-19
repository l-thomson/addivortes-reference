//! The `addivortes._native` extension module: a **config pass-through** over the
//! `addivortes` public API. Design rules:
//!
//! - This module enumerates nothing. It does not name a single hyperparameter,
//!   distance, response family or shelf item. It hands the core a `ConfigSpec`
//!   payload and asks it to map it, so a new shelf entry in the crate reaches
//!   Python on rebuild with no edit here. Anything that lists the shelf twice
//!   eventually lists it differently.
//! - Defaults live in Rust. An unset field in the payload is simply not applied,
//!   so the crate's default (and any future retune of it) has exactly one home.
//! - The ergonomic keyword layer is pure Python (`addivortes/__init__.py`). It
//!   is a convenience over the payload, not a second definition of the shelf,
//!   and it is the right place for it: keywords, docstrings and `dict`s are
//!   what Python users read.
//! - The shelf crosses the boundary; the traits do not. Authoring a *new*
//!   component stays a Rust-side activity: a trait impl is code,
//!   and no data payload can carry one.
//! - Bit-exact reproducibility survives the FFI. `f64`s cross unchanged (JSON
//!   with `float_roundtrip`); the pytest suite pins a fit against the same
//!   per-target golden vector the Rust tests use.
//! - Every crate error surfaces as `addivortes.AddiVortesError` (a
//!   `ValueError` subclass) carrying the crate's own message.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use addivortes::config_spec::{response_family_name, ConfigSpec};
use addivortes::{Data, FittedAddiVortes, ResponseFamily};

create_exception!(
    addivortes,
    AddiVortesError,
    PyValueError,
    "An error from the addivortes engine (invalid data, hyperparameters, or state)."
);

fn err(e: impl std::fmt::Display) -> PyErr {
    AddiVortesError::new_err(e.to_string())
}

/// Row-major copy of a 2-D NumPy array into the crate's `Data` (logical
/// order, so Fortran-ordered inputs convert correctly too).
fn to_data(x: &PyReadonlyArray2<'_, f64>) -> PyResult<Data> {
    let arr = x.as_array();
    let (n, p) = (arr.nrows(), arr.ncols());
    let values: Vec<f64> = arr.iter().copied().collect();
    Data::new(values, n, p).map_err(err)
}

/// A (lower, upper) pair of 1-D arrays: the interval return shape.
type IntervalArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn to_vec(y: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    y.as_array().iter().copied().collect()
}

/// The model specification: a `ConfigSpec` handed over as JSON.
///
/// This class enumerates nothing. It carries the payload the caller built and
/// asks the core to map it, so a new shelf entry in the crate is reachable from
/// Python on rebuild with no edit here. The ergonomic keyword layer lives in
/// pure Python (`addivortes/__init__.py`), where it belongs: it is a
/// convenience over the payload, not a second definition of the shelf.
#[pyclass(frozen, module = "addivortes")]
pub struct AddiVortes {
    spec: ConfigSpec,
    spec_json: String,
}

#[pymethods]
impl AddiVortes {
    /// Build from a `ConfigSpec` JSON payload.
    ///
    /// Validated here, at construction, so a bad value is reported where the
    /// user wrote it rather than at `fit`. The config itself cannot be built
    /// yet: the covariate-sized settings are sized by the data (see
    /// `ConfigSpec::into_config`), so that happens in `fit`.
    #[new]
    fn new(spec_json: &str) -> PyResult<Self> {
        let spec: ConfigSpec = serde_json::from_str(spec_json).map_err(err)?;
        spec.validate().map_err(err)?;
        Ok(Self {
            spec,
            spec_json: spec_json.to_string(),
        })
    }

    /// The spec payload, verbatim (round-trips through pickle).
    #[getter]
    fn spec_json(&self) -> &str {
        &self.spec_json
    }

    /// The selected response family name ('gaussian' | 'binary_probit' |
    /// 'robust_t').
    #[getter]
    fn response_family(&self) -> &str {
        self.spec.response_family.as_deref().unwrap_or("gaussian")
    }

    /// Fit on `x` (n×p, raw caller scale) and `y` (length n). Releases the
    /// GIL for the duration of the MCMC run.
    fn fit(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<FittedModel> {
        let data = to_data(&x)?;
        let response = to_vec(&y);
        // The data is what the covariate-sized settings are sized by, so the config
        // is assembled here, not at construction.
        let config = self.spec.clone().into_config(&data).map_err(err)?;
        let family = self.response_family().to_string();
        let fitted = py
            .detach(move || config.fit(&data, &response))
            .map_err(err)?;
        Ok(FittedModel {
            inner: fitted,
            response_family: family,
        })
    }

    /// Fit `n_chains` independent chains (seeds derived from the configured
    /// seed; chain 0 is bit-identical to a plain `fit`). Releases the GIL.
    fn fit_chains(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        n_chains: usize,
    ) -> PyResult<Vec<FittedModel>> {
        let data = to_data(&x)?;
        let response = to_vec(&y);
        let config = self.spec.clone().into_config(&data).map_err(err)?;
        let family = self.response_family().to_string();
        let fits = py
            .detach(move || config.fit_chains(&data, &response, n_chains))
            .map_err(err)?;
        Ok(fits
            .into_iter()
            .map(|inner| FittedModel {
                inner,
                response_family: family.clone(),
            })
            .collect())
    }
}

/// Validate a `ConfigSpec` JSON payload without any data, raising the crate's
/// own error message.
///
/// This is what lets the Python layer fail at the point of the mistake — a bad
/// `Distance.minkowski(0.5)` raises when it is written, not several lines later
/// at `fit`. The checks that need the covariate count (the *lengths* of the
/// per-covariate settings) necessarily wait for `fit`.
#[pyfunction]
fn validate_spec(spec_json: &str) -> PyResult<()> {
    let spec: ConfigSpec = serde_json::from_str(spec_json).map_err(err)?;
    spec.validate().map_err(err)
}

/// A fitted model: the posterior draws plus the fitted scaler, wrapping the
/// crate's `FittedAddiVortes` one-to-one.
#[pyclass(frozen, module = "addivortes")]
pub struct FittedModel {
    inner: FittedAddiVortes,
    response_family: String,
}

#[pymethods]
impl FittedModel {
    /// Posterior-mean predictions for `x`, one per row: response scale for
    /// gaussian/robust_t, probability scale for binary_probit. Deterministic
    /// (predict consumes no RNG).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data = to_data(&x)?;
        let predictions = self.inner.predict(&data).map_err(err)?;
        Ok(predictions.into_pyarray(py))
    }

    /// Per-draw predictions for `x`, shape (n_draws, n_rows): the full,
    /// unreduced posterior of the fit on the family's own scale (response
    /// scale for gaussian/robust_t, probability scale for binary_probit).
    /// `predict` is this matrix's column mean; posterior-predictive
    /// workflows (PPC, LOO/WAIC, per-draw partial dependence) consume the
    /// draw axis directly.
    fn predict_draws<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data = to_data(&x)?;
        let n_rows = data.n_rows();
        let draws = self.inner.predict_draws(&data).map_err(err)?;
        let n_draws = draws.len();
        let flat: Vec<f64> = draws.into_iter().flatten().collect();
        let array = Array2::from_shape_vec((n_draws, n_rows), flat)
            .expect("predict_draws is draw-major n_draws x n_rows");
        Ok(array.into_pyarray(py))
    }

    /// Pointwise log-likelihood ln p(y_i | draw d) against observed `y`,
    /// shape (n_draws, n_rows): the matrix PSIS-LOO/WAIC estimators
    /// consume. Per draw: N(fit, sigma^2) for gaussian, location-scale
    /// Student-t for robust_t, Bernoulli (probabilities clamped away from
    /// {0, 1}) for binary_probit.
    fn log_likelihood<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data = to_data(&x)?;
        let n_rows = data.n_rows();
        let response = to_vec(&y);
        let ll = self.inner.log_likelihood(&data, &response).map_err(err)?;
        let n_draws = ll.len();
        let flat: Vec<f64> = ll.into_iter().flatten().collect();
        let array = Array2::from_shape_vec((n_draws, n_rows), flat)
            .expect("log_likelihood is draw-major n_draws x n_rows");
        Ok(array.into_pyarray(py))
    }

    /// Posterior-predictive quantiles at `probs` for each row of `x`:
    /// shape (n_rows, len(probs)).
    fn predict_quantiles<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        probs: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data = to_data(&x)?;
        let quantiles = self.inner.predict_quantiles(&data, &probs).map_err(err)?;
        let n_rows = quantiles.n_rows();
        let n_probs = quantiles.probs().len();
        let (values, _, _) = quantiles.into_parts();
        let array = Array2::from_shape_vec((n_rows, n_probs), values)
            .expect("QuantilePredictions is row-major n_rows x n_probs");
        Ok(array.into_pyarray(py))
    }

    /// Central posterior-predictive interval for a new observation at each
    /// row of `x`, at the given level (e.g. 0.9): returns (lower, upper).
    fn prediction_interval<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        level: f64,
    ) -> PyResult<IntervalArrays<'py>> {
        let data = to_data(&x)?;
        let intervals = self.inner.prediction_interval(&data, level).map_err(err)?;
        let lower: Vec<f64> = intervals.iter().map(|i| i.lower).collect();
        let upper: Vec<f64> = intervals.iter().map(|i| i.upper).collect();
        Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
    }

    /// Central credible interval for the mean surface at each row of `x`,
    /// at the given level: returns (lower, upper).
    fn credible_interval<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        level: f64,
    ) -> PyResult<IntervalArrays<'py>> {
        let data = to_data(&x)?;
        let intervals = self.inner.credible_interval(&data, level).map_err(err)?;
        let lower: Vec<f64> = intervals.iter().map(|i| i.lower).collect();
        let upper: Vec<f64> = intervals.iter().map(|i| i.upper).collect();
        Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
    }

    /// Per-draw error standard deviation on the response scale.
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.sigma().into_pyarray(py)
    }

    /// Per-draw total cell count summed over the ensemble's tessellations.
    fn total_cells<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let posterior = self.inner.posterior();
        let counts: Vec<f64> = (0..posterior.n_draws())
            .map(|d| {
                posterior
                    .tessellations(d)
                    .iter()
                    .map(|t| t.n_cells() as f64)
                    .sum()
            })
            .collect();
        counts.into_pyarray(py)
    }

    /// BART-style per-covariate inclusion proportions (pre-encoding caller
    /// columns; one-hot groups aggregated onto their source column).
    fn variable_inclusion_proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.variable_inclusion_proportions().into_pyarray(py)
    }

    /// In-sample root-mean-square error (response scale).
    #[getter]
    fn in_sample_rmse(&self) -> f64 {
        self.inner.in_sample_rmse()
    }

    /// Number of kept posterior draws.
    #[getter]
    fn n_draws(&self) -> usize {
        self.inner.posterior().n_draws()
    }

    /// The fitted model's response family name.
    #[getter]
    fn response_family(&self) -> &str {
        &self.response_family
    }

    /// The Student-t degrees of freedom, or `None` unless the family is
    /// 'robust_t' (a dimensionless count).
    #[getter]
    fn t_df(&self) -> Option<f64> {
        match self.inner.config().response_family() {
            ResponseFamily::RobustT { df } => Some(df),
            _ => None,
        }
    }

    /// Number of caller-visible (pre-encoding) feature columns the model
    /// was fitted on. Prediction inputs must match this count.
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.scaler().n_raw_cols()
    }

    /// Fit warnings, as human-readable strings.
    fn warnings(&self) -> Vec<String> {
        self.inner
            .warnings()
            .iter()
            .map(|w| w.to_string())
            .collect()
    }

    /// Serialise to the crate's validated JSON format (format 2).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(err)
    }

    /// Deserialise from `to_json` output (payloads are validated on load;
    /// corrupt payloads raise, never panic).
    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        let inner: FittedAddiVortes = serde_json::from_str(payload).map_err(err)?;
        // The core names the family. `response_family_name` is exhaustive on
        // purpose: a new response family is a compile error here, never the
        // string "unknown" surfacing in Python.
        let response_family = response_family_name(inner.config().response_family()).to_string();
        Ok(Self {
            inner,
            response_family,
        })
    }

    /// Pickle support through the crate's validated JSON format: the
    /// payload is `to_json`, reconstruction is `from_json` (so loading
    /// re-validates, exactly like `load`). Models whose config carries a
    /// non-default shelf selection (custom distance, soft membership) refuse
    /// with the crate's own message, mirroring `save`.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let payload = self.to_json()?;
        let cls = py.get_type::<FittedModel>();
        Ok((cls.getattr("from_json")?, (payload,)))
    }

    /// Save to a JSON file (see `to_json`).
    fn save(&self, path: &str) -> PyResult<()> {
        let payload = self.to_json()?;
        std::fs::write(path, payload).map_err(err)
    }

    /// Load from a `save`d JSON file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let payload = std::fs::read_to_string(path).map_err(err)?;
        Self::from_json(&payload)
    }
}

/// The diagnostics take `&[Vec<f64>]` and *assert* their shape contract: at
/// least two chains, of equal length, at least four draws long. An assert is
/// right for a Rust caller and wrong here: a panic reaches Python as a bare
/// `PanicException`, which is not a `ValueError` and reads as a crash in the
/// engine rather than a mistake in the call. The contract is checked here,
/// where the caller typed the argument.
fn checked_chains(chains: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if chains.len() < 2 {
        return Err(err(format!(
            "r_hat and the ESS diagnostics compare chains, so they need at least 2; \
             got {}. Fit with chains=2 (or more) and pass the per-chain draws.",
            chains.len()
        )));
    }
    let n = chains[0].len();
    if chains.iter().any(|c| c.len() != n) {
        let lengths: Vec<String> = chains.iter().map(|c| c.len().to_string()).collect();
        return Err(err(format!(
            "every chain must have the same number of draws; got lengths {}",
            lengths.join(", ")
        )));
    }
    if n < 4 {
        return Err(err(format!(
            "need at least 4 draws per chain to split them; got {n}"
        )));
    }
    Ok(chains)
}

/// Split-R̂ (Vehtari et al. 2021) over chains of equal length.
#[pyfunction]
fn r_hat(chains: Vec<Vec<f64>>) -> PyResult<f64> {
    Ok(addivortes::diagnostics::r_hat(&checked_chains(chains)?))
}

/// Bulk effective sample size (Vehtari et al. 2021).
#[pyfunction]
fn ess_bulk(chains: Vec<Vec<f64>>) -> PyResult<f64> {
    Ok(addivortes::diagnostics::ess_bulk(&checked_chains(chains)?))
}

/// Tail effective sample size (Vehtari et al. 2021).
#[pyfunction]
fn ess_tail(chains: Vec<Vec<f64>>) -> PyResult<f64> {
    Ok(addivortes::diagnostics::ess_tail(&checked_chains(chains)?))
}

/// Predictive-QQ PIT values (sorted ascending): calibration of
/// the heteroscedastic Gaussian predictive N(fit_d(x_i), s_d(x_i)^2)
/// against observed `y`. `fit_draws` and `s_draws` are (n_draws, n_rows):
/// `predict_draws` output and the per-draw error SD broadcast per row.
/// Plot against uniform quantiles (i - 1/2)/n: a straight line is a
/// calibrated model. Shapes are validated here (the crate asserts).
#[pyfunction]
fn predictive_qq(
    y: PyReadonlyArray1<'_, f64>,
    fit_draws: PyReadonlyArray2<'_, f64>,
    s_draws: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<f64>> {
    let y = to_vec(&y);
    let fits = fit_draws.as_array();
    let sds = s_draws.as_array();
    if fits.nrows() == 0 || fits.dim() != sds.dim() {
        return Err(err(format!(
            "fit_draws {:?} and s_draws {:?} must be equal non-empty (n_draws, n_rows) shapes",
            fits.dim(),
            sds.dim()
        )));
    }
    if fits.ncols() != y.len() {
        return Err(err(format!(
            "fit_draws covers {} observations but y has {}",
            fits.ncols(),
            y.len()
        )));
    }
    if sds.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return Err(err("s_draws must be strictly positive error SDs"));
    }
    let fit_rows: Vec<Vec<f64>> = fits.rows().into_iter().map(|r| r.to_vec()).collect();
    let s_rows: Vec<Vec<f64>> = sds.rows().into_iter().map(|r| r.to_vec()).collect();
    Ok(addivortes::diagnostics::predictive_qq(
        &y, &fit_rows, &s_rows,
    ))
}

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AddiVortes>()?;
    m.add_class::<FittedModel>()?;
    m.add("AddiVortesError", py.get_type::<AddiVortesError>())?;
    m.add_function(wrap_pyfunction!(validate_spec, m)?)?;
    m.add_function(wrap_pyfunction!(r_hat, m)?)?;
    m.add_function(wrap_pyfunction!(ess_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(ess_tail, m)?)?;
    m.add_function(wrap_pyfunction!(predictive_qq, m)?)?;
    Ok(())
}
