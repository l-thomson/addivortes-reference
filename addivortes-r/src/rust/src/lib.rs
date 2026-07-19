//! The `addivortesr` native library: a thin, faithful wrapper over the
//! `addivortes` public API for R. Same design rules as `addivortes-py`:
//!
//! - Defaults live in Rust. R arguments default to `NULL` and only call
//!   the corresponding `with_*` when given, so the crate's defaults (and any
//!   future retune of them) have exactly one home.
//! - The shelf crosses the boundary; the traits do not. R selects built-in
//!   components by value, through the payload; authoring new components stays a
//!   Rust-side activity: a trait impl is code, and no data
//!   payload can carry one.
//! - Bit-exact reproducibility survives the FFI. `f64`s cross unchanged;
//!   the testthat suite pins a fit against the same per-target golden vector
//!   the Rust tests use.
//! - Every crate error surfaces as an R error carrying the crate's own
//!   message.
//!
//! This library enumerates nothing: no hyperparameter, no distance, no shelf
//! item. R hands over one `ConfigSpec` JSON payload and the core maps it, so a
//! new shelf entry in the crate reaches R on rebuild with no change here. The
//! R-side `avt_spec()` assembles the payload and `avt_distance()` /
//! `avt_membership()` are sugar over it.

use extendr_api::prelude::*;

use addivortes::config_spec::{response_family_name, ConfigSpec};
use addivortes::{Data, FittedAddiVortes, ResponseFamily};

fn rerr(e: impl std::fmt::Display) -> Error {
    Error::Other(e.to_string())
}

/// Surface an error as a proper R error carrying the crate's own message.
/// extendr's default `Result` handling panics with a generic "user function
/// panicked" message; raising the R condition here keeps the contract that
/// every crate error reaches R verbatim.
fn unwrap_r<T>(value: Result<T>) -> T {
    match value {
        Ok(v) => v,
        Err(e) => throw_r_error(e.to_string()),
    }
}

/// Column-major R matrix into the crate's `Data` (logical row-major order).
fn to_data(x: &RMatrix<f64>) -> Result<Data> {
    let (n, p) = (x.nrows(), x.ncols());
    let col_major = x.data();
    let mut values = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            values.push(col_major[j * n + i]);
        }
    }
    Data::new(values, n, p).map_err(rerr)
}

/// A fitted model: the posterior draws plus the fitted scaler, wrapping the
/// crate's `FittedAddiVortes` one-to-one.
#[extendr]
pub struct FittedModel {
    inner: FittedAddiVortes,
    response_family: String,
}

#[extendr]
impl FittedModel {
    /// Posterior-mean predictions for `x`, one per row: response scale for
    /// gaussian/robust_t, probability scale for binary_probit. Deterministic
    /// (predict consumes no RNG).
    fn predict(&self, x: RMatrix<f64>) -> Vec<f64> {
        let data = unwrap_r(to_data(&x));
        unwrap_r(self.inner.predict(&data).map_err(rerr))
    }

    /// Per-draw predictions for `x`, an (n_draws, n_rows) matrix: the full,
    /// unreduced posterior of the fit on the family's own scale. `predict`
    /// is this matrix's column mean; posterior-predictive workflows (PPC,
    /// LOO/WAIC, per-draw partial dependence) consume the draw axis
    /// directly.
    fn predict_draws(&self, x: RMatrix<f64>) -> RMatrix<f64> {
        let data = unwrap_r(to_data(&x));
        let n_rows = data.n_rows();
        let draws = unwrap_r(self.inner.predict_draws(&data).map_err(rerr));
        RMatrix::new_matrix(draws.len(), n_rows, |r, c| draws[r][c])
    }

    /// Pointwise log-likelihood ln p(y_i | draw d) against observed `y`, an
    /// (n_draws, n_rows) matrix: the matrix PSIS-LOO/WAIC estimators
    /// consume. Per draw: N(fit, sigma^2) for gaussian, location-scale
    /// Student-t for robust_t, Bernoulli (probabilities clamped away from
    /// {0, 1}) for binary_probit.
    fn log_likelihood(&self, x: RMatrix<f64>, y: Vec<f64>) -> RMatrix<f64> {
        let data = unwrap_r(to_data(&x));
        let n_rows = data.n_rows();
        let ll = unwrap_r(self.inner.log_likelihood(&data, &y).map_err(rerr));
        RMatrix::new_matrix(ll.len(), n_rows, |r, c| ll[r][c])
    }

    /// Posterior-predictive quantiles at `probs` for each row of `x`: an
    /// (n_rows, length(probs)) matrix.
    fn predict_quantiles(&self, x: RMatrix<f64>, probs: Vec<f64>) -> RMatrix<f64> {
        let data = unwrap_r(to_data(&x));
        let quantiles = unwrap_r(self.inner.predict_quantiles(&data, &probs).map_err(rerr));
        let n_rows = quantiles.n_rows();
        let n_probs = quantiles.probs().len();
        let (values, _, _) = quantiles.into_parts();
        RMatrix::new_matrix(n_rows, n_probs, |r, c| values[r * n_probs + c])
    }

    /// Central posterior-predictive interval for a NEW observation at each
    /// row of `x`, at the given level (e.g. 0.9): a list(lower=, upper=).
    fn prediction_interval(&self, x: RMatrix<f64>, level: f64) -> List {
        let data = unwrap_r(to_data(&x));
        let intervals = unwrap_r(self.inner.prediction_interval(&data, level).map_err(rerr));
        let lower: Vec<f64> = intervals.iter().map(|i| i.lower).collect();
        let upper: Vec<f64> = intervals.iter().map(|i| i.upper).collect();
        list!(lower = lower, upper = upper)
    }

    /// Central credible interval for the MEAN surface at each row of `x`,
    /// at the given level: a list(lower=, upper=).
    fn credible_interval(&self, x: RMatrix<f64>, level: f64) -> List {
        let data = unwrap_r(to_data(&x));
        let intervals = unwrap_r(self.inner.credible_interval(&data, level).map_err(rerr));
        let lower: Vec<f64> = intervals.iter().map(|i| i.lower).collect();
        let upper: Vec<f64> = intervals.iter().map(|i| i.upper).collect();
        list!(lower = lower, upper = upper)
    }

    /// Per-draw error standard deviation on the RESPONSE scale.
    fn sigma(&self) -> Vec<f64> {
        self.inner.sigma()
    }

    /// Per-draw total cell count summed over the ensemble's tessellations.
    fn total_cells(&self) -> Vec<f64> {
        let posterior = self.inner.posterior();
        (0..posterior.n_draws())
            .map(|d| {
                posterior
                    .tessellations(d)
                    .iter()
                    .map(|t| t.n_cells() as f64)
                    .sum()
            })
            .collect()
    }

    /// BART-style per-covariate inclusion proportions (pre-encoding caller
    /// columns; one-hot groups aggregated onto their source column).
    fn variable_inclusion_proportions(&self) -> Vec<f64> {
        self.inner.variable_inclusion_proportions()
    }

    /// In-sample root-mean-square error (response scale).
    fn in_sample_rmse(&self) -> f64 {
        self.inner.in_sample_rmse()
    }

    /// Number of kept posterior draws.
    fn n_draws(&self) -> i32 {
        self.inner.posterior().n_draws() as i32
    }

    /// The fitted model's response family name.
    fn response_family(&self) -> String {
        self.response_family.clone()
    }

    /// The Student-t degrees of freedom, or `NULL` unless the family is
    /// 'robust_t' (a dimensionless count).
    fn t_df(&self) -> Nullable<f64> {
        match self.inner.config().response_family() {
            ResponseFamily::RobustT { df } => Nullable::NotNull(df),
            _ => Nullable::Null,
        }
    }

    /// Number of caller-visible (pre-encoding) feature columns the model
    /// was fitted on; prediction inputs must match this count.
    fn n_features(&self) -> i32 {
        self.inner.scaler().n_raw_cols() as i32
    }

    /// Fit warnings, as human-readable strings.
    fn warnings(&self) -> Vec<String> {
        self.inner
            .warnings()
            .iter()
            .map(|w| w.to_string())
            .collect()
    }

    /// Serialise to the crate's validated JSON format (format 2). Models
    /// whose config carries a non-default shelf selection (custom distance, soft
    /// membership) refuse with the crate's own message, mirroring `save`.
    fn to_json(&self) -> String {
        unwrap_r(serde_json::to_string(&self.inner).map_err(rerr))
    }

    /// Deserialise from `to_json` output (payloads are validated on load;
    /// corrupt payloads raise an R error, never a panic).
    fn from_json(payload: &str) -> FittedModel {
        let inner: FittedAddiVortes = unwrap_r(serde_json::from_str(payload).map_err(rerr));
        let response_family = match inner.config().response_family() {
            ResponseFamily::Gaussian => "gaussian".to_string(),
            ResponseFamily::BinaryProbit => "binary_probit".to_string(),
            ResponseFamily::RobustT { .. } => "robust_t".to_string(),
            _ => "unknown".to_string(),
        };
        FittedModel {
            inner,
            response_family,
        }
    }

    /// Save to a JSON file (see `to_json`).
    fn save(&self, path: &str) {
        let payload = self.to_json();
        unwrap_r(std::fs::write(path, payload).map_err(rerr))
    }

    /// Load from a `save`d JSON file.
    fn load(path: &str) -> FittedModel {
        let payload = unwrap_r(std::fs::read_to_string(path).map_err(rerr));
        Self::from_json(&payload)
    }
}

/// Parse and validate a `ConfigSpec` JSON payload, without any data.
fn parse_spec(spec_json: &str) -> Result<ConfigSpec> {
    let spec: ConfigSpec = serde_json::from_str(spec_json).map_err(rerr)?;
    spec.validate().map_err(rerr)?;
    Ok(spec)
}

/// Validate a `ConfigSpec` payload with no data, so R can fail at the point the
/// user wrote the mistake rather than at `fit`. Carries the crate's own message.
/// @noRd
#[extendr]
fn avt_validate_spec(spec_json: String) {
    unwrap_r(parse_spec(&spec_json));
}

/// Fit from a `ConfigSpec` JSON payload.
///
/// The payload is the whole configuration. This function names no
/// hyperparameter and no shelf item, so a new shelf entry in the crate is
/// reachable from R on rebuild with no change here.
/// @noRd
#[extendr]
fn avt_fit(x: RMatrix<f64>, y: Vec<f64>, spec_json: String) -> FittedModel {
    let spec = unwrap_r(parse_spec(&spec_json));
    let data = unwrap_r(to_data(&x));
    // The covariate-sized settings are sized by the data, so the config is
    // assembled here rather than at construction.
    let config = unwrap_r(spec.into_config(&data).map_err(rerr));
    let family = response_family_name(config.response_family()).to_string();
    let fitted = unwrap_r(config.fit(&data, &y).map_err(rerr));
    FittedModel {
        inner: fitted,
        response_family: family,
    }
}

/// Fit `n_chains` independent chains (seeds derived from the configured seed;
/// chain 1 is bit-identical to a plain `avt_fit`). Returns a list of fitted
/// models.
/// @noRd
#[extendr]
fn avt_fit_chains(x: RMatrix<f64>, y: Vec<f64>, spec_json: String, n_chains: f64) -> List {
    if !(n_chains.is_finite() && n_chains >= 1.0 && n_chains.fract() == 0.0) {
        unwrap_r::<()>(Err(rerr("chains must be a whole number >= 1")));
    }
    let spec = unwrap_r(parse_spec(&spec_json));
    let data = unwrap_r(to_data(&x));
    let config = unwrap_r(spec.into_config(&data).map_err(rerr));
    let family = response_family_name(config.response_family()).to_string();
    let fits = unwrap_r(
        config
            .fit_chains(&data, &y, n_chains as usize)
            .map_err(rerr),
    );
    List::from_values(fits.into_iter().map(|inner| FittedModel {
        inner,
        response_family: family.clone(),
    }))
}

/// The diagnostics take `&[Vec<f64>]` and *assert* their shape contract: at
/// least two chains, of equal length, at least four draws long. An assert is
/// right for a Rust caller and wrong here: a panic crossing the C boundary is
/// not R's error mechanism, so the user gets a raw panic dump instead of a
/// condition they can catch. The contract is therefore checked here, where the
/// caller typed the argument, and returned as an ordinary R error.
fn chains_to_vecs(chains: List) -> Result<Vec<Vec<f64>>> {
    let chains: Vec<Vec<f64>> = chains
        .values()
        .map(|obj| {
            obj.as_real_vector()
                .ok_or_else(|| rerr("each chain must be a numeric (double) vector"))
        })
        .collect::<Result<_>>()?;

    if chains.len() < 2 {
        return Err(rerr(format!(
            "r_hat and the ESS diagnostics compare chains, so they need at \
             least 2; found {}. Fit with `chains = 2` (or more) and pass the \
             per-chain draws as a list.",
            chains.len()
        )));
    }
    let n = chains[0].len();
    if chains.iter().any(|c| c.len() != n) {
        let lengths: Vec<String> = chains.iter().map(|c| c.len().to_string()).collect();
        return Err(rerr(format!(
            "every chain must have the same number of draws; found lengths {}",
            lengths.join(", ")
        )));
    }
    if n < 4 {
        return Err(rerr(format!(
            "need at least 4 draws per chain to split them; found {n}"
        )));
    }
    Ok(chains)
}

/// Split-R̂ (Vehtari et al. 2021) over a list of equal-length numeric
/// chains.
/// @noRd
#[extendr]
fn avt_r_hat(chains: List) -> f64 {
    addivortes::diagnostics::r_hat(&unwrap_r(chains_to_vecs(chains)))
}

/// Bulk effective sample size (Vehtari et al. 2021).
/// @noRd
#[extendr]
fn avt_ess_bulk(chains: List) -> f64 {
    addivortes::diagnostics::ess_bulk(&unwrap_r(chains_to_vecs(chains)))
}

/// Tail effective sample size (Vehtari et al. 2021).
/// @noRd
#[extendr]
fn avt_ess_tail(chains: List) -> f64 {
    addivortes::diagnostics::ess_tail(&unwrap_r(chains_to_vecs(chains)))
}

/// Predictive-QQ PIT values (sorted ascending): calibration of
/// the heteroscedastic Gaussian predictive N(fit_d(x_i), s_d(x_i)^2)
/// against observed `y`. `fit_draws` and `s_draws` are (n_draws, n_rows)
/// matrices: `predict_draws` output and the per-draw error SD broadcast
/// per row. Plot against uniform quantiles (i - 1/2)/n: a straight line is
/// a calibrated model. Shapes are validated here (the crate asserts).
/// @noRd
#[extendr]
fn avt_predictive_qq(y: Vec<f64>, fit_draws: RMatrix<f64>, s_draws: RMatrix<f64>) -> Vec<f64> {
    let (n_draws, n_rows) = (fit_draws.nrows(), fit_draws.ncols());
    if n_draws == 0 || (s_draws.nrows(), s_draws.ncols()) != (n_draws, n_rows) {
        throw_r_error(format!(
            "fit_draws ({}x{}) and s_draws ({}x{}) must be equal non-empty \
             (n_draws, n_rows) shapes",
            n_draws,
            n_rows,
            s_draws.nrows(),
            s_draws.ncols()
        ));
    }
    if n_rows != y.len() {
        throw_r_error(format!(
            "fit_draws covers {n_rows} observations but y has {}",
            y.len()
        ));
    }
    let row = |data: &[f64], r: usize| -> Vec<f64> {
        (0..n_rows).map(|c| data[c * n_draws + r]).collect()
    };
    let s_data = s_draws.data();
    if s_data.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        throw_r_error("s_draws must be strictly positive error SDs");
    }
    let fit_data = fit_draws.data();
    let fit_rows: Vec<Vec<f64>> = (0..n_draws).map(|r| row(fit_data, r)).collect();
    let s_rows: Vec<Vec<f64>> = (0..n_draws).map(|r| row(s_data, r)).collect();
    addivortes::diagnostics::predictive_qq(&y, &fit_rows, &s_rows)
}

extendr_module! {
    mod addivortesr;
    impl FittedModel;
    fn avt_validate_spec;
    fn avt_fit;
    fn avt_fit_chains;
    fn avt_r_hat;
    fn avt_ess_bulk;
    fn avt_ess_tail;
    fn avt_predictive_qq;
}
