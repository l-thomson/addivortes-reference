#' Convergence diagnostics (Vehtari et al. 2021)
#'
#' Split-R-hat and bulk/tail effective sample size over a list of
#' equal-length numeric chains, the engine's own implementations, so R
#' and Python callers see identical numbers for identical draws. For a
#' fuller toolkit, export the fit with [as_draws_df.addivortes_fit()] and
#' use the posterior package.
#'
#' @param chains A list of equal-length numeric vectors (e.g.
#'   `lapply(fits, sigma)` over an `addivortes_chains` object).
#' @return A single numeric value.
#' @export
r_hat <- function(chains) {
  avt_r_hat(lapply(chains, as.numeric))
}

#' @rdname r_hat
#' @export
ess_bulk <- function(chains) {
  avt_ess_bulk(lapply(chains, as.numeric))
}

#' @rdname r_hat
#' @export
ess_tail <- function(chains) {
  avt_ess_tail(lapply(chains, as.numeric))
}

#' Predictive-QQ PIT values
#'
#' PIT values (sorted ascending) of the heteroscedastic Gaussian predictive
#' `N(fit_d(x_i), s_d(x_i)^2)` against observed `y`. Plot against uniform
#' quantiles `(i - 1/2) / n`: a straight line is a calibrated model.
#'
#' @param y Observed response, length `n_rows`.
#' @param fit_draws `n_draws` x `n_rows` matrix of per-draw fits
#'   (`predict(fit, x, type = "draws")`).
#' @param s_draws `n_draws` x `n_rows` matrix of per-draw error SDs
#'   (strictly positive), e.g. `matrix(sigma(fit), n_draws, n_rows)`.
#' @return Sorted PIT values, length `n_rows`.
#' @export
predictive_qq <- function(y, fit_draws, s_draws) {
  avt_predictive_qq(
    as.numeric(y),
    as_design_matrix(fit_draws),
    as_design_matrix(s_draws)
  )
}
