#' Export posterior draws to the posterior package
#'
#' Turns a fit (one chain) or an `addivortes_chains` object (chain
#' structure preserved) into a `draws_df` with the scalar chain traces:
#' `sigma` (error SD, response scale) and `total_cells` (ensemble
#' complexity). From there the whole posterior/bayesplot toolkit applies:
#' `summarise_draws()`, `rhat()`, `ess_bulk()`, `mcmc_trace()`, ...
#'
#' Requires the posterior package (Suggests); the methods register when
#' posterior is loaded.
#'
#' @param x An `addivortes_fit` or `addivortes_chains`.
#' @param ... Unused.
#' @return A [posterior::draws_df] with variables `sigma` and
#'   `total_cells`.
#' @exportS3Method posterior::as_draws_df
as_draws_df.addivortes_fit <- function(x, ...) {
  posterior::as_draws_df(draws_data_frame(list(x)))
}

#' @rdname as_draws_df.addivortes_fit
#' @exportS3Method posterior::as_draws_df
as_draws_df.addivortes_chains <- function(x, ...) {
  posterior::as_draws_df(draws_data_frame(x))
}

#' @rdname as_draws_df.addivortes_fit
#' @exportS3Method posterior::as_draws
as_draws.addivortes_fit <- function(x, ...) {
  as_draws_df.addivortes_fit(x, ...)
}

#' @rdname as_draws_df.addivortes_fit
#' @exportS3Method posterior::as_draws
as_draws.addivortes_chains <- function(x, ...) {
  as_draws_df.addivortes_chains(x, ...)
}

draws_data_frame <- function(fits) {
  draw_counts <- vapply(fits, function(f) f$ptr$n_draws(), integer(1))
  if (length(fits) == 0) {
    stop("need at least one chain", call. = FALSE)
  }
  if (length(unique(draw_counts)) != 1) {
    stop("chains have differing kept draw counts", call. = FALSE)
  }
  do.call(rbind, lapply(seq_along(fits), function(i) {
    fit <- fits[[i]]
    data.frame(
      .chain = i,
      .iteration = seq_len(draw_counts[i]),
      sigma = fit$ptr$sigma(),
      total_cells = fit$ptr$total_cells()
    )
  }))
}

#' PSIS-LOO / WAIC for AddiVortes fits
#'
#' Feeds the fit's pointwise [log_likelihood()] matrix to the loo package.
#' For multi-chain fits the relative efficiency `r_eff` is computed with
#' the correct chain labelling. Requires the loo package (Suggests); the
#' methods register when loo is loaded.
#'
#' @param x An `addivortes_fit` or `addivortes_chains`.
#' @param newdata Numeric matrix of predictors.
#' @param y Observed response, length `nrow(newdata)`.
#' @param ... Passed on to [loo::loo()] / [loo::waic()].
#' @return A `loo`/`waic` object from the loo package.
#' @exportS3Method loo::loo
loo.addivortes_fit <- function(x, newdata, y, ...) {
  loo_from_chains(list(x), newdata, y, ...)
}

#' @rdname loo.addivortes_fit
#' @exportS3Method loo::loo
loo.addivortes_chains <- function(x, newdata, y, ...) {
  loo_from_chains(x, newdata, y, ...)
}

#' @rdname loo.addivortes_fit
#' @exportS3Method loo::waic
waic.addivortes_fit <- function(x, newdata, y, ...) {
  loo::waic(log_likelihood(x, newdata, y), ...)
}

#' @rdname loo.addivortes_fit
#' @exportS3Method loo::waic
waic.addivortes_chains <- function(x, newdata, y, ...) {
  loo::waic(log_likelihood(x, newdata, y), ...)
}

loo_from_chains <- function(fits, newdata, y, ...) {
  log_lik <- do.call(rbind, lapply(fits, log_likelihood, newdata = newdata, y = y))
  chain_id <- rep(seq_along(fits), vapply(fits, function(f) f$ptr$n_draws(), integer(1)))
  r_eff <- loo::relative_eff(exp(log_lik), chain_id = chain_id)
  loo::loo(log_lik, r_eff = r_eff, ...)
}

#' Posterior-predictive replicates (yrep)
#'
#' Draws replicate responses from the posterior predictive, one row per
#' kept draw: the matrix bayesplot's `ppc_*` functions consume as `yrep`.
#' Per draw: `N(fit, sigma^2)` for gaussian, location-scale Student-t for
#' robust_t, Bernoulli labels for binary_probit. Noise is drawn with R's
#' RNG: call [set.seed()] first for reproducible replicates (the fit
#' itself is untouched; prediction consumes no engine RNG).
#'
#' @param object An `addivortes_fit` or `addivortes_chains`.
#' @param newdata Numeric matrix of predictors.
#' @return An `n_draws` x `n_rows` matrix (chains stacked row-wise,
#'   chain-major).
#' @export
predictive_draws <- function(object, newdata) {
  UseMethod("predictive_draws")
}

#' @export
predictive_draws.addivortes_fit <- function(object, newdata) {
  fits <- predict(object, newdata, type = "draws")
  n_draws <- nrow(fits)
  n_rows <- ncol(fits)
  family <- object$ptr$response_family()
  if (family == "binary_probit") {
    probabilities <- pmin(pmax(fits, 0), 1)
    matrix(
      stats::rbinom(length(fits), size = 1L, prob = probabilities),
      n_draws, n_rows
    )
  } else if (family == "robust_t") {
    # Per-draw sigma recycles down rows: element (d, i) gets sigma[d].
    fits + sigma(object) * matrix(
      stats::rt(length(fits), df = object$ptr$t_df()),
      n_draws, n_rows
    )
  } else {
    fits + sigma(object) * matrix(stats::rnorm(length(fits)), n_draws, n_rows)
  }
}

#' @export
predictive_draws.addivortes_chains <- function(object, newdata) {
  do.call(rbind, lapply(object, predictive_draws, newdata = newdata))
}
