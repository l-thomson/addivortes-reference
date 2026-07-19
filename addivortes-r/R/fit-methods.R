#' Predict from an AddiVortes fit
#'
#' @param object An `addivortes_fit` from [addivortes()].
#' @param newdata Numeric matrix of predictors (a single observation may be
#'   passed as a plain vector).
#' @param type `"mean"` (default) for posterior-mean predictions, one per
#'   row: response scale for gaussian/robust_t, probability scale for
#'   binary_probit; `"draws"` for the full per-draw prediction matrix
#'   (`n_draws` x `n_rows`), whose column means are `"mean"`; or
#'   `"quantiles"` for posterior-predictive quantiles at `probs`
#'   (`n_rows` x `length(probs)`).
#' @param probs Quantile probabilities, for `type = "quantiles"`.
#' @param ... Unused.
#' @return A vector (`"mean"`) or matrix (`"draws"`, `"quantiles"`).
#'   Prediction is deterministic (it consumes no RNG).
#' @export
predict.addivortes_fit <- function(object, newdata,
                                   type = c("mean", "draws", "quantiles"),
                                   probs = c(0.05, 0.5, 0.95), ...) {
  type <- match.arg(type)
  newdata <- prediction_matrix(object, newdata)
  switch(type,
    mean = object$ptr$predict(newdata),
    draws = object$ptr$predict_draws(newdata),
    quantiles = {
      probs <- as.numeric(probs)
      quantiles <- object$ptr$predict_quantiles(newdata, probs)
      colnames(quantiles) <- format(probs, trim = TRUE)
      quantiles
    }
  )
}

#' Predict from a multi-chain AddiVortes fit
#'
#' @param object An `addivortes_chains` from [addivortes()] with
#'   `chains > 1`.
#' @param newdata Numeric matrix of predictors.
#' @param type `"mean"` averages the chains' posterior means; `"draws"`
#'   stacks the chains' per-draw matrices row-wise (all chains' draws,
#'   chain-major).
#' @param ... Unused.
#' @export
predict.addivortes_chains <- function(object, newdata,
                                      type = c("mean", "draws"), ...) {
  type <- match.arg(type)
  per_chain <- lapply(object, predict, newdata = newdata, type = type)
  if (type == "mean") {
    Reduce(`+`, per_chain) / length(per_chain)
  } else {
    do.call(rbind, per_chain)
  }
}

#' Posterior-predictive interval for new observations
#'
#' The central interval for a NEW observation at each row of `newdata`
#' (fit uncertainty plus error noise). Compare [credible_interval()].
#'
#' @param object An `addivortes_fit`.
#' @param newdata Numeric matrix of predictors.
#' @param level Central coverage, e.g. 0.9.
#' @return A two-column matrix (`lower`, `upper`), one row per observation.
#' @export
prediction_interval <- function(object, newdata, level = 0.9) {
  UseMethod("prediction_interval")
}

#' @export
prediction_interval.addivortes_fit <- function(object, newdata, level = 0.9) {
  newdata <- prediction_matrix(object, newdata)
  interval <- object$ptr$prediction_interval(newdata, scalar_num(level, "level"))
  cbind(lower = interval$lower, upper = interval$upper)
}

#' Credible interval for the mean surface
#'
#' The central interval for the MEAN surface at each row of `newdata`
#' (fit uncertainty only). Compare [prediction_interval()].
#'
#' @inheritParams prediction_interval
#' @return A two-column matrix (`lower`, `upper`), one row per observation.
#' @export
credible_interval <- function(object, newdata, level = 0.9) {
  UseMethod("credible_interval")
}

#' @export
credible_interval.addivortes_fit <- function(object, newdata, level = 0.9) {
  newdata <- prediction_matrix(object, newdata)
  interval <- object$ptr$credible_interval(newdata, scalar_num(level, "level"))
  cbind(lower = interval$lower, upper = interval$upper)
}

# Neither interval has an `addivortes_chains` method, and both want one: the
# pooled interval is the one an R user running several chains actually wants.
# Both are engine-side by design and neither pools correctly from per-chain
# endpoints.
#
# credible_interval is the type-7 quantile of the per-draw fits, so pooling
# in R agrees with the engine to the last bit but not exactly: R's
# `(1-g)*lo + g*hi` and the engine's `lo + g*(hi-lo)` differ by 1 ULP. In a
# package that ships golden vectors that is a divergence, not a rounding
# detail.
#
# prediction_interval is a bisection on the equal-weight predictive mixture
# over draws (model.rs prediction_interval), so it cannot be recovered from
# per-chain endpoints at all, and doing the bisection in R would duplicate
# engine numerics.
#
# The fix for both is one engine-side entry point taking the pooled draws,
# mirrored in the Python binding so the two stay in step.

#' Pointwise log-likelihood matrix
#'
#' `log p(y_i | draw d)` against observed `y`, the matrix the PSIS-LOO /
#' WAIC estimators consume (see [loo.addivortes_fit()]). Per draw:
#' `N(fit, sigma^2)` for gaussian, location-scale Student-t for robust_t,
#' Bernoulli (probabilities clamped away from 0 and 1) for binary_probit.
#'
#' @param object An `addivortes_fit` or `addivortes_chains`.
#' @param newdata Numeric matrix of predictors.
#' @param y Observed response, length `nrow(newdata)`.
#' @param ... Unused.
#' @return An `n_draws` x `n_rows` matrix; for chains, the chains' matrices
#'   stacked row-wise (chain-major).
#' @export
log_likelihood <- function(object, newdata, y, ...) {
  UseMethod("log_likelihood")
}

#' @export
log_likelihood.addivortes_fit <- function(object, newdata, y, ...) {
  newdata <- prediction_matrix(object, newdata)
  object$ptr$log_likelihood(newdata, as.numeric(y))
}

#' @export
log_likelihood.addivortes_chains <- function(object, newdata, y, ...) {
  do.call(rbind, lapply(object, log_likelihood, newdata = newdata, y = y))
}

#' Per-draw error standard deviation
#'
#' The posterior draws of the error SD on the RESPONSE scale, as
#' [stats::sigma()] method.
#'
#' @param object An `addivortes_fit`.
#' @param ... Unused.
#' @return A numeric vector of length `n_draws`.
#' @importFrom stats predict sigma
#' @export
sigma.addivortes_fit <- function(object, ...) {
  object$ptr$sigma()
}

#' Per-draw total cell count
#'
#' The total number of Voronoi cells summed over the ensemble's
#' tessellations, per kept draw: the standard complexity trace.
#'
#' @param object An `addivortes_fit`.
#' @return A numeric vector of length `n_draws`.
#' @export
total_cells <- function(object) {
  UseMethod("total_cells")
}

#' @export
total_cells.addivortes_fit <- function(object) {
  object$ptr$total_cells()
}

#' Variable importance (inclusion proportions)
#'
#' BART-style per-covariate inclusion proportions (pre-encoding caller
#' columns; one-hot groups aggregated onto their source column), labelled
#' with the fit's feature names and sorted descending.
#'
#' @param object An `addivortes_fit`.
#' @return A named numeric vector summing to 1, sorted descending.
#' @export
variable_importance <- function(object) {
  UseMethod("variable_importance")
}

#' @export
variable_importance.addivortes_fit <- function(object) {
  proportions <- object$ptr$variable_inclusion_proportions()
  names(proportions) <- feature_labels(object, length(proportions))
  sort(proportions, decreasing = TRUE)
}

#' @export
print.addivortes_fit <- function(x, ...) {
  ptr <- x$ptr
  cat("AddiVortes fit (", ptr$response_family(), ")\n", sep = "")
  cat(
    "  features: ", ptr$n_features(),
    "   kept draws: ", ptr$n_draws(), "\n",
    sep = ""
  )
  cat("  in-sample RMSE: ", format(ptr$in_sample_rmse(), digits = 4), "\n", sep = "")
  fit_warnings <- ptr$warnings()
  if (length(fit_warnings) > 0) {
    cat("  fit warnings: ", length(fit_warnings), " (see summary())\n", sep = "")
  }
  invisible(x)
}

#' @export
print.addivortes_chains <- function(x, ...) {
  cat("AddiVortes fit, ", length(x), " chains (", x[[1]]$ptr$response_family(),
    "), ", x[[1]]$ptr$n_draws(), " kept draws each\n",
    sep = ""
  )
  invisible(x)
}

#' Summarise an AddiVortes fit
#'
#' @param object An `addivortes_fit`.
#' @param ... Unused.
#' @return A `summary.addivortes_fit` list: family, dimensions, in-sample
#'   RMSE, posterior quantiles of `sigma` and `total_cells`,
#'   [variable_importance()], and any fit warnings.
#' @export
summary.addivortes_fit <- function(object, ...) {
  ptr <- object$ptr
  structure(
    list(
      response_family = ptr$response_family(),
      n_features = ptr$n_features(),
      n_draws = ptr$n_draws(),
      t_df = ptr$t_df(),
      in_sample_rmse = ptr$in_sample_rmse(),
      sigma = stats::quantile(ptr$sigma(), c(0.05, 0.25, 0.5, 0.75, 0.95)),
      total_cells = stats::quantile(ptr$total_cells(), c(0.05, 0.5, 0.95)),
      variable_importance = variable_importance(object),
      warnings = ptr$warnings()
    ),
    class = "summary.addivortes_fit"
  )
}

#' @export
print.summary.addivortes_fit <- function(x, ...) {
  cat("AddiVortes fit (", x$response_family, ")\n", sep = "")
  if (!is.null(x$t_df)) cat("  t_df: ", x$t_df, "\n", sep = "")
  cat(
    "  features: ", x$n_features,
    "   kept draws: ", x$n_draws, "\n",
    sep = ""
  )
  cat("  in-sample RMSE: ", format(x$in_sample_rmse, digits = 4), "\n\n", sep = "")
  cat("Error SD (sigma) posterior quantiles:\n")
  print(x$sigma, digits = 4)
  cat("\nTotal cells posterior quantiles:\n")
  print(x$total_cells, digits = 4)
  cat("\nVariable importance (inclusion proportions):\n")
  print(x$variable_importance, digits = 3)
  if (length(x$warnings) > 0) {
    cat("\nFit warnings:\n")
    for (w in x$warnings) cat("  - ", w, "\n", sep = "")
  }
  invisible(x)
}

#' @export
sigma.addivortes_chains <- function(object, ...) {
  unlist(lapply(object, sigma), use.names = FALSE)
}

#' @export
total_cells.addivortes_chains <- function(object) {
  unlist(lapply(object, total_cells), use.names = FALSE)
}

#' @export
variable_importance.addivortes_chains <- function(object) {
  per_chain <- lapply(object, variable_importance)
  # Re-order to the first chain's labels before averaging: each chain sorts
  # its own proportions descending, so positions need not agree.
  labels <- names(per_chain[[1]])
  pooled <- Reduce(`+`, lapply(per_chain, function(v) v[labels])) / length(per_chain)
  sort(pooled, decreasing = TRUE)
}

#' Summarise a multi-chain AddiVortes fit
#'
#' As [summary.addivortes_fit()] over the pooled draws, plus the
#' between-chain convergence diagnostics: split-R-hat and bulk/tail ESS
#' for `sigma` and `total_cells`. R-hat near 1 means the chains agree;
#' materially above 1 means they have not converged and the pooled
#' summaries below it should not be trusted.
#'
#' @param object An `addivortes_chains`.
#' @param ... Unused.
#' @return A `summary.addivortes_chains` list.
#' @export
summary.addivortes_chains <- function(object, ...) {
  sigma_by_chain <- lapply(object, sigma)
  cells_by_chain <- lapply(object, total_cells)
  diagnose <- function(by_chain) {
    c(
      r_hat = r_hat(by_chain),
      ess_bulk = ess_bulk(by_chain),
      ess_tail = ess_tail(by_chain)
    )
  }
  structure(
    list(
      response_family = object[[1]]$ptr$response_family(),
      n_chains = length(object),
      n_draws = object[[1]]$ptr$n_draws(),
      sigma = stats::quantile(unlist(sigma_by_chain), c(0.05, 0.5, 0.95)),
      total_cells = stats::quantile(unlist(cells_by_chain), c(0.05, 0.5, 0.95)),
      sigma_diagnostics = diagnose(sigma_by_chain),
      total_cells_diagnostics = diagnose(cells_by_chain),
      variable_importance = variable_importance(object)
    ),
    class = "summary.addivortes_chains"
  )
}

#' @export
print.summary.addivortes_chains <- function(x, ...) {
  cat("AddiVortes fit (", x$response_family, "), ", x$n_chains,
    " chains, ", x$n_draws, " kept draws each\n\n",
    sep = ""
  )
  cat("Error SD (sigma) pooled quantiles:\n")
  print(x$sigma, digits = 4)
  cat("\nTotal cells pooled quantiles:\n")
  print(x$total_cells, digits = 4)
  cat("\nConvergence (sigma):\n")
  print(x$sigma_diagnostics, digits = 4)
  cat("\nConvergence (total cells):\n")
  print(x$total_cells_diagnostics, digits = 4)
  cat("\nVariable importance (chain-averaged):\n")
  print(x$variable_importance, digits = 3)
  invisible(x)
}

#' Serialise a fit to the engine's validated JSON format
#'
#' The payload round-trips with bit-identical predictions
#' (`avt_from_json()` / `avt_load()` re-validate on load, so a corrupt
#' payload is an error, never a crash). Models whose config carries a
#' non-default shelf selection (custom `distance`, soft `membership`) refuse
#' with the engine's own message. Feature names are an R-side convenience
#' and are not part of the payload. Note `saveRDS()` does not work on
#' fits: the model lives behind an external pointer; use these instead.
#'
#' @param object An `addivortes_fit`.
#' @return A JSON string.
#' @export
avt_to_json <- function(object) {
  object$ptr$to_json()
}

#' @rdname avt_to_json
#' @param json A string from [avt_to_json()].
#' @export
avt_from_json <- function(json) {
  new_addivortes_fit(FittedModel$from_json(json))
}

#' @rdname avt_to_json
#' @param path File path for the JSON payload.
#' @export
avt_save <- function(object, path) {
  invisible(object$ptr$save(path))
}

#' @rdname avt_to_json
#' @export
avt_load <- function(path) {
  new_addivortes_fit(FittedModel$load(path))
}

prediction_matrix <- function(object, newdata) {
  newdata <- as_design_matrix(newdata)
  expected <- object$ptr$n_features()
  if (ncol(newdata) != expected) {
    stop(
      sprintf(
        "newdata has %d features but the model was fitted on %d",
        ncol(newdata), expected
      ),
      call. = FALSE
    )
  }
  newdata
}

feature_labels <- function(object, n) {
  if (!is.null(object$feature_names)) {
    object$feature_names
  } else {
    paste0("x", seq_len(n))
  }
}
