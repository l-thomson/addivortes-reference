#' AddiVortes model specifications for parsnip
#'
#' Tidymodels front door: `addivortes_reg()` (regression, gaussian or
#' robust-t via engine arguments) and `addivortes_class()` (binary
#' classification through the probit family). One tunable main argument,
#' `trees` (the ensemble size `m`); every other engine hyperparameter
#' passes through `parsnip::set_engine()` (e.g.
#' `set_engine("addivortes", seed = 1, omega = 1.5)`).
#'
#' The engines register when the package loads with parsnip available
#' (or as soon as parsnip is loaded afterwards).
#'
#' @param mode `"regression"` / `"classification"`.
#' @param trees Ensemble size (the engine's `m`).
#' @param engine `"addivortes"` (the only engine).
#' @return A parsnip model specification.
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- addivortes_reg(trees = 20) |>
#'   set_engine("addivortes", seed = 1, burn_in = 100, draws = 200)
#' fitted <- fit(spec, y ~ ., data = my_data)
#' predict(fitted, my_data)
#' }
#' @export
addivortes_reg <- function(mode = "regression", trees = NULL,
                           engine = "addivortes") {
  check_parsnip_available()
  if (mode != "regression") {
    stop("addivortes_reg() only supports mode = 'regression'", call. = FALSE)
  }
  parsnip::new_model_spec(
    "addivortes_reg",
    args = list(trees = rlang::enquo(trees)),
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = engine
  )
}

#' @rdname addivortes_reg
#' @export
addivortes_class <- function(mode = "classification", trees = NULL,
                             engine = "addivortes") {
  check_parsnip_available()
  if (mode != "classification") {
    stop("addivortes_class() only supports mode = 'classification'", call. = FALSE)
  }
  parsnip::new_model_spec(
    "addivortes_class",
    args = list(trees = rlang::enquo(trees)),
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = engine
  )
}

check_parsnip_available <- function() {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("this function requires the parsnip package", call. = FALSE)
  }
}

# parsnip fit/predict shims. Exported because parsnip resolves them by
# (pkg, fun) name at fit/predict time; not part of the user-facing API.

#' parsnip engine internals
#'
#' Fit and prediction shims parsnip resolves by name; use
#' [addivortes_reg()] / [addivortes_class()] instead of calling these.
#'
#' @param x,y,new_data,object,seed,... Managed by parsnip.
#' @return See [addivortes()] and [predict.addivortes_fit()].
#' @keywords internal
#' @export
parsnip_addivortes_fit <- function(x, y, seed = 0, ...) {
  addivortes(as.matrix(x), y, seed = seed, ...)
}

#' @rdname parsnip_addivortes_fit
#' @export
parsnip_addivortes_fit_class <- function(x, y, seed = 0, ...) {
  y <- as.factor(y)
  class_levels <- levels(y)
  if (length(class_levels) != 2) {
    stop(
      sprintf(
        "Only binary classification is supported; got %d level(s)",
        length(class_levels)
      ),
      call. = FALSE
    )
  }
  fit <- addivortes(
    as.matrix(x), as.numeric(y == class_levels[2]),
    seed = seed, response_family = "binary_probit", ...
  )
  fit$class_levels <- class_levels
  fit
}

#' @rdname parsnip_addivortes_fit
#' @export
parsnip_addivortes_numeric <- function(object, new_data) {
  predict(object, as.matrix(new_data))
}

#' @rdname parsnip_addivortes_fit
#' @export
parsnip_addivortes_class <- function(object, new_data) {
  probability <- predict(object, as.matrix(new_data))
  factor(
    object$class_levels[(probability >= 0.5) + 1L],
    levels = object$class_levels
  )
}

#' @rdname parsnip_addivortes_fit
#' @export
parsnip_addivortes_prob <- function(object, new_data) {
  probability <- predict(object, as.matrix(new_data))
  out <- data.frame(1 - probability, probability)
  names(out) <- object$class_levels
  out
}

# Registration (idempotent; called from .onLoad when parsnip is present).
register_parsnip_models <- function() {
  existing <- parsnip::get_model_env()$models

  if (!"addivortes_reg" %in% existing) {
    parsnip::set_new_model("addivortes_reg")
    parsnip::set_model_mode("addivortes_reg", "regression")
    parsnip::set_model_engine("addivortes_reg", "regression", "addivortes")
    parsnip::set_dependency("addivortes_reg",
      eng = "addivortes", pkg = "addivortesr", mode = "regression"
    )
    parsnip::set_model_arg(
      model = "addivortes_reg", eng = "addivortes",
      parsnip = "trees", original = "m",
      func = list(pkg = "dials", fun = "trees"),
      has_submodel = FALSE
    )
    parsnip::set_encoding(
      model = "addivortes_reg", eng = "addivortes", mode = "regression",
      # Raw numeric predictors, no intercept, no auto-dummy expansion: the
      # engine encodes categoricals as integer columns through its metrics/
      # gower machinery, not one-hot, so it takes the design matrix as-is (the
      # tree-ensemble convention, matching parsnip::bart). "traditional" would
      # run model.matrix and inject a constant (Intercept) column the engine
      # rejects as zero-range.
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = FALSE,
        allow_sparse_x = FALSE
      )
    )
    parsnip::set_fit(
      model = "addivortes_reg", eng = "addivortes", mode = "regression",
      value = list(
        interface = "matrix",
        protect = c("x", "y"),
        func = c(pkg = "addivortesr", fun = "parsnip_addivortes_fit"),
        defaults = list()
      )
    )
    parsnip::set_pred(
      model = "addivortes_reg", eng = "addivortes", mode = "regression",
      type = "numeric",
      value = list(
        pre = NULL, post = NULL,
        func = c(pkg = "addivortesr", fun = "parsnip_addivortes_numeric"),
        args = list(object = quote(object$fit), new_data = quote(new_data))
      )
    )
  }

  if (!"addivortes_class" %in% existing) {
    parsnip::set_new_model("addivortes_class")
    parsnip::set_model_mode("addivortes_class", "classification")
    parsnip::set_model_engine("addivortes_class", "classification", "addivortes")
    parsnip::set_dependency("addivortes_class",
      eng = "addivortes", pkg = "addivortesr", mode = "classification"
    )
    parsnip::set_model_arg(
      model = "addivortes_class", eng = "addivortes",
      parsnip = "trees", original = "m",
      func = list(pkg = "dials", fun = "trees"),
      has_submodel = FALSE
    )
    parsnip::set_encoding(
      model = "addivortes_class", eng = "addivortes", mode = "classification",
      # See addivortes_reg above: raw predictors, no injected intercept.
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = FALSE,
        allow_sparse_x = FALSE
      )
    )
    parsnip::set_fit(
      model = "addivortes_class", eng = "addivortes", mode = "classification",
      value = list(
        interface = "matrix",
        protect = c("x", "y"),
        func = c(pkg = "addivortesr", fun = "parsnip_addivortes_fit_class"),
        defaults = list()
      )
    )
    parsnip::set_pred(
      model = "addivortes_class", eng = "addivortes", mode = "classification",
      type = "class",
      value = list(
        pre = NULL, post = NULL,
        func = c(pkg = "addivortesr", fun = "parsnip_addivortes_class"),
        args = list(object = quote(object$fit), new_data = quote(new_data))
      )
    )
    parsnip::set_pred(
      model = "addivortes_class", eng = "addivortes", mode = "classification",
      type = "prob",
      value = list(
        pre = NULL, post = NULL,
        func = c(pkg = "addivortesr", fun = "parsnip_addivortes_prob"),
        args = list(object = quote(object$fit), new_data = quote(new_data))
      )
    )
  }

  invisible()
}
