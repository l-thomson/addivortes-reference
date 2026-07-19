#' Fit an AddiVortes model
#'
#' Bayesian additive Voronoi tessellation regression (Stone & Gosling 2025,
#' JCGS): the response is modelled as a sum of `m` Voronoi tessellations
#' explored by a Gibbs backfitting sampler. This function wraps the
#' `addivortes` Rust engine; hyperparameters left `NULL` use the engine's
#' defaults, which live in the crate. The engine's `omega` default (3.0)
#' requires more than 3 features; pass e.g. `omega = 1.5` for narrower
#' designs.
#'
#' Chains are reproducible bit-for-bit given the same seed, engine version,
#' and compilation target (the testthat suite pins a fit against the same
#' golden vectors the Rust tests use).
#'
#' @param x Numeric predictor matrix (or something [as.matrix()] turns into
#'   one), one row per observation, raw caller scale.
#' @param y Numeric response of length `nrow(x)`; for
#'   `response_family = "binary_probit"` the values must be 0/1.
#' @param seed RNG seed (a non-negative whole number). Mandatory, as in the
#'   Rust and Python interfaces: two callers who omit it would otherwise
#'   share a chain and could mistake that for independent replication.
#' @param m Number of tessellations in the ensemble.
#' @param burn_in Number of burn-in sweeps to discard.
#' @param draws Number of posterior draws to keep.
#' @param thinning Keep every `thinning`-th post-burn-in draw.
#' @param nu,q Error-variance prior: degrees of freedom and quantile.
#' @param k Terminal-output prior spread.
#' @param lambda_c Poisson prior rate for cell counts.
#' @param omega Poisson prior rate for active dimensions (must be below the
#'   number of features).
#' @param sigma_c Coordinate proposal spread.
#' @param metrics Per-column metric names (`"euclidean"`, `"spherical"`,
#'   `"categorical"`, `"prepared"`), or `NULL` for all-Euclidean.
#' @param distance Assignment geometry: a kind name (`"euclidean"`,
#'   `"manhattan"`, `"cosine"`, `"spherical"`) or an [avt_distance()]
#'   specification for the parameterised kinds (`"minkowski"`, `"gower"`,
#'   `"mahalanobis"`). `NULL` uses the paper's per-column geometry.
#' @param membership Soft cell membership: an [avt_membership()]
#'   specification, a named list, or `NULL` (the default) for hard
#'   membership. Note models with a non-default extension point refuse JSON
#'   serialisation, with the engine's own message.
#' @param moves Structural moves and their selection weights: a list
#'   of `list(name =, weight =)`, or `NULL` for the paper's six-move set.
#' @param coords Centre-coordinate laws, one per raw column: a list
#'   of `list(type = "euclidean_normal", sigma_c =)` or `"wrapped_normal"`.
#'   `NULL` uses the per-metric defaults at `sigma_c`.
#' @param inclusion Covariate-inclusion model: e.g.
#'   `list(type = "dart", alpha = 0.5)`,
#'   `list(type = "weighted", weights = c(...))`, or `NULL` for uniform.
#' @param scale Error-variance model: e.g.
#'   `list(type = "h_variance", m_prime = 40)` for the heteroscedastic
#'   ensemble, or `list(type = "pinned", sigma_sq = 1)`. `NULL` uses the
#'   engine's calibrated global sigma.
#' @param count_priors Cell/dimension count priors; `NULL` uses the
#'   paper's pair.
#' @param basis Within-cell basis: e.g.
#'   `list(type = "linear", columns = c(0), sigma_beta_sq = 0.1)` for local
#'   linear cells. `NULL` gives a constant per cell.
#' @param response_family `"gaussian"` (default), `"binary_probit"`
#'   (predictions on the probability scale), or `"robust_t"` (requires
#'   `t_df`).
#' @param t_df Student-t degrees of freedom; only with
#'   `response_family = "robust_t"`.
#' @param chains Number of independent chains (seeds derived from `seed`;
#'   chain 1 is bit-identical to `chains = 1`).
#'
#' @return For `chains = 1` an object of class `addivortes_fit`; otherwise
#'   an `addivortes_chains` list of such fits. See
#'   [predict.addivortes_fit()], [prediction_interval()],
#'   [log_likelihood()], [sigma.addivortes_fit()], [total_cells()],
#'   [variable_importance()], [avt_save()].
#' @examples
#' x <- matrix(stats::runif(120), 60, 2)
#' y <- 2 * x[, 1] - x[, 2] + stats::rnorm(60, sd = 0.1)
#' fit <- addivortes(x, y, seed = 1, m = 8, burn_in = 20, draws = 30, omega = 1.5)
#' predict(fit, x[1:3, ])
#' @references Stone, A. and Gosling, J.P. (2025). AddiVortes: (Bayesian)
#'   Additive Voronoi Tessellations. JCGS 34(3), 859-871.
#' @export
addivortes <- function(x, y, seed, m = NULL, burn_in = NULL, draws = NULL,
                       thinning = NULL, nu = NULL, q = NULL, k = NULL,
                       lambda_c = NULL, omega = NULL, sigma_c = NULL,
                       metrics = NULL, moves = NULL, coords = NULL,
                       distance = NULL, inclusion = NULL, scale = NULL,
                       count_priors = NULL, basis = NULL, membership = NULL,
                       response_family = c("gaussian", "binary_probit", "robust_t"),
                       t_df = NULL, chains = 1L) {
  if (missing(seed)) {
    stop("`seed` is required: a chain is only reproducible against a stated seed",
      call. = FALSE
    )
  }
  response_family <- match.arg(response_family)
  x <- as_design_matrix(x)
  chains <- scalar_count(chains, "chains")
  spec <- avt_spec(
    seed = seed, m = m, burn_in = burn_in, draws = draws, thinning = thinning,
    nu = nu, q = q, k = k, lambda_c = lambda_c, omega = omega, sigma_c = sigma_c,
    metrics = metrics, moves = moves, coords = coords, distance = distance,
    inclusion = inclusion, response_family = response_family, t_df = t_df,
    scale = scale, count_priors = count_priors, basis = basis,
    membership = membership
  )
  json <- spec_json(spec)
  y <- as.numeric(y)
  if (chains == 1) {
    new_addivortes_fit(avt_fit(x, y, json), colnames(x))
  } else {
    ptrs <- avt_fit_chains(x, y, json, chains)
    structure(
      lapply(ptrs, new_addivortes_fit, feature_names = colnames(x)),
      class = "addivortes_chains"
    )
  }
}

#' The configuration payload handed to the engine
#'
#' Assembles the `ConfigSpec` the Rust core maps to a model configuration.
#' Arguments left `NULL` are omitted, so the engine's own defaults apply --
#' this package never restates a default.
#'
#' Each extension point takes a payload in the core's own vocabulary (a named
#' list), which is why a shelf entry added to the crate is selectable from R
#' the day it ships, with no change to this package.
#'
#' @inheritParams addivortes
#' @return A named list: the spec payload.
#' @keywords internal
avt_spec <- function(seed = 0, m = NULL, burn_in = NULL, draws = NULL,
                     thinning = NULL, nu = NULL, q = NULL, k = NULL,
                     lambda_c = NULL, omega = NULL, sigma_c = NULL,
                     metrics = NULL, moves = NULL, coords = NULL,
                     distance = NULL, inclusion = NULL,
                     response_family = NULL, t_df = NULL, scale = NULL,
                     count_priors = NULL, basis = NULL, membership = NULL) {
  spec <- list(
    seed = scalar_index(seed, "seed"),
    m = scalar_count(m, "m"),
    burn_in = scalar_index(burn_in, "burn_in"),
    draws = scalar_count(draws, "draws"),
    thinning = scalar_count(thinning, "thinning"),
    nu = scalar_num(nu, "nu"),
    q = scalar_num(q, "q"),
    k = scalar_num(k, "k"),
    lambda_c = scalar_num(lambda_c, "lambda_c"),
    omega = scalar_num(omega, "omega"),
    sigma_c = scalar_num(sigma_c, "sigma_c"),
    metrics = if (is.null(metrics)) NULL else as.character(metrics),
    moves = moves,
    coords = coords,
    distance = as_payload(distance, "distance"),
    inclusion = as_payload(inclusion, "inclusion"),
    response_family = response_family,
    t_df = scalar_num(t_df, "t_df"),
    scale = as_payload(scale, "scale"),
    count_priors = as_payload(count_priors, "count_priors"),
    basis = as_payload(basis, "basis"),
    membership = as_payload(membership, "membership")
  )
  spec[!vapply(spec, is.null, logical(1))]
}

#' Serialise a spec payload to the JSON the engine reads
#'
#' `digits = NA` is load-bearing, not stylistic: jsonlite's default
#' (`digits = 4`) would round every double on its way to the engine, silently
#' breaking the bit-exact reproducibility contract. `NA` keeps full precision.
#'
#' @param spec A spec payload (see `avt_spec()`).
#' @return A JSON string.
#' @keywords internal
spec_json <- function(spec) {
  as.character(jsonlite::toJSON(as_json_shape(spec),
    auto_unbox = TRUE, digits = NA, null = "null"
  ))
}

# Keys whose value is a JSON *array*, even when it holds one element.
#
# `auto_unbox = TRUE` is what makes `m = 8` serialise as `8` rather than `[8]`,
# and the engine needs that for every scalar. The cost is that it also unboxes a
# length-1 vector that was meant to be an array: `columns = 0` would go over as
# `0`, and `weights = 1` as `1`, which the engine rejects as "expected a
# sequence".
#
# So these keys are marked `I()` (jsonlite's "leave this alone"). This is a
# statement about JSON *shape*, not about the shelf: it says nothing about what
# distances or inclusion models exist, and a new shelf entry needs no entry here
# unless it introduces a new array-valued field.
.array_keys <- c(
  "metrics", "moves", "coords", "columns", "weights", "precision"
)

as_json_shape <- function(x, key = NULL) {
  if (is.list(x)) {
    shaped <- lapply(seq_along(x), function(i) {
      as_json_shape(x[[i]], key = names(x)[[i]])
    })
    names(shaped) <- names(x)
    # An unnamed list is already a JSON array; a named one is an object.
    if (!is.null(key) && key %in% .array_keys && is.null(names(x))) {
      return(I(shaped))
    }
    return(shaped)
  }
  if (!is.null(key) && key %in% .array_keys) {
    return(I(x))
  }
  x
}

#' Coerce an extension point argument to its payload list
#'
#' Accepts a bare kind name (`"manhattan"`), one of the `avt_*()` helpers, or a
#' raw named list in the core's vocabulary. The raw list is the escape hatch:
#' any shelf entry with no helper here is still reachable.
#' @noRd
as_payload <- function(value, field) {
  if (is.null(value)) {
    return(NULL)
  }
  if (is.character(value) && length(value) == 1L) {
    value <- list(type = value)
  }
  if (!is.list(value)) {
    helper <- if (field %in% c("distance", "membership")) {
      sprintf("an avt_%s() specification, ", field)
    } else {
      ""
    }
    stop(sprintf("`%s` must be a kind name, %sor a named list", field, helper),
      call. = FALSE
    )
  }
  value <- unclass(value)
  value <- value[!vapply(value, is.null, logical(1))]
  # Fail where the user wrote it, not several lines later at fit.
  probe <- list(seed = 0)
  probe[[field]] <- value
  avt_validate_spec(spec_json(probe))
  value
}

#' Assignment geometry (distance) specifications
#'
#' Selects the engine's built-in cell-assignment geometry (the distance
#' extension point's shelf). Plain kinds can also be passed to [addivortes()] directly as a
#' string; the parameterised kinds take their parameters here. Validation
#' (e.g. Mahalanobis symmetry and positive definiteness) happens in the
#' engine's own constructors at fit time.
#'
#' @param kind One of `"euclidean"`, `"manhattan"`, `"cosine"`,
#'   `"spherical"`, `"minkowski"`, `"gower"`, `"mahalanobis"`.
#' @param p Minkowski order (>= 1); `"minkowski"` only.
#' @param columns For `"gower"`: a character vector declaring each RAW
#'   column (in `metrics` order) as `"numeric"` or `"categorical"`.
#' @param levels For `"gower"`: an integer vector, the number of levels per
#'   column (ignored for numeric columns).
#' @param precision For `"mahalanobis"`: the square precision matrix over
#'   the ENCODED design width.
#' @return An `addivortes_distance` specification for [addivortes()].
#' @examples
#' avt_distance("minkowski", p = 3)
#' avt_distance("gower",
#'   columns = c("numeric", "categorical"),
#'   levels = c(0, 3)
#' )
#' @export
avt_distance <- function(kind = c(
                           "euclidean", "manhattan", "cosine", "spherical",
                           "minkowski", "gower", "mahalanobis"
                         ),
                         p = NULL, columns = NULL, levels = NULL,
                         precision = NULL) {
  kind <- match.arg(kind)
  if (kind == "minkowski" && is.null(p)) {
    stop("minkowski distance requires p (the order, >= 1)", call. = FALSE)
  }
  gower_kinds <- NULL
  gower_levels <- NULL
  if (kind == "gower") {
    if (is.null(columns)) {
      stop("gower distance requires columns ('numeric'/'categorical' per raw column)",
        call. = FALSE
      )
    }
    gower_kinds <- as.character(columns)
    gower_levels <- if (is.null(levels)) rep(0, length(gower_kinds)) else as.numeric(levels)
    if (length(gower_levels) != length(gower_kinds)) {
      stop("levels must match columns in length", call. = FALSE)
    }
  }
  if (kind == "mahalanobis") {
    if (is.null(precision)) {
      stop("mahalanobis distance requires the precision matrix", call. = FALSE)
    }
    precision <- as_design_matrix(precision)
  }
  payload <- list(type = kind)
  if (kind == "minkowski") {
    payload$p <- as.numeric(p)[[1]]
  }
  if (kind == "gower") {
    payload$columns <- lapply(seq_along(gower_kinds), function(i) {
      switch(gower_kinds[[i]],
        numeric = list(type = "numeric"),
        categorical = list(
          type = "categorical",
          levels = as.integer(gower_levels[[i]])
        ),
        stop(
          sprintf(
            "unknown gower column '%s': expected 'numeric' or 'categorical'",
            gower_kinds[[i]]
          ),
          call. = FALSE
        )
      )
    })
  }
  if (kind == "mahalanobis") {
    # Row-major list of rows: the core reads a square precision matrix.
    payload$precision <- lapply(seq_len(nrow(precision)), function(i) {
      as.numeric(precision[i, ])
    })
  }
  # Validated against the core here, so a bad order or a non-symmetric matrix
  # is reported where it was written rather than at fit.
  structure(as_payload(payload, "distance"), class = "addivortes_distance")
}

#' Soft cell-membership specification
#'
#' Selects the engine's softmax membership kernel (the membership extension
#' point's shelf): deterministic soft weights at fixed temperature `tau` instead of
#' hard nearest-centre assignment. Models fitted with soft membership
#' refuse JSON serialisation (the engine's rule for non-default selections).
#'
#' @param kernel Kernel name; only `"softmax"` is on the shelf.
#' @param tau Temperature (finite, > 0).
#' @return An `addivortes_membership` specification for [addivortes()].
#' @examples
#' avt_membership("softmax", tau = 2)
#' @export
avt_membership <- function(kernel = "softmax", tau) {
  payload <- list(type = kernel, tau = as.numeric(tau)[[1]])
  structure(as_payload(payload, "membership"), class = "addivortes_membership")
}

new_addivortes_fit <- function(ptr, feature_names = NULL, class_levels = NULL) {
  structure(
    list(ptr = ptr, feature_names = feature_names, class_levels = class_levels),
    class = "addivortes_fit"
  )
}

as_design_matrix <- function(x) {
  if (is.null(dim(x))) {
    x <- matrix(x, nrow = 1)
  }
  x <- as.matrix(x)
  if (!is.numeric(x)) {
    stop("x must be a numeric matrix", call. = FALSE)
  }
  storage.mode(x) <- "double"
  x
}

scalar_num <- function(value, name) {
  if (is.null(value)) {
    return(NULL)
  }
  value <- as.numeric(value)
  if (length(value) != 1 || is.na(value)) {
    stop(sprintf("%s must be a single non-missing number", name), call. = FALSE)
  }
  value
}

scalar_count <- function(value, name) {
  value <- scalar_num(value, name)
  if (is.null(value)) {
    return(NULL)
  }
  if (value < 1 || value != trunc(value)) {
    stop(sprintf("%s must be a whole number >= 1", name), call. = FALSE)
  }
  as.integer(value)
}

# A whole number >= 0, kept in R because R is where the user typed it: serde's
# own message for a negative seed is "invalid value: integer `-1`, expected u64",
# which never mentions which field was wrong.
scalar_index <- function(value, name) {
  value <- scalar_num(value, name)
  if (is.null(value)) {
    return(NULL)
  }
  if (value < 0 || value != trunc(value)) {
    stop(sprintf("%s must be a whole number >= 0", name), call. = FALSE)
  }
  as.integer(value)
}
