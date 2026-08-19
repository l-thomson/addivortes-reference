
# ---------------------------------------------------------------------------
# The config pass-through: the points no binding could reach before.
# ---------------------------------------------------------------------------

test_that("full f64 precision survives the JSON hand-off", {
  # jsonlite's DEFAULT is digits = 4, which would round every double on its way
  # to the engine and silently break the bit-exact reproducibility contract.
  # `spec_json()` pins digits = NA. This test is the guard on that.
  value <- 0.123456789012345
  json <- addivortesr:::spec_json(list(seed = 0, sigma_c = value))
  expect_match(json, "0.123456789012345", fixed = TRUE)

  round_tripped <- jsonlite::fromJSON(json)$sigma_c
  expect_identical(round_tripped, value)
})

test_that("each extension point is reachable as a payload", {
  set.seed(1)
  x <- matrix(stats::runif(80), 40, 2)
  y <- 2 * x[, 1] - x[, 2] + stats::rnorm(40, sd = 0.1)
  small <- list(m = 8, burn_in = 10, draws = 10, omega = 1.5, seed = 3)

  selections <- list(
    list(inclusion = list(type = "dart", alpha = 0.5)),
    list(inclusion = list(type = "weighted", weights = c(1, 3))),
    list(scale = list(type = "pinned", sigma_sq = 1)),
    list(scale = list(type = "h_variance", m_prime = 5)),
    list(count_priors = list(type = "shifted_poisson_binomial")),
    list(basis = list(type = "linear", columns = 0, sigma_beta_sq = 0.1)),
    list(membership = list(type = "softmax", tau = 0.2)),
    list(coords = list(
      list(type = "euclidean_normal", sigma_c = 0.8),
      list(type = "euclidean_normal", sigma_c = 0.8)
    )),
    list(moves = list(
      list(name = "add_centre", weight = 0.3),
      list(name = "remove_centre", weight = 0.3),
      list(name = "change", weight = 0.4)
    ))
  )

  for (selection in selections) {
    fit <- do.call(addivortes, c(list(x = x, y = y), small, selection))
    expect_true(all(is.finite(predict(fit, x))), info = names(selection))
  }
})

test_that("a bad dart alpha is an error, not a process abort", {
  # A non-positive alpha is rejected under the spec key before any constructor
  # sees it.
  expect_error(
    addivortes(matrix(stats::runif(20), 10, 2), stats::runif(10),
      seed = 1, inclusion = list(type = "dart", alpha = 0)
    ),
    "alpha"
  )
})

test_that("a wrong-length setting fails at fit, naming the covariate count", {
  x <- matrix(stats::runif(40), 20, 2)
  expect_error(
    addivortes(x, stats::runif(20),
      seed = 1, m = 4, burn_in = 5, draws = 5, omega = 1.5,
      inclusion = list(type = "weighted", weights = c(1))
    ),
    "p = 2"
  )
})
