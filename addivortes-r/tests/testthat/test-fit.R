# Core fit/predict surface: shapes, determinism, families, validation.

test_that("fit, predict, and intervals have the right shapes", {
  d <- regression_data()
  fit <- small_fit()
  predictions <- predict(fit, d$x)
  expect_length(predictions, nrow(d$x))
  expect_true(all(is.finite(predictions)))
  # In-sample R^2 on a clean signal.
  expect_gt(1 - var(d$y - predictions) / var(d$y), 0.5)

  draws <- predict(fit, d$x, type = "draws")
  expect_equal(dim(draws), c(SMALL$draws, nrow(d$x)))
  # predict is the draw matrix's column mean (up to accumulation order:
  # colMeans sums in long double, the engine in f64).
  expect_equal(colMeans(draws), predictions, tolerance = 1e-12)

  quantiles <- predict(fit, d$x, type = "quantiles", probs = c(0.25, 0.75))
  expect_equal(dim(quantiles), c(nrow(d$x), 2L))
  expect_true(all(quantiles[, 1] <= quantiles[, 2]))

  interval <- prediction_interval(fit, d$x, level = 0.9)
  expect_equal(colnames(interval), c("lower", "upper"))
  expect_true(all(interval[, "lower"] <= interval[, "upper"]))

  credible <- credible_interval(fit, d$x, level = 0.9)
  # The mean-surface band sits inside the new-observation band.
  expect_true(all(credible[, "lower"] >= interval[, "lower"]))
  expect_true(all(credible[, "upper"] <= interval[, "upper"]))
})

test_that("fits are bit-for-bit deterministic in the seed", {
  d <- regression_data()
  a <- predict(small_fit(seed = 5), d$x)
  b <- predict(small_fit(seed = 5), d$x)
  expect_identical_bits(a, b)
  c <- predict(small_fit(seed = 6), d$x)
  expect_false(identical(hex_bits(a), hex_bits(c)))
})

test_that("chain 1 of fit_chains is bit-identical to a plain fit", {
  d <- regression_data()
  chains <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 3, chains = 3), SMALL)
  )
  expect_s3_class(chains, "addivortes_chains")
  expect_length(chains, 3)
  single <- small_fit(seed = 3)
  expect_identical_bits(predict(chains[[1]], d$x), predict(single, d$x))
  # Other chains genuinely differ.
  expect_false(identical(
    hex_bits(predict(chains[[2]], d$x)),
    hex_bits(predict(chains[[1]], d$x))
  ))
  # Stacked draws are chain-major.
  stacked <- predict(chains, d$x, type = "draws")
  expect_equal(nrow(stacked), 3 * SMALL$draws)
})

test_that("accessors expose the posterior and metadata", {
  fit <- small_fit()
  expect_length(sigma(fit), SMALL$draws)
  expect_true(all(sigma(fit) > 0))
  expect_length(total_cells(fit), SMALL$draws)
  expect_true(all(total_cells(fit) >= SMALL$m))
  expect_identical(fit$ptr$n_draws(), as.integer(SMALL$draws))
  expect_identical(fit$ptr$n_features(), 3L)
  expect_identical(fit$ptr$response_family(), "gaussian")
  expect_null(fit$ptr$t_df())

  importance <- variable_importance(fit)
  expect_length(importance, 3)
  expect_equal(sum(importance), 1, tolerance = 1e-12)
  expect_true(all(diff(importance) <= 0))
  expect_true(all(grepl("^x[0-9]+$", names(importance))))
})

test_that("feature names flow from the design matrix", {
  d <- regression_data()
  colnames(d$x) <- c("alpha", "beta", "gamma")
  fit <- do.call(addivortes, c(list(x = d$x, y = d$y, seed = 1), SMALL))
  expect_setequal(names(variable_importance(fit)), c("alpha", "beta", "gamma"))
})

test_that("response families fit and report themselves", {
  d <- regression_data(seed = 3)
  robust <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 6, response_family = "robust_t", t_df = 4), SMALL)
  )
  expect_identical(robust$ptr$response_family(), "robust_t")
  expect_identical(robust$ptr$t_df(), 4)

  labels <- as.numeric(d$y > stats::median(d$y))
  probit <- do.call(
    addivortes,
    c(list(x = d$x, y = labels, seed = 7, response_family = "binary_probit"), SMALL)
  )
  probabilities <- predict(probit, d$x)
  expect_true(all(probabilities >= 0 & probabilities <= 1))
  expect_gt(mean((probabilities >= 0.5) == labels), 0.7)
})

test_that("hyperparameter and data errors carry the engine's messages", {
  d <- regression_data()
  expect_error(
    do.call(addivortes, c(list(x = d$x, y = d$y, seed = 1, nu = -1), SMALL[-4])),
    "nu"
  )
  expect_error(
    addivortes(d$x, d$y, seed = 1, m = 8, burn_in = 15, draws = 25),
    "omega" # default omega = 3 needs > 3 features
  )
  expect_error(
    do.call(addivortes, c(list(x = d$x, y = d$y[-1], seed = 1), SMALL)),
    "length|rows"
  )
  expect_error(
    do.call(
      addivortes,
      c(list(x = d$x, y = d$y, seed = 1, response_family = "robust_t"), SMALL)
    ),
    "t_df"
  )
  expect_error(
    do.call(addivortes, c(list(x = d$x, y = d$y, seed = -1), SMALL)),
    "seed"
  )
  fit <- small_fit()
  expect_error(predict(fit, d$x[, 1:2]), "features")
})

test_that("metrics validate and fit", {
  d <- regression_data()
  fit <- do.call(addivortes, c(
    list(
      x = d$x, y = d$y, seed = 2,
      metrics = c("euclidean", "euclidean", "spherical")
    ),
    SMALL
  ))
  expect_length(predict(fit, d$x), nrow(d$x))
  expect_error(
    do.call(
      addivortes,
      c(list(x = d$x, y = d$y, seed = 2, metrics = c("bogus", "euclidean", "euclidean")), SMALL)
    ),
    "unknown metric"
  )
})

test_that("print and summary render", {
  fit <- small_fit()
  expect_output(print(fit), "AddiVortes fit \\(gaussian\\)")
  s <- summary(fit)
  expect_s3_class(s, "summary.addivortes_fit")
  expect_output(print(s), "Variable importance")
  chains <- do.call(
    addivortes,
    c(list(x = regression_data()$x, y = regression_data()$y, seed = 1, chains = 2), SMALL)
  )
  expect_output(print(chains), "2 chains")
})

test_that("chains methods pool every kept draw", {
  d <- regression_data()
  chains <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 5, chains = 3), SMALL)
  )
  expect_length(sigma(chains), 3 * SMALL$draws)
  expect_length(total_cells(chains), 3 * SMALL$draws)
  expect_identical_bits(sigma(chains)[seq_len(SMALL$draws)], sigma(chains[[1]]))

  importance <- variable_importance(chains)
  expect_equal(sum(importance), 1)
  expect_named(importance, names(variable_importance(chains[[1]])),
    ignore.order = TRUE
  )
})

test_that("chains summary reports between-chain convergence", {
  d <- regression_data()
  chains <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 9, chains = 3), SMALL)
  )
  s <- summary(chains)
  expect_s3_class(s, "summary.addivortes_chains")
  expect_identical(s$n_chains, 3L)
  expect_true(all(is.finite(s$sigma_diagnostics)))
  expect_identical(
    unname(s$sigma_diagnostics[["r_hat"]]),
    r_hat(lapply(chains, sigma))
  )
  expect_output(print(s), "Convergence")
})
