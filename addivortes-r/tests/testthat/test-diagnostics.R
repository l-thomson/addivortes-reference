# The engine's own diagnostics across the FFI.

test_that("r_hat and ess run on chain traces", {
  d <- regression_data()
  chains <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 2, chains = 3), SMALL)
  )
  traces <- lapply(chains, sigma)
  expect_true(is.finite(r_hat(traces)))
  expect_gt(ess_bulk(traces), 0)
  expect_gt(ess_tail(traces), 0)
})

test_that("predictive_qq returns sorted PIT values and validates shapes", {
  d <- regression_data()
  fit <- small_fit()
  fit_draws <- predict(fit, d$x, type = "draws")
  s_draws <- matrix(sigma(fit), nrow(fit_draws), ncol(fit_draws))
  pit <- predictive_qq(d$y, fit_draws, s_draws)
  expect_length(pit, nrow(d$x))
  expect_true(!is.unsorted(pit))
  expect_true(all(pit >= 0 & pit <= 1))

  expect_error(predictive_qq(d$y, fit_draws, s_draws[, -1]), "equal")
  expect_error(predictive_qq(d$y[-1], fit_draws, s_draws), "observations")
  bad <- s_draws
  bad[1, 1] <- -1
  expect_error(predictive_qq(d$y, fit_draws, bad), "positive")
})

test_that("a bad chain list is an R error, never a panic across the boundary", {
  # The crate asserts this contract, which is right for a Rust caller and wrong
  # here: a panic is not R's error mechanism. Each must arrive catchable.
  one <- list(rnorm(50))
  expect_error(r_hat(one), "at least 2")
  expect_error(ess_bulk(one), "at least 2")
  expect_error(ess_tail(one), "at least 2")

  expect_error(ess_bulk(list(rnorm(50), rnorm(40))), "same number of draws")
  expect_error(ess_bulk(list(rnorm(3), rnorm(3))), "at least 4 draws")
})
