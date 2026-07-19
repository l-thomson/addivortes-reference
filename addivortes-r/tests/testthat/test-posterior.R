# The posterior/loo/bayesplot adapter (the ArviZ analogue).

test_that("as_draws_df exports single fits and chains", {
  skip_if_not_installed("posterior")
  d <- regression_data()
  fit <- small_fit()
  draws <- posterior::as_draws_df(fit)
  expect_s3_class(draws, "draws_df")
  expect_setequal(posterior::variables(draws), c("sigma", "total_cells"))
  expect_equal(posterior::ndraws(draws), SMALL$draws)

  chains <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 2, chains = 3), SMALL)
  )
  multi <- posterior::as_draws_df(chains)
  expect_equal(posterior::nchains(multi), 3)
  expect_equal(posterior::ndraws(multi), 3 * SMALL$draws)
  s <- posterior::summarise_draws(multi)
  expect_true(all(is.finite(s$rhat)))
})

test_that("mismatched chains are rejected", {
  skip_if_not_installed("posterior")
  d <- regression_data()
  a <- small_fit(seed = 3)
  b <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 3), modifyList(SMALL, list(draws = 10)))
  )
  expect_error(
    posterior::as_draws_df(structure(list(a, b), class = "addivortes_chains")),
    "draw counts"
  )
})

test_that("loo and waic run on the log-likelihood export", {
  skip_if_not_installed("loo")
  d <- regression_data(seed = 4)
  chains <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 5, chains = 2), SMALL)
  )
  ll <- log_likelihood(chains, d$x, d$y)
  expect_equal(dim(ll), c(2 * SMALL$draws, nrow(d$x)))
  expect_true(all(is.finite(ll)))

  loo_result <- loo::loo(chains, newdata = d$x, y = d$y)
  expect_s3_class(loo_result, "loo")
  expect_true(is.finite(loo_result$estimates["elpd_loo", "Estimate"]))

  waic_result <- loo::waic(chains, newdata = d$x, y = d$y)
  expect_true(is.finite(waic_result$estimates["elpd_waic", "Estimate"]))

  # Single-fit method too.
  single <- small_fit(seed = 5, data_seed = 4)
  expect_s3_class(loo::loo(single, newdata = d$x, y = d$y), "loo")
})

test_that("probit log-likelihood feeds loo", {
  skip_if_not_installed("loo")
  d <- regression_data(seed = 6)
  labels <- as.numeric(d$y > stats::median(d$y))
  fit <- do.call(
    addivortes,
    c(list(x = d$x, y = labels, seed = 7, response_family = "binary_probit"), SMALL)
  )
  result <- loo::loo(fit, newdata = d$x, y = labels)
  expect_true(is.finite(result$estimates["elpd_loo", "Estimate"]))
})

test_that("predictive_draws yields family-correct replicates", {
  d <- regression_data(seed = 7)
  fit <- small_fit(seed = 8, data_seed = 7)
  set.seed(1)
  yrep <- predictive_draws(fit, d$x)
  expect_equal(dim(yrep), c(SMALL$draws, nrow(d$x)))
  # Reproducible under R's RNG.
  set.seed(1)
  expect_identical(yrep, predictive_draws(fit, d$x))

  labels <- as.numeric(d$y > stats::median(d$y))
  probit <- do.call(
    addivortes,
    c(list(x = d$x, y = labels, seed = 9, response_family = "binary_probit"), SMALL)
  )
  reps <- predictive_draws(probit, d$x)
  expect_true(all(reps %in% c(0, 1)))

  chains <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 8, chains = 2), SMALL)
  )
  expect_equal(nrow(predictive_draws(chains, d$x)), 2 * SMALL$draws)
})
