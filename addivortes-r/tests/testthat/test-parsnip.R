# The parsnip adapter (the scikit-learn analogue).

test_that("addivortes_reg fits and predicts through parsnip", {
  skip_if_not_installed("parsnip")
  d <- regression_data()
  train <- as.data.frame(d$x)
  names(train) <- c("a", "b", "c")
  train$y <- d$y

  spec <- parsnip::set_engine(
    addivortes_reg(trees = 8),
    "addivortes",
    seed = 1, burn_in = 15, draws = 25, omega = 1.5
  )
  fitted <- parsnip::fit(spec, y ~ ., data = train)
  predictions <- predict(fitted, train)
  expect_named(predictions, ".pred")
  expect_equal(nrow(predictions), nrow(train))
  # The engine underneath is a plain addivortes_fit.
  expect_s3_class(fitted$fit, "addivortes_fit")
  expect_identical(fitted$fit$ptr$n_draws(), 25L)
})

test_that("addivortes_class fits factors and yields class + prob", {
  skip_if_not_installed("parsnip")
  d <- regression_data(seed = 4)
  train <- as.data.frame(d$x)
  names(train) <- c("a", "b", "c")
  train$y <- factor(ifelse(d$y > stats::median(d$y), "pos", "neg"))

  spec <- parsnip::set_engine(
    addivortes_class(trees = 8),
    "addivortes",
    seed = 2, burn_in = 15, draws = 25, omega = 1.5
  )
  fitted <- parsnip::fit(spec, y ~ ., data = train)

  classes <- predict(fitted, train, type = "class")
  expect_true(all(classes$.pred_class %in% c("neg", "pos")))
  expect_gt(mean(classes$.pred_class == train$y), 0.7)

  probabilities <- predict(fitted, train, type = "prob")
  expect_named(probabilities, c(".pred_neg", ".pred_pos"))
  expect_equal(
    probabilities$.pred_neg + probabilities$.pred_pos,
    rep(1, nrow(train)),
    tolerance = 1e-12
  )
})

test_that("classification rejects more than two levels", {
  skip_if_not_installed("parsnip")
  x <- matrix(stats::runif(90), 30, 3)
  y <- factor(rep(c("a", "b", "c"), 10))
  expect_error(
    parsnip_addivortes_fit_class(x, y, seed = 1, m = 8, burn_in = 15, draws = 25, omega = 1.5),
    "binary"
  )
})
