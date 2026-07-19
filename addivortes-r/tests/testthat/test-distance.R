# The distance shelf and soft membership across the FFI.

test_that("plain distance kinds fit, as strings or specifications", {
  d <- regression_data()
  for (kind in c("euclidean", "manhattan", "cosine")) {
    fit <- do.call(
      addivortes,
      c(list(x = d$x, y = d$y, seed = 1, distance = kind), SMALL)
    )
    expect_length(predict(fit, d$x), nrow(d$x))
  }
  # The default (per-column) geometry equals explicit all-Euclidean.
  base <- small_fit(seed = 4)
  euclid <- small_fit(seed = 4, distance = "euclidean")
  expect_identical_bits(
    predict(base, d$x),
    predict(euclid, d$x)
  )
})

test_that("minkowski takes an order and validates it", {
  d <- regression_data()
  fit <- do.call(
    addivortes,
    c(list(x = d$x, y = d$y, seed = 2, distance = avt_distance("minkowski", p = 3)), SMALL)
  )
  expect_length(predict(fit, d$x), nrow(d$x))
  expect_error(avt_distance("minkowski"), "requires p")
  expect_error(
    do.call(
      addivortes,
      c(list(x = d$x, y = d$y, seed = 2, distance = avt_distance("minkowski", p = 0.5)), SMALL)
    ),
    "p"
  )
})

test_that("gower handles mixed columns and validates the declaration", {
  set.seed(9)
  n <- 50
  x <- cbind(stats::runif(n), sample(0:2, n, replace = TRUE))
  y <- x[, 1] * 2 + (x[, 2] == 1) + stats::rnorm(n, sd = 0.2)
  spec <- avt_distance("gower",
    columns = c("numeric", "categorical"), levels = c(0, 3)
  )
  fit <- do.call(
    addivortes,
    c(
      list(
        x = x, y = y, seed = 3, distance = spec,
        metrics = c("euclidean", "categorical")
      ),
      SMALL
    )
  )
  expect_length(predict(fit, x), n)
  expect_error(avt_distance("gower"), "requires columns")
  expect_error(
    avt_distance("gower", columns = c("numeric"), levels = c(0, 1)),
    "match columns"
  )
  # A categorical column with no levels is caught where it is written, not
  # at fit, where it would surface only as a non-finite distance -- a true
  # statement about a symptom, not about the mistake.
  expect_error(
    avt_distance("gower", columns = c("numeric", "categorical"), levels = c(0, 0)),
    "levels"
  )
})

test_that("mahalanobis validates the precision matrix in the engine", {
  d <- regression_data()
  fit <- do.call(
    addivortes,
    c(
      list(
        x = d$x, y = d$y, seed = 4,
        distance = avt_distance("mahalanobis", precision = diag(3))
      ),
      SMALL
    )
  )
  expect_length(predict(fit, d$x), nrow(d$x))

  asymmetric <- diag(3)
  asymmetric[1, 2] <- 0.5
  expect_error(
    do.call(
      addivortes,
      c(
        list(
          x = d$x, y = d$y, seed = 4,
          distance = avt_distance("mahalanobis", precision = asymmetric)
        ),
        SMALL
      )
    ),
    "symmetric"
  )
  expect_error(avt_distance("mahalanobis"), "requires the precision")
})

test_that("softmax membership fits and validates tau", {
  d <- regression_data()
  fit <- do.call(
    addivortes,
    c(
      list(x = d$x, y = d$y, seed = 5, membership = avt_membership("softmax", tau = 2)),
      SMALL
    )
  )
  expect_length(predict(fit, d$x), nrow(d$x))
  # Soft membership genuinely changes the fit.
  expect_false(identical(
    hex_bits(predict(fit, d$x)),
    hex_bits(predict(small_fit(seed = 5), d$x))
  ))
  expect_error(
    do.call(
      addivortes,
      c(list(x = d$x, y = d$y, seed = 5, membership = avt_membership("softmax", tau = -1)), SMALL)
    ),
    "tau"
  )
  expect_error(avt_membership("gaussian", tau = 1))
})

test_that("distance argument rejects junk", {
  d <- regression_data()
  expect_error(
    do.call(addivortes, c(list(x = d$x, y = d$y, seed = 1, distance = 42), SMALL)),
    "avt_distance"
  )
  expect_error(avt_distance("hyperbolic"))
})
