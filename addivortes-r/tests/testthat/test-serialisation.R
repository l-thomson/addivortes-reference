# Persistence through the engine's validated JSON format.

test_that("JSON round trip is bit-identical", {
  d <- regression_data()
  fit <- small_fit(seed = 4)
  clone <- avt_from_json(avt_to_json(fit))
  expect_identical_bits(predict(clone, d$x), predict(fit, d$x))
  expect_identical(clone$ptr$n_features(), fit$ptr$n_features())
  expect_identical(clone$ptr$response_family(), "gaussian")
})

test_that("save/load round trip through a file", {
  d <- regression_data()
  fit <- small_fit(seed = 5)
  path <- tempfile(fileext = ".json")
  on.exit(unlink(path), add = TRUE)
  avt_save(fit, path)
  loaded <- avt_load(path)
  expect_identical_bits(predict(loaded, d$x), predict(fit, d$x))
})

test_that("corrupt payloads error instead of crashing", {
  expect_error(avt_from_json("{not json"))
  fit <- small_fit()
  payload <- avt_to_json(fit)
  expect_error(avt_from_json(sub("\"sigma", "\"sygma", payload, fixed = TRUE)))
})

test_that("non-default selections refuse serialisation with the engine's message", {
  d <- regression_data()
  soft <- do.call(
    addivortes,
    c(
      list(x = d$x, y = d$y, seed = 1, membership = avt_membership("softmax", tau = 2)),
      SMALL
    )
  )
  expect_error(avt_to_json(soft), "serialis|custom|membership")
  manhattan <- small_fit(distance = "manhattan")
  expect_error(avt_to_json(manhattan), "serialis|custom|distance")
})
