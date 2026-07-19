# Shared fixtures. SMALL keeps every fit fast; omega = 1.5 keeps narrow
# designs inside the engine's omega < n_features rule.
SMALL <- list(m = 8, burn_in = 15, draws = 25, omega = 1.5)

regression_data <- function(n = 50, seed = 1) {
  set.seed(seed)
  x <- matrix(stats::runif(n * 3), n, 3)
  y <- 4 * x[, 1] - 2 * x[, 2] + stats::rnorm(n, sd = 0.2)
  list(x = x, y = y)
}

small_fit <- function(seed = 1, data_seed = 1, ...) {
  d <- regression_data(seed = data_seed)
  do.call(addivortes, c(list(x = d$x, y = d$y, seed = seed), SMALL, list(...)))
}

# The exact bit pattern of each double, as python's struct '<Q' hex:
# the golden-vector encoding (rev() turns little-endian bytes into the
# big-endian hex string format('016x') produces).
hex_bits <- function(values) {
  paste(vapply(values, function(value) {
    bytes <- writeBin(as.numeric(value), raw(), size = 8L, endian = "little")
    paste(rev(sprintf("%02x", as.integer(bytes))), collapse = "")
  }, character(1)), collapse = ",")
}

expect_identical_bits <- function(a, b) {
  expect_identical(hex_bits(a), hex_bits(b))
}
