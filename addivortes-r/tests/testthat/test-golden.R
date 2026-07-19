# Cross-language bit-identity: the R binding reproduces the SAME
# per-target golden predict vector the Rust tests pin (tests/golden/), bit
# for bit. f64s cross the FFI unchanged, so any drift here is a real chain
# or predict-surface change: the same tripwire, now spanning a second
# boundary.

target_tag <- function() {
  machine <- Sys.info()[["machine"]]
  arch <- switch(machine,
    "AMD64" = "x86_64",
    "arm64" = "aarch64",
    machine
  )
  os <- switch(Sys.info()[["sysname"]],
    "Linux" = "linux",
    "Darwin" = "macos",
    "Windows" = "windows",
    tolower(Sys.info()[["sysname"]])
  )
  paste0(arch, "-", os)
}

# The exact fixture of tests/golden_chain.rs (arithmetic, no RNG).
golden_fixture <- function() {
  n <- 12
  i <- 0:(n - 1)
  a <- i / (n - 1)
  b <- ((i * 7) %% n) / n
  list(x = cbind(a, b), y = 2 * a - 1.5 * b + 0.25 * a * b)
}

test_that("R reproduces the Rust golden predict vector", {
  vector_file <- file.path(
    test_path(), "..", "..", "..", "tests", "golden",
    paste0("predict-", target_tag(), ".txt")
  )
  skip_if_not(
    file.exists(vector_file),
    sprintf("no golden vector for target %s", target_tag())
  )

  fixture <- golden_fixture()
  fit <- addivortes(fixture$x, fixture$y,
    seed = 777, m = 4, omega = 1.5, burn_in = 5, draws = 10
  )

  new_x <- matrix(c(0.1, 0.9, 0.5, 0.5, 0.95, 0.05), ncol = 2, byrow = TRUE)
  predictions <- predict(fit, new_x)
  quantiles <- predict(fit, new_x, type = "quantiles", probs = c(0.25, 0.5, 0.75))

  lines <- readLines(vector_file)
  expected <- list()
  for (line in lines) {
    split_at <- regexpr(" ", line)
    expected[[substr(line, 1, split_at - 1)]] <-
      substr(line, split_at + 1, nchar(line))
  }

  expect_identical(hex_bits(predictions), expected$predict)
  # The file stores quantiles row-major (row = observation).
  expect_identical(hex_bits(as.vector(t(quantiles))), expected$quantiles)
  expect_identical(hex_bits(fit$ptr$in_sample_rmse()), expected$in_sample_rmse)
})
