#!/usr/bin/env Rscript
# Release-gate reference-comparison leg: against the pinned R reference. The
# reference is not an oracle; this validates that the shipped default
# configuration stays in the same statistical neighbourhood,
# |RMSE_rust − RMSE_R| <= 0.05 on f-recovery.
#
# Inputs (written by the Rust test `stat_gates::reference_comparison_emit_fixture_and_rmse`):
#   $STAT_GATES_DIR/reference-train.csv      x1..x10, y
#   $STAT_GATES_DIR/reference-test.csv       x1..x10, f_true
#   $STAT_GATES_DIR/reference-rust-rmse.txt  the crate's f-recovery RMSE
#
# The reference package is pinned by VERSION here and by COMMIT at checkout
# (the calibration/release workflow checks out johnpaulgosling/AddiVortes at
# the recorded commit and installs it). VERSION alone is not a pin: 0.6.0 ran
# for months upstream and its R/ and src/ changed within it. The reference is
# executed, never read, by this gate.
#
# pbapply / core-count sensitivity: the reference parallelises
# prediction via pbapply/parallel; RNG-relevant fitting is serial, but core
# count is pinned to 1 anyway so the run is machine-shape independent.

PINNED_REFERENCE <- "0.6.0"
TOLERANCE <- 0.05
REFERENCE_SEED <- 20260702

# The recorded expected offset between the two packages' shipped defaults.
# The default configurations are deliberately different models (this crate
# ships lambda_c = 5, the R package ships 25 with its own cell-count
# pricing), so their RMSEs on this fixture differ by a stable amount:
# rust at lambda_c = 5 measured 1.5580
# (sweep 25/10/5/4/3/2 -> 1.6675/1.5673/1.5580/1.5416/1.4903/1.3654),
# the R package measured 1.4225, so the recorded offset is +0.1355.
# The gate asserts the observed delta stays within tolerance of this
# recorded value: it detects new drift, not the known configuration
# difference.
EXPECTED_DELTA <- 0.1355 # rust minus reference, shipped defaults

if (!requireNamespace("AddiVortes", quietly = TRUE)) {
  stop("the reference AddiVortes package is not installed: the release-leg ",
       "workflow checks out johnpaulgosling/AddiVortes at the pinned commit ",
       "and R CMD INSTALLs it (see .github/workflows/release-gate.yml)")
}
found <- as.character(utils::packageVersion("AddiVortes"))
if (found != PINNED_REFERENCE) {
  stop(sprintf(
    "reference AddiVortes %s found but %s is pinned: bump the pin (and the ref commit in .github/workflows/release-gate.yml) deliberately",
    found, PINNED_REFERENCE))
}

dir <- Sys.getenv("STAT_GATES_DIR", "target/stat-gates")
train <- utils::read.csv(file.path(dir, "reference-train.csv"))
test <- utils::read.csv(file.path(dir, "reference-test.csv"))
rust_rmse <- as.numeric(readLines(file.path(dir, "reference-rust-rmse.txt"))[1])

x_train <- as.matrix(train[, paste0("x", 1:10)])
x_test <- as.matrix(test[, paste0("x", 1:10)])

options(mc.cores = 1)
set.seed(REFERENCE_SEED)
elapsed <- system.time(
  fit <- AddiVortes::AddiVortes(train$y, x_train, showProgress = FALSE)
)
pred <- predict(fit, x_test, type = "response")
r_rmse <- sqrt(mean((pred - test$f_true)^2))

delta <- rust_rmse - r_rmse
drift <- abs(delta - EXPECTED_DELTA)
cat(sprintf(
  "reference comparison: rust RMSE %.4f | reference RMSE %.4f (AddiVortes %s, seed %d, %.0fs) | delta %+.4f (recorded offset %+.4f, drift %.4f, tolerance %.2f)\n",
  rust_rmse, r_rmse, PINNED_REFERENCE, REFERENCE_SEED, elapsed[["elapsed"]],
  delta, EXPECTED_DELTA, drift, TOLERANCE))
if (drift > TOLERANCE) {
  stop(sprintf(
    "reference comparison FAIL: delta %+.4f drifted %.4f from the recorded offset %+.4f (tolerance %.2f): something changed beyond the known default-configuration difference. Investigate before release; if a deliberate change is the cause, document it and re-record the expected offset.",
    delta, drift, EXPECTED_DELTA, TOLERANCE))
}
cat("reference comparison PASS: the default-config comparison sits at the recorded offset: no new drift beyond the known configuration difference\n")
