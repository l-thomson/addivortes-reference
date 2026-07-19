#!/usr/bin/env Rscript
# The SBC uniformity verdict: the ECDF simultaneous-confidence-band
# test of Säilynoja, Bürkner & Vehtari (2022, Statistics and Computing 32:32),
# applied to the rank CSVs the Rust battery emits (src/stat_gates.rs).
#
# This procedure is deliberately not ported to Rust. The band computation is
# delegated to the method authors' own reference implementation, maintained
# inside the `bayesplot` package (adjust_gamma / ecdf_intervals in
# R/helpers-ppc.R; Säilynoja is a bayesplot author): a merge-blocking
# validator must be more trusted than the code it judges, not an
# equally-novel hand port. Those helpers are internal (:::), so the package
# version is PINNED below and --self-test re-validates the critical values'
# defining coverage property on every run before anything real is judged.
#
# Usage:
#   Rscript ci/sbc-ecdf-check.R --self-test
#   Rscript ci/sbc-ecdf-check.R [--alpha 0.05] [--summary out.md] \
#       [--expect-fail bad.csv] good1.csv good2.csv ...
#
# CSV contract (per file): header `quantity,rank,n_draws`; one row per SBC
# replication and quantity; ranks in 0..n_draws. The band's per-quantity
# coverage is Bonferroni-split over every quantity judged in the invocation,
# so adding files tightens (never loosens) each individual band.
#
# Exit status: 0 iff every non-flagged file passes every quantity AND every
# --expect-fail file fails at least one quantity (the injection teeth-check).

PINNED_BAYESPLOT <- "1.15.0"

# --- pinned-environment guard ------------------------------------------------
if (!requireNamespace("bayesplot", quietly = TRUE)) {
  stop("bayesplot is not installed: the calibration workflow provisions it ",
       "from the date-pinned Posit snapshot (see .github/workflows/calibration.yml)")
}
found <- as.character(utils::packageVersion("bayesplot"))
if (found != PINNED_BAYESPLOT) {
  stop(sprintf(
    "bayesplot %s found but %s is pinned: the band computation calls internal",
    found, PINNED_BAYESPLOT),
    " helpers (bayesplot:::adjust_gamma / :::ecdf_intervals), so the version ",
    "is part of the gate's provenance. Bump the pin deliberately, regenerate ",
    "GAMMA_ANCHOR, and re-run --self-test (see this script's header)")
}
adjust_gamma  <- get("adjust_gamma",  envir = asNamespace("bayesplot"))
ecdf_intervals <- get("ecdf_intervals", envir = asNamespace("bayesplot"))

# --- the band test -----------------------------------------------------------

# Scaled-ECDF counts of SBC ranks (0..L) at the K = L+1 evaluation points
# z_k = k/K. With u = (rank+1)/(L+1), P(u <= z_k) = z_k EXACTLY under the
# null, so the binomial band is exact at every evaluation point (no
# discreteness slack, which K != L+1 grids would introduce).
ecdf_counts <- function(ranks, n_draws) {
  k_points <- n_draws + 1
  cumsum(tabulate(ranks + 1, nbins = k_points))
}

# One quantity's verdict: counts within [lower, upper] at every z_k.
band_check <- function(ranks, n_draws, prob) {
  n <- length(ranks)
  k_points <- n_draws + 1
  gamma <- adjust_gamma(N = n, L = 1, K = k_points, prob = prob,
                        interpolate_adj = FALSE)
  lims <- ecdf_intervals(gamma = gamma, N = n, K = k_points, L = 1)
  counts <- ecdf_counts(ranks, n_draws)
  # lims run over z = 0/K .. K/K (K+1 values); counts run over z_1..z_K.
  lower <- lims$lower[-1]
  upper <- lims$upper[-1]
  violations <- sum(counts < lower | counts > upper)
  list(pass = violations == 0, violations = violations, gamma = gamma)
}

judge_file <- function(path, prob) {
  data <- utils::read.csv(path, stringsAsFactors = FALSE)
  stopifnot(all(c("quantity", "rank", "n_draws") %in% names(data)))
  results <- lapply(split(data, data$quantity), function(rows) {
    n_draws <- unique(rows$n_draws)
    stopifnot(length(n_draws) == 1)
    stopifnot(all(rows$rank >= 0 & rows$rank <= n_draws))
    band_check(rows$rank, n_draws, prob)
  })
  results
}

# --- self-test: trust the band before it gates anything -----------------------

self_test <- function() {
  n <- 300; n_draws <- 99; k_points <- n_draws + 1

  # 1. Critical-value regression anchor: the adjusted gamma for the canonical
  #    calibration gate shape, computed by the reference implementation's exact
  #    optimisation method. Pinned so a silent behaviour change in the
  #    internal helpers (a version bump that slips past the pin) turns red.
  gamma <- adjust_gamma(N = n, L = 1, K = k_points, prob = 0.99,
                        interpolate_adj = FALSE)
  cat(sprintf("self-test: adjust_gamma(N=300, K=100, prob=0.99) = %.10f\n", gamma))
  expected_gamma <- GAMMA_ANCHOR
  if (abs(gamma - expected_gamma) > 1e-8) {
    stop(sprintf("gamma regression anchor drifted: %.10f != %.10f", gamma,
                 expected_gamma))
  }

  # 2. The defining property of the critical values (Säilynoja et al. §2:
  #    simultaneous coverage 1−alpha under the uniform null), reproduced by
  #    Monte Carlo: the observed rejection rate over M null rank sets must be
  #    statistically consistent with alpha. This is the check that makes the
  #    band trustworthy BEFORE it gates anything real.
  set.seed(20260702)
  alpha <- 0.05
  m_sims <- 2000
  # The band is a fixed object for fixed (N, K, prob): compute once, then
  # count Monte Carlo violations against it.
  gamma_mc <- adjust_gamma(N = n, L = 1, K = k_points, prob = 1 - alpha,
                           interpolate_adj = FALSE)
  lims_mc <- ecdf_intervals(gamma = gamma_mc, N = n, K = k_points, L = 1)
  lower_mc <- lims_mc$lower[-1]
  upper_mc <- lims_mc$upper[-1]
  rejections <- 0L
  for (sim in seq_len(m_sims)) {
    ranks <- as.integer(floor(stats::runif(n) * k_points))
    counts <- ecdf_counts(ranks, n_draws)
    if (any(counts < lower_mc | counts > upper_mc)) {
      rejections <- rejections + 1L
    }
  }
  rate <- rejections / m_sims
  tolerance <- 3 * sqrt(alpha * (1 - alpha) / m_sims)
  cat(sprintf("self-test: null rejection rate %.4f (target %.3f +/- %.4f)\n",
              rate, alpha, tolerance))
  if (abs(rate - alpha) > tolerance) {
    stop("uniform-null coverage does not reproduce the published critical values")
  }

  # 3. Power sanity: an over-dispersed rank set (posterior too narrow; the
  #    classic broken-sampler signature) must be rejected.
  set.seed(1)
  u <- stats::rbeta(n, 0.5, 0.5)
  biased <- as.integer(floor(u * k_points))
  if (band_check(biased, n_draws, prob = 1 - alpha)$pass) {
    stop("the band failed to reject an over-dispersed fixture: no power")
  }
  cat("self-test: over-dispersed fixture rejected as expected\n")
  cat("self-test: PASS\n")
}

# Anchor for step 1 (computed by this script once at wiring time, 2026-07-02,
# bayesplot 1.15.0; regenerate deliberately if the pin is ever bumped).
GAMMA_ANCHOR <- 0.0005572809

# --- CLI ----------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("no arguments: pass --self-test or rank CSV paths")
}
if (identical(args, "--self-test")) {
  self_test()
  quit(status = 0)
}

alpha <- 0.05
summary_path <- NULL
expect_fail <- character()
files <- character()
i <- 1
while (i <= length(args)) {
  if (args[i] == "--alpha") {
    alpha <- as.numeric(args[i + 1]); i <- i + 2
  } else if (args[i] == "--summary") {
    summary_path <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--expect-fail") {
    expect_fail <- c(expect_fail, args[i + 1]); i <- i + 2
  } else {
    files <- c(files, args[i]); i <- i + 1
  }
}
all_files <- c(files, expect_fail)
if (length(all_files) == 0) stop("no rank CSVs given")

# Always re-validate the band before judging anything real.
self_test()

# Bonferroni across every quantity judged in this invocation.
n_tests <- sum(vapply(all_files, function(path) {
  length(unique(utils::read.csv(path)$quantity))
}, integer(1)))
prob <- 1 - alpha / n_tests
cat(sprintf("judging %d quantities across %d files at per-quantity band prob %.6f\n",
            n_tests, length(all_files), prob))

lines <- c("### SBC ECDF-band verdict (Säilynoja–Bürkner–Vehtari via bayesplot)", "")
ok <- TRUE
for (path in all_files) {
  flagged <- path %in% expect_fail
  results <- judge_file(path, prob)
  failed <- names(results)[!vapply(results, `[[`, logical(1), "pass")]
  for (quantity in names(results)) {
    r <- results[[quantity]]
    cat(sprintf("%s %s %s: %s (violations=%d)\n",
                if (r$pass) "PASS" else "FAIL",
                basename(path), quantity,
                if (r$pass) "rank ECDF inside the simultaneous band"
                else "rank ECDF LEAVES the simultaneous band",
                r$violations))
  }
  if (flagged) {
    if (length(failed) == 0) {
      ok <- FALSE
      lines <- c(lines, sprintf(
        "- **%s**: :rotating_light: injection fixture PASSED the band: the gate has no teeth",
        basename(path)))
    } else {
      lines <- c(lines, sprintf(
        "- **%s**: injection correctly rejected (%s): teeth confirmed",
        basename(path), paste(failed, collapse = ", ")))
    }
  } else if (length(failed) > 0) {
    ok <- FALSE
    lines <- c(lines, sprintf(
      "- **%s**: :x: rank uniformity REJECTED for %s: the sampler does not recover its own prior for this config",
      basename(path), paste(failed, collapse = ", ")))
  } else {
    lines <- c(lines, sprintf("- **%s**: :white_check_mark: all quantities inside the band",
                              basename(path)))
  }
}
lines <- c(lines, "", sprintf(
  "Per-quantity band coverage %.6f (alpha %.3f Bonferroni over %d quantities); bayesplot %s pinned.",
  prob, alpha, n_tests, PINNED_BAYESPLOT))

if (!is.null(summary_path)) {
  writeLines(lines, summary_path)
}
cat(paste(lines, collapse = "\n"), "\n")
if (!ok) quit(status = 1)
