# addivortesr: R bindings for the addivortes crate

An R package over the same Rust engine as the crate and the Python
binding: Bayesian additive Voronoi tessellation regression
(Stone & Gosling 2025, JCGS). The crate's shelf is selectable from R, and
the full posterior is available.

> **Naming note.** The method's original authors publish their own R
> implementation as `AddiVortes`
> (github.com/johnpaulgosling/AddiVortes). This package is NOT that
> implementation; it wraps the `addivortes` Rust crate. The package name
> `addivortesr` is provisional.

## What you get

- `addivortes(x, y, seed, ...)`: fit with the engine's defaults (the
  defaults live in Rust, in exactly one place); `chains = n` for
  independent chains, chain 1 bit-identical to a single fit.
- Shelf selection by value: `response_family` (`gaussian`, `robust_t`,
  `binary_probit`), `metrics`, `distance` (`avt_distance()` for
  minkowski/gower/mahalanobis), `membership` (`avt_membership("softmax",
  tau)`). Authoring new components stays a Rust-side activity.
- Posterior access: `predict(fit, x, type = "draws")` (per-draw
  fits), `log_likelihood(fit, x, y)` (pointwise, LOO/WAIC-ready),
  `sigma(fit)`, `total_cells(fit)`, `variable_importance(fit)`,
  quantiles and posterior-predictive / credible intervals.
- **posterior/loo/bayesplot adapters** (Suggests): `as_draws_df(fit)`,
  `loo(fit, newdata, y)`, `waic(...)` (methods register when those
  packages load), `predictive_draws(fit, x)` for `ppc_*` yrep matrices.
- **parsnip engines** (Suggests): `addivortes_reg()` /
  `addivortes_class()` with `trees` tunable and every engine
  hyperparameter through `set_engine("addivortes", ...)`.
- Persistence through the crate's validated JSON format (`avt_save` /
  `avt_load` / `avt_to_json` / `avt_from_json`), interchangeable with
  Rust and Python, bit-identical predictions after reload. `saveRDS` does
  not work (external pointer); models with custom distance or soft
  membership refuse serialisation, with the engine's message.

The bit-exact reproducibility contract crosses this FFI too: the testthat
suite pins an R fit against the same per-target golden vectors
(`tests/golden/`) the Rust and Python tests use, bit for bit.

## Quick start

```r
# From the repo root (dev install; compiles the Rust staticlib):
#   R CMD INSTALL addivortes-r
library(addivortesr)

x <- matrix(runif(200), 100, 2)
y <- 2 * x[, 1] - x[, 2] + rnorm(100, sd = 0.1)

fit <- addivortes(x, y, seed = 1, m = 20, burn_in = 200, draws = 500, omega = 1.5)
predict(fit, x[1:5, ])
prediction_interval(fit, x[1:5, ], level = 0.9)
summary(fit)

# Multi-chain diagnostics via posterior:
chains <- addivortes(x, y, seed = 1, chains = 4, m = 20, omega = 1.5)
posterior::summarise_draws(posterior::as_draws_df(chains))

# Model comparison via loo:
loo::loo(chains, newdata = x, y = y)
```

## Layout and build

`src/rust/` is the extendr wrapper crate (`addivortesr`), a staticlib with
a path dependency on the repo's crate; at release it flips to the
published crates.io version, exactly like `addivortes-py`. `R/extendr-wrappers.R` and `man/` are generated
(`justfile` recipe `r` documents the regeneration commands); the
user-facing R surface is hand-written in the other `R/` files.

`tools/cran-check.sh` is the CRAN rehearsal: it stages a self-contained
tarball (crate source embedded via `cargo package`, registry dependencies
vendored, cargo forced offline, since CRAN forbids install-time network) and
runs `R CMD check --as-cran`.
