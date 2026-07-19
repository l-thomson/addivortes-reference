# addivortes

Bayesian additive Voronoi-tessellation regression: a Rust implementation,
with Python and R bindings.

## About

`addivortes` implements the AddiVortes method (Bayesian additive Voronoi
tessellation regression), introduced by Stone & Gosling (2025), JCGS
34(3):859–871, together with its published variants: Binary-AddiVortes
(probit classification) and H-AddiVortes (heteroscedastic variance). All
credit for the method belongs to its authors; the original R package is
[`AddiVortes`](https://github.com/johnpaulgosling/AddiVortes).

The crate is licensed MIT OR Apache-2.0; `NOTICE` carries the attribution
and licensing statement.

The model is `Y = Σ_{j=1..m} g(x | T_j, M_j) + ε` with `ε ~ N(0, σ²)`: a sum of
`m` Voronoi tessellations, each partitioning a random subspace of the
covariates, explored by a Gibbs backfitting sampler with six structural
Metropolis–Hastings moves.

## Rust

```rust
use addivortes::{AddiVortesConfig, Data};

fn main() -> addivortes::Result<()> {
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
    let x = Data::new(xs, n, 1)?;

    let model = AddiVortesConfig::new(42) // the chain seed is mandatory
        .with_m(100)
        .fit(&x, &y)?;

    let predictions = model.predict(&x)?;
    let intervals = model.credible_interval(&x, 0.9)?; // uncertainty of the mean fit
    let predictive = model.prediction_interval(&x, 0.9)?; // range for a NEW observation
    let importance = model.variable_inclusion_proportions(); // which columns mattered
    println!(
        "rmse {} interval {:?} predictive {:?} importance {:?}",
        model.in_sample_rmse(),
        intervals[0],
        predictive[0],
        importance
    );
    Ok(())
}
```

The full API documentation, including every extension point, is the rustdoc:
`cargo doc --open`.

Per-column input types via `with_metrics`: ordinary numbers (`Euclidean`),
angles (`Spherical`), categories (`Categorical`, one-hot encoded internally),
and caller-prepared pass-through columns (`Prepared`: you own the
preparation; see the `distance` module docs).

With the `serde` cargo feature, fitted models save and load through any serde
format with bit-identical predictions (for JSON, enable `serde_json`'s
`float_roundtrip` feature); loading validates the payload, and models fitted
with custom extension points refuse to serialise.

## Python

[`addivortes-py/`](addivortes-py/) is the Python binding (PyO3/maturin):

```python
import numpy as np
from addivortes import AddiVortes

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 5))
y = np.sin(np.pi * x[:, 0] * x[:, 1]) + rng.normal(scale=0.1, size=200)

model = AddiVortes(seed=42).fit(x, y)
mean = model.predict(x)
lower, upper = model.prediction_interval(x, 0.9)
```

A NumPy-native class over the same engine, with shelf selection, crate-format
save/load (and pickle) interchangeable with Rust, per-draw posterior access,
and optional adapters: scikit-learn estimators that pass the
`check_estimator` battery, an ArviZ export, and interpretability helpers
(variable importance, PDP/ICE with credible bands). Install and details:
its [README](addivortes-py/README.md).

## R

[`addivortes-r/`](addivortes-r/) is the R binding (extendr):

```r
library(addivortesr)

x <- matrix(runif(500), 100, 5)
y <- 2 * x[, 1] - x[, 2] + rnorm(100, sd = 0.1)

fit <- addivortes(x, y, seed = 1)
predict(fit, x[1:5, ])
prediction_interval(fit, x[1:5, ], level = 0.9)
```

An idiomatic `addivortes()` fit function over the same engine with S3
`predict`/`summary` methods, shelf selection, crate-format save/load
interchangeable with Rust and Python, per-draw posterior access, and
optional adapters: posterior/loo methods and parsnip engines. The package
name is provisional: the original authors' own R implementation is called
`AddiVortes`. Install and details: its [README](addivortes-r/README.md).

## Reproducibility

A chain is reproducible given the same seed, the same addivortes version, and
the same compilation target, built with this crate's default release profile
and no overriding RUSTFLAGS (in particular no -Ctarget-cpu=native and no
target-feature=+fma). Any change that alters the sampled chain for a fixed
seed bumps the 0.y minor version and regenerates the golden vectors
deliberately; patch releases guarantee bit-identical chains, enforced by a
golden-chain regression test in CI (Linux x86_64, macOS ARM, Windows). Any
major bump of rand, rand_core, rand_distr, or libm is treated as
chain-altering by definition.

The same contract crosses both FFIs: the Python and R test suites pin a fit
against the same per-target golden vectors (`tests/golden/`) the Rust tests
use, bit for bit.

## Extending

The engine exposes ten extension points, all public: structural moves,
coordinate laws, distance/assignment, variable inclusion, cell payload
family, response family, scale/precision, count priors, cell basis, and
membership. Every point has the same shape: a trait you implement, a shelf
of shipped implementations beside it, a copy-paste template under
`examples/`, and a one-command conformance check, with the public Geweke/SBC
validation battery behind them for components destined for real inference.
External crates validate custom components with no engine access:
`tests/calibration_acceptance.rs` is the worked example, and it compiles
against the public surface only.

The crate-level rustdoc's "Extending" section is the entry point; each
point's module documents what you implement, what the engine provides, what
is on the shelf, and the sources.

## Defaults

Hyperparameter defaults follow the paper, with one deliberate exception:
λ_c ships at 5 rather than the paper's 25. The reference implementation
thins the cell-count prior by 1/(b+1), so its nominal 25 corresponds to
roughly 5 cells in practice; this crate implements the shifted Poisson
directly, and 5 reproduces the reference's effective behaviour. The paper's
nominal setting: `.with_lambda_c(25.0)`.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT licence ([`LICENSE-MIT`](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 licence, shall be
dual licensed as above, without any additional terms or conditions.
