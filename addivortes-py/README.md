# addivortes (Python)

Python bindings for the [`addivortes`](..) Rust crate: Bayesian additive
Voronoi tessellation regression (AddiVortes, Stone & Gosling 2025), over
the crate's engine, with a NumPy API.

The engine, its defaults, and its bit-exact reproducibility contract
live in Rust; this package selects and drives them. The same seed gives
the same chain from Python and from Rust on any supported platform (the
pytest suite pins a fit against the same per-target golden vectors the
Rust tests use).

## Install (development)

```sh
cd addivortes-py
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release
```

## Use

```python
import numpy as np
from addivortes import AddiVortes, Distance

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 5))
y = np.sin(np.pi * x[:, 0] * x[:, 1]) + rng.normal(scale=0.1, size=200)

model = AddiVortes(seed=42).fit(x, y)          # crate defaults throughout
mean = model.predict(x)
lower, upper = model.prediction_interval(x, 0.9)
model.save("model.json")                        # crate-native format;
                                                # loadable from Rust too
```

The full posterior is available:

```python
draws = model.predict_draws(x)                  # (n_draws, n_rows), family scale
loglik = model.log_likelihood(x, y)             # (n_draws, n_rows), for LOO/WAIC
```

Fitted models pickle (through the crate's validated JSON format, so
loading re-validates), and the package ships type stubs (`py.typed`).

Shelf selection (every hyperparameter left unset uses the Rust default):

```python
AddiVortes(seed=1, lambda_c=25.0)                                 # the paper's value (default 5)
AddiVortes(seed=1, distance=Distance.mahalanobis(precision))      # the distance point
AddiVortes(seed=1, metrics=["euclidean", "categorical"],
           distance=Distance.gower(["numeric", ("categorical", 3)]))
AddiVortes(seed=1, response_family="robust_t", t_df=4.0)          # the response point
AddiVortes(seed=1, response_family="binary_probit")               # Binary-AddiVortes
AddiVortes(seed=1, membership=("softmax", 0.1))                   # the membership point
```

(Serialisation note, mirroring the crate: models fitted with a
non-default distance or soft membership refuse `save`/pickle; trait
objects have no portable form yet.)

## Optional extras

### `addivortes[sklearn]`

`addivortes.sklearn.AddiVortesRegressor` / `AddiVortesClassifier`: the
full scikit-learn contract: `get_params`/`clone`, `validate_data`
(`n_features_in_`, `feature_names_in_`), `NotFittedError`, estimator
pickling (so `joblib` and parallel `GridSearchCV` work), and
`check_estimator` passes in CI. Posterior uncertainty uses the
standard Bayesian-regressor idiom:

```python
mean, std = est.predict(X, return_std=True)     # posterior-predictive SD
lower, upper = est.prediction_interval(X, 0.9)
```

### `addivortes[arviz]`

`addivortes.arviz.to_inference_data(model | chains, X=x, y=y)` exports
every standard `InferenceData` group: `posterior` (`sigma`,
`total_cells`, and `mu`, the per-draw fit at `X`),
`posterior_predictive` (seeded replicates from each draw's family
predictive law), `log_likelihood` (the engine's own pointwise densities),
and `observed_data`/`constant_data` with labelled dims. Then:

```python
chains = AddiVortes(seed=42).fit_chains(x, y, 4)   # chain 0 matches fit()
idata = to_inference_data(chains, X=x, y=y)
az.summary(idata)     # R-hat / ESS
az.plot_ppc(idata)    # posterior-predictive checks
az.loo(idata)         # PSIS-LOO; az.waic / az.compare likewise
```

### `addivortes.interpret` (no extra dependencies)

NumPy-only interpretability, in the spirit of PyMC-BART but decoupled from any
one plotting stack, returning arrays for
whatever plotting stack you use: `variable_importance` (labelled, sorted
inclusion proportions), `partial_dependence` (bands are genuine credible
bands, since rows are averaged within each posterior draw), `ice`. All
accept native models or fitted sklearn estimators.

## Scope

Python selects the built-in shelf (metrics, distances, response
families, soft membership, hyperparameters). Authoring new components
(moves, cell models, membership kernels) is a Rust-side activity against
the crate's public traits (the "Extending" section of the crate docs);
custom components do not cross the FFI boundary.
