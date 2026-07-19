"""Bayesian additive Voronoi tessellation regression (AddiVortes).

Python bindings for the ``addivortes`` Rust crate (Stone & Gosling 2025).
The engine, its defaults, and its bit-exact reproducibility contract live
in Rust; this package selects and drives them. Same seed, same chain:
in Python, in Rust, on any platform.

Quickstart::

    import numpy as np
    from addivortes import AddiVortes

    rng = np.random.default_rng(0)
    x = rng.uniform(size=(200, 5))
    y = np.sin(np.pi * x[:, 0] * x[:, 1]) + rng.normal(scale=0.1, size=200)

    model = AddiVortes(seed=42).fit(x, y)
    mean = model.predict(x)
    lower, upper = model.prediction_interval(x, 0.9)
    draws = model.predict_draws(x)        # (n_draws, n_rows) posterior of the fit
    loglik = model.log_likelihood(x, y)   # pointwise, for LOO/WAIC

Shelf selection. Nine of the ten extension points are chosen with a payload in
the core's own vocabulary (the cell payload family is Rust-only by design: its
prior variance is engine-derived, so a hand-typed value would silently override
the engine's own calibration), so a shelf entry added to the Rust crate is
selectable from Python the day it ships — this package never lists the shelf a
second time::

    from addivortes import AddiVortes, Distance

    AddiVortes(seed=1, distance=Distance.manhattan())
    AddiVortes(seed=1, distance=Distance.gower(["numeric", ("categorical", 3)]),
               metrics=["euclidean", "categorical"])
    AddiVortes(seed=1, response_family="robust_t", t_df=4.0)
    AddiVortes(seed=1, response_family="binary_probit")
    AddiVortes(seed=1, membership=("softmax", 0.1))

    # …and the points that no binding could reach before:
    AddiVortes(seed=1, inclusion={"type": "dart", "alpha": 0.5})
    AddiVortes(seed=1, scale={"type": "h_variance", "m_prime": 40})
    AddiVortes(seed=1, basis={"type": "linear", "columns": [0], "sigma_beta_sq": 0.1})
    AddiVortes(seed=1, moves=[{"name": "add_centre", "weight": 0.3},
                              {"name": "remove_centre", "weight": 0.3},
                              {"name": "change", "weight": 0.4}])

Authoring a *new* component (your own move, cell model or membership kernel) is a
Rust-side activity: a trait implementation is code, and no data payload can
carry one. See the Rust crate documentation's "Extending" section.

Fitted models pickle (via the crate's validated JSON format) and save/load
files interchangeable with Rust.

Optional extras: ``addivortes[sklearn]`` for scikit-learn estimators
(:mod:`addivortes.sklearn`), ``addivortes[arviz]`` for the full
``InferenceData`` export, PPC and LOO/WAIC (:mod:`addivortes.arviz`);
:mod:`addivortes.interpret` (NumPy-only) for variable importance and
PDP/ICE with credible bands.
"""

from addivortes._model import AddiVortes
from addivortes._native import (
    AddiVortesError,
    FittedModel,
    ess_bulk,
    ess_tail,
    predictive_qq,
    r_hat,
)
from addivortes._shelf import Distance

__version__ = "0.1.0"

__all__ = [
    "AddiVortes",
    "AddiVortesError",
    "Distance",
    "FittedModel",
    "ess_bulk",
    "ess_tail",
    "predictive_qq",
    "r_hat",
    "__version__",
]
