"""ArviZ interop: posterior draws as an :class:`arviz.InferenceData`.

Requires the ``arviz`` extra (``pip install addivortes[arviz]``).

Given a fitted model (one chain) or the list from
:meth:`addivortes.AddiVortes.fit_chains` (many), :func:`to_inference_data`
builds an ``InferenceData`` carrying every group the standard ArviZ
workflows consume:

- ``posterior``: ``sigma`` (error SD, response scale; pinned at the
  latent unit scale for the probit family), ``total_cells`` (ensemble
  structure-complexity trace), and, when ``X`` is given, ``mu``, the
  per-draw fit at each row of ``X`` on the family's own scale.
- ``posterior_predictive``: ``y`` replicates sampled from each draw's
  predictive law (Gaussian, location-scale Student-t, or Bernoulli),
  feeding ``az.plot_ppc``. Sampling happens NumPy-side from the exported
  posterior (seeded via ``random_seed``); the engine's bit-exact contract
  covers the chain itself, not these replicates.
- ``log_likelihood``: pointwise ln p(yᵢ | draw) from the engine's own
  family densities, feeding ``az.loo`` / ``az.waic`` / ``az.compare``.
- ``observed_data`` / ``constant_data``: ``y`` and ``X``, dimension-linked
  (``obs``, ``feature``) so summaries and plots label themselves;
  ``feature_names`` overrides the default ``x0..x{p-1}`` labels.

Minimal call (diagnostics only)::

    idata = to_inference_data(model)            # sigma, total_cells

Full workflow::

    chains = AddiVortes(seed=1).fit_chains(x, y, 4)
    idata = to_inference_data(chains, X=x, y=y)
    az.summary(idata)      # R-hat / ESS over chains
    az.plot_ppc(idata)     # posterior-predictive check
    az.loo(idata)          # PSIS-LOO, pointwise
"""

from __future__ import annotations

import inspect

import numpy as np

try:
    import arviz as az
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "addivortes.arviz requires arviz: pip install 'addivortes[arviz]'"
    ) from exc

# ArviZ 1.x refactored ``from_dict`` to take one positional mapping of
# groups; the classic API takes per-group keyword arguments. Detect once by
# the first parameter's name ('data' new, 'posterior' classic) and support
# both; the tests run against whichever generation is installed.
_FROM_DICT_PARAMS = inspect.signature(az.from_dict).parameters
_NEW_FROM_DICT = next(iter(_FROM_DICT_PARAMS)) == "data"


def _sample_predictive(mu, sigma, family, t_df, rng):
    """One replicate per draw from the family's predictive law: shape of
    ``mu`` = (chain, draw, obs); ``sigma`` = (chain, draw)."""
    if family == "binary_probit":
        return (rng.random(mu.shape) < mu).astype(np.float64)
    noise = (
        rng.standard_t(t_df, size=mu.shape)
        if family == "robust_t"
        else rng.standard_normal(mu.shape)
    )
    return mu + sigma[..., None] * noise


def to_inference_data(
    models,
    *,
    X=None,
    y=None,
    feature_names=None,
    posterior_predictive=True,
    log_likelihood=True,
    random_seed=0,
):
    """Build an ``arviz.InferenceData`` from fitted AddiVortes model(s).

    Parameters
    ----------
    models : FittedModel or sequence of FittedModel
        One chain, or the equal-length chains from ``fit_chains``.
    X : array-like of shape (n_obs, n_features), optional
        Prediction inputs (typically the training design). Enables the
        ``mu`` posterior variable, ``posterior_predictive`` and (with
        ``y``) ``log_likelihood``; stored in ``constant_data``.
    y : array-like of shape (n_obs,), optional
        Observed responses, stored in ``observed_data``.
    feature_names : sequence of str, optional
        Labels for the ``feature`` coordinate (defaults to ``x0..x{p-1}``).
    posterior_predictive : bool, default True
        Sample ``y`` replicates per draw (needs ``X``).
    log_likelihood : bool, default True
        Export the pointwise log-likelihood matrix (needs ``X`` and ``y``).
    random_seed : int, default 0
        Seed for the NumPy generator behind ``posterior_predictive``.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    if not models:
        raise ValueError("at least one fitted model is required")
    draws = {m.n_draws for m in models}
    if len(draws) != 1:
        raise ValueError(f"chains disagree on draw counts: {sorted(draws)}")
    families = {m.response_family for m in models}
    if len(families) != 1:
        raise ValueError(f"chains disagree on response family: {sorted(families)}")
    family = models[0].response_family

    posterior = {
        "sigma": np.stack([np.asarray(m.sigma()) for m in models]),
        "total_cells": np.stack([np.asarray(m.total_cells()) for m in models]),
    }
    coords = {}
    dims = {}
    groups = {}

    y_arr = None
    if y is not None:
        y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
        groups["observed_data"] = {"y": y_arr}
        coords["obs"] = np.arange(len(y_arr))
        dims["y"] = ["obs"]

    if X is not None:
        x_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if x_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {x_arr.shape}")
        if y_arr is not None and len(y_arr) != x_arr.shape[0]:
            raise ValueError(
                f"X has {x_arr.shape[0]} rows but y has {len(y_arr)} entries"
            )
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(x_arr.shape[1])]
        )
        if len(names) != x_arr.shape[1]:
            raise ValueError(
                f"feature_names has {len(names)} entries for {x_arr.shape[1]} columns"
            )
        coords["obs"] = np.arange(x_arr.shape[0])
        coords["feature"] = names
        dims["X"] = ["obs", "feature"]
        groups["constant_data"] = {"X": x_arr}

        # (chain, draw, obs): each chain's unreduced posterior of the fit.
        mu = np.stack([np.asarray(m.predict_draws(x_arr)) for m in models])
        posterior["mu"] = mu
        dims["mu"] = ["obs"]

        if posterior_predictive:
            rng = np.random.default_rng(random_seed)
            t_df = models[0].t_df
            groups["posterior_predictive"] = {
                "y": _sample_predictive(mu, posterior["sigma"], family, t_df, rng)
            }
        if log_likelihood and y_arr is not None:
            groups["log_likelihood"] = {
                "y": np.stack(
                    [np.asarray(m.log_likelihood(x_arr, y_arr)) for m in models]
                )
            }

    groups["posterior"] = posterior

    if _NEW_FROM_DICT:
        kwargs = {
            name: value
            for name, value in (("coords", coords), ("dims", dims))
            if value and name in _FROM_DICT_PARAMS
        }
        return az.from_dict(groups, **kwargs)
    return az.from_dict(
        posterior=groups["posterior"],
        posterior_predictive=groups.get("posterior_predictive"),
        log_likelihood=groups.get("log_likelihood"),
        observed_data=groups.get("observed_data"),
        constant_data=groups.get("constant_data"),
        coords=coords or None,
        dims=dims or None,
    )
