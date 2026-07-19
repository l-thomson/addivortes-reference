"""Interpretability helpers: variable importance, partial dependence, ICE.

NumPy-only (no plotting dependency): every function returns arrays ready
for matplotlib/seaborn/plotly, in the spirit of PyMC-BART's ``plot_pdp`` /
``plot_ice`` / ``plot_variable_importance`` but decoupled from any one
plotting stack.

All helpers accept either a fitted :class:`addivortes.FittedModel` or a
fitted ``addivortes.sklearn`` estimator (the native model is unwrapped
from ``model_``, and ``feature_names_in_`` labels resolve automatically).
Outputs are on the family's own prediction scale: response scale for
gaussian/robust_t, probability scale for binary_probit.
"""

from __future__ import annotations

import numpy as np


def _as_model(model):
    """Unwrap a fitted sklearn estimator to its native model, or pass a
    native FittedModel through."""
    if hasattr(model, "predict_draws"):
        return model
    inner = getattr(model, "model_", None)
    if inner is not None and hasattr(inner, "predict_draws"):
        return inner
    raise ValueError(
        "expected a fitted addivortes model (FittedModel or a fitted "
        "addivortes.sklearn estimator); got an object without predict_draws"
    )


def _default_names(model, feature_names):
    if feature_names is not None:
        names = [str(n) for n in feature_names]
    elif getattr(model, "feature_names_in_", None) is not None:
        names = [str(n) for n in model.feature_names_in_]
    else:
        names = None
    return names


def _resolve_feature(feature, names, n_features):
    if isinstance(feature, str):
        if names is None:
            raise ValueError(
                f"feature {feature!r} is a name but no feature names are "
                "available; pass feature_names or a column index"
            )
        try:
            return names.index(feature)
        except ValueError:
            raise ValueError(f"feature {feature!r} is not in {names}") from None
    index = int(feature)
    if not 0 <= index < n_features:
        raise ValueError(f"feature index {index} out of range for {n_features} columns")
    return index


def _as_x(X):
    x = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {x.shape}")
    return x


def _grid_for(column, grid, grid_points):
    if grid is not None:
        g = np.asarray(grid, dtype=np.float64)
        if g.ndim != 1 or len(g) == 0:
            raise ValueError("grid must be a non-empty 1-D array")
        return g
    unique = np.unique(column)
    if len(unique) <= grid_points:
        return unique
    return np.linspace(column.min(), column.max(), grid_points)


def variable_importance(model, feature_names=None):
    """Labelled posterior variable-inclusion proportions, sorted descending.

    The BART-style covariate-importance summary: the share of all active
    tessellation dimensions across kept draws that map to each
    caller-visible column (one-hot groups aggregate onto their source
    column). Returns ``{name: proportion}`` in descending-importance order;
    proportions are dimensionless shares in [0, 1] summing to 1.
    """
    names = _default_names(model, feature_names)
    native = _as_model(model)
    proportions = np.asarray(native.variable_inclusion_proportions())
    if names is None:
        names = [f"x{i}" for i in range(len(proportions))]
    if len(names) != len(proportions):
        raise ValueError(
            f"feature_names has {len(names)} entries for {len(proportions)} columns"
        )
    order = np.argsort(proportions)[::-1]
    return {names[i]: float(proportions[i]) for i in order}


def partial_dependence(
    model, X, feature, *, grid=None, grid_points=20, level=0.9, feature_names=None
):
    """Bayesian partial dependence of the fit on one feature, with a
    credible band.

    For each grid value g the feature column of ``X`` is set to g and each
    posterior draw's predictions are averaged over rows, giving a full
    posterior distribution of the partial-dependence value, not just a
    point estimate (the draw axis is preserved through the average, so the
    band is a genuine credible band).

    Parameters
    ----------
    model : FittedModel or fitted sklearn estimator
    X : array-like of shape (n_obs, n_features)
        The background dataset (typically the training design).
    feature : int or str
        Column index, or name (resolved via ``feature_names`` /
        ``feature_names_in_``).
    grid : array-like, optional
        Evaluation values; defaults to the column's unique values when few,
        else ``grid_points`` evenly spaced values over its range.
    grid_points : int, default 20
    level : float, default 0.9
        Central credible level of the band.
    feature_names : sequence of str, optional

    Returns
    -------
    grid : ndarray of shape (n_grid,)
    mean : ndarray of shape (n_grid,), posterior-mean partial dependence
    lower, upper : ndarrays of shape (n_grid,), the credible band
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be inside (0, 1), got {level}")
    names = _default_names(model, feature_names)
    native = _as_model(model)
    x = _as_x(X)
    index = _resolve_feature(feature, names, x.shape[1])
    g = _grid_for(x[:, index], grid, grid_points)

    # One native call: stack the grid-modified copies of X.
    stacked = np.repeat(x[None, :, :], len(g), axis=0)
    stacked[:, :, index] = g[:, None]
    draws = np.asarray(native.predict_draws(stacked.reshape(-1, x.shape[1])))
    # (n_draws, n_grid, n_obs) -> average rows per draw: the PD posterior.
    pd_draws = draws.reshape(draws.shape[0], len(g), x.shape[0]).mean(axis=2)

    tail = 0.5 * (1.0 - level)
    mean = pd_draws.mean(axis=0)
    lower = np.quantile(pd_draws, tail, axis=0)
    upper = np.quantile(pd_draws, 1.0 - tail, axis=0)
    return g, mean, lower, upper


def ice(model, X, feature, *, grid=None, grid_points=20, feature_names=None):
    """Individual conditional expectation curves for one feature.

    For each row of ``X`` and each grid value g, the posterior-mean
    prediction with that row's feature set to g. The per-row analogue of
    :func:`partial_dependence` (whose mean curve is the row-average of
    these).

    Returns
    -------
    grid : ndarray of shape (n_grid,)
    curves : ndarray of shape (n_obs, n_grid)
    """
    names = _default_names(model, feature_names)
    native = _as_model(model)
    x = _as_x(X)
    index = _resolve_feature(feature, names, x.shape[1])
    g = _grid_for(x[:, index], grid, grid_points)

    stacked = np.repeat(x[None, :, :], len(g), axis=0)
    stacked[:, :, index] = g[:, None]
    mean = np.asarray(native.predict(stacked.reshape(-1, x.shape[1])))
    # (n_grid, n_obs) -> (n_obs, n_grid)
    return g, mean.reshape(len(g), x.shape[0]).T
