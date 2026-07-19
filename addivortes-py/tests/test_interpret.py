"""The interpretability helpers: variable importance, PDP, ICE."""

import numpy as np
import pytest

from addivortes import AddiVortes
from addivortes.interpret import ice, partial_dependence, variable_importance

SMALL = dict(m=10, burn_in=20, draws=30)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(1)
    n = 80
    x = rng.uniform(size=(n, 5))
    # Strong signal on x0, none on the rest.
    y = 6.0 * x[:, 0] ** 2 + rng.normal(scale=0.1, size=n)
    return AddiVortes(seed=3, **SMALL).fit(x, y), x, y


def test_variable_importance_is_sorted_and_labelled(fitted):
    model, x, _ = fitted
    importance = variable_importance(model, feature_names=["a", "b", "c", "d", "e"])
    assert list(importance) == sorted(importance, key=importance.get, reverse=True)
    assert np.isclose(sum(importance.values()), 1.0)
    assert importance["a"] == max(importance.values())  # the signal column wins
    # Default labels are x0..x4.
    default = variable_importance(model)
    assert set(default) == {"x0", "x1", "x2", "x3", "x4"}
    with pytest.raises(ValueError, match="feature_names"):
        variable_importance(model, feature_names=["too", "few"])


def test_partial_dependence_tracks_the_signal(fitted):
    model, x, _ = fitted
    grid, mean, lower, upper = partial_dependence(model, x, 0, grid_points=8)
    assert grid.shape == mean.shape == lower.shape == upper.shape == (8,)
    assert np.all(lower <= mean) and np.all(mean <= upper)
    # 6 x^2 is increasing on [0, 1]: the PD curve ends well above its start.
    assert mean[-1] - mean[0] > 2.0
    # A noise column's PD curve is comparatively flat.
    _, flat, _, _ = partial_dependence(model, x, 3, grid_points=8)
    assert (flat.max() - flat.min()) < (mean.max() - mean.min()) / 2


def test_partial_dependence_named_feature_and_custom_grid(fitted):
    model, x, _ = fitted
    grid = np.array([0.1, 0.5, 0.9])
    g, mean, _, _ = partial_dependence(
        model, x, "a", grid=grid, feature_names=["a", "b", "c", "d", "e"]
    )
    assert np.array_equal(g, grid)
    assert mean.shape == (3,)
    with pytest.raises(ValueError, match="not in"):
        partial_dependence(model, x, "zz", feature_names=["a", "b", "c", "d", "e"])
    with pytest.raises(ValueError, match="level"):
        partial_dependence(model, x, 0, level=1.5)


def test_ice_curves_average_to_pdp_mean_curve(fitted):
    model, x, _ = fitted
    grid = np.array([0.2, 0.8])
    g, curves = ice(model, x, 0, grid=grid)
    assert curves.shape == (len(x), 2)
    _, pd_mean, _, _ = partial_dependence(model, x, 0, grid=grid)
    assert np.allclose(curves.mean(axis=0), pd_mean)


def test_helpers_accept_sklearn_estimators():
    sklearn = pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from addivortes.sklearn import AddiVortesRegressor

    rng = np.random.default_rng(2)
    x = rng.uniform(size=(60, 4))
    y = 4.0 * x[:, 1] + rng.normal(scale=0.2, size=60)
    frame = pd.DataFrame(x, columns=["p", "q", "r", "s"])
    est = AddiVortesRegressor(seed=4, **SMALL).fit(frame, y)
    importance = variable_importance(est)  # names resolve from the frame
    assert set(importance) == {"p", "q", "r", "s"}
    assert importance["q"] == max(importance.values())
    g, mean, lower, upper = partial_dependence(est, x, "q", grid_points=5)
    assert mean.shape == (5,)
    with pytest.raises(ValueError, match="predict_draws"):
        variable_importance(AddiVortesRegressor(seed=1))  # unfitted
