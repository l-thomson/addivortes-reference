"""Core surface: determinism, validation, predictions, intervals,
serialisation, diagnostics."""

import numpy as np
import pytest

from addivortes import AddiVortes, AddiVortesError, ess_bulk, ess_tail, r_hat


def friedman(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 6))
    signal = (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
    )
    return x, signal + rng.normal(scale=1.0, size=n)


SMALL = dict(m=10, burn_in=20, draws=30)


@pytest.fixture(scope="module")
def fitted():
    x, y = friedman(60, 1)
    return AddiVortes(seed=7, **SMALL).fit(x, y), x, y


def test_same_seed_is_bit_identical():
    x, y = friedman(40, 2)
    a = AddiVortes(seed=11, **SMALL).fit(x, y).predict(x)
    b = AddiVortes(seed=11, **SMALL).fit(x, y).predict(x)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))


def test_different_seeds_differ():
    x, y = friedman(40, 2)
    a = AddiVortes(seed=11, **SMALL).fit(x, y).predict(x)
    b = AddiVortes(seed=12, **SMALL).fit(x, y).predict(x)
    assert not np.array_equal(a, b)


def test_invalid_hyperparameters_raise_value_error():
    with pytest.raises(AddiVortesError, match="m"):
        AddiVortes(seed=1, m=0)
    with pytest.raises(ValueError):  # AddiVortesError subclasses ValueError
        AddiVortes(seed=1, q=2.0)
    with pytest.raises(AddiVortesError, match="response_family"):
        AddiVortes(seed=1, response_family="cauchy")
    with pytest.raises(AddiVortesError, match="t_df"):
        AddiVortes(seed=1, response_family="robust_t")
    with pytest.raises(AddiVortesError, match="t_df"):
        AddiVortes(seed=1, response_family="gaussian", t_df=4.0)
    with pytest.raises(AddiVortesError, match="metric"):
        AddiVortes(seed=1, metrics=["euclidan"])


def test_shape_errors_surface_cleanly(fitted):
    model, x, _ = fitted
    with pytest.raises(AddiVortesError):
        model.predict(x[:, :3])  # wrong feature count


def test_predictions_and_intervals(fitted):
    model, x, y = fitted
    mean = model.predict(x)
    assert mean.shape == (len(y),)
    assert np.all(np.isfinite(mean))

    quantiles = model.predict_quantiles(x, [0.1, 0.5, 0.9])
    assert quantiles.shape == (len(y), 3)
    assert np.all(np.diff(quantiles, axis=1) >= 0)  # monotone in probability

    lower, upper = model.prediction_interval(x, 0.9)
    assert np.all(lower <= upper)
    c_lower, c_upper = model.credible_interval(x, 0.9)
    assert np.all(c_lower <= c_upper)
    # A new observation's interval is at least as wide as the mean's.
    assert np.all((upper - lower) >= (c_upper - c_lower) - 1e-12)


def test_posterior_accessors(fitted):
    model, x, y = fitted
    assert model.n_draws == SMALL["draws"]
    sigma = model.sigma()
    assert sigma.shape == (SMALL["draws"],)
    assert np.all(sigma > 0)
    cells = model.total_cells()
    assert cells.shape == (SMALL["draws"],)
    assert np.all(cells >= SMALL["m"])  # at least one cell per tessellation
    proportions = model.variable_inclusion_proportions()
    assert proportions.shape == (x.shape[1],)
    assert np.isclose(proportions.sum(), 1.0)
    assert model.warnings() == []
    assert model.response_family == "gaussian"


def test_more_features_than_observations_warns():
    x, y = friedman(60, 3)
    model = AddiVortes(seed=5, **SMALL).fit(x[:5], y[:5])
    assert any("more features" in w for w in model.warnings())


def test_fortran_ordered_input_matches_c_ordered():
    x, y = friedman(40, 4)
    c_fit = AddiVortes(seed=9, **SMALL).fit(np.ascontiguousarray(x), y)
    f_fit = AddiVortes(seed=9, **SMALL).fit(np.asfortranarray(x), y)
    assert np.array_equal(
        c_fit.predict(x).view(np.uint64), f_fit.predict(x).view(np.uint64)
    )


def test_json_round_trip_is_bit_identical(fitted, tmp_path):
    model, x, _ = fitted
    reloaded = type(model).from_json(model.to_json())
    assert np.array_equal(
        model.predict(x).view(np.uint64), reloaded.predict(x).view(np.uint64)
    )
    assert reloaded.response_family == "gaussian"

    path = tmp_path / "model.json"
    model.save(str(path))
    from_disk = type(model).load(str(path))
    assert np.array_equal(
        model.predict(x).view(np.uint64), from_disk.predict(x).view(np.uint64)
    )


def test_corrupt_json_raises_not_panics():
    from addivortes import FittedModel

    with pytest.raises(AddiVortesError):
        FittedModel.from_json("{}")


def test_fit_chains_first_chain_matches_plain_fit():
    x, y = friedman(40, 5)
    spec = AddiVortes(seed=21, **SMALL)
    chains = spec.fit_chains(x, y, 3)
    assert len(chains) == 3
    single = spec.fit(x, y)
    assert np.array_equal(
        chains[0].predict(x).view(np.uint64), single.predict(x).view(np.uint64)
    )
    # Different chains genuinely differ.
    assert not np.array_equal(chains[0].predict(x), chains[1].predict(x))


def test_diagnostics_functions():
    x, y = friedman(40, 6)
    chains = AddiVortes(seed=31, **SMALL).fit_chains(x, y, 2)
    sigma_chains = [list(c.sigma()) for c in chains]
    assert np.isfinite(r_hat(sigma_chains))
    assert ess_bulk(sigma_chains) > 0
    assert ess_tail(sigma_chains) > 0


def test_predict_draws_mean_is_predict(fitted):
    model, x, _ = fitted
    draws = model.predict_draws(x)
    assert draws.shape == (SMALL["draws"], len(x))
    assert np.allclose(draws.mean(axis=0), model.predict(x))


def test_log_likelihood_matrix_and_validation(fitted):
    model, x, y = fitted
    ll = model.log_likelihood(x, y)
    assert ll.shape == (SMALL["draws"], len(y))
    assert np.all(np.isfinite(ll))
    # Gaussian closed form at draw 0, row 0.
    fits = model.predict_draws(x)
    sigma = model.sigma()
    z = (y[0] - fits[0, 0]) / sigma[0]
    expected = -0.5 * np.log(2 * np.pi) - np.log(sigma[0]) - 0.5 * z * z
    assert np.isclose(ll[0, 0], expected)
    with pytest.raises(AddiVortesError):
        model.log_likelihood(x, y[:-1])  # row-count mismatch


def test_probit_log_likelihood_is_bernoulli():
    x, y = friedman(60, 7)
    labels = (y > np.median(y)).astype(np.float64)
    model = AddiVortes(seed=13, **SMALL, response_family="binary_probit").fit(
        x, labels
    )
    ll = model.log_likelihood(x, labels)
    p = model.predict_draws(x)
    eps = np.finfo(np.float64).eps
    pc = np.clip(p, eps, 1 - eps)
    expected = np.where(labels[None, :] == 1.0, np.log(pc), np.log1p(-pc))
    assert np.allclose(ll, expected)
    with pytest.raises(AddiVortesError, match="response"):
        model.log_likelihood(x, labels + 0.5)  # non-{0,1} labels


def test_fitted_model_pickle_round_trip_is_bit_identical(fitted):
    import pickle

    model, x, _ = fitted
    clone = pickle.loads(pickle.dumps(model))
    assert np.array_equal(
        model.predict(x).view(np.uint64), clone.predict(x).view(np.uint64)
    )
    assert clone.response_family == model.response_family


def test_distance_pickle_and_value_equality():
    import pickle

    from addivortes import Distance

    for d in [
        Distance.euclidean(),
        Distance.manhattan(),
        Distance.cosine(),
        Distance.spherical(),
        Distance.minkowski(3.0),
        Distance.gower(["numeric", ("categorical", 3)]),
        Distance.mahalanobis(np.eye(2)),
    ]:
        assert pickle.loads(pickle.dumps(d)) == d
    assert Distance.minkowski(3.0) != Distance.minkowski(2.0)
    assert Distance.euclidean() != Distance.manhattan()


def test_fitted_metadata_getters(fitted):
    model, x, _ = fitted
    assert model.n_features == x.shape[1]
    assert model.t_df is None
    x2, y2 = friedman(50, 8)
    robust = AddiVortes(seed=17, **SMALL, response_family="robust_t", t_df=4.0).fit(
        x2, y2
    )
    assert robust.t_df == 4.0


def test_softmax_membership_fits_and_differs(fitted):
    model, x, y = fitted
    soft = AddiVortes(seed=7, **SMALL, membership=("softmax", 0.1)).fit(x, y)
    assert not np.array_equal(soft.predict(x), model.predict(x))
    # The selection is readable from the spec payload. There is deliberately no
    # per-point `.membership` getter: ten of those would be the same hand-written
    # mirror this binding just stopped keeping.
    assert AddiVortes(seed=1, membership=("softmax", 0.2)).spec["membership"] == {
        "type": "softmax",
        "tau": 0.2,
    }
    with pytest.raises(AddiVortesError, match="membership|unknown variant"):
        AddiVortes(seed=1, membership=("gaussian", 0.1))
    with pytest.raises(AddiVortesError, match="tau"):
        AddiVortes(seed=1, membership=("softmax", -1.0))
    # Soft-membership models refuse serialisation with the crate's message.
    import pickle

    with pytest.raises(AddiVortesError, match="custom extension points"):
        pickle.dumps(soft)


def test_predictive_qq_is_sorted_pit(fitted):
    from addivortes import predictive_qq

    model, x, y = fitted
    fits = model.predict_draws(x)
    s = np.repeat(model.sigma()[:, None], len(y), axis=1)
    q = np.asarray(predictive_qq(y, fits, s))
    assert q.shape == (len(y),)
    assert np.all(np.diff(q) >= 0)
    assert np.all((q > 0) & (q < 1))
    with pytest.raises(AddiVortesError, match="shape"):
        predictive_qq(y, fits, s[:, :-1])


def test_bad_chain_list_raises_rather_than_panics():
    """The crate asserts this contract; a panic would reach Python as a bare
    PanicException, reading as an engine crash rather than a bad call."""
    one = [list(np.random.default_rng(0).normal(size=50))]
    for fn in (r_hat, ess_bulk, ess_tail):
        with pytest.raises(AddiVortesError, match="at least 2"):
            fn(one)

    rng = np.random.default_rng(0)
    with pytest.raises(AddiVortesError, match="same number of draws"):
        ess_bulk([list(rng.normal(size=50)), list(rng.normal(size=40))])
    with pytest.raises(AddiVortesError, match="at least 4 draws"):
        ess_bulk([list(rng.normal(size=3)), list(rng.normal(size=3))])
