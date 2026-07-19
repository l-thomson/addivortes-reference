"""The scikit-learn adapter: estimator contract, regression, classification."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone

from addivortes.sklearn import AddiVortesClassifier, AddiVortesRegressor

SMALL = dict(m=10, burn_in=20, draws=30)


def regression_data(n=60, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 4))
    y = 5.0 * x[:, 0] + 2.0 * x[:, 1] ** 2 + rng.normal(scale=0.2, size=n)
    return x, y


def test_regressor_fit_predict_score():
    x, y = regression_data()
    est = AddiVortesRegressor(seed=1, **SMALL).fit(x, y)
    assert est.predict(x).shape == y.shape
    assert est.score(x, y) > 0.5  # in-sample R^2 on a clean signal
    assert est.n_features_in_ == 4
    lower, upper = est.prediction_interval(x, 0.9)
    assert np.all(lower <= upper)


def test_estimator_contract_params_and_clone():
    est = AddiVortesRegressor(seed=3, m=12, lambda_c=5.0)
    params = est.get_params()
    assert params["seed"] == 3
    assert params["m"] == 12
    assert params["lambda_c"] == 5.0

    cloned = clone(est)
    assert cloned.get_params() == params

    est.set_params(m=20)
    assert est.get_params()["m"] == 20


def test_clone_then_fit_is_deterministic():
    x, y = regression_data(seed=2)
    est = AddiVortesRegressor(seed=5, **SMALL)
    a = est.fit(x, y).predict(x)
    b = clone(est).fit(x, y).predict(x)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))


def test_regressor_robust_t_family():
    x, y = regression_data(seed=3)
    est = AddiVortesRegressor(seed=6, response_family="robust_t", t_df=4.0, **SMALL)
    est.fit(x, y)
    assert est.model_.response_family == "robust_t"


def test_unfitted_predict_raises_not_fitted_error():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        AddiVortesRegressor(seed=1).predict(np.zeros((2, 2)))


def test_predict_return_std():
    x, y = regression_data()
    est = AddiVortesRegressor(seed=2, **SMALL).fit(x, y)
    mean, std = est.predict(x, return_std=True)
    assert mean.shape == std.shape == y.shape
    assert np.all(std > 0)
    # Law of total variance: std is at least the smallest error SD draw.
    assert np.all(std >= np.asarray(est.model_.sigma()).min() - 1e-12)
    # Plain predict is unchanged by the flag.
    assert np.array_equal(mean, est.predict(x))


def test_predict_return_std_robust_t_low_df_raises():
    x, y = regression_data(seed=5)
    est = AddiVortesRegressor(
        seed=2, response_family="robust_t", t_df=2.0, **SMALL
    ).fit(x, y)
    with pytest.raises(ValueError, match="t_df > 2"):
        est.predict(x, return_std=True)


def test_estimator_pickle_round_trip():
    import pickle

    x, y = regression_data()
    est = AddiVortesRegressor(seed=4, **SMALL).fit(x, y)
    clone_est = pickle.loads(pickle.dumps(est))
    assert np.array_equal(
        est.predict(x).view(np.uint64), clone_est.predict(x).view(np.uint64)
    )
    assert clone_est.n_features_in_ == est.n_features_in_


def test_feature_names_in_from_dataframe():
    pd = pytest.importorskip("pandas")
    x, y = regression_data()
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    est = AddiVortesRegressor(seed=1, **SMALL).fit(frame, y)
    assert list(est.feature_names_in_) == ["f0", "f1", "f2", "f3"]
    est.predict(frame)  # names accepted at predict
    with pytest.warns(UserWarning, match="feature names"):
        est.predict(x)  # fitted with names, predicting without warns


def test_predict_validates_feature_count():
    x, y = regression_data()
    est = AddiVortesRegressor(seed=1, **SMALL).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        est.predict(x[:, :2])


def test_predict_quantiles_passthrough():
    x, y = regression_data()
    est = AddiVortesRegressor(seed=1, **SMALL).fit(x, y)
    q = est.predict_quantiles(x, [0.25, 0.75])
    assert q.shape == (len(y), 2)
    assert np.all(q[:, 0] <= q[:, 1])


def test_classifier_binary_labels_and_probabilities():
    rng = np.random.default_rng(4)
    n = 60
    x = rng.uniform(size=(n, 3))
    labels = np.where(x[:, 0] + x[:, 1] > 1.0, "pos", "neg")

    est = AddiVortesClassifier(seed=7, omega=1.5, **SMALL).fit(x, labels)
    assert list(est.classes_) == ["neg", "pos"]

    proba = est.predict_proba(x)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    predicted = est.predict(x)
    assert set(predicted) <= {"neg", "pos"}
    assert (predicted == labels).mean() > 0.7


def test_classifier_rejects_multiclass():
    x = np.random.default_rng(5).uniform(size=(30, 2))
    y = np.arange(30) % 3
    with pytest.raises(ValueError, match="binary"):
        AddiVortesClassifier(seed=8, **SMALL).fit(x, y)


# --- The scikit-learn conformance battery -----------------------------------
# Every applicable check from sklearn.utils.estimator_checks, run against
# small-budget instances (omega=1.5 keeps narrow check datasets inside the
# engine's omega < n_features rule).
from sklearn.utils.estimator_checks import parametrize_with_checks

_CHECK_INSTANCES = [
    AddiVortesRegressor(seed=1, omega=1.5, **SMALL),
    AddiVortesClassifier(seed=1, omega=1.5, **SMALL),
]


@parametrize_with_checks(_CHECK_INSTANCES)
def test_sklearn_conformance(estimator, check):
    check(estimator)
