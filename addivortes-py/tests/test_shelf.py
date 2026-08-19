"""Shelf selection through the binding: every distance factory fits, the
guards fire with the crate's own messages, and the non-Gaussian response
families behave on their own scales."""

import numpy as np
import pytest

from addivortes import AddiVortes, AddiVortesError, Distance

SMALL = dict(m=8, burn_in=15, draws=20)


def linear_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 2))
    return x, 3.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=n)


@pytest.mark.parametrize(
    "distance",
    [
        Distance.euclidean(),
        Distance.manhattan(),
        Distance.cosine(),
        Distance.minkowski(3.0),
        Distance.mahalanobis(np.array([[2.0, -0.5], [-0.5, 1.0]])),
    ],
    ids=lambda d: repr(d),
)
def test_each_distance_fits(distance):
    x, y = linear_data(30, 1)
    model = AddiVortes(seed=3, omega=1.5, distance=distance, **SMALL).fit(x, y)
    assert np.all(np.isfinite(model.predict(x)))


def test_gower_mixed_fit():
    n = 30
    rng = np.random.default_rng(2)
    numeric = rng.uniform(size=n)
    category = (np.arange(n) % 2).astype(float)
    x = np.column_stack([numeric, category])
    y = 2.0 * numeric + 0.5 * category
    model = AddiVortes(
        seed=4,
        omega=1.5,
        metrics=["euclidean", "categorical"],
        distance=Distance.gower(["numeric", ("categorical", 2)]),
        **SMALL,
    ).fit(x, y)
    assert np.all(np.isfinite(model.predict(x)))


def test_distance_guards_raise():
    with pytest.raises(AddiVortesError, match="minkowski_p"):
        Distance.minkowski(0.5)
    with pytest.raises(AddiVortesError, match="square"):
        Distance.mahalanobis(np.ones((2, 3)))
    with pytest.raises(AddiVortesError, match="positive definite"):
        Distance.mahalanobis(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(AddiVortesError, match="symmetric"):
        Distance.mahalanobis(np.array([[1.0, 0.5], [-0.5, 1.0]]))
    with pytest.raises(AddiVortesError, match="gower column"):
        Distance.gower(["numerical"])


def test_mahalanobis_wrong_width_fails_the_fit_loudly():
    x, y = linear_data(30, 5)
    spec = AddiVortes(
        seed=5, omega=1.5, distance=Distance.mahalanobis(np.eye(3)), **SMALL
    )
    with pytest.raises(AddiVortesError):
        spec.fit(x, y)  # design has 2 encoded columns, matrix declares 3


def test_binary_probit_predicts_probabilities():
    rng = np.random.default_rng(6)
    n = 40
    x = rng.uniform(size=(n, 2))
    labels = (x[:, 0] > 0.5).astype(float)
    model = AddiVortes(
        seed=6, omega=1.5, response_family="binary_probit", **SMALL
    ).fit(x, labels)
    p = model.predict(x)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert model.response_family == "binary_probit"
    # Round-trips keep the family.
    assert type(model).from_json(model.to_json()).response_family == "binary_probit"


def test_binary_probit_rejects_non_binary_response():
    x, y = linear_data(30, 7)
    spec = AddiVortes(seed=7, omega=1.5, response_family="binary_probit", **SMALL)
    with pytest.raises(AddiVortesError):
        spec.fit(x, y)


def test_robust_t_fits_and_keeps_its_family():
    x, y = linear_data(40, 8)
    y = y.copy()
    y[::13] += 25.0  # gross outliers
    model = AddiVortes(
        seed=8, omega=1.5, response_family="robust_t", t_df=4.0, **SMALL
    ).fit(x, y)
    assert np.all(np.isfinite(model.predict(x)))
    lower, upper = model.prediction_interval(x, 0.9)
    assert np.all(lower <= upper)
    assert model.response_family == "robust_t"
    assert type(model).from_json(model.to_json()).response_family == "robust_t"


# ---------------------------------------------------------------------------
# The points reachable only through the config pass-through: `moves`,
# `coords`, `inclusion`, `scale`, `count_priors` and `basis` (DART, the
# heteroscedastic variance ensemble, the whole cell-basis point). Each of
# these fits through the payload alone — no per-point code exists in the
# binding for any of them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "point,payload",
    [
        ("inclusion", {"type": "uniform"}),
        ("inclusion", {"type": "weighted", "weights": [1.0, 3.0]}),
        ("inclusion", {"type": "dart", "alpha": 0.5}),
        ("scale", {"type": "pinned", "sigma_sq": 1.0}),
        ("scale", {"type": "h_variance", "m_prime": 5}),
        ("count_priors", {"type": "shifted_poisson_binomial"}),
        ("basis", {"type": "linear", "columns": [0], "sigma_beta_sq": 0.1}),
        ("membership", {"type": "softmax", "tau": 0.2}),
        ("coords", [{"type": "euclidean_normal", "sigma_c": 0.8}] * 2),
        (
            "moves",
            [
                {"name": "add_centre", "weight": 0.3},
                {"name": "remove_centre", "weight": 0.3},
                {"name": "change", "weight": 0.4},
            ],
        ),
    ],
    ids=lambda v: str(v)[:40],
)
def test_every_point_is_reachable_as_a_payload(point, payload):
    x, y = linear_data(40, 11)
    model = AddiVortes(seed=9, omega=1.5, **SMALL, **{point: payload}).fit(x, y)
    assert np.all(np.isfinite(model.predict(x)))


def test_dart_alpha_is_an_error_not_a_process_abort():
    # A non-positive alpha is rejected under the spec key before any
    # constructor sees it.
    with pytest.raises(AddiVortesError, match="alpha"):
        AddiVortes(seed=1, inclusion={"type": "dart", "alpha": 0.0})


def test_a_typod_key_is_loud_rather_than_ignored():
    with pytest.raises(TypeError, match="lambda_C"):
        AddiVortes(seed=1, lambda_C=5.0)


def test_a_wrong_length_setting_fails_at_fit_with_the_covariate_count():
    # Data-free validation cannot know p, so it does not guess; the error lands
    # at fit, naming the count it expected.
    x, y = linear_data(30, 12)
    spec = AddiVortes(seed=1, **SMALL, inclusion={"type": "weighted", "weights": [1.0]})
    with pytest.raises(AddiVortesError, match="p = 2"):
        spec.fit(x, y)


def test_the_spec_round_trips_through_pickle():
    import pickle

    spec = AddiVortes(seed=3, m=5, inclusion={"type": "dart", "alpha": 0.5})
    assert pickle.loads(pickle.dumps(spec)) == spec
