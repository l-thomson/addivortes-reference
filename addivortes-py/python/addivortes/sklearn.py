"""scikit-learn estimators wrapping AddiVortes.

Requires the ``sklearn`` extra (``pip install addivortes[sklearn]``).

Two estimators, following the scikit-learn contract in full:
``get_params`` / ``set_params`` / ``clone``, ``validate_data`` input
checking (``n_features_in_`` / ``feature_names_in_``), ``NotFittedError``,
``__sklearn_tags__``, and pickling of fitted estimators (the native model
round-trips through the crate's validated JSON):

- :class:`AddiVortesRegressor`: Gaussian or robust-t regression, with
  ``predict(X, return_std=True)`` posterior-predictive uncertainty in the
  ``BayesianRidge`` / ``GaussianProcessRegressor`` idiom.
- :class:`AddiVortesClassifier`: binary classification through the
  Binary-AddiVortes probit family (``predict_proba`` on the probability
  scale).

Hyperparameters left as ``None`` use the Rust engine's defaults, so the
defaults have exactly one home, in the crate. The engine's ``omega``
default (3.0) requires ``n_features > 3``; pass e.g. ``omega=1.5`` for
narrower designs.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.multiclass import check_classification_targets, type_of_target
    from sklearn.utils.validation import check_is_fitted
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "addivortes.sklearn requires scikit-learn: pip install 'addivortes[sklearn]'"
    ) from exc

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover - scikit-learn 1.3-1.5

    def validate_data(estimator, X, y="no_validation", **kwargs):
        return estimator._validate_data(X, y=y, **kwargs)


from addivortes import AddiVortes

# The one place a parameter list survives, and it is forced: scikit-learn
# discovers an estimator's hyperparameters by *introspecting its ``__init__``
# signature* (`get_params`, `clone`, `set_params`, the conformance suite all
# depend on it), so `**kwargs` is not an option here. Every other layer of this
# binding names nothing.
#
# The component *values* are still opaque payloads passed straight to the core,
# so a new shelf entry needs no change here either — only a brand-new eleventh
# extension point would.
_PARAM_NAMES = (
    "seed",
    "m",
    "burn_in",
    "draws",
    "thinning",
    "nu",
    "q",
    "k",
    "lambda_c",
    "omega",
    "sigma_c",
    "metrics",
    "moves",
    "coords",
    "distance",
    "inclusion",
    "scale",
    "count_priors",
    "basis",
    "membership",
)


class _AddiVortesBase(BaseEstimator):
    """Shared plumbing: parameters held verbatim (sklearn convention), the
    model spec built at ``fit`` time."""

    def __init__(
        self,
        seed=0,
        m=None,
        burn_in=None,
        draws=None,
        thinning=None,
        nu=None,
        q=None,
        k=None,
        lambda_c=None,
        omega=None,
        sigma_c=None,
        metrics=None,
        moves=None,
        coords=None,
        distance=None,
        inclusion=None,
        scale=None,
        count_priors=None,
        basis=None,
        membership=None,
    ):
        self.seed = seed
        self.m = m
        self.burn_in = burn_in
        self.draws = draws
        self.thinning = thinning
        self.nu = nu
        self.q = q
        self.k = k
        self.lambda_c = lambda_c
        self.omega = omega
        self.sigma_c = sigma_c
        self.metrics = metrics
        self.moves = moves
        self.coords = coords
        self.distance = distance
        self.inclusion = inclusion
        self.scale = scale
        self.count_priors = count_priors
        self.basis = basis
        self.membership = membership

    def _spec(self, response_family, t_df=None):
        kwargs = {name: getattr(self, name) for name in _PARAM_NAMES if name != "seed"}
        return AddiVortes(
            self.seed, response_family=response_family, t_df=t_df, **kwargs
        )

    def _validate_fit(self, X, y, **kwargs):
        """``validate_data`` for fit: float64, dense, finite; sets
        ``n_features_in_`` / ``feature_names_in_``."""
        X, y = validate_data(self, X, y, dtype=np.float64, order="C", **kwargs)
        if X.shape[0] < 2:
            raise ValueError(
                f"n_samples = {X.shape[0]}: AddiVortes needs at least 2 samples"
            )
        return X, y

    def _validate_predict(self, X):
        check_is_fitted(self)
        return validate_data(
            self, X, dtype=np.float64, order="C", reset=False
        )


class AddiVortesRegressor(RegressorMixin, _AddiVortesBase):
    """AddiVortes regression with Gaussian (default) or Student-t errors.

    Parameters mirror :class:`addivortes.AddiVortes`; ``response_family``
    is ``'gaussian'`` or ``'robust_t'`` (the latter requires ``t_df``).

    After ``fit``, the native :class:`addivortes.FittedModel` is available
    as ``model_`` (posterior draws, per-draw predictions, pointwise
    log-likelihood, save/load).
    """

    def __init__(
        self,
        seed=0,
        m=None,
        burn_in=None,
        draws=None,
        thinning=None,
        nu=None,
        q=None,
        k=None,
        lambda_c=None,
        omega=None,
        sigma_c=None,
        metrics=None,
        moves=None,
        coords=None,
        distance=None,
        inclusion=None,
        scale=None,
        count_priors=None,
        basis=None,
        membership=None,
        response_family="gaussian",
        t_df=None,
    ):
        super().__init__(
            seed=seed,
            m=m,
            burn_in=burn_in,
            draws=draws,
            thinning=thinning,
            nu=nu,
            q=q,
            k=k,
            lambda_c=lambda_c,
            omega=omega,
            sigma_c=sigma_c,
            metrics=metrics,
            moves=moves,
            coords=coords,
            distance=distance,
            inclusion=inclusion,
            scale=scale,
            count_priors=count_priors,
            basis=basis,
            membership=membership,
        )
        self.response_family = response_family
        self.t_df = t_df

    def fit(self, X, y):
        """Fit on ``X`` (n_samples, n_features) and numeric ``y``."""
        X, y = self._validate_fit(X, y, y_numeric=True)
        y = np.ascontiguousarray(y, dtype=np.float64)
        spec = self._spec(self.response_family, self.t_df)
        self.model_ = spec.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        """Posterior-mean predictions; with ``return_std=True`` also the
        posterior-predictive standard deviation per point (the law of total
        variance over kept draws: fit variance + expected error variance).

        Under ``response_family='robust_t'`` the error variance is
        sigma^2 * t_df / (t_df - 2), so ``return_std`` requires
        ``t_df > 2`` (below that the Student-t predictive has no finite
        variance; use ``prediction_interval`` instead).
        """
        X = self._validate_predict(X)
        mean = self.model_.predict(X)
        if not return_std:
            return mean
        draws = self.model_.predict_draws(X)
        sigma = np.asarray(self.model_.sigma())
        if self.model_.response_family == "robust_t":
            df = self.model_.t_df
            if df <= 2.0:
                raise ValueError(
                    "return_std requires t_df > 2 (the Student-t predictive "
                    "has no finite variance below that); use "
                    "prediction_interval instead"
                )
            noise_var = (sigma**2 * df / (df - 2.0)).mean()
        else:
            noise_var = (sigma**2).mean()
        std = np.sqrt(draws.var(axis=0) + noise_var)
        return mean, std

    def predict_quantiles(self, X, probs):
        """Posterior-predictive quantiles at ``probs`` per row:
        shape (n_samples, len(probs))."""
        X = self._validate_predict(X)
        return self.model_.predict_quantiles(X, list(probs))

    def prediction_interval(self, X, level=0.9):
        """Posterior-predictive interval for new observations: (lower, upper)."""
        X = self._validate_predict(X)
        return self.model_.prediction_interval(X, level)

    def credible_interval(self, X, level=0.9):
        """Credible interval for the mean surface: (lower, upper)."""
        X = self._validate_predict(X)
        return self.model_.credible_interval(X, level)


class AddiVortesClassifier(ClassifierMixin, _AddiVortesBase):
    """Binary classification through the Binary-AddiVortes probit family.

    Labels must be exactly two classes; they are mapped to {0, 1} in sorted
    order and ``predict_proba`` returns ``P(class)`` columns in
    ``classes_`` order.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        """Fit on ``X`` (n_samples, n_features) and binary labels ``y``."""
        X, y = self._validate_fit(X, y)
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y")
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the "
                f"target is {y_type}."
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                "Classifier can't train when only one class is present: "
                f"got {self.classes_[0]!r}"
            )
        labels = np.ascontiguousarray(
            (y == self.classes_[1]).astype(np.float64)
        )
        spec = self._spec("binary_probit")
        self.model_ = spec.fit(X, labels)
        return self

    def predict_proba(self, X):
        """P(class) per row, columns in ``classes_`` order."""
        X = self._validate_predict(X)
        p1 = self.model_.predict(X)
        return np.column_stack([1.0 - p1, p1])

    def predict_log_proba(self, X):
        """Log of :meth:`predict_proba`, probabilities clamped away from 0."""
        proba = self.predict_proba(X)
        eps = np.finfo(np.float64).eps
        return np.log(np.clip(proba, eps, 1.0))

    def predict(self, X):
        """The most probable class per row."""
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]
