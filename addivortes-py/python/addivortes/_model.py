"""The ``AddiVortes`` model spec: keywords in, one config payload out."""

from addivortes._shelf import as_payload, normalise_membership
from addivortes._spec import build_spec, native

# Fields whose value may be given as a helper object (or a raw dict).
_PAYLOAD_FIELDS = ("distance", "inclusion", "scale", "count_priors", "basis")


class AddiVortes:
    """A model specification: seed, hyperparameters, and shelf selection.

    Immutable once constructed. ``fit`` builds a fresh config each time, so one
    spec can fit many datasets.

    Every hyperparameter defaults to the crate's default — passing ``None``
    means "do not override" — so the defaults have exactly one home, in Rust.

    Nine of the ten extension points are selected with a payload in the core's
    own vocabulary (the cell payload family is Rust-only by design), which means
    a shelf entry added to the crate is reachable from here the day it ships,
    with no change to this file::

        AddiVortes(seed=1, distance={"type": "manhattan"})
        AddiVortes(seed=1, inclusion={"type": "dart", "alpha": 0.5})
        AddiVortes(seed=1, scale={"type": "h_variance", "m_prime": 40})
        AddiVortes(seed=1, basis={"type": "linear", "columns": [0], "sigma_beta_sq": 0.1})
        AddiVortes(seed=1, membership={"type": "softmax", "tau": 0.1})
        AddiVortes(seed=1, moves=[{"name": "add_centre", "weight": 0.3}, ...])
        AddiVortes(seed=1, coords=[{"type": "euclidean_normal", "sigma_c": 0.8}, ...])

    ``Distance`` (and the ``("softmax", tau)`` tuple) are sugar over exactly
    these payloads.

    Authoring a *new* component — your own move, cell model or membership kernel —
    is a Rust-side activity: a trait implementation is code, and no payload can
    carry one. See the Rust crate documentation's "Extending" section.
    """

    def __init__(
        self,
        seed,
        *,
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
        response_family=None,
        t_df=None,
        scale=None,
        count_priors=None,
        basis=None,
        membership=None,
    ):
        fields = {
            "moves": moves,
            "coords": coords,
            "distance": distance,
            "inclusion": inclusion,
            "scale": scale,
            "count_priors": count_priors,
            "basis": basis,
            "membership": (
                normalise_membership(membership) if membership is not None else None
            ),
        }
        for name in _PAYLOAD_FIELDS:
            if fields[name] is not None:
                fields[name] = as_payload(fields[name])

        self.spec = build_spec(
            seed,
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
            response_family=response_family,
            t_df=t_df,
            **fields,
        )
        # Validated at construction, against the core's own rules: a bad value is
        # reported where it was written. What cannot be checked yet is anything
        # sized by the covariate count (how many coordinate laws, how many
        # inclusion weights) — the data settles that, so it is checked at `fit`.
        self._native = native(self.spec)

    @property
    def response_family(self):
        """The selected response family name."""
        return self._native.response_family

    def fit(self, x, y):
        """Fit on ``x`` (n×p, raw caller scale) and ``y`` (length n)."""
        return self._native.fit(x, y)

    def fit_chains(self, x, y, n_chains):
        """Fit ``n_chains`` independent chains (chain 0 is bit-identical to
        ``fit``)."""
        return self._native.fit_chains(x, y, n_chains)

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self.spec.items())
        return f"AddiVortes({inner})"

    def __eq__(self, other):
        return isinstance(other, AddiVortes) and self.spec == other.spec

    def __reduce__(self):
        # The payload IS the object: rebuilding from it is exact, and needs no
        # per-parameter pickle mirror.
        return (_from_spec, (self.spec,))


def _from_spec(spec):
    seed = spec["seed"]
    kwargs = {k: v for k, v in spec.items() if k != "seed"}
    return AddiVortes(seed, **kwargs)
