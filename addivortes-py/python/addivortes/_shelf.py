"""Keyword sugar over the config payload.

Nothing here *defines* a shelf entry — the core does that, once. These helpers
only spell a payload in a way that reads well in Python, and validate it eagerly
so a mistake is reported where it was written rather than at ``fit``.

Every one of them is optional. The raw dict always works::

    AddiVortes(seed=1, distance={"type": "minkowski", "p": 3.0})

which is why a shelf entry with no sugar here is still reachable from Python the
day the crate ships it.
"""

from addivortes._native import AddiVortesError
from addivortes._spec import validate


def _validated(payload, field):
    """Check one extension point payload against the core, on its own."""
    validate({"seed": 0, field: payload})
    return payload


class Distance:
    """An assignment geometry.

    Construct with the factories; each validates against the core's own rules
    at the point of construction, with the core's own message.
    """

    __slots__ = ("payload",)

    def __init__(self, payload):
        self.payload = _validated(dict(payload), "distance")

    def __repr__(self):
        kind = self.payload["type"]
        extra = {k: v for k, v in self.payload.items() if k != "type"}
        return f"Distance.{kind}({extra})" if extra else f"Distance.{kind}()"

    def __eq__(self, other):
        return isinstance(other, Distance) and self.payload == other.payload

    def as_spec(self):
        return dict(self.payload)

    @staticmethod
    def euclidean():
        """Squared Euclidean — the paper's geometry."""
        return Distance({"type": "euclidean"})

    @staticmethod
    def manhattan():
        """Manhattan (L1)."""
        return Distance({"type": "manhattan"})

    @staticmethod
    def cosine():
        """Cosine distance from the mid-range origin of scaled space."""
        return Distance({"type": "cosine"})

    @staticmethod
    def spherical():
        """Great-circle geometry, for an all-angular design."""
        return Distance({"type": "spherical"})

    @staticmethod
    def minkowski(p):
        """Minkowski (L_p) of order ``p >= 1``."""
        return Distance({"type": "minkowski", "p": float(p)})

    @staticmethod
    def gower(columns):
        """Gower mixed numeric/categorical geometry.

        ``columns`` has one entry per *raw* column, in ``metrics`` order:
        ``"numeric"``, or ``("categorical", n_levels)``.

        You state the level count because the engine cannot infer intent from a
        sample: a level absent from *this* data is still a level.
        """
        parsed = []
        for column in columns:
            if column == "numeric":
                parsed.append({"type": "numeric"})
            elif (
                isinstance(column, (tuple, list))
                and len(column) == 2
                and column[0] == "categorical"
            ):
                parsed.append({"type": "categorical", "levels": int(column[1])})
            else:
                raise AddiVortesError(
                    f"unknown gower column {column!r}: expected 'numeric' or "
                    f"('categorical', n_levels)"
                )
        return Distance({"type": "gower", "columns": parsed})

    @staticmethod
    def mahalanobis(precision):
        """Mahalanobis geometry from a square precision matrix over the
        **encoded** (post one-hot) design width.

        Validated here as square, finite, symmetric and positive definite. The
        *width* can only be checked against the fitted design, so a matrix of
        the wrong size is an error at ``fit``.
        """
        rows = [[float(v) for v in row] for row in precision]
        return Distance({"type": "mahalanobis", "precision": rows})


def normalise_membership(value):
    """Accept ``("softmax", tau)`` as well as the raw payload."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        name, tau = value
        return {"type": str(name), "tau": float(tau)}
    return value


def as_payload(value):
    """Unwrap a helper object to its dict; pass a raw dict straight through."""
    return value.as_spec() if hasattr(value, "as_spec") else value
