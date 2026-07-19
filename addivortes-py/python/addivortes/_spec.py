"""The Python side of the config pass-through.

Everything here is a convenience over one payload: a ``ConfigSpec`` dict that
the Rust core maps to a model configuration. This module names the *keywords*
Python users type; it does not define the shelf. The shelf is defined once, in
Rust, and reaches Python through the payload — so a new distance, inclusion
model or scale model is selectable from Python the moment the crate ships it,
with no edit here.

That is the point of the design, and it is worth being blunt about why: any
binding that lists the shelf a second time eventually lists it *differently*.
Nothing fails when you forget to update the copy.

The escape hatch is always available. Every component keyword accepts a raw dict in
the core's own vocabulary::

    AddiVortes(seed=1, distance={"type": "manhattan"})
    AddiVortes(seed=1, inclusion={"type": "dart", "alpha": 0.5})

so a shelf entry this module has no sugar for is still reachable today.
"""

import json

from addivortes._native import AddiVortes as _NativeAddiVortes
from addivortes._native import validate_spec as _validate_spec

# The keys that take a structured payload. Listed here only so a stray keyword
# is a clean TypeError from Python rather than a confusing error from serde;
# the *contents* of each payload are the core's business, not this module's.
_PAYLOAD_KEYS = (
    "moves",
    "coords",
    "distance",
    "inclusion",
    "scale",
    "count_priors",
    "basis",
    "membership",
)

_SCALAR_KEYS = (
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
    "response_family",
    "t_df",
)


def _clean(value):
    """Recursively drop ``None`` values, so an unset keyword means "use the
    crate default" rather than "set this to null"."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def build_spec(seed, **kwargs):
    """Assemble a ``ConfigSpec`` payload from keyword arguments.

    Unknown keywords raise, rather than being silently dropped — the same
    contract the core enforces with ``deny_unknown_fields``. A silently ignored
    ``lambda_C`` is exactly the failure a config surface exists to prevent.
    """
    allowed = set(_SCALAR_KEYS) | set(_PAYLOAD_KEYS)
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(
            f"unknown keyword argument(s) {unknown}; expected any of "
            f"{sorted(allowed)}"
        )
    spec = {"seed": seed}
    for key, value in kwargs.items():
        if value is None:
            continue
        spec[key] = _clean(value)
    return spec


def validate(spec):
    """Validate a spec payload against the core, without any data.

    Raises ``AddiVortesError`` with the crate's own message. The checks that
    need the covariate count (how many coordinate laws, how many inclusion
    weights) can only happen at ``fit``, where the data is.
    """
    _validate_spec(json.dumps(spec))


def native(spec):
    """The native model object for a spec payload (validated on construction)."""
    return _NativeAddiVortes(json.dumps(spec))
