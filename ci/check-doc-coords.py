#!/usr/bin/env python3
"""Doc gate: every public function returning f64 data must state its
coordinate system in its rustdoc, checked mechanically from rustdoc
JSON, not by eye.

Usage: check-doc-coords.py target/doc/addivortes.json
"""

import json
import sys

# A doc comment counts as stating the coordinate system if it contains any of
# these (case-insensitive). Deliberately generous: the gate exists to catch
# accessors with NO statement at all.
KEYWORDS = (
    "scaled space",
    "scaled-space",
    "response scale",
    "response-scale",
    "raw",
    "radians",
    "coordinate system",
    "caller",
    "log",  # log-density / log-ratio functions state their own scale
    "probabilit",
    "weight",  # inclusion weights are dimensionless relative preferences
    "count",  # integer-valued counts carried as f64
)

# Functions that return f64 machinery rather than data in a coordinate system.
ALLOWLIST = {
    "eq",  # PartialEq plumbing
    # mathsfn: pinned pure-maths wrappers over libm; dimensionless numbers in,
    # dimensionless numbers out; each states its own mathematical domain.
    "exp",
    "exp_m1",
    "powf",
    "powi",
    "erfc",
}


def returns_f64(type_tree) -> bool:
    return '"f64"' in json.dumps(type_tree)


def trait_impl_members(doc) -> set:
    """Ids of items living inside trait impls: the coordinate-system contract
    is stated once, on the trait declaration (which this gate checks); the
    impls inherit it and routinely carry no rustdoc of their own."""
    members = set()
    for item in doc["index"].values():
        inner = item.get("inner")
        if isinstance(inner, dict) and "impl" in inner:
            impl = inner["impl"]
            if impl.get("trait") is not None:
                members.update(impl.get("items") or [])
    return members


def main(path: str) -> int:
    with open(path) as f:
        doc = json.load(f)
    skip = trait_impl_members(doc)
    failures = []
    for item_id, item in doc["index"].items():
        inner = item.get("inner")
        if not isinstance(inner, dict) or "function" not in inner:
            continue
        try:
            if int(item_id) in skip or item_id in skip:
                continue
        except ValueError:
            if item_id in skip:
                continue
        name = item.get("name") or "?"
        if name in ALLOWLIST:
            continue
        output = inner["function"]["sig"].get("output")
        if output is None or not returns_f64(output):
            continue
        docs = (item.get("docs") or "").lower()
        if not docs:
            failures.append(f"{name}: no rustdoc at all")
        elif not any(k in docs for k in KEYWORDS):
            failures.append(f"{name}: rustdoc does not state a coordinate system")
    if failures:
        print("numeric accessors missing coordinate-system statements:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("doc-coords: all public f64-returning functions state their coordinate system")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
