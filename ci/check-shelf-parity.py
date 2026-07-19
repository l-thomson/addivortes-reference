#!/usr/bin/env python3
"""Shelf-parity gate: every shelf entry is either reachable from `ConfigSpec`
(and therefore from every language binding) or explicitly recorded as not
exposed, with a reason.

Usage: check-shelf-parity.py target/doc/addivortes.json

WHY THIS EXISTS
---------------
Without a gate, each binding hand-mirrors the shelf, nothing fails when a new
shelf entry is added and a binding is not updated, and the mirrors drift. That
is not a discipline problem; it is what hand-maintained parallel lists do,
and no amount of care in review reliably catches the omission.

The pass-through removes the mirrors. This gate is what stops them growing back:
a new shelf entry now fails CI until somebody decides whether a binding user can
select it.

WHY IT IS NOT CIRCULAR
----------------------
The reference — the set of types implementing each extension point trait — is read out of
rustdoc JSON, i.e. from the compiler. It does not come from `ConfigSpec`, which
is the thing under test. A check whose oracle is derived from its subject can
only prove self-consistency; the conformance suite in this crate learned that the
expensive way, and the lesson transfers.

So: adding `struct Chebyshev; impl PairwiseDistance for Chebyshev` is enough to
red this gate. Nobody has to remember to update a list.
"""

import json
import sys

# The ten extension-point traits. A type implementing one of these is a shelf entry.
POINT_TRAITS = {
    "ProposalMove": "moves",
    "CoordinateDistribution": "coord",
    "PairwiseDistance": "distance",
    "InclusionModel": "inclusion",
    "CellModel": "cell_model",
    "ResponseModel": "response",
    "ScaleModel": "scale",
    "CountPriors": "count_priors",
    "CellBasis": "basis",
    "MembershipKernel": "membership",
}

# Every shelf entry, classified. The value is either:
#   ("spec", "<SpecEnum>::<Variant>")  -- reachable as data from any binding
#   ("spec", "moves:<name>")           -- reachable by name through MoveSpec
#   ("not-exposed", "<reason>")        -- deliberately Rust-only; say why
#
# A shelf entry missing from this table fails the gate. That is the whole point:
# the decision is forced, rather than defaulting to "no binding can use it".
CLASSIFICATION = {
    # --- moves. Selected by name, with a weight. ---
    "AddCentre": ("spec", "moves:add_centre"),
    "RemoveCentre": ("spec", "moves:remove_centre"),
    "AddDimension": ("spec", "moves:add_dimension"),
    "RemoveDimension": ("spec", "moves:remove_dimension"),
    "Change": ("spec", "moves:change"),
    "Swap": ("spec", "moves:swap"),
    # --- coord ---
    "EuclideanNormal": ("spec", "CoordSpec::EuclideanNormal"),
    "WrappedNormal": ("spec", "CoordSpec::WrappedNormal"),
    # --- distance ---
    "Euclidean": ("spec", "DistanceSpec::Euclidean"),
    "Manhattan": ("spec", "DistanceSpec::Manhattan"),
    "Cosine": ("spec", "DistanceSpec::Cosine"),
    "Spherical": ("spec", "DistanceSpec::Spherical"),
    "Minkowski": ("spec", "DistanceSpec::Minkowski"),
    "Gower": ("spec", "DistanceSpec::Gower"),
    "Mahalanobis": ("spec", "DistanceSpec::Mahalanobis"),
    "ColumnMetrics": (
        "not-exposed",
        "the fit-time default, assembled by the engine from the encoded metric "
        "list; a caller selects it through `metrics`, not as a geometry",
    ),
    # --- inclusion ---
    "UniformInclusion": ("spec", "InclusionSpec::Uniform"),
    "WeightedInclusion": ("spec", "InclusionSpec::Weighted"),
    "DartInclusion": ("spec", "InclusionSpec::Dart"),
    # --- cell_model. Deliberately not exposed as data (see below). ---
    "GaussianCellModel": (
        "not-exposed",
        "sigma_mu_sq is engine-derived from k and m (and widened again for a "
        "probit fit); a hand-typed value would silently override the engine's "
        "own calibration",
    ),
    "WeightedGaussianModel": (
        "not-exposed",
        "same engine-derived sigma_mu_sq; the engine attaches this itself "
        "whenever the weights are fractional (RobustT, heteroscedastic scale)",
    ),
    "InvChiSqCellModel": (
        "not-exposed",
        "lambda is the engine's data-calibrated value; used as HVariance's "
        "variance-cell family, which the scale point wires",
    ),
    "LinearGaussianModel": (
        "spec",
        "BasisSpec::Linear",  # wired as the payload half of the basis point, q derived
    ),
    # --- response. The family route assembles the trio; the raw seam
    # is not exposed, and on its own it is a footgun (a bare RobustTStep feeds
    # fractional weights to the hard-assignment cell statistic). ---
    "AlbertChibProbit": (
        "not-exposed",
        "selected declaratively via response_family='binary_probit', which "
        "assembles the matching cell model and pinned scale",
    ),
    "RobustTStep": (
        "not-exposed",
        "selected declaratively via response_family='robust_t' + t_df, which "
        "assembles the matching weighted cell model and scale draw",
    ),
    # --- scale ---
    "PinnedSigma": ("spec", "ScaleSpec::Pinned"),
    "HVariance": ("spec", "ScaleSpec::HVariance"),
    "GlobalSigma": (
        "not-exposed",
        "takes the engine's data-calibrated lambda and IS what the engine "
        "attaches when the point is unset",
    ),
    "WeightedGlobalSigma": (
        "not-exposed",
        "as GlobalSigma; the engine attaches it for a RobustT response",
    ),
    # --- count_priors ---
    "ShiftedPoissonBinomial": ("spec", "CountPriorsSpec::ShiftedPoissonBinomial"),
    # --- basis ---
    "LinearBasis": ("spec", "BasisSpec::Linear"),
    # --- membership ---
    "SoftmaxKernel": ("spec", "MembershipSpec::Softmax"),
}

# The spec enums whose variants are checked to exist. `moves:` entries are
# matched by string in `MoveSpec::build`, not by a variant, and are covered by a
# Rust test (`every_builtin_move_is_reachable_from_a_spec`) instead.
SPEC_ENUMS = (
    "CoordSpec",
    "DistanceSpec",
    "InclusionSpec",
    "ScaleSpec",
    "CountPriorsSpec",
    "BasisSpec",
    "MembershipSpec",
)


def shelf_entries(doc):
    """{trait name: {implementing type names}}, read from the compiler."""
    found = {trait: set() for trait in POINT_TRAITS}
    for item in doc["index"].values():
        inner = item.get("inner")
        if not isinstance(inner, dict) or "impl" not in inner:
            continue
        impl = inner["impl"]
        trait = impl.get("trait")
        if not trait:
            continue
        name = trait.get("path", "").split("::")[-1]
        if name not in found:
            continue
        target = impl.get("for", {})
        # A blanket impl (`impl<T: PairwiseDistance> CellAssigner for T`) has a
        # generic, not a concrete path: it is not a shelf entry.
        resolved = target.get("resolved_path") if isinstance(target, dict) else None
        if not resolved:
            continue
        found[name].add(resolved["path"].split("::")[-1])
    return found


def spec_variants(doc):
    """{enum name: {variant names}} for the ConfigSpec enums."""
    variants = {}
    for item in doc["index"].values():
        inner = item.get("inner")
        if not isinstance(inner, dict) or "enum" not in inner:
            continue
        name = item.get("name")
        if name not in SPEC_ENUMS:
            continue
        names = set()
        for vid in inner["enum"].get("variants") or []:
            variant = doc["index"].get(str(vid)) or doc["index"].get(vid)
            if variant and variant.get("name"):
                names.add(variant["name"])
        variants[name] = names
    return variants


def main(path: str) -> int:
    with open(path) as f:
        doc = json.load(f)

    shelf = shelf_entries(doc)
    variants = spec_variants(doc)
    failures = []

    # 1. Every shelf entry the compiler can see must be classified.
    for trait, types in sorted(shelf.items()):
        point = POINT_TRAITS[trait]
        for type_name in sorted(types):
            if type_name not in CLASSIFICATION:
                failures.append(
                    f"{point}: `{type_name}` implements `{trait}` but is not classified.\n"
                    f"      It is a new shelf entry. Either give it a ConfigSpec variant "
                    f"(so Python and R can select it),\n"
                    f"      or record in ci/check-shelf-parity.py why it is deliberately "
                    f"Rust-only."
                )

    # 2. Every entry claimed as exposed must actually have its spec variant.
    #    This catches the drift in the other direction: a variant deleted or
    #    renamed while the shelf entry stays.
    for type_name, (kind, detail) in sorted(CLASSIFICATION.items()):
        if kind != "spec" or detail.startswith("moves:"):
            continue
        enum_name, variant = detail.split("::")
        if enum_name not in variants:
            failures.append(
                f"`{type_name}` is claimed reachable via `{detail}`, but the enum "
                f"`{enum_name}` was not found in the public API."
            )
        elif variant not in variants[enum_name]:
            failures.append(
                f"`{type_name}` is claimed reachable via `{detail}`, but `{enum_name}` "
                f"has no variant `{variant}` (renamed or removed?). "
                f"Bindings can no longer select this shelf entry."
            )

    # 3. A classified entry that no longer exists is stale bookkeeping.
    live = {t for types in shelf.values() for t in types}
    for type_name in sorted(CLASSIFICATION):
        if type_name not in live:
            failures.append(
                f"`{type_name}` is classified in ci/check-shelf-parity.py but no longer "
                f"implements any extension-point trait; remove the stale entry."
            )

    if failures:
        print("shelf parity: the shelf and the binding config surface disagree\n")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    exposed = sum(1 for k, _ in CLASSIFICATION.values() if k == "spec")
    total = len(CLASSIFICATION)
    print(
        f"shelf-parity: all {total} shelf entries classified "
        f"({exposed} reachable from every binding, {total - exposed} Rust-only by decision)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
