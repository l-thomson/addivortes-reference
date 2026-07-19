# Developer command shortcuts. Run `just check` before committing.

# The full local pre-commit gate (mirrors the CI fast-PR jobs that need no
# extra toolchain: fmt, clippy, lockfile, tests incl. serde, every extension point
# template).
check: fmt-check clippy lockcheck test templates

# Everything `check` runs plus the snapshot gates (public API, doc
# coordinate-system audit, feature pins, R man pages). Needs the pinned
# nightly and cargo-public-api 0.52.0 (see the `public-api` job in ci.yml),
# plus R with roxygen2 8.0.0 (see the `r-docs` job).
check-full: check api-check doc-coords pins-check r-docs-check

# Formatting must be clean.
fmt-check:
    cargo fmt --check

# Lints must pass with no warnings (includes the mathsfn/libm
# disallowed-methods ban). CI additionally runs clippy on a pinned recent
# stable (1.96.0) with more lints than 1.85's; install it once with
# `rustup toolchain install 1.96.0` and this recipe will still work either way.
clippy:
    cargo clippy --all-targets --locked -- -D warnings

# The committed Cargo.lock must be up to date.
lockcheck:
    cargo build --locked

# Unit tests (via nextest) plus documentation tests (nextest does not run
# doc-tests). The serde leg matters: the save/load surface only compiles
# behind the feature, and CI runs it, so skipping it locally is how
# serde-only breakage reaches CI unseen.
test:
    cargo nextest run --locked
    cargo nextest run --locked --features serde
    cargo test --doc --locked

# Every extension-point template must pass its conformance/self-checks end-to-end,
# unedited, plus the embed walkthrough: exactly the set CI runs.
templates:
    cargo run --example template_moves --locked
    cargo run --example template_coord --locked
    cargo run --example template_distance --locked
    cargo run --example template_inclusion --locked
    cargo run --example template_cell_model --locked
    cargo run --example template_response --locked
    cargo run --example template_scale --locked
    cargo run --example template_basis --locked
    cargo run --example template_count_priors --locked
    cargo run --example template_membership --locked
    cargo run --example template_embed --locked

# --- The validation ladder's upper rungs -----------------------------------
# Rungs 1, 2 and 6 (oracles, conformance, golden chain) are ordinary tests:
# `just check` runs them. The rungs below are `#[ignore]`d because they are
# slow, so they need naming explicitly. CONTRIBUTING.md documents the ladder.

# Rung 3, calibration: the SBC + Geweke joint-distribution gates, at the
# calibration leg's sizes. Emits SBC rank CSVs to target/stat-gates; the
# uniformity verdict is R's (see `just sbc-verdict`). Selector discipline
# mirrors CI: one run per selector with `--no-tests fail`, never a union, so
# a stale clause goes red instead of being silently dropped.
[doc("ladder rung 3: SBC + Geweke calibration gates (slow; then `just sbc-verdict`)")]
calibration:
    cargo nextest run --locked --cargo-profile determinism --run-ignored ignored-only \
        --no-fail-fast --no-tests fail -E 'test(/^stat_gates::(sbc_ranks|geweke|pure_prior)/)'
    cargo nextest run --locked --cargo-profile determinism --run-ignored ignored-only \
        --no-fail-fast --no-tests fail -E 'test(/^stat_gates::battery_public_driver/)'
    cargo nextest run --locked --cargo-profile determinism --run-ignored ignored-only \
        --no-fail-fast --no-tests fail -E 'binary(calibration_acceptance)'

# Rung 4, interval coverage: Friedman n=150 p=10 at the paper defaults, score
# test of H0: coverage = 0.90. Coverage of the credible intervals, not code
# coverage. Override the fit count with INTERVAL_COVERAGE_FITS.
[doc("ladder rung 4: frequentist coverage of the credible intervals (not code coverage)")]
interval-coverage:
    cargo nextest run --locked --cargo-profile determinism --run-ignored ignored-only \
        --no-capture --no-tests fail -E 'test(/^stat_gates::interval_coverage_friedman$/)'

# Rung 5, reference comparison: |RMSE_rust - RMSE_R| against the authors' R
# package at the pinned commit. A comparison, never an oracle.
#
# Needs R plus the pinned reference installed to a scratch library. Do NOT
# install it into your default library: that clobbers whatever AddiVortes you
# already have, and the version guard in ci/reference-compare.R exists to
# catch exactly that mix-up. Note R_LIBS, not R_LIBS_USER — a ~/.Renviron
# setting R_LIBS_USER beats the environment variable, so R_LIBS_USER is
# silently ignored on such a machine while R_LIBS still prepends.
[doc("ladder rung 5: RMSE vs the authors' R package at the pinned commit (needs R; comparison, not oracle)")]
reference-comparison:
    #!/usr/bin/env bash
    set -euo pipefail
    ref_lib=target/reference-lib
    # Anchor on the variable name: the workflow also pins action SHAs, and a
    # bare 40-hex grep picks up actions/checkout's instead.
    ref_commit=$(grep -oE 'REF_COMMIT: [0-9a-f]{40}' .github/workflows/release-gate.yml | grep -oE '[0-9a-f]{40}')
    if [ -z "${ref_commit}" ]; then
        echo "could not read REF_COMMIT from .github/workflows/release-gate.yml" >&2
        exit 1
    fi
    if ! Rscript -e "q(status = !requireNamespace('AddiVortes', quietly = TRUE, lib.loc = '${ref_lib}'))" 2>/dev/null; then
        echo "installing the pinned reference (johnpaulgosling/AddiVortes @ ${ref_commit}) into ${ref_lib}"
        rm -rf target/reference-src && mkdir -p "${ref_lib}"
        git clone --quiet https://github.com/johnpaulgosling/AddiVortes target/reference-src
        git -C target/reference-src checkout --quiet "${ref_commit}"
        R CMD INSTALL --library="${ref_lib}" target/reference-src
    fi
    cargo nextest run --locked --cargo-profile determinism --run-ignored ignored-only \
        --no-capture --no-tests fail -E 'test(/^stat_gates::reference_comparison_emit_fixture_and_rmse$/)'
    R_LIBS="${ref_lib}" Rscript ci/reference-compare.R

# The SBC uniformity verdict for the ranks `just calibration` emitted: the
# ECDF simultaneous-confidence-band test, delegated to R's bayesplot so the
# validator is more trusted than the code it judges.
[doc("the R ECDF-band verdict on the SBC ranks `just calibration` emitted (needs R + bayesplot)")]
sbc-verdict:
    Rscript ci/sbc-ecdf-check.R \
        --expect-fail target/stat-gates/sbc-injection-rd.csv \
        target/stat-gates/sbc-small-p.csv \
        target/stat-gates/sbc-spherical.csv

# Build the Python binding into its venv and run the pytest suite (incl.
# the cross-language golden bit-identity test). One-time setup:
#   python3 -m venv addivortes-py/.venv
#   addivortes-py/.venv/bin/pip install maturin pytest numpy scikit-learn arviz
python:
    cargo clippy --manifest-path addivortes-py/Cargo.toml --all-targets --locked -- -D warnings
    cd addivortes-py && .venv/bin/maturin develop --release && .venv/bin/python -m pytest tests -q

# Build the R binding and run the testthat suite (incl. the cross-language
# golden bit-identity test). Needs R with testthat, plus posterior, loo,
# and parsnip for the adapter tests (they skip when absent). Wrappers and
# man pages are generated, not hand-edited: after changing the extendr
# surface, reinstall and regenerate the wrappers with
#   Rscript -e '.Call("wrap__make_addivortesr_wrappers", TRUE, "addivortesr")'
# and the man pages with `just r-docs` (which CI diffs against).
r:
    cargo clippy --manifest-path addivortes-r/src/rust/Cargo.toml --all-targets --locked -- -D warnings
    R CMD INSTALL --no-multiarch addivortes-r
    Rscript -e 'library(addivortesr); testthat::test_dir("addivortes-r/tests/testthat", stop_on_failure = TRUE)'

# Regenerate the pinned feature-tree snapshot (ci/feature-pins.txt) after a
# deliberate dependency or feature change; CI diffs against it.
pins:
    cargo tree -e features --locked | sed 's| (/[^)]*)||' > ci/feature-pins.txt

# Verify the feature-tree snapshot without regenerating (what CI does).
pins-check:
    cargo tree -e features --locked | sed 's| (/[^)]*)||' | diff -u ci/feature-pins.txt -

# Regenerate the public-API snapshot (ci/public-api.txt) after a deliberate API
# change; CI diffs against it. Needs the pinned nightly and cargo-public-api
# 0.52.0 installed (see the `public-api` job in .github/workflows/ci.yml).
api:
    RUSTUP_TOOLCHAIN=nightly-2026-06-01 cargo public-api --simplified --all-features > ci/public-api.txt

# Verify the public-API snapshot without regenerating (what CI does).
api-check:
    RUSTUP_TOOLCHAIN=nightly-2026-06-01 cargo public-api --simplified --all-features | diff -u ci/public-api.txt -

# The doc gate: every public f64-returning function must state its
# coordinate system (checked from rustdoc JSON, mirrors the CI step).
doc-coords:
    RUSTUP_TOOLCHAIN=nightly-2026-06-01 cargo rustdoc --lib -- -Zunstable-options --output-format json
    python3 ci/check-doc-coords.py target/doc/addivortes.json

# Regenerate the R man pages + NAMESPACE from the roxygen blocks in
# addivortes-r/R/ after editing any `#'` comment; CI diffs against them.
# Pinned to roxygen2 8.0.0: Rd output is version-dependent and 8.0.0 rewrites
# Config/roxygen2/version in DESCRIPTION, so regenerating with another version
# reds CI on an otherwise-current tree.
# `load_code = "source"` is passed explicitly because roxygenise() resolves its
# load strategy before reading DESCRIPTION: the package's own `load = "source"`
# is ignored, and the default pkgload strategy compiles the Rust staticlib just
# to write text files (~2.5s becomes minutes).
r-docs:
    Rscript -e 'stopifnot(packageVersion("roxygen2") == "8.0.0"); roxygen2::roxygenise("addivortes-r", load_code = "source")'

# Verify the man pages are current (what CI does). Note this regenerates in
# place: if it fails, the fix is already in your working tree, so review the
# diff and commit it.
r-docs-check:
    #!/usr/bin/env bash
    set -euo pipefail
    just r-docs
    paths="addivortes-r/man addivortes-r/NAMESPACE addivortes-r/DESCRIPTION"
    # --intent-to-add first: a plain `git diff` does not see untracked files,
    # so an entirely missing man page would pass unnoticed.
    git add -A --intent-to-add -- $paths
    if ! git diff --exit-code -- $paths; then
        echo "addivortes-r/man is stale (regenerated above; review and commit)" >&2
        exit 1
    fi
