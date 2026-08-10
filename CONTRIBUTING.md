# Contributing to AddiVortes

Thank you for your interest in contributing. This crate is a from-the-paper Rust
implementation of AddiVortes (Stone & Gosling, 2025). Statistical correctness and
reproducibility come before everything else, so a few rules below are stricter than a
typical crate's; please read them.

## Toolchain

- **Minimum supported Rust version (MSRV): 1.85**, edition 2024. `rust-toolchain.toml`
  pins local builds to `1.85.1` (the dev and reproducibility anchor). Rustup installs
  it automatically on first use, so there is nothing to set up by hand.
- CI additionally runs clippy on a fixed recent stable (more lints than 1.85's clippy),
  so a locally-green `just check` can occasionally still fail the CI clippy job.

## Extending the model

Most contributions are new entries on one of the ten extension points
(moves, coordinate laws, distance, inclusion, cell payloads, response
families, scale models, count priors, cell bases, membership). Start with
the crate documentation's "Extending" section and the extension point's own
module docs (what you implement, what the engine provides, and what already
sits on the shelf). Every point
has a copy-paste template under `examples/` and a one-command conformance check;
components destined for real inference also run the Geweke/SBC battery
(`calibration::getting_it_right`; see the worked examples in
`tests/calibration_acceptance.rs`).

## Landing a shelf entry

A merged shelf entry touches a fixed set of files; this is the list. In order:

1. The shelf file itself: one file per approach in `src/extensions/<point>/`,
   named for what it is (`manhattan.rs`), never for its provenance.
2. `src/extensions/<point>.rs`: the `mod` line and the re-export.
3. `src/lib.rs`: the crate-root re-export and the crate-map row.
4. `src/config_spec.rs`: a spec variant, mapping arm and test, so the entry
   is selectable from R and Python — or a recorded exclusion, with its
   reason, in `ci/check-shelf-parity.py`.
5. The `CLASSIFICATION` table in `ci/check-shelf-parity.py`: the parity gate
   fails the build on any shelf entry it cannot classify.
6. `ci/public-api.txt`: regenerate with `just api`. The snapshot gates
   (public API, shelf parity, doc coordinates) read rustdoc JSON and need
   the pinned nightly named in the justfile; on stable they fail everything
   they look at — that is the toolchain, not your code.
7. The extension point's module docs: add the entry to its shelf list.

Dependency policy: the crate has six mandatory direct dependencies (plus
optional `serde`), and cargo-deny gates the graph. A shelf entry that needs
a new dependency is a proposal-stage conversation (open a shelf-entry
issue), not a `Cargo.toml` edit.

## The check before you commit

Run the full local gate with [`just`](https://github.com/casey/just):

```sh
just check
```

It runs, in order:

1. `cargo fmt --check`: formatting must be clean.
2. `cargo clippy --all-targets --locked -- -D warnings`: no lint warnings allowed.
3. `cargo build --locked`: the committed `Cargo.lock` must be up to date.
4. `cargo nextest run --locked`, the same again with `--features serde` (the
   save/load surface only compiles behind the feature), then
   `cargo test --doc --locked`.
5. every extension-point template under `examples/` plus the embed walkthrough: each
   must run end-to-end, unedited.

`just check-full` additionally verifies the three snapshot gates (public-API
surface, doc coordinate-system audit, feature pins); it needs the pinned
nightly toolchain and `cargo-public-api` 0.52.0. See the `public-api` job in
`.github/workflows/ci.yml`.

Install the two runners once:

```sh
cargo install just
cargo install cargo-nextest
```

## The validation ladder

The model is validated in six rungs, from the narrowest check to the broadest. Each
rung can fail in a way the others cannot, which is why they all exist. `just check`
runs rungs 1, 2 and 6 on every commit; the middle three are slow, so they are
`#[ignore]`d and have to be asked for by name.

| # | Rung | What it establishes | Run it with |
|---|---|---|---|
| 1 | **oracles** | Hand-derived closed-form values: the move ratios computed on paper, plus the detailed-balance telescoping (AD×RD = 1). Catches arithmetic drift to 1e-10. | `just check` |
| 2 | **conformance** | Per-component necessary conditions, in seconds — the checks an extension author runs before anything expensive. Public API: `addivortes::conformance`. | `just check` |
| 3 | **calibration** | That the sampler targets the right posterior: SBC ([Talts et al. 2018]) and the Geweke joint-distribution test ([Geweke 2004], "Getting It Right"). Catches wrong-but-self-consistent samplers. | `just calibration`, then `just sbc-verdict` |
| 4 | **interval coverage** | That the credible intervals are honest: a frequentist score test of H0: coverage = 0.90 on the Friedman benchmark. *Interval* coverage — nothing to do with code coverage. | `just interval-coverage` |
| 5 | **reference comparison** | That the shipped default sits in the same statistical neighbourhood as the original authors' R package, at a pinned commit. A **comparison, never an oracle**: the reference has known bugs, and where we differ from it deliberately the offset is recorded, not zero. | `just reference-comparison` |
| 6 | **golden chain** | Bit-exact reproducibility per target: the sampled chain and prediction surface are pinned byte for byte. | `just check` |

Rungs 3–5 are what CI runs: the `calibration` workflow's `battery` job runs rung 3 on
every commit that touches the sampler, and `release-gate` runs `calibration-full`,
`interval-coverage` and `reference-comparison` at release sizes before a version bump.
Anything touching a sampling kernel needs rung 3 before it merges — a conformance pass
(rung 2) is a necessary condition, not evidence of correctness.

Two things worth knowing before you run rung 5. It needs R, and it installs the pinned
reference package — the recipe puts it in `target/reference-lib` rather than your
default library on purpose, because clobbering your own `AddiVortes` install is exactly
what the script's version guard is there to catch. And it uses `R_LIBS`, not
`R_LIBS_USER`: if your `~/.Renviron` sets `R_LIBS_USER`, that beats the environment
variable and your override is silently ignored.

A note on `L1`/`L2` in this codebase: they mean the **norms** (Manhattan, Euclidean) in
the distance modules, and nothing else. The rungs above have names, not numbers.

[Talts et al. 2018]: https://arxiv.org/abs/1804.06788
[Geweke 2004]: https://doi.org/10.1198/016214504000001132

## Reproducibility rules (please read)

The crate guarantees that the same seed yields the same result, bit for bit, on a given
target. Two rules protect that, and both are enforced mechanically:

1. **Never call the standard-library floating-point transcendentals**: `f64::ln`,
   `f64::exp`, `f64::acos`, `f64::powf`, and so on. The standard library is permitted to
   produce platform-dependent results for these. Use the deterministic wrappers in the
   `mathsfn` module instead (`mathsfn::ln`, `mathsfn::exp`, …). A clippy lint fails the
   build if a banned call slips in; if you need a transcendental that `mathsfn` does not
   yet expose, add a wrapper there.
2. **Do not override `RUSTFLAGS`** when building or testing: in particular no
   `-Ctarget-cpu=native` and no `target-feature=+fma`. The crate pins
   `-Ctarget-cpu=generic` in `.cargo/config.toml`; overriding it can change results.

## Platform note

The performance and supply-chain gates run on Linux x86_64 and are authoritative
there. The reproducibility (golden) gate runs on the full target matrix — Linux
x86_64, macOS ARM, Windows — on every PR, and weekly against fresh runner images
(`scheduled.yml`). Developing on macOS or Windows is fine.

## Golden-vector capture runbook (per-target)

The reproducibility contract is enforced by per-target golden vectors in
`tests/golden/{chain,predict}-<arch>-<os>.txt`. To capture the first vector for
a new target
(or to regenerate after a deliberate chain-altering change):

1. Use the pinned CI image for that target (`ubuntu-24.04`, `macos-15`,
   `windows-2025`) or the matching local machine; toolchain comes from
   `rust-toolchain.toml` (1.85.1); the math path is `libm`, so vectors are
   independent of the host CPU within a target.
2. Run `GOLDEN_WRITE=1 cargo test --test golden_chain`: the test writes the
   vectors and then deliberately fails so a capture can never look like a pass.
3. Re-run `cargo test --test golden_chain` without the variable; both golden
   tests must now pass.
4. Commit the new files. All targets' vectors regenerate together in one
   reviewed change, alongside the 0.y minor bump. A lone drifting vector is
   a bug, not a re-baseline.

The target-independent anchors (splitmix64 key bytes, raw ChaCha8 stream) must
never change with a re-baseline; if they did, the seed expansion itself was
altered: a different, more serious event.
