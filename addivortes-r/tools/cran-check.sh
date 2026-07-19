#!/usr/bin/env bash
# CRAN rehearsal for the R binding, runnable entirely in the private repo.
#
# R CMD check copies the package out of the repo, so the dev-time path
# dependency on the crate (src/rust/Cargo.toml -> ../../..) cannot survive.
# This script stages what the PUBLISHED package will be: the addivortes
# crate source embedded in the package (via `cargo package`, i.e. exactly
# the .crate file a crates.io release ships), the path dependency rewritten
# to it, and the registry dependencies (extendr-api & co.) vendored so the
# build runs offline, since CRAN forbids network access at install time.
#
# At release the path dependency flips to the published crates.io
# version instead and `cargo vendor` picks the
# crate up like any other registry dependency; the embed step here stands
# in for that while the crate is unpublished.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

echo "== staging self-contained package in $stage"
cp -r "$repo_root/addivortes-r" "$stage/addivortesr"
rm -rf "$stage/addivortesr/src/rust/target" \
  "$stage/addivortesr/src/"*.o "$stage/addivortesr/src/"*.so \
  "$stage/addivortesr/tools"

echo "== embedding the addivortes crate (cargo package output)"
(cd "$repo_root" && cargo package --no-verify --allow-dirty --quiet)
crate_archive=$(ls "$repo_root"/target/package/addivortes-*.crate | sort | tail -1)
mkdir -p "$stage/addivortesr/src/rust/vendor-local"
tar -xzf "$crate_archive" -C "$stage/addivortesr/src/rust/vendor-local"
crate_dir=$(basename "$crate_archive" .crate)
sed -i.bak "s|path = \"../../..\"|path = \"./vendor-local/$crate_dir\"|" \
  "$stage/addivortesr/src/rust/Cargo.toml"
rm "$stage/addivortesr/src/rust/Cargo.toml.bak"

echo "== vendoring registry dependencies for the offline build"
(cd "$stage/addivortesr/src/rust" && cargo vendor --locked vendor >/dev/null)
# The dev .Rbuildignore drops .cargo dirs (working state); the staged
# package NEEDS its offline config, so keep it under a visible name.
sed -i.bak '/cargo/d' "$stage/addivortesr/.Rbuildignore" && rm "$stage/addivortesr/.Rbuildignore.bak"
# Note: relative paths in a `--config <file>` resolve against cargo's cwd,
# which is src/ (where Makevars runs), hence rust/vendor, not vendor.
cat > "$stage/addivortesr/src/rust/cargo-vendor-config.toml" <<'EOF'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "rust/vendor"
EOF
# Point cargo at the package-local config and forbid the network.
sed -i.bak \
  's|cargo build --lib --release --locked|cargo build --lib --release --locked --offline --config ./rust/cargo-vendor-config.toml|' \
  "$stage/addivortesr/src/Makevars"
rm "$stage/addivortesr/src/Makevars.bak"

echo "== R CMD build + check (offline)"
cd "$stage"
R CMD build addivortesr
# Don't hard-require Suggests (adapter tests skip cleanly when a suggested
# package is missing on the checking machine), and skip the CRAN-incoming
# remote probes. `--as-cran` additionally fetches CRAN metadata over the
# network during the dependency check, so it is opt-in (AS_CRAN=true on a
# machine with CRAN access); the plain check battery runs everywhere.
export _R_CHECK_CRAN_INCOMING_=false
export _R_CHECK_CRAN_INCOMING_REMOTE_=false
export _R_CHECK_FORCE_SUGGESTS_=false
check_flags=(--no-manual)
if [ "${AS_CRAN:-false}" = "true" ]; then
  check_flags+=(--as-cran)
fi
R CMD check "${check_flags[@]}" addivortesr_*.tar.gz || status=$?
echo "== full check log"
cat addivortesr.Rcheck/00check.log
if [ -n "${status:-}" ] && [ -f addivortesr.Rcheck/00install.out ]; then
  echo "== install log tail"
  tail -40 addivortesr.Rcheck/00install.out
fi
exit "${status:-0}"
