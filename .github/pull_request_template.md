# Summary

<!-- What changes and why, in a sentence or two. -->

## Extension point

<!-- Which extension point (folder under src/extensions/) this touches, or
     "engine" / "docs" / "CI" if none. -->

## Validation

<!-- Tick what ran; delete lines that do not apply.
- [ ] `just check` green locally (fmt, clippy, lockfile, tests incl. serde, templates)
- [ ] conformance check for the touched extension point (name it)
- [ ] Geweke/SBC battery leg (required for anything touching a sampling kernel; paste the D statistics)
- [ ] golden chain untouched (a red golden test means the change is chain-altering; see below)
-->

## Chain-altering?

<!-- The same seed must produce the same chain, bit for bit, within a patch
     version. If this change alters the sampled chain for a fixed seed, say
     so here: it needs the 0.y minor bump and a deliberate golden-vector
     regeneration in the same PR (CONTRIBUTING.md runbook). If not, state
     "No, golden chain green". -->
