---
name: Bug report
about: Something is wrong: a crash, a wrong result, or a broken reproducibility promise
labels: bug
---

## What happened

<!-- What you observed, and what you expected instead. -->

## Reproduction

<!-- The smallest config + data shape that shows it. Because chains are
     seeded, a full reproduction is usually tiny. Please include: -->

- seed:
- addivortes version / commit:
- target (OS + arch) and toolchain:
- config (the `with_*` calls, or "defaults"):
- custom extension components in play (or "none"):

```rust
// minimal reproduction here
```

## Reproducibility note

<!-- If two runs with the same seed disagree, or the same seed differs
     across machines, say so explicitly: that class of bug outranks
     everything else in this crate. Include both outputs if you can. -->
