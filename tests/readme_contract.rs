//! The README carries the reproducibility contract verbatim (checked
//! mechanically, not by eye). The same paragraph must appear in the
//! crate-level rustdoc.

/// The normative reproducibility contract text (do not edit one copy alone).
const CONTRACT: &[&str] = &[
    "A chain is reproducible given the same seed, the same addivortes version,",
    "the same compilation target, built with this crate's default release",
    "no overriding RUSTFLAGS (in particular no -Ctarget-cpu=native",
    "no target-feature=+fma).",
    "Any change that alters the sampled chain for a fixed",
    "seed bumps the 0.y minor version and regenerates the golden vectors",
    "deliberately; patch releases guarantee bit-identical chains",
    "enforced by a golden-chain",
    "regression test in CI (Linux x86_64, macOS ARM, Windows).",
    "Any major bump of rand, rand_core, rand_distr, or libm is treated as",
    "chain-altering by definition.",
];

fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn readme_contains_the_reproducibility_contract_verbatim() {
    let readme = normalise(include_str!("../README.md"));
    for fragment in CONTRACT {
        let fragment = normalise(fragment);
        assert!(
            readme.contains(&fragment),
            "README.md lost this piece of the reproducibility contract: {fragment:?}"
        );
    }
}

#[test]
fn crate_docs_contain_the_reproducibility_contract_verbatim() {
    let lib = include_str!("../src/lib.rs");
    let docs: String = normalise(
        &lib.lines()
            .filter_map(|l| l.trim_start().strip_prefix("//!"))
            .collect::<Vec<_>>()
            .join(" "),
    );
    for fragment in CONTRACT {
        let fragment = normalise(fragment);
        assert!(
            docs.contains(&fragment),
            "crate docs lost this piece of the reproducibility contract: {fragment:?}"
        );
    }
}
