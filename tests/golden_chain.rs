//! Per-target golden regression tests (the last rung of the validation ladder, and the
//! reproducibility contract's enforcement): the sampled chain and the prediction surface are pinned bit for bit
//! per compilation target, through the public API only.
//!
//! Encoding is lossless (`f64::to_bits()` hex, never rounded formatting,
//! which would mask sub-precision drift) plus a SHA-256 line for one-line
//! diffs. Vector files live in `tests/golden/<arch>-<os>.txt`; when no vector
//! exists for the running target the test skips with guidance.
//! Regenerate deliberately with
//! `GOLDEN_WRITE=1 cargo test --test golden_chain`: a regeneration is a
//! chain-altering event and bumps the 0.y minor version.

use std::fmt::Write as _;

use addivortes::{AddiVortesConfig, Data, Sampler};
use sha2::{Digest, Sha256};

fn target_tag() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

fn vector_path(kind: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{kind}-{}.txt", target_tag()))
}

fn hex_bits(values: impl IntoIterator<Item = f64>, out: &mut String) {
    for (i, v) in values.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write!(out, "{:016x}", v.to_bits()).unwrap();
    }
    out.push('\n');
}

fn finish(mut body: String) -> String {
    let digest = Sha256::digest(body.as_bytes());
    body.push_str("sha256 ");
    for byte in digest {
        write!(body, "{byte:02x}").unwrap();
    }
    body.push('\n');
    body
}

/// Compare against (or, with GOLDEN_WRITE=1, write) the per-target vector.
fn check_or_write(kind: &str, rendered: String) {
    let path = vector_path(kind);
    if std::env::var_os("GOLDEN_WRITE").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, rendered).unwrap();
        panic!(
            "golden vector {} (re)written: commit it and re-run without GOLDEN_WRITE \
             (a regeneration is a deliberate 0.y-bump event)",
            path.display()
        );
    }
    let Ok(expected) = std::fs::read_to_string(&path) else {
        // Skip-with-guidance: no vector for this target yet.
        eprintln!(
            "golden[{kind}]: no vector for target {}: skipping. Capture one on a \
             pinned runner with GOLDEN_WRITE=1.",
            target_tag()
        );
        return;
    };
    if expected != rendered {
        let summary = |s: &str| s.lines().last().unwrap_or("").to_string();
        panic!(
            "golden[{kind}] drift on {}: {} != {}: a chain-altering change. If \
             deliberate, regenerate every target's vector in one PR and bump the 0.y \
             minor.",
            target_tag(),
            summary(&rendered),
            summary(&expected),
        );
    }
}

/// Fixed, arithmetic-generated fixture (no RNG involved in the data itself).
fn fixture() -> (Data, Vec<f64>) {
    let n = 12;
    let mut values = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64 / (n - 1) as f64;
        let b = ((i * 7) % n) as f64 / n as f64;
        values.push(a);
        values.push(b);
        y.push(2.0 * a - 1.5 * b + 0.25 * a * b);
    }
    (Data::new(values, n, 2).unwrap(), y)
}

/// The chain golden: 20 raw sweeps of the sampler, everything bit-serialised.
#[test]
fn golden_chain_vector_matches() {
    let (x, y) = fixture();
    let config = AddiVortesConfig::new(20_260_702).with_m(5).with_omega(1.5);
    let mut sampler = Sampler::new(config, &x, &y).unwrap();

    let mut body = String::new();
    for sweep in 0..20 {
        let draw = sampler.step().unwrap();
        write!(body, "sweep {sweep} sigma_sq ").unwrap();
        hex_bits([draw.sigma_sq], &mut body);
        for (j, tessellation) in draw.tessellations.iter().enumerate() {
            write!(body, "t {j} dims ").unwrap();
            for (k, dim) in tessellation.dims().iter().enumerate() {
                if k > 0 {
                    body.push(',');
                }
                write!(body, "{dim}").unwrap();
            }
            body.push('\n');
            write!(body, "t {j} centres ").unwrap();
            hex_bits(tessellation.centres().iter().copied(), &mut body);
            write!(body, "t {j} mus ").unwrap();
            hex_bits(tessellation.mus().iter().copied(), &mut body);
        }
    }
    check_or_write("chain", finish(body));
}

/// The prediction golden: fit → predict + predict_quantiles, bit-serialised
/// (predict consumes no RNG; this pins that surface too).
#[test]
fn golden_predict_vector_matches() {
    let (x, y) = fixture();
    let model = AddiVortesConfig::new(777)
        .with_m(4)
        .with_omega(1.5)
        .with_burn_in(5)
        .with_draws(10)
        .fit(&x, &y)
        .unwrap();

    let new_x = Data::from_rows(&[[0.1, 0.9], [0.5, 0.5], [0.95, 0.05]]).unwrap();
    let predictions = model.predict(&new_x).unwrap();
    // Repeat-call determinism (no RNG in predict): identical bits.
    let again = model.predict(&new_x).unwrap();
    assert_eq!(
        predictions.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );

    let quantiles = model.predict_quantiles(&new_x, &[0.25, 0.5, 0.75]).unwrap();

    let mut body = String::new();
    body.push_str("predict ");
    hex_bits(predictions, &mut body);
    body.push_str("quantiles ");
    hex_bits(quantiles.values().iter().copied(), &mut body);
    body.push_str("in_sample_rmse ");
    hex_bits([model.in_sample_rmse()], &mut body);
    check_or_write("predict", finish(body));
}
