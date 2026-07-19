"""Cross-language bit-identity: the Python binding reproduces the same
per-target golden predict vector the Rust tests pin (tests/golden/), bit
for bit. f64s cross the FFI unchanged, so any drift here is a real chain
or predict-surface change: the same tripwire, now spanning the boundary.
"""

import platform
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from addivortes import AddiVortes

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "tests" / "golden"


def _target_tag():
    machine = platform.machine()
    arch = {"AMD64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    os_name = {"linux": "linux", "darwin": "macos", "win32": "windows"}[sys.platform]
    return f"{arch}-{os_name}"


def _hex_bits(values):
    return ",".join(
        format(struct.unpack("<Q", struct.pack("<d", v))[0], "016x") for v in values
    )


def _fixture():
    """The exact fixture of tests/golden_chain.rs (arithmetic, no RNG)."""
    n = 12
    x = np.empty((n, 2))
    y = np.empty(n)
    for i in range(n):
        a = i / (n - 1)
        b = ((i * 7) % n) / n
        x[i] = (a, b)
        y[i] = 2.0 * a - 1.5 * b + 0.25 * a * b
    return x, y


def test_python_reproduces_the_rust_golden_predict_vector():
    vector = GOLDEN_DIR / f"predict-{_target_tag()}.txt"
    if not vector.exists():
        pytest.skip(f"no golden vector for target {_target_tag()}")

    x, y = _fixture()
    model = AddiVortes(seed=777, m=4, omega=1.5, burn_in=5, draws=10).fit(x, y)

    new_x = np.array([[0.1, 0.9], [0.5, 0.5], [0.95, 0.05]])
    predictions = model.predict(new_x)
    quantiles = model.predict_quantiles(new_x, [0.25, 0.5, 0.75])

    expected = {}
    for line in vector.read_text().splitlines():
        key, _, payload = line.partition(" ")
        expected[key] = payload

    assert _hex_bits(predictions) == expected["predict"]
    assert _hex_bits(quantiles.ravel()) == expected["quantiles"]
    assert _hex_bits([model.in_sample_rmse]) == expected["in_sample_rmse"]
