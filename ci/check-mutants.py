#!/usr/bin/env python3
"""Release-gate mutation leg: aggregate sharded cargo-mutants outcomes into
the kill-rate verdict.

Rules:
- kill-rate = caught / (caught + missed), EXCLUDING unviable and timeout;
- run INVALID (exit 3, distinct from red) if timeouts > 5% of viable mutants;
- survivors on the reviewed allowlist (ci/mutants-allowlist.txt) are excluded
  from `missed`;
- floor: kill-rate >= 0.90.

Usage: check-mutants.py <outcomes.json> [<outcomes.json> ...]
where each file is a shard's mutants.out/outcomes.json.
"""

import json
import sys
from pathlib import Path

ALLOWLIST = Path(__file__).parent / "mutants-allowlist.txt"
FLOOR = 0.90
TIMEOUT_INVALID_FRACTION = 0.05


def mutant_key(outcome: dict) -> str:
    scenario = outcome.get("scenario", {})
    mutant = scenario.get("Mutant", scenario) or {}
    return "{}:{}: {}".format(
        mutant.get("file", "?"),
        (mutant.get("span", {}).get("start", {}) or {}).get("line", "?"),
        mutant.get("name", outcome.get("log_path", "?")),
    )


def main(paths: list[str]) -> int:
    allow = set()
    if ALLOWLIST.exists():
        for line in ALLOWLIST.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                allow.add(line)

    caught = missed = unviable = timeout = 0
    survivors: list[str] = []
    allowed_hits: list[str] = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        for outcome in data.get("outcomes", []):
            summary = outcome.get("summary")
            if summary == "CaughtMutant":
                caught += 1
            elif summary == "MissedMutant":
                key = mutant_key(outcome)
                if key in allow:
                    allowed_hits.append(key)
                else:
                    missed += 1
                    survivors.append(key)
            elif summary == "Unviable":
                unviable += 1
            elif summary == "Timeout":
                timeout += 1
            # baseline/other outcomes are not mutants

    viable = caught + missed + len(allowed_hits)
    print(
        f"mutants: caught={caught} missed={missed} allowlisted={len(allowed_hits)} "
        f"unviable={unviable} timeout={timeout}"
    )
    for key in survivors:
        print(f"SURVIVOR: {key}")
    for key in allowed_hits:
        print(f"allowlisted survivor: {key}")

    if viable == 0:
        print("INVALID: no viable mutants were exercised")
        return 3
    if timeout > TIMEOUT_INVALID_FRACTION * viable:
        print(
            f"INVALID: {timeout} timeouts exceed {TIMEOUT_INVALID_FRACTION:.0%} of "
            f"{viable} viable mutants: the run's kill-rate is not interpretable "
            "(raise --timeout or fix the slow tests, then re-run)"
        )
        return 3

    kill_rate = caught / (caught + missed) if (caught + missed) else 1.0
    print(f"kill-rate (excl. unviable+timeout+allowlisted): {kill_rate:.3f} (floor {FLOOR})")
    if kill_rate < FLOOR:
        print(
            "FAIL: kill-rate below the floor: kill the survivors with sharper "
            "oracles or add them to ci/mutants-allowlist.txt WITH a review note"
        )
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
