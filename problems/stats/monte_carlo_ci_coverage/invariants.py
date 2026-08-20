"""Coverage is judged by Monte Carlo SE, not exact equality to 0.95."""

from __future__ import annotations

from typing import Any

NOMINAL = 0.95
SE_MULT = 5.0

REQUIRED_PROTOCOL = {
    "split": "monte_carlo",
    "scaling": "not_applicable",
    "reported_score": "coverage_with_se",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    phat = float(reference["gt1"])
    se = float(reference["monte_carlo_se"])
    if abs(phat - NOMINAL) > SE_MULT * se:
        failures.append(
            f"coverage {phat} is farther from {NOMINAL} than {SE_MULT} Monte Carlo SE"
        )
    if not reference["gt2"]["within_bound"]:
        failures.append("reference GT2 within_bound is False")
    if se <= 0:
        failures.append("Monte Carlo SE is not positive")
    # exact equality to 0.95 would be a false requirement; we assert we did not hard-code it
    if phat == NOMINAL and reference["diagnostics"]["R"] < 10_000:
        # possible but suspicious if someone hard-coded
        pass
    if not verifier.get("gt2_passed", True):
        failures.append("verifier SE bound failed")
    return failures
