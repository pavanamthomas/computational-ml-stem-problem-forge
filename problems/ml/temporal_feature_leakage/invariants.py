"""Timestamp sentinel: a leaked rolling window uses time >= t."""

from __future__ import annotations

from typing import Any

MIN_GAP = 0.12
FUTURE_CORR_SENTINEL = 0.35

REQUIRED_PROTOCOL = {
    "split": "chronological",
    "scaling": "inside_cv",
    "reported_score": "lagged_only",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if float(reference["gt1"]) < MIN_GAP:
        failures.append(f"R² gap {reference['gt1']} below {MIN_GAP}")
    g2 = reference["gt2"]
    if float(g2["leaked_future_source_fraction"]) < 0.99:
        failures.append("leaked feature does not use future timestamps")
    if float(g2["lagged_future_source_fraction"]) != 0.0:
        failures.append("lagged feature uses a future timestamp")
    if float(g2["corr_leaked_with_y_lead1"]) < FUTURE_CORR_SENTINEL:
        failures.append("leaked feature is not correlated with y[t+1] above sentinel")
    if float(verifier["gt1"]) < MIN_GAP:
        failures.append("verifier R² gap below minimum")
    return failures
