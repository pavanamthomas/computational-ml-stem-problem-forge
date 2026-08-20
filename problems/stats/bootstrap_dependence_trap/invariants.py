"""Variance ratio under clustered intercepts."""

from __future__ import annotations

from typing import Any

MIN_RATIO = 1.8

REQUIRED_PROTOCOL = {
    "split": "cluster_bootstrap",
    "scaling": "not_applicable",
    "reported_score": "cluster_se",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if float(reference["gt1"]) < MIN_RATIO:
        failures.append(f"variance ratio {reference['gt1']} below {MIN_RATIO}")
    if float(reference["diagnostics"]["var_cluster"]) <= float(
        reference["diagnostics"]["var_iid"]
    ):
        failures.append("cluster bootstrap variance is not larger than iid")
    if float(verifier["gt1"]) < MIN_RATIO * 0.8:
        failures.append("analytic CRVE/iid ratio is too small")
    return failures
