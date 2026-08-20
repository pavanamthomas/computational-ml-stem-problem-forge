"""AUC gap and ECE gap must both be positive."""

from __future__ import annotations

from typing import Any

MIN_AUC_GAP = 0.04
MIN_ECE_GAP = 0.02

REQUIRED_PROTOCOL = {
    "split": "in_sample_scores",
    "scaling": "none",
    "reported_score": "auc_and_ece",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if float(reference["gt1"]) < MIN_AUC_GAP:
        failures.append(f"AUC_B - AUC_A = {reference['gt1']} below {MIN_AUC_GAP}")
    ece_gap = float(reference["gt2"]["ece_gap"])
    if ece_gap < MIN_ECE_GAP:
        failures.append(f"ECE_B - ECE_A = {ece_gap} below {MIN_ECE_GAP}")
    diag = reference["diagnostics"]
    if diag["auc_b"] <= diag["auc_a"]:
        failures.append("B does not discriminate more than A")
    if diag["ece_b"] <= diag["ece_a"]:
        failures.append("B is not worse calibrated than A")
    if float(verifier["gt1"]) < MIN_AUC_GAP:
        failures.append("verifier AUC gap below minimum")
    return failures
