"""Majority-classifier identities at low prevalence."""

from __future__ import annotations

from typing import Any

MIN_GAP = 0.15

REQUIRED_PROTOCOL = {
    "split": "in_sample_geometry",
    "scaling": "none",
    "reported_score": "pr_auc_and_cost",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if float(reference["gt1"]) < MIN_GAP:
        failures.append(f"ROC-AUC - PR-AUC = {reference['gt1']} below {MIN_GAP}")
    g2 = reference["gt2"]
    if float(g2["majority_recall"]) != 0.0:
        failures.append("majority recall is not 0")
    if abs(float(g2["accuracy_identity_error"])) > 1e-12:
        failures.append("majority accuracy is not 1-prevalence")
    if not (0.01 <= float(g2["prevalence"]) <= 0.04):
        failures.append(f"prevalence {g2['prevalence']} is not near 2%")
    half = reference["diagnostics"]["metrics_threshold_0.5"]
    if half["accuracy"] < 0.9:
        failures.append("accuracy at 0.5 is not high on this DGP")
    if half["recall"] > 0.6:
        failures.append("recall at 0.5 is unexpectedly high; DGP may have drifted")
    if not verifier.get("gt3_passed", False):
        failures.append("fixture confusion identity failed")
    return failures
