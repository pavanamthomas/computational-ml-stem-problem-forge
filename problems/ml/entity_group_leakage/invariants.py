"""Invariants for entity-group leakage.

GT2 is a split property: naive KFold shares entities across the fold
boundary; GroupKFold does not. That count is not a re-computation of the
accuracy gap.
"""

from __future__ import annotations

from typing import Any

MIN_GAP = 0.08

REQUIRED_PROTOCOL = {
    "split": "group_kfold",
    "scaling": "inside_cv",
    "reported_score": "grouped_cv",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    gap = float(reference["gt1"])
    if gap < MIN_GAP:
        failures.append(f"GT1 gap {gap:.4f} is below required {MIN_GAP}")
    diag = reference["diagnostics"]
    if float(diag["naive_cv_acc"]) <= float(diag["grouped_cv_acc"]):
        failures.append("naive CV accuracy is not strictly larger than grouped")
    sentinel = reference["gt2"]
    if float(sentinel["naive_leaked_entity_occurrences"]) <= 0:
        failures.append("naive split leaked-entity sentinel is not positive")
    if float(sentinel["grouped_leaked_entity_occurrences"]) != 0.0:
        failures.append("GroupKFold leaked-entity sentinel is not zero")
    vgap = float(verifier["gt1"])
    if vgap < MIN_GAP:
        failures.append(f"verifier gap {vgap:.4f} is below required {MIN_GAP}")
    if seed != reference.get("seed"):
        failures.append("reference did not echo the requested seed")
    return failures
