"""Leaked AUC is high; legitimate AUC is not. Permuting X is the sentinel."""

from __future__ import annotations

from typing import Any

REQUIRED_PROTOCOL = {
    "split": "held_out",
    "scaling": "none",
    "reported_score": "legitimate_auc",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    leaked = float(reference["gt2"]["auc_leaked"])
    legit = float(reference["gt1"])
    if not (0.82 <= leaked <= 0.97):
        failures.append(f"leaked AUC {leaked} is not in the convincing band")
    if legit >= leaked - 0.08:
        failures.append("legitimate AUC is not materially lower than leaked")
    if float(reference["gt2"]["auc_leaked_perm_X"]) < 0.80:
        failures.append("leaked AUC collapsed after permuting X; leak is not label-driven")
    if float(reference["gt2"]["auc_legit_perm_X"]) > 0.70:
        failures.append("legitimate AUC did not collapse after permuting X")
    if not verifier.get("gt3_passed", False):
        failures.append("independent residual-correlation leak test failed")
    return failures
