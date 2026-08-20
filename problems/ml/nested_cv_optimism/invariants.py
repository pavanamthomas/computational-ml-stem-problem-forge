"""Selected C is an identity. Inner best score is not nested outer accuracy."""

from __future__ import annotations

from typing import Any

MIN_OPTIMISM = 0.01
C_GRID = [0.02, 0.1, 1.0, 50.0]

REQUIRED_PROTOCOL = {
    "split": "stratified_kfold",
    "scaling": "inside_cv",
    "reported_score": "nested_outer",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if float(reference["gt1"]) < MIN_OPTIMISM:
        failures.append(f"optimism {reference['gt1']} is not positive enough")
    chosen = float(reference["gt2"]["selected_C"])
    if chosen not in C_GRID:
        failures.append(f"selected C {chosen} is not in the grid")
    inner = float(reference["diagnostics"]["inner_best_score"])
    nested = float(reference["diagnostics"]["nested_outer_score"])
    if inner <= nested:
        failures.append("inner best score is not strictly larger than nested outer")
    if abs(chosen - inner) < 1e-12:
        failures.append("selected C was stored as if it were the accuracy")
    if float(verifier["gt1"]) < MIN_OPTIMISM:
        failures.append("verifier optimism is not positive")
    return failures
