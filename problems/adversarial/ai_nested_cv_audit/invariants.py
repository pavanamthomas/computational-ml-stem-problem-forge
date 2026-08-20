"""Protocol stages for nested grouped CV. Order is part of the specification."""

from __future__ import annotations

from typing import Any

REQUIRED_PROTOCOL = {
    "split": "group_kfold",
    "scaling": "inside_cv",
    "reported_score": "nested_outer",
}

EXPECTED_EARLIEST = {
    "c1_random_split_despite_groups": "split_integrity",
    "c2_scaling_outside_cv": "preprocessing_isolation",
    "c3_inner_cv_as_final": "estimation_protocol",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    proto = reference["gt2"]
    for key, val in REQUIRED_PROTOCOL.items():
        if proto.get(key) != val:
            failures.append(f"reference protocol {key}={proto.get(key)!r} != {val!r}")
    earliest = (verifier.get("gt3") or {}).get("earliest", {})
    for cid, stage in EXPECTED_EARLIEST.items():
        if earliest.get(cid) != stage:
            failures.append(
                f"{cid} diagnosed as {earliest.get(cid)!r}, expected {stage!r}"
            )
    if float(reference["gt1"]) < 0.45 or float(reference["gt1"]) > 0.95:
        failures.append(f"nested grouped score {reference['gt1']} is outside a sane band")
    return failures
