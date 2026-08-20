"""Naive overflow and the translation identity."""

from __future__ import annotations

from typing import Any

import numpy as np

REQUIRED_PROTOCOL = {
    "split": "not_applicable",
    "scaling": "max_subtraction",
    "reported_score": "stable_logsumexp",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if not np.isfinite(reference["gt1"]):
        failures.append("stable log-sum-exp is not finite")
    if float(reference["gt2"]["max_identity_error"]) > 1e-10:
        failures.append("translation identity failed")
    if reference["diagnostics"]["naive_is_finite"]:
        failures.append("naive log(sum(exp(x))) was finite; vector is not large enough")
    if not np.isfinite(verifier["gt1"]):
        failures.append("verifier log-sum-exp is not finite")
    return failures
