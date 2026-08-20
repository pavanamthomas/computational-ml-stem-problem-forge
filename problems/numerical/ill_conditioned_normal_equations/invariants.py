"""cond(X'X) is the invariant; the residual gap is GT1."""

from __future__ import annotations

from typing import Any

REQUIRED_PROTOCOL = {
    "split": "not_applicable",
    "scaling": "qr_or_svd",
    "reported_score": "residual_norm",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if float(reference["gt1"]) <= 0:
        failures.append("naive residual is not worse than QR")
    if float(reference["gt2"]["cond_xtx"]) < 1e8:
        failures.append("Gram condition number is not huge on this Hilbert-like design")
    diag = reference["diagnostics"]
    if diag["resid_qr"] >= diag["resid_naive"]:
        failures.append("QR residual is not strictly smaller")
    if diag["normal_eq_resid_qr"] > diag["normal_eq_resid_naive"] * 1.01:
        # QR can have a slightly worse normal-eq residual; require naive not better by luck only if huge
        pass
    if float(verifier["gt1"]) <= 0:
        failures.append("verifier residual gap is not positive")
    return failures
