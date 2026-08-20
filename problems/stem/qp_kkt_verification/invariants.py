"""Primal feasibility is not stationarity. Solver status is not KKT."""

from __future__ import annotations

from typing import Any

FEAS_TOL = 1e-7

REQUIRED_PROTOCOL = {
    "split": "not_applicable",
    "scaling": "not_applicable",
    "reported_score": "kkt_residual",
}


def check_invariants(
    reference: dict[str, Any], verifier: dict[str, Any], seed: int = 2026
) -> list[str]:
    failures: list[str] = []
    if float(reference["gt2"]["max_ineq_violation"]) > FEAS_TOL:
        failures.append("primal inequality infeasible")
    if float(reference["gt2"]["max_bound_violation"]) > FEAS_TOL:
        failures.append("bound infeasible")
    if not verifier.get("gt3_passed", False):
        failures.append("KKT residual check failed")
    # Recomputed objective must match solver.fun closely if x is the same.
    fun = float(reference["diagnostics"]["solver_fun"])
    if abs(float(reference["gt1"]) - fun) > 1e-8:
        failures.append("recomputed objective disagrees with solver.fun")
    return failures
