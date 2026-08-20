"""Solve a small convex QP; do not treat solver status as optimality.

What problem is being solved?
    Minimise a strictly convex quadratic with linear inequalities and verify
    KKT residuals independently of the solver message.

What assumptions are required?
    Q SPD, Gx <= h, x >= 0, SLSQP.

Why was this method chosen?
    The KKT system is small enough to assemble by hand. That is the check.

What alternative method could have been used?
    An active-set QP solver, or CVXPY. Different codes, same KKT object.

What can go wrong?
    Trusting status. Not recomputing the objective. Dropping slackness.

How is correctness independently checked?
    Objective from x. Feasibility of Gx-h and x. Stationarity with
    independently recovered multipliers.

What can legitimately be concluded?
    On this instance the returned x is feasible and approximately stationary.

What cannot be concluded?
    Global behaviour of SLSQP, or nonconvex QPs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize

from problemforge.rng import get_rng

N = 4
N_INEQ = 3


def make_qp(seed: int = 2026) -> dict[str, NDArray[np.float64]]:
    rng = get_rng(seed)
    A = rng.normal(0.0, 1.0, size=(N, N))
    Q = A.T @ A + 2.5 * np.eye(N)
    c = rng.normal(0.0, 1.0, size=N)
    G = rng.normal(0.0, 1.0, size=(N_INEQ, N))
    x0 = 0.2 * np.ones(N)
    h = G @ x0 + rng.uniform(0.4, 1.2, size=N_INEQ)
    return {"Q": Q, "c": c, "G": G, "h": h}


def objective(x: NDArray[np.float64], Q: NDArray[np.float64], c: NDArray[np.float64]) -> float:
    return float(0.5 * x @ Q @ x + c @ x)


def solve(seed: int = 2026) -> dict[str, object]:
    qp = make_qp(seed)
    Q, c, G, h = qp["Q"], qp["c"], qp["G"], qp["h"]

    def fun(x):
        return 0.5 * x @ Q @ x + c @ x

    def jac(x):
        return Q @ x + c

    cons = LinearConstraint(G, -np.inf, h)
    bounds = [(0.0, None)] * N
    res = minimize(
        fun,
        x0=np.full(N, 0.2),
        jac=jac,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    x = np.asarray(res.x, dtype=float)
    obj = objective(x, Q, c)
    viol = np.maximum(G @ x - h, 0.0)
    bound_viol = np.maximum(-x, 0.0)
    return {
        "gt1": obj,
        "gt2": {
            "max_ineq_violation": float(viol.max()),
            "max_bound_violation": float(bound_viol.max()),
        },
        "seed": int(seed),
        "diagnostics": {
            "x": x.tolist(),
            "solver_success": bool(res.success),
            "solver_message": str(res.message),
            "solver_fun": float(res.fun),
            "Q": Q.tolist(),
            "c": c.tolist(),
            "G": G.tolist(),
            "h": h.tolist(),
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    x = np.array(out["diagnostics"]["x"], dtype=float)  # type: ignore[index]
    x_bad = x + 0.35
    Q = np.array(out["diagnostics"]["Q"], dtype=float)  # type: ignore[index]
    c = np.array(out["diagnostics"]["c"], dtype=float)  # type: ignore[index]
    out["gt1"] = objective(x_bad, Q, c)
    out["diagnostics"]["x"] = x_bad.tolist()  # type: ignore[index]
    out["broken"] = True
    return out
