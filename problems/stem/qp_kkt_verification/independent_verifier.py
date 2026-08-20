"""KKT residual assembled without the solver's success flag.

Multipliers are recovered by non-negative least squares on the stationarity
equation, which is a different code path from SLSQP.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from problemforge.rng import get_rng

N = 4
N_INEQ = 3
KKT_TOL = 1e-5
FEAS_TOL = 1e-7


def _qp(seed: int):
    rng = get_rng(seed)
    A = rng.normal(0.0, 1.0, size=(N, N))
    Q = A.T @ A + 2.5 * np.eye(N)
    c = rng.normal(0.0, 1.0, size=N)
    G = rng.normal(0.0, 1.0, size=(N_INEQ, N))
    x0 = 0.2 * np.ones(N)
    h = G @ x0 + rng.uniform(0.4, 1.2, size=N_INEQ)
    return Q, c, G, h


def _solve_x(Q, c, G, h) -> np.ndarray:
    from scipy.optimize import LinearConstraint, minimize

    def fun(x):
        return 0.5 * x @ Q @ x + c @ x

    def jac(x):
        return Q @ x + c

    res = minimize(
        fun,
        x0=np.full(N, 0.2),
        jac=jac,
        bounds=[(0.0, None)] * N,
        constraints=LinearConstraint(G, -np.inf, h),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    return np.asarray(res.x, dtype=float)


def verify(seed: int = 2026) -> dict[str, object]:
    Q, c, G, h = _qp(seed)
    x = _solve_x(Q, c, G, h)
    obj = float(0.5 * x @ Q @ x + c @ x)
    slack = h - G @ x
    feas_ineq = float(np.max(np.maximum(-slack, 0.0)))
    feas_bnd = float(np.max(np.maximum(-x, 0.0)))
    # Stationarity: Qx + c + G.T λ - μ = 0, λ>=0, μ>=0.
    # Recover (λ, μ) by NNLS: [G.T, -I] [λ; μ] ≈ -(Qx+c)
    A_eq = np.column_stack([G.T, -np.eye(N)])
    rhs = -(Q @ x + c)
    mult, _ = nnls(A_eq, rhs)
    lam, mu = mult[:N_INEQ], mult[N_INEQ:]
    stat = A_eq @ mult - rhs
    stat_norm = float(np.linalg.norm(stat))
    cs_ineq = float(np.max(np.abs(lam * slack)))
    cs_bnd = float(np.max(np.abs(mu * x)))
    gt2_passed = feas_ineq <= FEAS_TOL and feas_bnd <= FEAS_TOL
    gt3_passed = stat_norm <= KKT_TOL and cs_ineq <= 1e-4 and cs_bnd <= 1e-4
    return {
        "gt1": obj,
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": f"ineq viol={feas_ineq:.3e}, bound viol={feas_bnd:.3e}",
            "max_ineq_violation": feas_ineq,
            "max_bound_violation": feas_bnd,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": (
                f"stationarity L2={stat_norm:.3e}, "
                f"CS ineq={cs_ineq:.3e}, CS bound={cs_bnd:.3e}; nnls multipliers"
            ),
            "stationarity": stat_norm,
            "cs_ineq": cs_ineq,
            "cs_bnd": cs_bnd,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
    }
