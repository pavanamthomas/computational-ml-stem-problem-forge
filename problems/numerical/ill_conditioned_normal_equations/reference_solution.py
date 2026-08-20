"""Naive normal equations versus QR on a Hilbert-like design.

What problem is being solved?
    Show that (X'X)^{-1}X'y can return finite, ordinary-looking coefficients
    while losing residual accuracy relative to QR/lstsq.

What assumptions are required?
    Hilbert-like columns, float64, a tiny y perturbation.

Why was this method chosen?
    Hilbert matrices are the classic ill-conditioned least-squares example.
    The residual and the normal-equation residual are the checks.

What alternative method could have been used?
    SVD (used as GT3), or a regularised inverse (a different estimand).

What can go wrong?
    Trusting finite β. Forming the Gram matrix. Ignoring ||X'Xβ - X'y||.

How is correctness independently checked?
    SVD/lstsq residual. Condition number of X'X.

What can legitimately be concluded?
    On this design the naive inverse is a worse residual minimiser than QR.

What cannot be concluded?
    That inv(X'X) is always unusable, or that QR recovers a causal slope.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng

N = 12
P = 8
# When LAPACK refuses inv(X'X), the naive method has already failed.
# Use a finite residual so GT1 remains a number on every platform.
INVERSION_FAIL_RESIDUAL = 1.0e16


def hilbert_like(n: int = N, p: int = P) -> NDArray[np.float64]:
    i = np.arange(1, n + 1)[:, None]
    j = np.arange(1, p + 1)[None, :]
    return 1.0 / (i + j - 1.0)


def generate(seed: int = 2026) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = get_rng(seed)
    X = hilbert_like()
    beta_true = np.array([1.0, -0.5, 0.25, -0.1, 0.05, -0.02, 0.01, -0.005])
    y = X @ beta_true + 1e-10 * rng.normal(size=N)
    return X, y


def naive_beta(X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """Form (X'X)^{-1} X'y. Returns (beta, inversion_failed).

    Some LAPACK builds return a finite garbage inverse. Others raise
    ``LinAlgError`` on the same Gram matrix. Both are the pathology:
    forming the inverse of X'X is not a well-defined float64 operation
    on this design.
    """
    gram = X.T @ X
    try:
        beta = np.linalg.inv(gram) @ (X.T @ y)
        return np.asarray(beta, dtype=float), False
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1], dtype=float), True


def qr_beta(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    q, r = np.linalg.qr(X, mode="reduced")
    return np.linalg.solve(r, q.T @ y)


def solve(seed: int = 2026) -> dict[str, object]:
    X, y = generate(seed)
    b_naive, inversion_failed = naive_beta(X, y)
    b_qr = qr_beta(X, y)
    if inversion_failed:
        resid_naive = INVERSION_FAIL_RESIDUAL
        ne_naive = INVERSION_FAIL_RESIDUAL
    else:
        resid_naive = float(np.linalg.norm(X @ b_naive - y))
        ne_naive = float(np.linalg.norm(X.T @ X @ b_naive - X.T @ y))
    resid_qr = float(np.linalg.norm(X @ b_qr - y))
    ne_qr = float(np.linalg.norm(X.T @ X @ b_qr - X.T @ y))
    cond = float(np.linalg.cond(X.T @ X))
    return {
        "gt1": float(resid_naive - resid_qr),
        "gt2": {"cond_xtx": cond, "cond_x": float(np.linalg.cond(X))},
        "seed": int(seed),
        "diagnostics": {
            "resid_naive": resid_naive,
            "resid_qr": resid_qr,
            "normal_eq_resid_naive": ne_naive,
            "normal_eq_resid_qr": ne_qr,
            "naive_inversion_failed": inversion_failed,
            "beta_naive": b_naive.tolist(),
            "beta_qr": b_qr.tolist(),
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    out["gt1"] = -abs(float(out["gt1"]))
    out["gt2"] = {"cond_xtx": 1.0, "cond_x": 1.0}
    out["broken"] = True
    return out
