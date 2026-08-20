"""SVD / lstsq path. Does not form (X'X)^{-1}."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd as scipy_svd

from problemforge.rng import get_rng

N = 12
P = 8


INVERSION_FAIL_RESIDUAL = 1.0e16


def _design(seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = get_rng(seed)
    i = np.arange(1, N + 1)[:, None]
    j = np.arange(1, P + 1)[None, :]
    X = 1.0 / (i + j - 1.0)
    beta_true = np.array([1.0, -0.5, 0.25, -0.1, 0.05, -0.02, 0.01, -0.005])
    y = X @ beta_true + 1e-10 * rng.normal(size=N)
    return X, y


def _svd_solve(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    U, s, Vt = scipy_svd(X, full_matrices=False)
    # Truncate only true zeros; keep small singular values (this is still SVD LS).
    s_inv = np.array([1.0 / si if si > 1e-18 else 0.0 for si in s])
    return (Vt.T * s_inv) @ (U.T @ y)


def verify(seed: int = 2026) -> dict[str, object]:
    X, y = _design(seed)
    b_svd = _svd_solve(X, y)
    b_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    gram = X.T @ X
    try:
        b_naive = np.linalg.inv(gram) @ (X.T @ y)
        r_naive = float(np.linalg.norm(X @ b_naive - y))
    except np.linalg.LinAlgError:
        r_naive = INVERSION_FAIL_RESIDUAL
    r_svd = float(np.linalg.norm(X @ b_svd - y))
    r_lstsq = float(np.linalg.norm(X @ b_lstsq - y))
    cond = float(np.linalg.cond(gram))
    gap = r_naive - r_svd
    gt2_passed = cond > 1e8
    gt3_passed = r_svd < r_naive and r_lstsq < r_naive
    return {
        "gt1": float(gap),
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": f"cond(X'X)={cond:.3e}",
            "cond_xtx": cond,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": (
                f"SVD resid={r_svd:.3e}, lstsq resid={r_lstsq:.3e}, "
                f"naive resid={r_naive:.3e}"
            ),
            "resid_svd": r_svd,
            "resid_lstsq": r_lstsq,
            "resid_naive": r_naive,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
    }
