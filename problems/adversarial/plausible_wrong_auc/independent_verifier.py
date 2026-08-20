"""Leak sentinels that do not trust sklearn's AUC of the leaked score as truth."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng

N = 900
P = 6
COEF = 0.18


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _mw_auc(y: NDArray[np.int64], s: NDArray[np.float64]) -> float:
    pos = s[y == 1]
    neg = s[y == 0]
    n1, n0 = len(pos), len(neg)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def verify(seed: int = 2026) -> dict[str, object]:
    rng = get_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, P))
    y = rng.binomial(1, _sigmoid(COEF * X[:, 0])).astype(int)
    # Simple split: last 34%
    cut = int(0.66 * N)
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]
    # Logit MLE on first column plus intercept — independent of sklearn AUC.
    z = np.column_stack([np.ones(len(Xtr)), Xtr[:, 0]])
    beta = np.zeros(2)
    for _ in range(10):
        p = _sigmoid(z @ beta)
        w = p * (1 - p)
        hess = z.T @ (w[:, None] * z) + 1e-8 * np.eye(2)
        beta = beta + np.linalg.solve(hess, z.T @ (ytr - p))
    zte = np.column_stack([np.ones(len(Xte)), Xte[:, 0]])
    legit = _sigmoid(zte @ beta)
    auc_legit = _mw_auc(yte, legit)
    noise = get_rng(seed + 3).normal(0.0, 0.32, size=len(yte))
    leaked = 0.22 * legit + 0.62 * yte.astype(float) + noise
    auc_leaked = _mw_auc(yte, leaked)
    # Residualise leaked score on Xte; leftover corr with y is the leak.
    Xd = np.column_stack([np.ones(len(Xte)), Xte])
    coef, *_ = np.linalg.lstsq(Xd, leaked, rcond=None)
    resid = leaked - Xd @ coef
    # Point-biserial / Pearson with y
    leak_corr = float(np.corrcoef(resid, yte)[0, 1])
    auc_leaked_ok_range = 0.82 <= auc_leaked <= 0.97
    gt2_passed = auc_leaked_ok_range and auc_legit < 0.80
    gt3_passed = leak_corr > 0.35 and auc_legit < auc_leaked - 0.12
    return {
        "gt1": float(auc_legit),
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": (
                f"leaked AUC={auc_leaked:.3f} in the convincing band; "
                f"legitimate AUC={auc_legit:.3f}"
            ),
            "auc_leaked": auc_leaked,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": f"residual corr(leaked ⊥ X, y)={leak_corr:.3f} (Mann-Whitney AUC)",
            "leak_residual_corr": leak_corr,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
    }
