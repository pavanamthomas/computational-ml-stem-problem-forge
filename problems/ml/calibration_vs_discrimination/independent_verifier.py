"""Mann-Whitney AUC and ECE from bin counts. No sklearn.metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng

N = 4000
N_BINS = 10


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _draws(seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = get_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=N)
    x2 = rng.normal(0.0, 1.0, size=N)
    y = rng.binomial(1, _sigmoid(0.35 * x1 + 0.90 * x2)).astype(float)
    p_a = np.clip(_sigmoid(0.35 * x1), 1e-6, 1 - 1e-6)
    p_b = np.clip(_sigmoid(6.0 * (0.35 * x1 + 0.90 * x2)), 1e-6, 1 - 1e-6)
    return y, p_a, p_b


def mw_auc(y: NDArray[np.float64], s: NDArray[np.float64]) -> float:
    pos = s[y == 1]
    neg = s[y == 0]
    n1, n0 = len(pos), len(neg)
    # Vectorised: ranks via argsort
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(y: NDArray[np.float64], p: NDArray[np.float64]) -> float:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    total = 0.0
    for k in range(N_BINS):
        mask = (p >= edges[k]) & (p <= edges[k + 1] if k == N_BINS - 1 else p < edges[k + 1])
        if not np.any(mask):
            continue
        total += (mask.sum() / len(p)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def verify(seed: int = 2026) -> dict[str, object]:
    y, p_a, p_b = _draws(seed)
    auc_a = mw_auc(y, p_a)
    auc_b = mw_auc(y, p_b)
    ece_a = ece(y, p_a)
    ece_b = ece(y, p_b)
    gap = float(auc_b - auc_a)
    gt2_passed = ece_b > ece_a + 0.02
    gt3_passed = auc_b > auc_a + 0.03
    return {
        "gt1": gap,
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": f"ECE_B={ece_b:.4f} > ECE_A={ece_a:.4f} (weighted |acc-conf|)",
            "ece_a": ece_a,
            "ece_b": ece_b,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": "Mann-Whitney rank AUC; B discriminates more than A",
            "auc_a": auc_a,
            "auc_b": auc_b,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
    }
