"""Analytic cluster-robust variance of the mean. Not a bootstrap."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng

G = 40
M = 15
A_SD = 1.0
E_SD = 0.35
MIN_RATIO = 1.8
N_BOOT = 400


def _data(seed: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = get_rng(seed)
    a = rng.normal(0.0, A_SD, size=G)
    y = np.empty(G * M)
    groups = np.empty(G * M, dtype=int)
    for g in range(G):
        sl = slice(g * M, (g + 1) * M)
        y[sl] = a[g] + rng.normal(0.0, E_SD, size=M)
        groups[sl] = g
    return y, groups


def cluster_robust_var_mean(y: NDArray[np.float64], groups: NDArray[np.int64]) -> float:
    """Sandwich variance of the grand mean with cluster sums of residuals."""
    n = len(y)
    ybar = float(y.mean())
    g = int(np.unique(groups).size)
    acc = 0.0
    for u in np.unique(groups):
        resid = y[groups == u] - ybar
        acc += float(resid.sum() ** 2)
    return float((g / (g - 1)) * acc / n**2)


def iid_var_mean(y: NDArray[np.float64]) -> float:
    n = len(y)
    return float(y.var(ddof=1) / n)


def verify(seed: int = 2026) -> dict[str, object]:
    y, groups = _data(seed)
    v_cr = cluster_robust_var_mean(y, groups)
    v_iid = iid_var_mean(y)
    ratio = float(v_cr / v_iid)
    # Independent cluster bootstrap with a different stream (seed+7) as a check
    rng = get_rng(seed + 7)
    uniq = np.unique(groups)
    buckets = [y[groups == u] for u in uniq]
    stats = []
    for _ in range(N_BOOT):
        draw = rng.integers(0, G, size=G)
        stats.append(float(np.concatenate([buckets[i] for i in draw]).mean()))
    v_boot = float(np.var(stats, ddof=1))
    gt2_passed = ratio >= MIN_RATIO
    gt3_passed = v_cr > v_iid * MIN_RATIO * 0.7 and abs(np.log(v_cr) - np.log(v_boot)) < 1.2
    return {
        "gt1": ratio,
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": f"analytic CRVE/iid = {ratio:.3f} (min {MIN_RATIO})",
            "ratio": ratio,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": (
                f"CRVE={v_cr:.6f}, iid formula={v_iid:.6f}, "
                f"independent cluster boot={v_boot:.6f}"
            ),
            "analytic_crve": v_cr,
            "iid_formula": v_iid,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
    }
