"""iid versus cluster bootstrap of a mean under a random intercept.

What problem is being solved?
    Show that resampling rows as if they were iid understates the variance of
    the grand mean when a cluster intercept is present.

What assumptions are required?
    iid clusters, iid rows given intercept. Estimand is E[Y].

Why was this method chosen?
    The mean is a linear functional, so an analytic cluster-robust variance
    exists as GT3.

What alternative method could have been used?
    A pairs cluster bootstrap of a slope, or a block bootstrap for time series.
    Those are different dependence models.

What can go wrong?
    Quoting the iid SE. Treating cluster bootstrap as a generic 'robust SE'.

How is correctness independently checked?
    Analytic CRVE for the mean. A ratio lower bound.

What can legitimately be concluded?
    On this DGP the iid bootstrap variance is too small.

What cannot be concluded?
    Coverage of a bootstrap percentile interval, or validity for serial dependence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng

G = 40
M = 15
N_BOOT = 400
A_SD = 1.0
E_SD = 0.35
MIN_RATIO = 1.8


def generate(seed: int = 2026) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = get_rng(seed)
    a = rng.normal(0.0, A_SD, size=G)
    y = np.empty(G * M)
    groups = np.empty(G * M, dtype=int)
    for g in range(G):
        sl = slice(g * M, (g + 1) * M)
        y[sl] = a[g] + rng.normal(0.0, E_SD, size=M)
        groups[sl] = g
    return y, groups


def iid_bootstrap_var(y: NDArray[np.float64], rng: np.random.Generator, n_boot: int = N_BOOT) -> float:
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = float(y[idx].mean())
    return float(np.var(stats, ddof=1))


def cluster_bootstrap_var(
    y: NDArray[np.float64], groups: NDArray[np.int64], rng: np.random.Generator, n_boot: int = N_BOOT
) -> float:
    uniq = np.unique(groups)
    g = len(uniq)
    stats = np.empty(n_boot)
    buckets = [y[groups == u] for u in uniq]
    for b in range(n_boot):
        draw = rng.integers(0, g, size=g)
        sample = np.concatenate([buckets[i] for i in draw])
        stats[b] = float(sample.mean())
    return float(np.var(stats, ddof=1))


def solve(seed: int = 2026) -> dict[str, object]:
    y, groups = generate(seed)
    rng = get_rng(seed + 99)
    v_iid = iid_bootstrap_var(y, rng)
    v_clu = cluster_bootstrap_var(y, groups, rng)
    ratio = float(v_clu / v_iid)
    return {
        "gt1": ratio,
        "gt2": {"ratio": ratio, "min_ratio": MIN_RATIO, "passed": ratio >= MIN_RATIO},
        "seed": int(seed),
        "diagnostics": {
            "var_iid": v_iid,
            "var_cluster": v_clu,
            "mean": float(y.mean()),
            "n": int(len(y)),
            "G": G,
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    out["gt1"] = 1.0
    out["gt2"] = {"ratio": 1.0, "min_ratio": MIN_RATIO, "passed": True}
    out["diagnostics"]["var_cluster"] = out["diagnostics"]["var_iid"]  # type: ignore[index]
    out["broken"] = True
    return out
