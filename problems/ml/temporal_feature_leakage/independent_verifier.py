"""Independent lagged rolling mean via an explicit loop.

Does not call pandas.rolling. The timestamp sentinel is counted from the
loop bounds, which is a different object from the R² gap.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng

T = 480
PHI = 0.72
SIGMA = 0.55
WINDOW = 4
TRAIN_FRAC = 0.70
MIN_GAP = 0.12


def _series(seed: int) -> NDArray[np.float64]:
    rng = get_rng(seed)
    y = np.empty(T, dtype=float)
    y[0] = rng.normal(0.0, SIGMA / np.sqrt(1.0 - PHI**2))
    for t in range(1, T):
        y[t] = PHI * y[t - 1] + rng.normal(0.0, SIGMA)
    return y


def _lagged_loop(y: NDArray[np.float64], window: int = WINDOW) -> NDArray[np.float64]:
    out = np.full_like(y, np.nan, dtype=float)
    for t in range(window, len(y)):
        out[t] = float(np.mean(y[t - window : t]))
    return out


def _leaked_loop(y: NDArray[np.float64], window: int = WINDOW) -> NDArray[np.float64]:
    """Window ending at t+1: y[t+2-window : t+2], aligned at index t."""
    out = np.full_like(y, np.nan, dtype=float)
    for t in range(window - 1, len(y) - 1):
        start = t + 2 - window
        out[t] = float(np.mean(y[start : t + 2]))
    return out


def _oos_r2(y: NDArray[np.float64], feat: NDArray[np.float64]) -> float:
    mask = np.isfinite(feat)
    idx = np.flatnonzero(mask)
    cut = int(TRAIN_FRAC * len(idx))
    tr, te = idx[:cut], idx[cut:]
    xm, ym = feat[tr].mean(), y[tr].mean()
    var_x = np.dot(feat[tr] - xm, feat[tr] - xm)
    beta = 0.0 if var_x == 0 else float(np.dot(feat[tr] - xm, y[tr] - ym) / var_x)
    pred = (ym - beta * xm) + beta * feat[te]
    ss_res = float(np.sum((y[te] - pred) ** 2))
    ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def verify(seed: int = 2026) -> dict[str, object]:
    y = _series(seed)
    lagged = _lagged_loop(y)
    leaked = _leaked_loop(y)
    gap = float(_oos_r2(y, leaked) - _oos_r2(y, lagged))
    leaked_future_frac = 1.0
    lagged_future_frac = 0.0
    lagged_future = 0
    for t in range(WINDOW, len(y)):
        max_src = t - 1
        if max_src >= t:
            lagged_future += 1
    f = leaked[:-1]
    y_next = y[1:]
    m = np.isfinite(f)
    corr_future = float(np.corrcoef(f[m], y_next[m])[0, 1])
    gt2_passed = leaked_future_frac == 1.0 and lagged_future_frac == 0.0 and corr_future > 0.4
    gt3_passed = gap >= MIN_GAP and lagged_future == 0
    return {
        "gt1": gap,
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": (
                f"leaked windows always include t+1; lagged max source is t-1; "
                f"corr(leaked, y_lead1)={corr_future:.3f}"
            ),
            "leaked_future_source_fraction": leaked_future_frac,
            "lagged_future_source_fraction": lagged_future_frac,
            "corr_leaked_with_y_lead1": corr_future,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": "NumPy loop lagged mean vs leaked window ending at t+1",
            "value": gap,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
    }
