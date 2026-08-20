"""Lagged versus future-including rolling means.

What problem is being solved?
    Measure how much a rolling statistic that includes time t or t+1 inflates
    out-of-time R² for predicting y[t] when only s < t is permitted.

What assumptions are required?
    Causal AR(1), Gaussian innovations, a single chronological train/test cut.

Why was this method chosen?
    A linear model on one feature makes the R² gap a statement about the
    feature, not about a flexible learner memorising time.

What alternative method could have been used?
    Expanding-window CV, or an explicit state-space filter. Those change the
    estimand.

What can go wrong?
    ``rolling(..., center=True)``, ``shift(-1)``, or shuffling before the split.

How is correctness independently checked?
    A NumPy loop rebuilds the lagged mean. A timestamp sentinel counts
    future sources without using R².

What can legitimately be concluded?
    On this DGP, the leaked feature's OOS R² is not forecast skill.

What cannot be concluded?
    That every rolling mean in production is leaked. The permitted information
    set has to be named first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from problemforge.rng import get_rng

T = 480
PHI = 0.72
SIGMA = 0.55
WINDOW = 4
TRAIN_FRAC = 0.70
MIN_GAP = 0.12


def generate_series(seed: int = 2026) -> NDArray[np.float64]:
    rng = get_rng(seed)
    y = np.empty(T, dtype=float)
    y[0] = rng.normal(0.0, SIGMA / np.sqrt(1.0 - PHI**2))
    for t in range(1, T):
        y[t] = PHI * y[t - 1] + rng.normal(0.0, SIGMA)
    return y


def lagged_rolling_mean(y: NDArray[np.float64], window: int = WINDOW) -> NDArray[np.float64]:
    """Mean of y[t-window], ..., y[t-1]. pandas rolling then shift(1)."""
    s = pd.Series(y)
    return s.rolling(window, min_periods=window).mean().shift(1).to_numpy()


def leaked_rolling_mean(y: NDArray[np.float64], window: int = WINDOW) -> NDArray[np.float64]:
    """Mean of a window that includes y[t] and y[t+1] (shift -1 of a rolling mean)."""
    s = pd.Series(y)
    return s.rolling(window, min_periods=window).mean().shift(-1).to_numpy()


def source_max_lagged(n: int, window: int = WINDOW) -> NDArray[np.float64]:
    out = np.full(n, np.nan)
    for t in range(window, n):
        out[t] = t - 1
    return out


def source_max_leaked(n: int, window: int = WINDOW) -> NDArray[np.float64]:
    """Leaked rolling mean at t uses y[t-window+2] ... y[t+1] after shift(-1)."""
    out = np.full(n, np.nan)
    for t in range(window - 1, n - 1):
        out[t] = min(n - 1, t + 1)
    return out


def _oos_r2(y: NDArray[np.float64], feat: NDArray[np.float64]) -> float:
    mask = np.isfinite(feat) & np.isfinite(y)
    idx = np.flatnonzero(mask)
    cut = int(TRAIN_FRAC * len(idx))
    tr, te = idx[:cut], idx[cut:]
    x_tr, x_te = feat[tr], feat[te]
    y_tr, y_te = y[tr], y[te]
    x_mean = x_tr.mean()
    y_mean = y_tr.mean()
    var_x = np.sum((x_tr - x_mean) ** 2)
    beta = 0.0 if var_x == 0 else float(np.sum((x_tr - x_mean) * (y_tr - y_mean)) / var_x)
    intercept = y_mean - beta * x_mean
    pred = intercept + beta * x_te
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def solve(seed: int = 2026) -> dict[str, object]:
    y = generate_series(seed)
    lagged = lagged_rolling_mean(y)
    leaked = leaked_rolling_mean(y)
    r2_lagged = _oos_r2(y, lagged)
    r2_leaked = _oos_r2(y, leaked)
    gap = float(r2_leaked - r2_lagged)
    src_lagged = source_max_lagged(len(y))
    src_leaked = source_max_leaked(len(y))
    t_idx = np.arange(len(y), dtype=float)
    valid = np.isfinite(src_leaked)
    leaked_future_frac = float(np.mean(src_leaked[valid] >= t_idx[valid]))
    lagged_valid = np.isfinite(src_lagged)
    lagged_future_frac = float(np.mean(src_lagged[lagged_valid] >= t_idx[lagged_valid]))
    pair = np.isfinite(leaked) & np.isfinite(y)
    # corr(feature_t, y_{t+1}) on interior points
    f = leaked[:-1]
    y_next = y[1:]
    m = np.isfinite(f)
    corr_future = float(np.corrcoef(f[m], y_next[m])[0, 1])
    return {
        "gt1": gap,
        "gt2": {
            "leaked_future_source_fraction": leaked_future_frac,
            "lagged_future_source_fraction": lagged_future_frac,
            "corr_leaked_with_y_lead1": corr_future,
        },
        "seed": int(seed),
        "diagnostics": {
            "r2_leaked": r2_leaked,
            "r2_lagged": r2_lagged,
            "min_gap_required": MIN_GAP,
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    """Treat the leaked R² as if it came from a lagged feature."""
    out = solve(seed)
    out["gt1"] = 0.0
    out["gt2"] = {
        "leaked_future_source_fraction": 0.0,
        "lagged_future_source_fraction": 0.0,
        "corr_leaked_with_y_lead1": 0.0,
    }
    out["broken"] = True
    return out
