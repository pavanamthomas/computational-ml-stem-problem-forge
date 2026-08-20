"""Monte Carlo coverage of a Student-t mean interval.

What problem is being solved?
    Estimate the repeated-sampling coverage of the 95% t-interval for a Normal
    mean, and judge the estimate by its Monte Carlo SE.

What assumptions are required?
    iid N(0,1), n=30, R=2000, two-sided t interval.

Why was this method chosen?
    Coverage is a Monte Carlo functional. Exact equality to 0.95 is the wrong
    null.

What alternative method could have been used?
    A z-interval, or a bootstrap percentile interval. Different objects.

What can go wrong?
    Declaring failure because phat=0.946. Interpreting one interval as a
    posterior probability.

How is correctness independently checked?
    A loop with t.ppf. The SE bound |phat-0.95| <= 5 * sqrt(p(1-p)/R).

What can legitimately be concluded?
    Under this DGP the simulated coverage is consistent with 95% within MC error.

What cannot be concluded?
    Coverage under a skewed parent, or for a single realised interval.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from problemforge.rng import get_rng

N = 30
R = 2000
NOMINAL = 0.95
SE_MULT = 5.0


def solve(seed: int = 2026) -> dict[str, object]:
    rng = get_rng(seed)
    draws = rng.normal(0.0, 1.0, size=(R, N))
    means = draws.mean(axis=1)
    s = draws.std(axis=1, ddof=1)
    crit = float(stats.t.ppf(0.5 + NOMINAL / 2.0, df=N - 1))
    half = crit * s / np.sqrt(N)
    covered = (means - half <= 0.0) & (means + half >= 0.0)
    phat = float(covered.mean())
    se = float(np.sqrt(NOMINAL * (1.0 - NOMINAL) / R))
    return {
        "gt1": phat,
        "gt2": {
            "abs_err_vs_nominal": abs(phat - NOMINAL),
            "monte_carlo_se": se,
            "bound": SE_MULT * se,
            "within_bound": bool(abs(phat - NOMINAL) <= SE_MULT * se),
        },
        "monte_carlo_se": se,
        "seed": int(seed),
        "diagnostics": {"n": N, "R": R, "critical_t": crit, "nominal": NOMINAL},
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    out["gt1"] = 1.0
    out["gt2"] = {
        "abs_err_vs_nominal": 0.05,
        "monte_carlo_se": 0.0,
        "bound": 0.0,
        "within_bound": True,
    }
    out["broken"] = True
    return out
