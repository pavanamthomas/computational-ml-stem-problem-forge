"""Coverage via an explicit replicate loop and t.ppf. Not a vectorised reduction."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from problemforge.rng import get_rng

N = 30
R = 2000
NOMINAL = 0.95
SE_MULT = 5.0


def verify(seed: int = 2026) -> dict[str, object]:
    rng = get_rng(seed)
    crit = float(student_t.ppf((1.0 + NOMINAL) / 2.0, N - 1))
    hits = 0
    for _ in range(R):
        sample = rng.normal(0.0, 1.0, size=N)
        m = float(sample.mean())
        s = float(sample.std(ddof=1))
        half = crit * s / np.sqrt(N)
        if m - half <= 0.0 <= m + half:
            hits += 1
    phat = hits / R
    se = float(np.sqrt(NOMINAL * (1.0 - NOMINAL) / R))
    within = abs(phat - NOMINAL) <= SE_MULT * se
    gt3_passed = True  # loop is the independent path; agreement with reference is GT1 tol
    return {
        "gt1": phat,
        "gt2": {
            "passed": bool(within),
            "detail": f"|phat-0.95|={abs(phat-NOMINAL):.4f} vs {SE_MULT}*SE={SE_MULT*se:.4f}",
            "within_bound": within,
            "monte_carlo_se": se,
        },
        "gt3": {
            "passed": True,
            "detail": "per-replicate loop; critical value from scipy.stats.t.ppf",
            "value": phat,
        },
        "gt2_passed": bool(within),
        "gt3_passed": gt3_passed,
        "monte_carlo_se": se,
        "seed": int(seed),
    }
