"""Stable log-sum-exp by max subtraction.

What problem is being solved?
    Evaluate log(sum(exp(x))) for vectors that overflow the naive formula.

What assumptions are required?
    float64. The reduction is a scalar log-sum-exp.

Why was this method chosen?
    Max subtraction is the standard stable reduction. It is not a clip.

What alternative method could have been used?
    scipy.special.logsumexp (used as GT3, not as the reference formula).

What can go wrong?
    Naive sum-exp. Subtracting the min. Overflow interpreted as a DGP issue.

How is correctness independently checked?
    Translation identity, SciPy, and a documented non-finite naive value.

What can legitimately be concluded?
    On this vector the naive formula is non-finite and the stable value is not.

What cannot be concluded?
    Behaviour of other reductions (log1p, hypot) or of GPU float32 kernels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng


def make_vector(seed: int = 2026) -> NDArray[np.float64]:
    rng = get_rng(seed)
    return 705.0 + rng.uniform(0.0, 18.0, size=12)


def naive_logsumexp(x: NDArray[np.float64]) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.log(np.sum(np.exp(x))))


def stable_logsumexp(x: NDArray[np.float64]) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def solve(seed: int = 2026) -> dict[str, object]:
    x = make_vector(seed)
    value = stable_logsumexp(x)
    shifts = [0.0, 10.0, -25.0, float(np.max(x))]
    ident_err = [
        abs(stable_logsumexp(x) - (c + stable_logsumexp(x - c))) for c in shifts
    ]
    naive = naive_logsumexp(x)
    return {
        "gt1": float(value),
        "gt2": {
            "max_identity_error": float(max(ident_err)),
            "identity_errors": ident_err,
        },
        "seed": int(seed),
        "diagnostics": {
            "naive_value": naive,
            "naive_is_finite": bool(np.isfinite(naive)),
            "x_max": float(np.max(x)),
            "x_min": float(np.min(x)),
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    x = make_vector(seed)
    naive = naive_logsumexp(x)
    return {
        "gt1": float(naive) if np.isfinite(naive) else 0.0,
        "gt2": {"max_identity_error": 1.0, "identity_errors": [1.0]},
        "seed": int(seed),
        "diagnostics": {"naive_is_finite": True},
        "broken": True,
    }
