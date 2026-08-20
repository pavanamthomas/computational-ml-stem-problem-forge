"""SciPy log-sum-exp versus an independent compensated loop."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp as scipy_logsumexp

from problemforge.rng import get_rng


def _vector(seed: int) -> NDArray[np.float64]:
    rng = get_rng(seed)
    return 705.0 + rng.uniform(0.0, 18.0, size=12)


def _loop_lse(x: NDArray[np.float64]) -> float:
    m = x[0]
    for v in x[1:]:
        if v > m:
            m = float(v)
    acc = 0.0
    for v in x:
        acc += float(np.exp(v - m))
    return float(m + np.log(acc))


def verify(seed: int = 2026) -> dict[str, object]:
    x = _vector(seed)
    loop = _loop_lse(x)
    sci = float(scipy_logsumexp(x))
    c = 33.0
    ident = abs(loop - (c + _loop_lse(x - c)))
    with np.errstate(over="ignore", invalid="ignore"):
        naive = float(np.log(np.sum(np.exp(x))))
    gt2_passed = ident < 1e-10
    gt3_passed = abs(loop - sci) < 1e-10 and not np.isfinite(naive)
    return {
        "gt1": loop,
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": f"shift identity error={ident:.3e}",
            "max_identity_error": ident,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": f"|loop-scipy|={abs(loop-sci):.3e}; naive finite={np.isfinite(naive)}",
            "scipy": sci,
            "loop": loop,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
    }
