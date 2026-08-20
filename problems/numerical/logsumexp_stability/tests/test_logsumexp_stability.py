from __future__ import annotations

import numpy as np

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_naive_overflow_stable_finite() -> None:
    mods = load_problem_modules("numerical/logsumexp_stability")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert np.isfinite(out["gt1"])
    assert not out["diagnostics"]["naive_is_finite"]
    assert out["gt2"]["max_identity_error"] < 1e-10


def test_seed_and_runner() -> None:
    mods = load_problem_modules("numerical/logsumexp_stability")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("numerical/logsumexp_stability").ok


def test_broken_fails() -> None:
    mods = load_problem_modules("numerical/logsumexp_stability")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)
