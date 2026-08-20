from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_naive_worse_residual_huge_cond() -> None:
    mods = load_problem_modules("numerical/ill_conditioned_normal_equations")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt1"] > 0
    assert out["gt2"]["cond_xtx"] > 1e8
    assert out["diagnostics"]["resid_naive"] > out["diagnostics"]["resid_qr"]


def test_seed_and_runner() -> None:
    mods = load_problem_modules("numerical/ill_conditioned_normal_equations")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("numerical/ill_conditioned_normal_equations").ok


def test_broken_fails() -> None:
    mods = load_problem_modules("numerical/ill_conditioned_normal_equations")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)
