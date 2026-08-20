from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_cluster_var_larger() -> None:
    mods = load_problem_modules("stats/bootstrap_dependence_trap")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt1"] > 1.8
    assert out["diagnostics"]["var_cluster"] > out["diagnostics"]["var_iid"]


def test_seed_and_runner() -> None:
    mods = load_problem_modules("stats/bootstrap_dependence_trap")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("stats/bootstrap_dependence_trap").ok


def test_broken_fails() -> None:
    mods = load_problem_modules("stats/bootstrap_dependence_trap")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)
