from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_optimism_positive_and_C_is_not_a_score() -> None:
    mods = load_problem_modules("ml/nested_cv_optimism")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt1"] > 0.0
    assert out["gt2"]["selected_C"] in [0.02, 0.1, 1.0, 50.0]
    assert out["diagnostics"]["inner_best_score"] > out["diagnostics"]["nested_outer_score"]


def test_seed_regenerates_gt1() -> None:
    mods = load_problem_modules("ml/nested_cv_optimism")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]


def test_broken_zero_optimism_fails() -> None:
    mods = load_problem_modules("ml/nested_cv_optimism")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)


def test_runner_ok() -> None:
    assert run_problem("ml/nested_cv_optimism").ok
