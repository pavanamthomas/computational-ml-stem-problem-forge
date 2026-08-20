from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_gap_is_substantially_positive() -> None:
    mods = load_problem_modules("ml/entity_group_leakage")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt1"] > 0.08
    assert out["diagnostics"]["naive_cv_acc"] > out["diagnostics"]["grouped_cv_acc"]


def test_leakage_sentinel_naive_positive_grouped_zero() -> None:
    mods = load_problem_modules("ml/entity_group_leakage")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    sentinel = out["gt2"]
    assert sentinel["naive_leaked_entity_occurrences"] > 0
    assert sentinel["grouped_leaked_entity_occurrences"] == 0.0


def test_seed_regenerates_gt1() -> None:
    mods = load_problem_modules("ml/entity_group_leakage")
    a = mods.reference.solve(seed=DEFAULT_SEED)["gt1"]
    b = mods.reference.solve(seed=DEFAULT_SEED)["gt1"]
    assert a == b


def test_broken_solver_fails_invariants() -> None:
    mods = load_problem_modules("ml/entity_group_leakage")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    honest = mods.reference.solve(seed=DEFAULT_SEED)
    failures = mods.invariants.check_invariants(broken, mods.verifier.verify(seed=DEFAULT_SEED))
    assert failures
    assert honest["gt1"] > broken["gt1"]


def test_runner_ok() -> None:
    report = run_problem("ml/entity_group_leakage")
    assert report.ok
    assert all(c.passed for c in report.unit_checks)
