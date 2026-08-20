from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_auc_and_ece_disagree() -> None:
    mods = load_problem_modules("ml/calibration_vs_discrimination")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt1"] > 0.04
    assert out["gt2"]["ece_gap"] > 0.02
    assert out["diagnostics"]["auc_b"] > out["diagnostics"]["auc_a"]
    assert out["diagnostics"]["ece_b"] > out["diagnostics"]["ece_a"]


def test_seed_and_runner() -> None:
    mods = load_problem_modules("ml/calibration_vs_discrimination")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("ml/calibration_vs_discrimination").ok


def test_broken_fails() -> None:
    mods = load_problem_modules("ml/calibration_vs_discrimination")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)
