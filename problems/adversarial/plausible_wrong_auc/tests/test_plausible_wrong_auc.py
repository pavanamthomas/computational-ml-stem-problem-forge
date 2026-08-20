from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_leaked_auc_convincing_legitimate_lower() -> None:
    mods = load_problem_modules("adversarial/plausible_wrong_auc")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert 0.82 <= out["gt2"]["auc_leaked"] <= 0.97
    assert out["gt1"] < out["gt2"]["auc_leaked"] - 0.08
    assert out["gt2"]["auc_leaked_perm_X"] > 0.80
    assert out["gt2"]["auc_legit_perm_X"] < 0.70


def test_seed_and_runner() -> None:
    mods = load_problem_modules("adversarial/plausible_wrong_auc")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("adversarial/plausible_wrong_auc").ok


def test_broken_reports_leaked_as_gt1() -> None:
    mods = load_problem_modules("adversarial/plausible_wrong_auc")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)
