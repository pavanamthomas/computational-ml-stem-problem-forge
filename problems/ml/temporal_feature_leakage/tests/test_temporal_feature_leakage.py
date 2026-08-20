from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_r2_gap_and_future_sentinel() -> None:
    mods = load_problem_modules("ml/temporal_feature_leakage")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt1"] > 0.12
    assert out["gt2"]["leaked_future_source_fraction"] > 0.99
    assert out["gt2"]["lagged_future_source_fraction"] == 0.0
    assert out["gt2"]["corr_leaked_with_y_lead1"] > 0.35


def test_seed_regenerates_gt1() -> None:
    mods = load_problem_modules("ml/temporal_feature_leakage")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]


def test_broken_hides_sentinel() -> None:
    mods = load_problem_modules("ml/temporal_feature_leakage")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)


def test_runner_ok() -> None:
    assert run_problem("ml/temporal_feature_leakage").ok
