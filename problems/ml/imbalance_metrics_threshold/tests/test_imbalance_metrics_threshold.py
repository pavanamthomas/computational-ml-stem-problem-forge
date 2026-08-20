from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_majority_identity_and_metric_gap() -> None:
    mods = load_problem_modules("ml/imbalance_metrics_threshold")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt1"] > 0.15
    assert out["gt2"]["majority_recall"] == 0.0
    assert out["gt2"]["accuracy_identity_error"] < 1e-12
    assert 0.01 <= out["gt2"]["prevalence"] <= 0.04
    assert out["diagnostics"]["metrics_threshold_0.5"]["accuracy"] > 0.9


def test_cost_threshold_not_default_half() -> None:
    mods = load_problem_modules("ml/imbalance_metrics_threshold")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["diagnostics"]["cost_at_cost_threshold"] <= out["diagnostics"]["cost_at_0.5"]


def test_seed_and_runner() -> None:
    mods = load_problem_modules("ml/imbalance_metrics_threshold")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("ml/imbalance_metrics_threshold").ok


def test_broken_fails() -> None:
    mods = load_problem_modules("ml/imbalance_metrics_threshold")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert mods.invariants.check_invariants(broken, ver)
