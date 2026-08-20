from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem
from problemforge.validate import check_tolerance


def test_coverage_near_nominal_not_exact() -> None:
    mods = load_problem_modules("stats/monte_carlo_ci_coverage")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    phat = out["gt1"]
    se = out["monte_carlo_se"]
    assert abs(phat - 0.95) <= 5 * se
    assert 0.92 < phat < 0.98


def test_seed_regenerates() -> None:
    mods = load_problem_modules("stats/monte_carlo_ci_coverage")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]


def test_runner_ok() -> None:
    assert run_problem("stats/monte_carlo_ci_coverage").ok


def test_se_policy_boundaries() -> None:
    policy = {"kind": "monte_carlo_se", "se_mult": 5.0}
    se = 0.01
    # just inside
    assert check_tolerance(0.95 + 5.0 * se, 0.95, policy, monte_carlo_se=se).passed
    # just outside
    assert not check_tolerance(
        0.95 + 5.0 * se + 1e-12, 0.95, policy, monte_carlo_se=se
    ).passed
