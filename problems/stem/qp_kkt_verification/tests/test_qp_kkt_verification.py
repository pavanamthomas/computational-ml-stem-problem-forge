from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_feasible_and_kkt() -> None:
    mods = load_problem_modules("stem/qp_kkt_verification")
    out = mods.reference.solve(seed=DEFAULT_SEED)
    assert out["gt2"]["max_ineq_violation"] < 1e-7
    assert out["gt2"]["max_bound_violation"] < 1e-7
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    assert ver["gt3"]["passed"]


def test_seed_and_runner() -> None:
    mods = load_problem_modules("stem/qp_kkt_verification")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("stem/qp_kkt_verification").ok


def test_shifted_x_fails_kkt() -> None:
    mods = load_problem_modules("stem/qp_kkt_verification")
    broken = mods.reference.deliberately_broken_solve(seed=DEFAULT_SEED)
    ver = mods.verifier.verify(seed=DEFAULT_SEED)
    # broken x is not the verifier x; invariants on broken reference should fail feasibility or obj match
    failures = mods.invariants.check_invariants(broken, ver)
    assert failures
